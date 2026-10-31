/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

//! Peak KV need over the most recent requests. Trimming the mapped KV to this peak, instead of to the current request, keeps a
//! short request between long conversation turns from shrinking the pool the next long turn needs again.
class KvNeedWindow
{
public:
    using Clock = std::chrono::steady_clock;

    //! @param maxRequests Entries kept; values below 1 are treated as 1 (the current request alone).
    //! @param maxAge Entries older than this are dropped when a new one is recorded; the new entry is always kept.
    KvNeedWindow(size_t maxRequests, std::chrono::seconds maxAge)
        : mMaxRequests(std::max<size_t>(maxRequests, 1))
        , mMaxAge(maxAge)
    {
    }

    //! Record a request's need in tokens and return the peak need within the window, this request included.
    int32_t record(Clock::time_point now, int32_t tokens)
    {
        mEntries.emplace_back(now, std::max(tokens, 0));
        while (mEntries.size() > mMaxRequests || (mEntries.size() > 1 && now - mEntries.front().first > mMaxAge))
        {
            mEntries.pop_front();
        }
        return peak();
    }

    int32_t peak() const
    {
        int32_t result = 0;
        for (auto const& entry : mEntries)
        {
            result = std::max(result, entry.second);
        }
        return result;
    }

    size_t size() const
    {
        return mEntries.size();
    }

private:
    size_t mMaxRequests;
    std::chrono::seconds mMaxAge;
    std::deque<std::pair<Clock::time_point, int32_t>> mEntries;
};

//! Integer from the environment; @p fallback when unset, malformed or below @p minimum.
inline int64_t kvEnvInteger(char const* name, int64_t fallback, int64_t minimum)
{
    char const* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return fallback;
    }
    char* end = nullptr;
    long long const parsed = std::strtoll(value, &end, 10);
    return (end != value && *end == '\0' && parsed >= minimum) ? parsed : fallback;
}

//! Window configured by EDGELLM_KV_SHRINK_WINDOW_REQUESTS (default 32, at least 1) and EDGELLM_KV_SHRINK_WINDOW_SECONDS (default 900).
inline KvNeedWindow makeKvNeedWindowFromEnv()
{
    return KvNeedWindow(static_cast<size_t>(kvEnvInteger("EDGELLM_KV_SHRINK_WINDOW_REQUESTS", 32, 1)),
        std::chrono::seconds(kvEnvInteger("EDGELLM_KV_SHRINK_WINDOW_SECONDS", 900, 0)));
}

} // namespace rt
} // namespace trt_edgellm
