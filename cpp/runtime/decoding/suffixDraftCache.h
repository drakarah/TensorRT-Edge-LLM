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
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Model-free draft source: an n-gram index over generated tokens of all requests (outputs only), proposing
//! the continuation of the most recent earlier occurrence of the context's last n tokens (n = 4, 3, 2,
//! longest first). Structured outputs repeat across requests (same schemas, keys, SQL shapes), so whole
//! stretches can be proposed at once, beyond what a fixed-depth model drafter reaches in one round.
class SuffixDraftCache
{
public:
    static constexpr int32_t kSeparator{-1}; //!< Between requests: no n-gram spans two requests.

    //! Up to 3 index entries per token (~56 B each): the default 2^18 tokens (~2,400 app answers) stays under
    //! ~45 MB of host memory, which Jetson shares with the GPU. When full, the cache starts over.
    explicit SuffixDraftCache(size_t maxTokens = size_t{1} << 18)
        : mMaxTokens(maxTokens)
    {
    }

    //! Start a new request: later n-grams do not continue the previous one.
    void beginSequence()
    {
        if (!mTokens.empty() && mTokens.back() != kSeparator)
        {
            append(kSeparator);
        }
    }

    void append(int32_t token)
    {
        if (mTokens.size() >= mMaxTokens)
        {
            mTokens.clear();
            mLast.clear();
        }
        mTokens.push_back(token);
        int64_t const end = static_cast<int64_t>(mTokens.size());
        for (int32_t n : kNgrams)
        {
            if (end < n)
            {
                continue;
            }
            uint64_t const key = hashNgram(mTokens.data() + end - n, n);
            if (key == kInvalidKey)
            {
                continue;
            }
            auto& entry = mLast[key];
            entry.second = entry.first;
            entry.first = end;
        }
    }

    //! Up to maxLen tokens that followed the most recent earlier occurrence of context's last n tokens.
    //! The context's own end (the cache's current end, when the context is the sequence just appended) is
    //! skipped, so a sequence never proposes from itself at its own position.
    std::vector<int32_t> propose(int32_t const* context, int32_t contextLen, int32_t maxLen) const
    {
        std::vector<int32_t> proposal;
        int64_t const ownEnd = static_cast<int64_t>(mTokens.size());
        for (int32_t n : kNgrams)
        {
            if (contextLen < n)
            {
                continue;
            }
            uint64_t const key = hashNgram(context + contextLen - n, n);
            auto const it = mLast.find(key);
            if (key == kInvalidKey || it == mLast.end())
            {
                continue;
            }
            int64_t const end = it->second.first == ownEnd ? it->second.second : it->second.first;
            if (end <= 0 || !matches(end, context + contextLen - n, n))
            {
                continue;
            }
            for (int64_t pos = end; pos < ownEnd && static_cast<int32_t>(proposal.size()) < maxLen; ++pos)
            {
                if (mTokens[pos] == kSeparator)
                {
                    break;
                }
                proposal.push_back(mTokens[pos]);
            }
            if (!proposal.empty())
            {
                return proposal;
            }
        }
        return proposal;
    }

private:
    static constexpr int32_t kNgrams[] = {4, 3, 2};
    static constexpr uint64_t kInvalidKey{0};

    //! Order-sensitive hash of n tokens with n mixed in; 0 marks an n-gram that spans a separator.
    static uint64_t hashNgram(int32_t const* tokens, int32_t n)
    {
        uint64_t hash = 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(n);
        for (int32_t i = 0; i < n; ++i)
        {
            if (tokens[i] == kSeparator)
            {
                return kInvalidKey;
            }
            hash ^= static_cast<uint64_t>(static_cast<uint32_t>(tokens[i])) + 0x9E3779B97F4A7C15ULL + (hash << 6)
                + (hash >> 2);
        }
        return hash == kInvalidKey ? 1 : hash;
    }

    //! Guards against hash collisions: the indexed n-gram ending at end equals the context's.
    bool matches(int64_t end, int32_t const* ngram, int32_t n) const
    {
        return end >= n && std::equal(ngram, ngram + n, mTokens.begin() + (end - n));
    }

    size_t mMaxTokens;
    std::vector<int32_t> mTokens;
    std::unordered_map<uint64_t, std::pair<int64_t, int64_t>> mLast; //!< n-gram -> (latest end, previous end)
};

} // namespace rt
} // namespace trt_edgellm
