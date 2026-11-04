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

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

//! How far a KV growth got.
enum class KvGrowOutcome
{
    kFull,    //!< The requested size is mapped.
    kMinimum, //!< Only the smaller floor size is mapped.
    kFailed,  //!< Nothing more could be mapped.
};

struct KvGrowResult
{
    KvGrowOutcome outcome{KvGrowOutcome::kFailed};
    int64_t released{0}; //!< Expert cache bytes handed over, over all attempts.
    int32_t releases{0}; //!< Release calls made.
};

//! Maps KV for @p wanted tokens, taking memory from the expert caches when the pool cannot grow by itself. Memory freed in slot-sized
//! pieces is not always usable for the pool's mapping granules, so a release that is not enough is followed by up to @p maxExtra
//! releases of @p extraBytes each. If @p wanted still does not fit, @p minimum is tried before giving up.
//! @param ensure Maps the given token count; true when mapped.
//! @param bytesToMap Device bytes mapping the given token count would newly take.
//! @param release Frees at least the given bytes from the expert caches; returns the bytes freed, 0 when nothing can go.
template <typename EnsureFn, typename BytesFn, typename ReleaseFn>
KvGrowResult growKvWithRelease(
    EnsureFn&& ensure, BytesFn&& bytesToMap, ReleaseFn&& release, int32_t wanted, int32_t minimum, int64_t extraBytes, int32_t maxExtra)
{
    KvGrowResult result;
    if (ensure(wanted))
    {
        result.outcome = KvGrowOutcome::kFull;
        return result;
    }
    int64_t freed = release(static_cast<int64_t>(bytesToMap(wanted)));
    ++result.releases;
    result.released += freed > 0 ? freed : 0;
    bool full = ensure(wanted);
    for (int32_t attempt = 0; !full && freed > 0 && attempt < maxExtra && extraBytes > 0; ++attempt)
    {
        freed = release(extraBytes);
        ++result.releases;
        result.released += freed > 0 ? freed : 0;
        full = ensure(wanted);
    }
    if (full)
    {
        result.outcome = KvGrowOutcome::kFull;
    }
    else if (minimum > 0 && minimum < wanted && ensure(minimum))
    {
        result.outcome = KvGrowOutcome::kMinimum;
    }
    return result;
}

} // namespace rt
} // namespace trt_edgellm
