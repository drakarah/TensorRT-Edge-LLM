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

#include "runtime/state/boundedSwaKVPageManager.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class KVCacheManager;
class KVPageTable;

//! Host-resident snapshots of the KV state a bounded-SWA sequence has at fixed token boundaries, so a later request
//! that starts with the same tokens can skip their prefill. A snapshot holds the pages of the full-capacity layers for
//! [0, length) and the retained window pages of the bounded sliding-window layers. Entries are pinned host buffers
//! evicted least-recently-used under a byte budget.
class SwaPrefixCache final
{
public:
    struct Hit
    {
        int32_t length{}; //!< Tokens of the request covered by the snapshot.
        void const* handle{};
    };

    SwaPrefixCache(KVCacheManager& kvCache, KVPageTable const& basePageTable, size_t budgetBytes);
    ~SwaPrefixCache() noexcept;
    SwaPrefixCache(SwaPrefixCache const&) = delete;
    SwaPrefixCache& operator=(SwaPrefixCache const&) = delete;

    //! Copies the state of `slot` (whose sliding-window pages are `state`) holding exactly `prefix.size()` tokens.
    //! The caller guarantees the device work that produced the state has completed or is ordered on `stream`.
    void capture(int32_t slot, std::vector<int32_t> const& prefix, SwaKVCacheState const& state, cudaStream_t stream);

    //! Longest snapshot whose tokens are a strict prefix of `tokens`.
    std::optional<Hit> lookup(std::vector<int32_t> const& tokens);

    //! Copies the snapshot back into `slot`: global pages for [0, restoreLength) and the window pages currently bound
    //! in `state` (the manager has mapped them for `restoreLength`). Ordered on `stream`.
    void restore(Hit const& hit, int32_t slot, int32_t restoreLength, SwaKVCacheState const& state, cudaStream_t stream);

    size_t usedBytes() const noexcept;
    int64_t hits() const noexcept;
    int64_t lookups() const noexcept;

private:
    struct Entry;
    void evictFor(size_t bytes);

    KVCacheManager& mKvCache;
    KVPageTable const& mBasePageTable;
    size_t const mBudgetBytes;
    size_t mUsedBytes{};
    int64_t mHits{};
    int64_t mLookups{};
    std::list<std::unique_ptr<Entry>> mEntries; //!< Front = most recently used.
};

} // namespace rt
} // namespace trt_edgellm
