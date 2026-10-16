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

#include "runtime/state/swaPrefixCache.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "runtime/kvCacheManager.h"
#include "runtime/state/kvPageTable.h"

#include <algorithm>
#include <cstring>

namespace trt_edgellm
{
namespace rt
{

struct SwaPrefixCache::Entry
{
    std::vector<int32_t> tokens;
    std::vector<int32_t> windowLogicalPages; //!< Logical pages of the bounded layers held by the snapshot.
    void* host{};                            //!< Pinned buffer, layers in order, K then V per page.
    size_t bytes{};
    std::vector<size_t> layerOffsets;        //!< Per-layer byte offset into `host` (SIZE_MAX: layer not stored).
    int32_t globalPages{};                   //!< Pages per plane held for each full-capacity layer.
};

namespace
{

size_t pageBytes(KVCacheManager const& kv, int32_t layer)
{
    KVLayerConfig const& lc = kv.getLayerConfig(layer);
    return static_cast<size_t>(kTOKENS_PER_PAGE) * lc.numKVHeads * lc.headDim
        * rt::utils::getTypeSize(kv.getConfig().kvCacheType);
}

bool isBoundedLayer(KVCacheManager const& kv, int32_t layer)
{
    return kv.hasReducedKVCache()
        && isReducedKvCacheCapacity(kv.getLayerConfig(layer).kvCacheCapacity, kv.getConfig().maxSequenceLength);
}

//! K page `kPage` of the layer's pool; the V page of the same table entry follows `numPages` pages later.
char* kPagePtr(KVCacheManager const& kv, int32_t layer, int64_t kPage)
{
    return static_cast<char*>(kv.kPoolPtr(layer)) + kPage * static_cast<int64_t>(pageBytes(kv, layer));
}

char* vPagePtr(KVCacheManager const& kv, int32_t layer, int64_t kPage)
{
    return static_cast<char*>(kv.vPoolPtr(layer)) + kPage * static_cast<int64_t>(pageBytes(kv, layer));
}

} // namespace

SwaPrefixCache::SwaPrefixCache(KVCacheManager& kvCache, KVPageTable const& basePageTable, size_t budgetBytes)
    : mKvCache(kvCache)
    , mBasePageTable(basePageTable)
    , mBudgetBytes(budgetBytes)
{
}

SwaPrefixCache::~SwaPrefixCache() noexcept
{
    for (auto& entry : mEntries)
    {
        if (entry->host != nullptr)
        {
            cudaFreeHost(entry->host);
        }
    }
}

size_t SwaPrefixCache::usedBytes() const noexcept
{
    return mUsedBytes;
}

int64_t SwaPrefixCache::hits() const noexcept
{
    return mHits;
}

int64_t SwaPrefixCache::lookups() const noexcept
{
    return mLookups;
}

void SwaPrefixCache::evictFor(size_t bytes)
{
    while (!mEntries.empty() && mUsedBytes + bytes > mBudgetBytes)
    {
        auto& victim = mEntries.back();
        mUsedBytes -= victim->bytes;
        cudaFreeHost(victim->host);
        mEntries.pop_back();
    }
}

void SwaPrefixCache::capture(
    int32_t slot, std::vector<int32_t> const& prefix, SwaKVCacheState const& state, cudaStream_t stream)
{
    int32_t const length = static_cast<int32_t>(prefix.size());
    if (length <= 1 || state.exactResidentTokenCount != length)
    {
        return;
    }
    for (auto const& entry : mEntries)
    {
        if (entry->tokens == prefix)
        {
            return;
        }
    }

    auto entry = std::make_unique<Entry>();
    entry->tokens = prefix;
    entry->globalPages = (length + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;
    for (SwaPageBinding const& binding : state.pageBindings)
    {
        entry->windowLogicalPages.push_back(binding.logicalPage);
    }

    int32_t const numLayers = mKvCache.numLayers();
    entry->layerOffsets.assign(static_cast<size_t>(numLayers), SIZE_MAX);
    size_t total = 0;
    for (int32_t layer = 0; layer < numLayers; ++layer)
    {
        if (mKvCache.isKVSharingRecipient(layer))
        {
            continue;
        }
        size_t const pages = isBoundedLayer(mKvCache, layer) ? entry->windowLogicalPages.size()
                                                              : static_cast<size_t>(entry->globalPages);
        entry->layerOffsets[static_cast<size_t>(layer)] = total;
        total += 2 * pages * pageBytes(mKvCache, layer);
    }
    entry->bytes = total;
    if (total > mBudgetBytes)
    {
        return;
    }
    evictFor(total);
    if (cudaMallocHost(&entry->host, total) != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
        return;
    }

    int32_t const maxPagesPerSeq = mBasePageTable.maxPagesPerSeq();
    char* const host = static_cast<char*>(entry->host);
    for (int32_t layer = 0; layer < numLayers; ++layer)
    {
        size_t const offset = entry->layerOffsets[static_cast<size_t>(layer)];
        if (offset == SIZE_MAX)
        {
            continue;
        }
        size_t const pb = pageBytes(mKvCache, layer);
        char* dst = host + offset;
        if (isBoundedLayer(mKvCache, layer))
        {
            for (SwaPageBinding const& binding : state.pageBindings)
            {
                CUDA_CHECK(cudaMemcpyAsync(dst, kPagePtr(mKvCache, layer, binding.page), pb, cudaMemcpyDeviceToHost, stream));
                dst += pb;
                CUDA_CHECK(cudaMemcpyAsync(dst, vPagePtr(mKvCache, layer, binding.page), pb, cudaMemcpyDeviceToHost, stream));
                dst += pb;
            }
        }
        else
        {
            for (int32_t page = 0; page < entry->globalPages; ++page)
            {
                int64_t const physical = static_cast<int64_t>(slot) * maxPagesPerSeq + page;
                CUDA_CHECK(cudaMemcpyAsync(dst, kPagePtr(mKvCache, layer, physical), pb, cudaMemcpyDeviceToHost, stream));
                dst += pb;
                CUDA_CHECK(cudaMemcpyAsync(dst, vPagePtr(mKvCache, layer, physical), pb, cudaMemcpyDeviceToHost, stream));
                dst += pb;
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    mUsedBytes += total;
    mEntries.push_front(std::move(entry));
}

std::optional<SwaPrefixCache::Hit> SwaPrefixCache::lookup(std::vector<int32_t> const& tokens)
{
    ++mLookups;
    Entry* best = nullptr;
    for (auto const& entry : mEntries)
    {
        size_t const n = entry->tokens.size();
        if (n < tokens.size() && (best == nullptr || n > best->tokens.size())
            && std::equal(entry->tokens.begin(), entry->tokens.end(), tokens.begin()))
        {
            best = entry.get();
        }
    }
    if (best == nullptr)
    {
        return std::nullopt;
    }
    // Most recently used first.
    auto it = std::find_if(mEntries.begin(), mEntries.end(), [&](auto const& e) { return e.get() == best; });
    mEntries.splice(mEntries.begin(), mEntries, it);
    ++mHits;
    return Hit{static_cast<int32_t>(best->tokens.size()), best};
}

void SwaPrefixCache::restore(
    Hit const& hit, int32_t slot, int32_t restoreLength, SwaKVCacheState const& state, cudaStream_t stream)
{
    auto const* entry = static_cast<Entry const*>(hit.handle);
    ELLM_CHECK(entry != nullptr && restoreLength > 0 && restoreLength <= static_cast<int32_t>(entry->tokens.size()),
        "SwaPrefixCache restore length is outside the snapshot");
    int32_t const maxPagesPerSeq = mBasePageTable.maxPagesPerSeq();
    int32_t const restorePages = (restoreLength + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;
    char const* const host = static_cast<char const*>(entry->host);
    for (int32_t layer = 0; layer < mKvCache.numLayers(); ++layer)
    {
        size_t const offset = entry->layerOffsets[static_cast<size_t>(layer)];
        if (offset == SIZE_MAX)
        {
            continue;
        }
        size_t const pb = pageBytes(mKvCache, layer);
        char const* src = host + offset;
        if (isBoundedLayer(mKvCache, layer))
        {
            for (SwaPageBinding const& binding : state.pageBindings)
            {
                auto const found = std::find(
                    entry->windowLogicalPages.begin(), entry->windowLogicalPages.end(), binding.logicalPage);
                if (found == entry->windowLogicalPages.end())
                {
                    continue; // Page outside the snapshot's window: never read for the restored endpoint.
                }
                size_t const index = static_cast<size_t>(found - entry->windowLogicalPages.begin());
                char const* k = src + 2 * index * pb;
                CUDA_CHECK(cudaMemcpyAsync(kPagePtr(mKvCache, layer, binding.page), k, pb, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    vPagePtr(mKvCache, layer, binding.page), k + pb, pb, cudaMemcpyHostToDevice, stream));
            }
        }
        else
        {
            for (int32_t page = 0; page < restorePages; ++page)
            {
                int64_t const physical = static_cast<int64_t>(slot) * maxPagesPerSeq + page;
                char const* k = src + 2 * static_cast<size_t>(page) * pb;
                CUDA_CHECK(cudaMemcpyAsync(kPagePtr(mKvCache, layer, physical), k, pb, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(vPagePtr(mKvCache, layer, physical), k + pb, pb, cudaMemcpyHostToDevice, stream));
            }
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
