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

#include "runtime/state/contextCache/hostKvPageStore.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <limits>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace
{

size_t pageBytesOf(KVLayerConfig const& layer, nvinfer1::DataType type)
{
    size_t const tokenElements = static_cast<size_t>(layer.numKVHeads) * static_cast<size_t>(layer.headDim);
    return tokenElements * static_cast<size_t>(kTOKENS_PER_PAGE) * utils::getTypeSize(type);
}

void* byteOffset(void* base, size_t bytes)
{
    return static_cast<uint8_t*>(base) + bytes;
}

} // namespace

size_t HostKvPageStore::bytesPerSlot(KVCacheManager::Config const& config)
{
    size_t total = 0;
    for (KVLayerConfig const& layer : config.layerConfigs)
    {
        total += 2 * pageBytesOf(layer, config.kvCacheType);
    }
    return total;
}

HostKvPageStore::HostKvPageStore(KVCacheManager const& kvCache, int32_t slotCount)
    : mKvCache(kvCache)
    , mSlotCount(slotCount)
{
    ELLM_CHECK(slotCount >= 0, "Host KV page store slot count must be non-negative");
    ELLM_CHECK(kvCache.numLayers() > 0, "Host KV page store needs at least one attention layer");
    ELLM_CHECK(!kvCache.getConfig().packedV, "The host KV tier does not support packed V rows");
    for (int32_t layer = 0; layer < kvCache.numLayers(); ++layer)
    {
        KVLayerConfig const& config = kvCache.getLayerConfig(layer);
        mPageBytes.push_back(pageBytesOf(config, kvCache.getConfig().kvCacheType));
        if (slotCount > 0)
        {
            Coords const shape{slotCount, kTOKENS_PER_PAGE, config.numKVHeads, config.headDim};
            std::string const suffix = std::to_string(layer);
            mKeys.emplace_back(shape, DeviceType::kCPU, kvCache.getConfig().kvCacheType, "HostKvPageStore::k_" + suffix);
            mValues.emplace_back(shape, DeviceType::kCPU, kvCache.getConfig().kvCacheType, "HostKvPageStore::v_" + suffix);
        }
    }
    int64_t hostBytes = 0;
    for (std::vector<Tensor> const* slabs : {&mKeys, &mValues})
    {
        for (Tensor const& slab : *slabs)
        {
            hostBytes += slab.getMemoryCapacity();
        }
    }
    mHostReservation = HostMemoryReservation(HostMemoryCategory::kKvTier, hostBytes);
}

int32_t HostKvPageStore::slotCount() const noexcept
{
    return mSlotCount;
}

void HostKvPageStore::capture(std::vector<HostKvPageCopy> const& copies, cudaStream_t stream)
{
    copy(copies, cudaMemcpyDeviceToHost, stream);
}

void HostKvPageStore::restore(std::vector<HostKvPageCopy> const& copies, cudaStream_t stream)
{
    copy(copies, cudaMemcpyHostToDevice, stream);
}

void HostKvPageStore::copy(std::vector<HostKvPageCopy> const& copies, cudaMemcpyKind kind, cudaStream_t stream)
{
    for (HostKvPageCopy const& item : copies)
    {
        ELLM_CHECK(item.slot >= 0 && item.slot < mSlotCount, "Host KV page store slot is out of range");
        ELLM_CHECK(item.page >= 0 && item.page < mKvCache.numPages(), "Host KV page store device page is out of range");
    }
    size_t begin = 0;
    while (begin < copies.size())
    {
        size_t end = begin + 1;
        while (end < copies.size() && copies[end].page == copies[end - 1].page + 1
            && copies[end].slot == copies[end - 1].slot + 1)
        {
            ++end;
        }
        size_t const run = end - begin;
        for (int32_t layer = 0; layer < mKvCache.numLayers(); ++layer)
        {
            size_t const pageBytes = mPageBytes[static_cast<size_t>(layer)];
            size_t const deviceOffset = static_cast<size_t>(copies[begin].page) * pageBytes;
            size_t const hostOffset = static_cast<size_t>(copies[begin].slot) * pageBytes;
            void* const deviceKeys = byteOffset(mKvCache.kPoolPtr(layer), deviceOffset);
            void* const deviceValues = byteOffset(mKvCache.vPoolPtr(layer), deviceOffset);
            void* const hostKeys = byteOffset(mKeys[static_cast<size_t>(layer)].rawPointer(), hostOffset);
            void* const hostValues = byteOffset(mValues[static_cast<size_t>(layer)].rawPointer(), hostOffset);
            bool const toHost = kind == cudaMemcpyDeviceToHost;
            CUDA_CHECK(cudaMemcpyAsync(toHost ? hostKeys : deviceKeys, toHost ? deviceKeys : hostKeys, run * pageBytes, kind, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                toHost ? hostValues : deviceValues, toHost ? deviceValues : hostValues, run * pageBytes, kind, stream));
        }
        begin = end;
    }
}

} // namespace rt
} // namespace trt_edgellm
