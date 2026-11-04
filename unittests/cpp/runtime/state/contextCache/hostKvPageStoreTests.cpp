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

#include "common/cudaMacros.h"
#include "runtime/hybridCacheManager.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm::rt;

namespace
{

HybridCacheManager::Config makeHybridConfig()
{
    HybridCacheManager::Config config;
    config.layerTypes = {HybridCacheManager::LayerType::kAttention, HybridCacheManager::LayerType::kMamba,
        HybridCacheManager::LayerType::kAttention, HybridCacheManager::LayerType::kMamba};
    config.kvConfig = KVCacheManager::Config{/*.numAttentionLayers=*/2, /*.maxBatchSize=*/2,
        /*.maxSequenceLength=*/256, /*.layerConfigs=*/{KVLayerConfig{2, 4}, KVLayerConfig{1, 8}},
        /*.kvCacheType=*/DataType::kHALF, /*.numPages=*/8};
    config.mambaConfig = MambaCacheManager::Config{/*.numRecurrentLayers=*/2, /*.maxBatchSize=*/2,
        /*.recurrentStateNumHeads=*/2, /*.recurrentStateHeadDim=*/3, /*.recurrentStateSize=*/4,
        /*.convDim=*/5, /*.convKernel=*/6, /*.maxIntermediateSeqLen=*/0,
        /*.recurrentStateType=*/DataType::kHALF, /*.convStateType=*/DataType::kHALF};
    config.maxBatchSize = 2;
    return config;
}

size_t pageBytes(KVCacheManager const& kv, int32_t layer)
{
    KVLayerConfig const& config = kv.getLayerConfig(layer);
    return static_cast<size_t>(config.numKVHeads) * static_cast<size_t>(config.headDim)
        * static_cast<size_t>(kTOKENS_PER_PAGE) * sizeof(half);
}

uint8_t pattern(int32_t layer, int32_t page, int32_t half, int32_t salt)
{
    return static_cast<uint8_t>(1 + layer * 31 + page * 7 + half * 13 + salt * 3);
}

void fillPage(KVCacheManager const& kv, int32_t layer, int32_t page, int32_t salt, cudaStream_t stream)
{
    size_t const bytes = pageBytes(kv, layer);
    auto* keys = static_cast<uint8_t*>(kv.kPoolPtr(layer)) + static_cast<size_t>(page) * bytes;
    auto* values = static_cast<uint8_t*>(kv.vPoolPtr(layer)) + static_cast<size_t>(page) * bytes;
    CUDA_CHECK(cudaMemsetAsync(keys, pattern(layer, page, 0, salt), bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(values, pattern(layer, page, 1, salt), bytes, stream));
}

bool pageHolds(KVCacheManager const& kv, int32_t layer, int32_t page, int32_t sourcePage, int32_t salt)
{
    size_t const bytes = pageBytes(kv, layer);
    std::vector<uint8_t> keys(bytes);
    std::vector<uint8_t> values(bytes);
    EXPECT_EQ(cudaMemcpy(keys.data(), static_cast<uint8_t*>(kv.kPoolPtr(layer)) + static_cast<size_t>(page) * bytes,
                  bytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(cudaMemcpy(values.data(), static_cast<uint8_t*>(kv.vPoolPtr(layer)) + static_cast<size_t>(page) * bytes,
                  bytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    return std::all_of(keys.begin(), keys.end(), [&](uint8_t v) { return v == pattern(layer, sourcePage, 0, salt); })
        && std::all_of(values.begin(), values.end(), [&](uint8_t v) { return v == pattern(layer, sourcePage, 1, salt); });
}

void zeroPage(KVCacheManager const& kv, int32_t layer, int32_t page, cudaStream_t stream)
{
    size_t const bytes = pageBytes(kv, layer);
    CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(kv.kPoolPtr(layer)) + static_cast<size_t>(page) * bytes, 0, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(static_cast<uint8_t*>(kv.vPoolPtr(layer)) + static_cast<size_t>(page) * bytes, 0, bytes, stream));
}

} // namespace

TEST(HostKvPageStoreTests, ReportsBytesPerSlot)
{
    HybridCacheManager::Config const config = makeHybridConfig();
    EXPECT_EQ(HostKvPageStore::bytesPerSlot(config.kvConfig), 8192U);
}

TEST(HostKvPageStoreTests, CaptureAndRestoreAreByteExactAcrossRunsAndGaps)
{
    cudaStream_t stream{};
    HybridCacheManager cacheManager(makeHybridConfig(), stream);
    KVCacheManager& kv = cacheManager.getKVCacheManager();
    HostKvPageStore store(kv, 6);
    std::vector<int32_t> const sources{1, 2, 3, 6, 4};
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        for (int32_t page : sources)
        {
            fillPage(kv, layer, page, 0, stream);
        }
    }
    // pages 1-3 form one run into slots 0-2, page 6 goes to slot 5, page 4 to slot 3
    store.capture({{1, 0}, {2, 1}, {3, 2}, {6, 5}, {4, 3}}, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        for (int32_t page : sources)
        {
            zeroPage(kv, layer, page, stream);
        }
    }
    // different destinations: slots 0-2 to pages 5-7 (a run), slot 5 to page 0, slot 3 to page 4
    store.restore({{5, 0}, {6, 1}, {7, 2}, {0, 5}, {4, 3}}, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        EXPECT_TRUE(pageHolds(kv, layer, 5, 1, 0)) << "layer " << layer;
        EXPECT_TRUE(pageHolds(kv, layer, 6, 2, 0)) << "layer " << layer;
        EXPECT_TRUE(pageHolds(kv, layer, 7, 3, 0)) << "layer " << layer;
        EXPECT_TRUE(pageHolds(kv, layer, 0, 6, 0)) << "layer " << layer;
        EXPECT_TRUE(pageHolds(kv, layer, 4, 4, 0)) << "layer " << layer;
    }
}

TEST(HostKvPageStoreTests, CaptureLeavesTheDevicePagesUntouched)
{
    cudaStream_t stream{};
    HybridCacheManager cacheManager(makeHybridConfig(), stream);
    KVCacheManager& kv = cacheManager.getKVCacheManager();
    HostKvPageStore store(kv, 2);
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        fillPage(kv, layer, 2, 5, stream);
    }
    store.capture({{2, 1}}, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        EXPECT_TRUE(pageHolds(kv, layer, 2, 2, 5));
    }
}

TEST(HostKvPageStoreTests, ASlotIsOverwrittenByTheNextCapture)
{
    cudaStream_t stream{};
    HybridCacheManager cacheManager(makeHybridConfig(), stream);
    KVCacheManager& kv = cacheManager.getKVCacheManager();
    HostKvPageStore store(kv, 1);
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        fillPage(kv, layer, 1, 0, stream);
        fillPage(kv, layer, 2, 0, stream);
    }
    store.capture({{1, 0}}, stream);
    store.capture({{2, 0}}, stream);
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        zeroPage(kv, layer, 3, stream);
    }
    store.restore({{3, 0}}, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (int32_t layer = 0; layer < kv.numLayers(); ++layer)
    {
        EXPECT_TRUE(pageHolds(kv, layer, 3, 2, 0));
    }
}

TEST(HostKvPageStoreTests, RejectsSlotsAndPagesOutOfRange)
{
    cudaStream_t stream{};
    HybridCacheManager cacheManager(makeHybridConfig(), stream);
    KVCacheManager& kv = cacheManager.getKVCacheManager();
    HostKvPageStore store(kv, 2);
    EXPECT_THROW(store.capture({{0, 2}}, stream), std::runtime_error);
    EXPECT_THROW(store.capture({{0, -1}}, stream), std::runtime_error);
    EXPECT_THROW(store.capture({{8, 0}}, stream), std::runtime_error);
    EXPECT_THROW(store.restore({{-1, 0}}, stream), std::runtime_error);
    EXPECT_THROW(store.restore({{0, 0}, {0, 5}}, stream), std::runtime_error);
}

TEST(HostKvPageStoreTests, EmptyCopyListsDoNothingAndAZeroSlotStoreRefusesCopies)
{
    cudaStream_t stream{};
    HybridCacheManager cacheManager(makeHybridConfig(), stream);
    KVCacheManager& kv = cacheManager.getKVCacheManager();
    HostKvPageStore empty(kv, 0);
    EXPECT_EQ(empty.slotCount(), 0);
    EXPECT_NO_THROW(empty.capture({}, stream));
    EXPECT_NO_THROW(empty.restore({}, stream));
    EXPECT_THROW(empty.capture({{0, 0}}, stream), std::runtime_error);
    EXPECT_THROW(HostKvPageStore(kv, -1), std::runtime_error);
}
