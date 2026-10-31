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
// Unit tests for KVCacheManager's on-demand mapping (Config::initialMappedTokens).

#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "common/vmmRegion.h"
#include "runtime/kvCacheManager.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// 2 KV heads x 256 dims, INT8: 64 KiB per page, so one 2 MiB mapping granule is 32 pages (4,096 tokens).
rt::KVCacheManager::Config makeConfig(int32_t layers, int32_t maxBatch, int32_t maxSeq, int32_t initialTokens)
{
    std::vector<rt::KVLayerConfig> layerConfigs(static_cast<size_t>(layers), rt::KVLayerConfig{2, 256});
    rt::KVCacheManager::Config config{layers, maxBatch, maxSeq, layerConfigs, DataType::kINT8};
    config.initialMappedTokens = initialTokens;
    return config;
}

void fill(void* device, uint8_t value, size_t bytes)
{
    ASSERT_EQ(cudaMemset(device, value, bytes), cudaSuccess);
}

uint8_t readByte(void const* device)
{
    uint8_t value = 0;
    EXPECT_EQ(cudaMemcpy(&value, device, 1, cudaMemcpyDeviceToHost), cudaSuccess);
    return value;
}

} // namespace

TEST(KvCacheManagerGrowTest, DisabledByDefaultMapsEverything)
{
    rt::KVCacheManager kv(makeConfig(2, 1, 16384, 0), nullptr);
    EXPECT_FALSE(kv.isGrowable());
    EXPECT_EQ(kv.mappedTokens(), 16384);
    EXPECT_TRUE(kv.ensureTokens(16384));
    EXPECT_EQ(kv.bytesToMap(16384), 0u);
}

TEST(KvCacheManagerGrowTest, StartsAtTheInitialSizeAndGrowsWithStablePointers)
{
    rt::KVCacheManager kv(makeConfig(2, 1, 16384, 4096), nullptr);
    ASSERT_TRUE(kv.isGrowable());
    EXPECT_EQ(kv.mappedTokens(), 4096);
    void* const k0 = kv.kPoolPtr(0);
    void* const v0 = kv.vPoolPtr(0);
    void* const k1 = kv.kPoolPtr(1);

    ASSERT_TRUE(kv.ensureTokens(9000));
    EXPECT_EQ(kv.mappedTokens(), 9088) << "backed in whole pages";
    EXPECT_EQ(kv.kPoolPtr(0), k0);
    EXPECT_EQ(kv.vPoolPtr(0), v0);
    EXPECT_EQ(kv.kPoolPtr(1), k1);
    size_t const halfBytes = static_cast<size_t>(kv.numPages(0)) * rt::kTOKENS_PER_PAGE * 2 * 256;
    EXPECT_EQ(static_cast<uint8_t*>(v0) - static_cast<uint8_t*>(k0), static_cast<ptrdiff_t>(halfBytes));
}

TEST(KvCacheManagerGrowTest, ContentsSurviveGrowth)
{
    rt::KVCacheManager kv(makeConfig(1, 1, 16384, 4096), nullptr);
    size_t const pageBytes = static_cast<size_t>(rt::kTOKENS_PER_PAGE) * 2 * 256;
    fill(kv.kPoolPtr(0), 0xA1, pageBytes);
    fill(kv.vPoolPtr(0), 0xB2, pageBytes);

    ASSERT_TRUE(kv.ensureTokens(16384));
    EXPECT_EQ(readByte(kv.kPoolPtr(0)), 0xA1);
    EXPECT_EQ(readByte(kv.vPoolPtr(0)), 0xB2);
    uint8_t* const lastK = static_cast<uint8_t*>(kv.kPoolPtr(0)) + (kv.numPages(0) - 1) * pageBytes;
    fill(lastK, 0xC3, pageBytes);
    EXPECT_EQ(readByte(lastK + pageBytes - 1), 0xC3);
}

TEST(KvCacheManagerGrowTest, RequestsBeyondTheMaximumAreCapped)
{
    rt::KVCacheManager kv(makeConfig(1, 1, 16384, 4096), nullptr);
    ASSERT_TRUE(kv.ensureTokens(1 << 30));
    EXPECT_EQ(kv.mappedTokens(), 16384);
    EXPECT_EQ(kv.bytesToMap(1 << 30), 0u);
}

TEST(KvCacheManagerGrowTest, InitialSizeAboveTheMaximumMapsEverything)
{
    rt::KVCacheManager kv(makeConfig(1, 1, 8192, 1 << 20), nullptr);
    EXPECT_EQ(kv.mappedTokens(), 8192);
}

TEST(KvCacheManagerGrowTest, MultiSlotEnginesKeepTheFullPool)
{
    rt::KVCacheManager kv(makeConfig(1, 2, 8192, 4096), nullptr);
    EXPECT_FALSE(kv.isGrowable());
    EXPECT_EQ(kv.mappedTokens(), 8192);
}

TEST(KvCacheManagerGrowTest, PoolsThatMissTheMappingGranuleKeepTheFullPool)
{
    rt::KVCacheManager kv(makeConfig(1, 1, 1000, 128), nullptr);
    EXPECT_FALSE(kv.isGrowable());
    EXPECT_EQ(kv.mappedTokens(), 1000);
}

TEST(KvCacheManagerGrowTest, BytesToMapCountsEveryLayerAndBothHalves)
{
    rt::KVCacheManager kv(makeConfig(3, 1, 16384, 4096), nullptr);
    size_t const granule = rt::VmmRegion::granularity(0);
    EXPECT_EQ(kv.bytesToMap(4096), 0u);
    EXPECT_EQ(kv.bytesToMap(4097), 3 * 2 * granule);
    EXPECT_EQ(kv.bytesToMap(8192), 3 * 2 * granule);
    EXPECT_EQ(kv.bytesToMap(8193), 3 * 2 * 2 * granule);
}

TEST(KvCacheManagerGrowTest, FailedGrowthRollsBackEveryLayer)
{
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&freeBytes, &totalBytes), cudaSuccess);
    // Each layer's full pool is 4 GiB (2 x 2 GiB halves); eight layers exceed any current GPU the tests run on.
    int32_t const maxSeq = 4 * 1024 * 1024;
    int32_t const layers = static_cast<int32_t>(totalBytes / (4ull << 30)) + 2;
    rt::KVCacheManager kv(makeConfig(layers, 1, maxSeq, 4096), nullptr);
    ASSERT_TRUE(kv.isGrowable());
    fill(kv.kPoolPtr(0), 0x42, 1);

    EXPECT_FALSE(kv.ensureTokens(maxSeq));
    EXPECT_EQ(kv.mappedTokens(), 4096);
    EXPECT_EQ(readByte(kv.kPoolPtr(0)), 0x42);
    EXPECT_TRUE(kv.ensureTokens(8192));
    EXPECT_EQ(kv.mappedTokens(), 8192);
}

TEST(KvCacheManagerGrowTest, RetainedExtraPagesKeepTheFullPool)
{
    rt::KVCacheManager::Config config = makeConfig(1, 1, 16384, 4096);
    config.numPages = 128 + 32;
    rt::KVCacheManager kv(config, nullptr);
    EXPECT_FALSE(kv.isGrowable()) << "pages past the active range would be handed out unmapped";
    EXPECT_EQ(kv.mappedTokens(), 16384);
    EXPECT_EQ(kv.numPages(), 160);
}

TEST(KvCacheManagerGrowTest, NonPositiveAndSmallerRequestsChangeNothing)
{
    rt::KVCacheManager kv(makeConfig(1, 1, 16384, 4096), nullptr);
    EXPECT_TRUE(kv.ensureTokens(0));
    EXPECT_TRUE(kv.ensureTokens(-5));
    EXPECT_EQ(kv.mappedTokens(), 4096);
    EXPECT_EQ(kv.bytesToMap(-5), 0u);

    ASSERT_TRUE(kv.ensureTokens(12000));
    int32_t const grown = kv.mappedTokens();
    EXPECT_TRUE(kv.ensureTokens(100));
    EXPECT_EQ(kv.mappedTokens(), grown) << "the mapping only ratchets up";
    EXPECT_TRUE(kv.ensureTokens(12000));
    EXPECT_EQ(kv.mappedTokens(), grown);
}

TEST(KvCacheManagerGrowTest, ExactPageAndGranuleBoundaries)
{
    rt::KVCacheManager kv(makeConfig(1, 1, 16384, 4096), nullptr);
    ASSERT_TRUE(kv.ensureTokens(4096));
    EXPECT_EQ(kv.mappedTokens(), 4096);
    ASSERT_TRUE(kv.ensureTokens(4097));
    EXPECT_EQ(kv.mappedTokens(), 4224) << "one more page";
    ASSERT_TRUE(kv.ensureTokens(8192));
    EXPECT_EQ(kv.mappedTokens(), 8192);
    ASSERT_TRUE(kv.ensureTokens(16384));
    EXPECT_EQ(kv.mappedTokens(), 16384);
}

TEST(KvCacheManagerGrowTest, InitialSizeOfOneTokenBacksOnePage)
{
    rt::KVCacheManager kv(makeConfig(1, 1, 16384, 1), nullptr);
    ASSERT_TRUE(kv.isGrowable());
    EXPECT_EQ(kv.mappedTokens(), rt::kTOKENS_PER_PAGE);
    uint8_t* const k = static_cast<uint8_t*>(kv.kPoolPtr(0));
    fill(k, 0x6D, static_cast<size_t>(rt::kTOKENS_PER_PAGE) * 2 * 256);
    EXPECT_EQ(readByte(k + static_cast<size_t>(rt::kTOKENS_PER_PAGE) * 2 * 256 - 1), 0x6D);
}

TEST(KvCacheManagerGrowTest, MovedManagerKeepsTheMappings)
{
    rt::KVCacheManager first(makeConfig(1, 1, 16384, 4096), nullptr);
    void* const k = first.kPoolPtr(0);
    rt::KVCacheManager second(std::move(first));
    EXPECT_TRUE(second.isGrowable());
    EXPECT_EQ(second.kPoolPtr(0), k);
    ASSERT_TRUE(second.ensureTokens(9000));
    EXPECT_EQ(second.kPoolPtr(0), k);

    rt::KVCacheManager third(makeConfig(1, 1, 8192, 0), nullptr);
    third = std::move(second);
    EXPECT_TRUE(third.isGrowable());
    EXPECT_EQ(third.mappedTokens(), 9088);
    EXPECT_FALSE(first.isGrowable());
}

TEST(KvCacheManagerGrowTest, KvSharingRecipientsOwnNoRegion)
{
    rt::KVCacheManager::Config config = makeConfig(3, 1, 16384, 4096);
    config.kvSharingDonors = {-1, 0, 0};
    rt::KVCacheManager kv(config, nullptr);
    ASSERT_TRUE(kv.isGrowable());
    size_t const granule = rt::VmmRegion::granularity(0);
    EXPECT_EQ(kv.bytesToMap(4097), 1 * 2 * granule) << "only the donor layer maps memory";
    ASSERT_TRUE(kv.ensureTokens(8000));
    EXPECT_EQ(kv.kPoolPtr(1), kv.kPoolPtr(0));
}
