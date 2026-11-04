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

// Unit tests for KVCacheManager's packed-V layout (Config::packedV): the V plane holds 160-byte rows after the full-size K plane.

#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "kernels/posEncoding/kvInt4Quant.h"
#include "runtime/kvCacheManager.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

constexpr size_t kK_PAGE_BYTES = static_cast<size_t>(rt::kTOKENS_PER_PAGE) * 2 * 256; // 2 KV heads x 256 channels, INT8
constexpr size_t kV_PAGE_BYTES = static_cast<size_t>(rt::kTOKENS_PER_PAGE) * 2 * kernel::kVInt4PackedUsedBytes;

rt::KVCacheManager::Config makeConfig(
    int32_t layers, int32_t maxBatch, int32_t maxSeq, int32_t initialTokens, bool packed)
{
    std::vector<rt::KVLayerConfig> layerConfigs(static_cast<size_t>(layers), rt::KVLayerConfig{2, 256});
    rt::KVCacheManager::Config config{layers, maxBatch, maxSeq, layerConfigs, DataType::kINT8};
    config.initialMappedTokens = initialTokens;
    config.packedV = packed;
    return config;
}

uint8_t readByte(void const* device)
{
    uint8_t value = 0;
    EXPECT_EQ(cudaMemcpy(&value, device, 1, cudaMemcpyDeviceToHost), cudaSuccess);
    return value;
}

} // namespace

TEST(KvCacheManagerPackedVTest, PageSizesAndPlaneLayout)
{
    rt::KVCacheManager kv(makeConfig(2, 1, 4096, 0, true), nullptr);
    EXPECT_EQ(kv.kPageBytes(0), kK_PAGE_BYTES);
    EXPECT_EQ(kv.vPageBytes(0), kV_PAGE_BYTES);
    EXPECT_EQ(kV_PAGE_BYTES * 8, kK_PAGE_BYTES * 5);
    int32_t const pages = kv.numPages(0);
    EXPECT_EQ(static_cast<uint8_t*>(kv.vPoolPtr(0)) - static_cast<uint8_t*>(kv.kPoolPtr(0)),
        static_cast<ptrdiff_t>(pages * kK_PAGE_BYTES));
    EXPECT_EQ(kv.capacityBytes(), 2 * static_cast<size_t>(pages) * (kK_PAGE_BYTES + kV_PAGE_BYTES));
    rt::Tensor const& pool = kv.getCombinedKVCache(0);
    EXPECT_EQ(pool.getShape(), (rt::Coords{2, pages, rt::kTOKENS_PER_PAGE, 2, 256}));
    EXPECT_EQ(pool.rawPointer(), kv.kPoolPtr(0));
}

TEST(KvCacheManagerPackedVTest, UnpackedManagerKeepsEqualPlanes)
{
    rt::KVCacheManager kv(makeConfig(1, 1, 4096, 0, false), nullptr);
    EXPECT_EQ(kv.vPageBytes(0), kv.kPageBytes(0));
    EXPECT_EQ(kv.capacityBytes(), 2 * static_cast<size_t>(kv.numPages(0)) * kK_PAGE_BYTES);
}

TEST(KvCacheManagerPackedVTest, CompactAllocationIsSmallerAndFullyUsable)
{
    rt::KVCacheManager packed(makeConfig(2, 1, 8192, 0, true), nullptr);
    rt::KVCacheManager plain(makeConfig(2, 1, 8192, 0, false), nullptr);
    EXPECT_EQ(plain.capacityBytes() - packed.capacityBytes(), 2u * 64 * (kK_PAGE_BYTES - kV_PAGE_BYTES));
    // The last byte of the V plane is the last byte of the compact allocation: both ends must be writable and the planes
    // must not overlap.
    int32_t const pages = packed.numPages(1);
    uint8_t* const vEnd = static_cast<uint8_t*>(packed.vPoolPtr(1)) + pages * kV_PAGE_BYTES - 1;
    ASSERT_EQ(cudaMemset(vEnd, 0x5A, 1), cudaSuccess);
    ASSERT_EQ(cudaMemset(packed.kPoolPtr(1), 0xA5, pages * kK_PAGE_BYTES), cudaSuccess);
    EXPECT_EQ(readByte(vEnd), 0x5A);
}

TEST(KvCacheManagerPackedVTest, RejectsUnsupportedConfigurations)
{
    auto cfg = makeConfig(1, 1, 4096, 0, true);
    cfg.kvCacheType = DataType::kHALF;
    EXPECT_THROW(rt::KVCacheManager(cfg, nullptr), std::exception);
    cfg = makeConfig(1, 1, 4096, 0, true);
    cfg.layerConfigs[0].headDim = 128;
    EXPECT_THROW(rt::KVCacheManager(cfg, nullptr), std::exception);
    cfg = makeConfig(2, 1, 4096, 0, true);
    cfg.kvSharingDonors = {-1, 0};
    EXPECT_THROW(rt::KVCacheManager(cfg, nullptr), std::exception);
}

TEST(KvCacheManagerPackedVTest, GrowableMapsEachPlaneAtItsOwnStride)
{
    rt::KVCacheManager kv(makeConfig(2, 1, 16384, 4096, true), nullptr);
    ASSERT_TRUE(kv.isGrowable());
    void* const k0 = kv.kPoolPtr(0);
    void* const v0 = kv.vPoolPtr(0);
    size_t const initial = kv.mappedBytes();
    // 32 pages: K is exactly one 2 MiB granule, V (1.25 MiB) rounds up to one granule.
    EXPECT_EQ(initial, 2u * 2 * (2u << 20));

    size_t const need = kv.bytesToMap(16384);
    ASSERT_TRUE(kv.ensureTokens(16384));
    EXPECT_EQ(kv.mappedBytes(), initial + need);
    EXPECT_EQ(kv.kPoolPtr(0), k0);
    EXPECT_EQ(kv.vPoolPtr(0), v0);
    // 128 pages: K 8 MiB, V 5 MiB mapped as 6 MiB.
    EXPECT_EQ(kv.mappedBytes(), 2u * ((8u << 20) + (6u << 20)));
    int32_t const pages = kv.numPages(0);
    uint8_t* const vEnd = static_cast<uint8_t*>(v0) + pages * kV_PAGE_BYTES - 1;
    ASSERT_EQ(cudaMemset(vEnd, 0x3C, 1), cudaSuccess);
    EXPECT_EQ(readByte(vEnd), 0x3C);

    size_t const released = kv.shrinkTokens(4096);
    EXPECT_EQ(released, 2u * ((8u << 20) + (6u << 20)) - initial);
    EXPECT_EQ(kv.mappedBytes(), initial);
    EXPECT_EQ(kv.bytesToMap(16384), need);
}

TEST(KvCacheManagerPackedVTest, GrowableUsesLessMemoryThanUnpackedAtTheSameLength)
{
    rt::KVCacheManager packed(makeConfig(2, 1, 16384, 4096, true), nullptr);
    rt::KVCacheManager plain(makeConfig(2, 1, 16384, 4096, false), nullptr);
    ASSERT_TRUE(packed.ensureTokens(16384));
    ASSERT_TRUE(plain.ensureTokens(16384));
    EXPECT_EQ(plain.mappedBytes(), 2u * 2 * (8u << 20));
    EXPECT_LT(packed.mappedBytes(), plain.mappedBytes());
    EXPECT_EQ(plain.mappedBytes() - packed.mappedBytes(), 2u * (2u << 20));
}
