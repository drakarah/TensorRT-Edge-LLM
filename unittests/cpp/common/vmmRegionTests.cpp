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
#include "common/vmmRegion.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;

namespace
{

size_t granule()
{
    return rt::VmmRegion::granularity(0);
}

void fill(void* device, uint8_t value, size_t bytes)
{
    ASSERT_EQ(cudaMemset(device, value, bytes), cudaSuccess);
}

std::vector<uint8_t> read(void const* device, size_t bytes)
{
    std::vector<uint8_t> host(bytes);
    EXPECT_EQ(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    return host;
}

} // namespace

TEST(VmmRegionTest, RejectsHalfSizesThatAreNotGranuleMultiples)
{
    EXPECT_THROW(rt::VmmRegion(0, 0), std::runtime_error);
    EXPECT_THROW(rt::VmmRegion(granule() + 1, 0), std::runtime_error);
}

TEST(VmmRegionTest, GrowsInGranulesAndKeepsBaseAndContents)
{
    size_t const g = granule();
    rt::VmmRegion region(4 * g, 0);
    void* const base = region.base();
    EXPECT_EQ(region.mappedPerHalf(), 0u);

    ASSERT_TRUE(region.growTo(1));
    EXPECT_EQ(region.mappedPerHalf(), g);
    uint8_t* const k = static_cast<uint8_t*>(base);
    uint8_t* const v = k + region.halfBytes();
    fill(k, 0x11, g);
    fill(v, 0x22, g);

    ASSERT_TRUE(region.growTo(3 * g - 1));
    EXPECT_EQ(region.mappedPerHalf(), 3 * g);
    EXPECT_EQ(region.base(), base);
    fill(k + g, 0x33, 2 * g);
    fill(v + g, 0x44, 2 * g);

    EXPECT_EQ(read(k, 1)[0], 0x11);
    EXPECT_EQ(read(v, 1)[0], 0x22);
    EXPECT_EQ(read(k + 3 * g - 1, 1)[0], 0x33);
    EXPECT_EQ(read(v + 3 * g - 1, 1)[0], 0x44);
}

TEST(VmmRegionTest, GrowingToAMappedSizeOrLessChangesNothing)
{
    size_t const g = granule();
    rt::VmmRegion region(2 * g, 0);
    ASSERT_TRUE(region.growTo(g));
    EXPECT_TRUE(region.growTo(g));
    EXPECT_TRUE(region.growTo(0));
    EXPECT_EQ(region.mappedPerHalf(), g);
}

TEST(VmmRegionTest, GrowthIsCappedAtTheHalfSize)
{
    size_t const g = granule();
    rt::VmmRegion region(2 * g, 0);
    ASSERT_TRUE(region.growTo(100 * g));
    EXPECT_EQ(region.mappedPerHalf(), 2 * g);
}

TEST(VmmRegionTest, ShrinkReleasesWholeChunksAtOrAboveTheRequestedSize)
{
    size_t const g = granule();
    rt::VmmRegion region(8 * g, 0);
    ASSERT_TRUE(region.growTo(2 * g));
    ASSERT_TRUE(region.growTo(5 * g));
    EXPECT_EQ(region.mappedPerHalf(), 5 * g);

    region.shrinkTo(4 * g);
    EXPECT_EQ(region.mappedPerHalf(), 5 * g) << "the last chunk spans 2g..5g and cannot be split";
    region.shrinkTo(2 * g);
    EXPECT_EQ(region.mappedPerHalf(), 2 * g);
    region.shrinkTo(0);
    EXPECT_EQ(region.mappedPerHalf(), 0u);

    ASSERT_TRUE(region.growTo(g));
    fill(region.base(), 0x5A, g);
    EXPECT_EQ(read(region.base(), 1)[0], 0x5A);
}

TEST(VmmRegionTest, FailedGrowthLeavesTheMappingUnchanged)
{
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&freeBytes, &totalBytes), cudaSuccess);
    size_t const g = granule();
    size_t const half = (totalBytes + g - 1) / g * g;
    rt::VmmRegion region(half, 0);
    ASSERT_TRUE(region.growTo(g));

    EXPECT_FALSE(region.growTo(half)) << "two halves of the whole device memory cannot be backed";
    EXPECT_EQ(region.mappedPerHalf(), g);
    fill(region.base(), 0x77, g);
    EXPECT_EQ(read(region.base(), 1)[0], 0x77);
    EXPECT_TRUE(region.growTo(2 * g));
}
