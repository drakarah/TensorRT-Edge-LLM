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

#include "common/cudaUtils.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
struct Counters
{
    uint32_t nonFinite;
    float maxAbs;
    uint32_t nearLimit;
};

Counters watch(std::vector<half> const& values)
{
    half* dValues{};
    uint32_t* dCounters{};
    CUDA_CHECK(cudaMalloc(&dValues, values.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dCounters, 3 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dValues, values.data(), values.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dCounters, 0, 3 * sizeof(uint32_t)));
    launchFiniteWatch(dValues, static_cast<int64_t>(values.size()), dCounters, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t raw[3];
    CUDA_CHECK(cudaMemcpy(raw, dCounters, sizeof(raw), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dValues));
    CUDA_CHECK(cudaFree(dCounters));
    Counters c{};
    c.nonFinite = raw[0];
    std::memcpy(&c.maxAbs, &raw[1], sizeof(float));
    c.nearLimit = raw[2];
    return c;
}
} // namespace

TEST(FiniteWatchTest, CountsNonFiniteValuesAndTracksTheLargestFiniteMagnitude)
{
    std::vector<half> v(100000, __float2half(1.0F));
    v[3] = __float2half(-40000.0F);
    v[50000] = __float2half(std::numeric_limits<float>::quiet_NaN());
    v[50001] = __float2half(std::numeric_limits<float>::infinity());
    v[99999] = __float2half(-std::numeric_limits<float>::infinity());
    auto const c = watch(v);
    EXPECT_EQ(c.nonFinite, 3U);
    EXPECT_FLOAT_EQ(c.maxAbs, 40000.0F) << "infinities and NaN do not count as the largest finite value";
    EXPECT_EQ(c.nearLimit, 1U);
}

TEST(FiniteWatchTest, CleanInputLeavesTheCountersAtZero)
{
    std::vector<half> v(4096, __float2half(0.5F));
    auto const c = watch(v);
    EXPECT_EQ(c.nonFinite, 0U);
    EXPECT_FLOAT_EQ(c.maxAbs, 0.5F);
    EXPECT_EQ(c.nearLimit, 0U);
}

TEST(FiniteWatchTest, CountersAccumulateAcrossLaunches)
{
    std::vector<half> v(1000, __float2half(31000.0F));
    half* dValues{};
    uint32_t* dCounters{};
    CUDA_CHECK(cudaMalloc(&dValues, v.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dCounters, 3 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dValues, v.data(), v.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dCounters, 0, 3 * sizeof(uint32_t)));
    for (int32_t i = 0; i < 3; ++i)
    {
        launchFiniteWatch(dValues, static_cast<int64_t>(v.size()), dCounters, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t raw[3];
    CUDA_CHECK(cudaMemcpy(raw, dCounters, sizeof(raw), cudaMemcpyDeviceToHost));
    EXPECT_EQ(raw[0], 0U);
    EXPECT_EQ(raw[2], 3000U);
    CUDA_CHECK(cudaFree(dValues));
    CUDA_CHECK(cudaFree(dCounters));
}

TEST(FiniteWatchTest, SizesThatAreNotAMultipleOfTheBlockOrTheWarp)
{
    for (size_t n : {1U, 2U, 31U, 32U, 33U, 255U, 256U, 257U, 8191U, 8192U, 8193U, 100003U})
    {
        std::vector<half> v(n, __float2half(2.0F));
        v[n - 1] = __float2half(-7.0F); // the last element lives in a partial warp
        auto const c = watch(v);
        EXPECT_EQ(c.nonFinite, 0U) << n;
        EXPECT_FLOAT_EQ(c.maxAbs, 7.0F) << "n = " << n;
        EXPECT_EQ(c.nearLimit, 0U) << n;
    }
}

TEST(FiniteWatchTest, EmptyInputTouchesNothing)
{
    half* dValues{};
    uint32_t* dCounters{};
    CUDA_CHECK(cudaMalloc(&dValues, sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dCounters, 3 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dCounters, 0, 3 * sizeof(uint32_t)));
    launchFiniteWatch(dValues, 0, dCounters, nullptr);
    launchFiniteWatch(dValues, -5, dCounters, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t raw[3];
    CUDA_CHECK(cudaMemcpy(raw, dCounters, sizeof(raw), cudaMemcpyDeviceToHost));
    EXPECT_EQ(raw[0], 0U);
    EXPECT_EQ(raw[1], 0U);
    EXPECT_EQ(raw[2], 0U);
    CUDA_CHECK(cudaFree(dValues));
    CUDA_CHECK(cudaFree(dCounters));
}

TEST(FiniteWatchTest, TheNearLimitThresholdIsInclusive)
{
    // fp16 spacing near 30000 is 16: 29984 is below the threshold, 30000 is on it, 30016 above
    std::vector<half> v{__float2half(29984.0F), __float2half(30000.0F), __float2half(30016.0F), __float2half(-30000.0F), __float2half(-29984.0F)};
    auto const c = watch(v);
    EXPECT_EQ(c.nearLimit, 3U);
    EXPECT_FLOAT_EQ(c.maxAbs, 30016.0F);
}

TEST(FiniteWatchTest, ExtremeFiniteValuesAreNotNonFinite)
{
    std::vector<half> v{__float2half(65504.0F), __float2half(-65504.0F), __float2half(0.0F), __float2half(-0.0F), __ushort_as_half(0x0001), __ushort_as_half(0x03FF)};
    auto const c = watch(v);
    EXPECT_EQ(c.nonFinite, 0U) << "the largest fp16 value, zeros and subnormals are finite";
    EXPECT_FLOAT_EQ(c.maxAbs, 65504.0F);
    EXPECT_EQ(c.nearLimit, 2U);
}

TEST(FiniteWatchTest, EveryKindOfNonFiniteBitPatternIsCounted)
{
    std::vector<half> v{__ushort_as_half(0x7C00), __ushort_as_half(0xFC00), // +inf, -inf
        __ushort_as_half(0x7E00), __ushort_as_half(0xFE00),               // quiet NaNs
        __ushort_as_half(0x7C01), __ushort_as_half(0xFDFF),               // signalling NaN patterns
        __float2half(1.0F)};
    auto const c = watch(v);
    EXPECT_EQ(c.nonFinite, 6U);
    EXPECT_FLOAT_EQ(c.maxAbs, 1.0F);
}

TEST(FiniteWatchTest, AllNonFiniteInputLeavesTheMagnitudeAtZero)
{
    std::vector<half> v(5000, __ushort_as_half(0x7E00));
    auto const c = watch(v);
    EXPECT_EQ(c.nonFinite, 5000U);
    EXPECT_EQ(c.maxAbs, 0.0F);
    EXPECT_EQ(c.nearLimit, 0U);
}

TEST(FiniteWatchTest, TenMillionValuesWithTheMaximumInTheLastBlock)
{
    size_t const n = 10000000;
    std::vector<half> v(n, __float2half(0.25F));
    v[n - 1] = __float2half(12344.0F);
    v[n / 2] = __float2half(-100.0F);
    auto const c = watch(v);
    EXPECT_EQ(c.nonFinite, 0U);
    EXPECT_FLOAT_EQ(c.maxAbs, 12344.0F);
}
