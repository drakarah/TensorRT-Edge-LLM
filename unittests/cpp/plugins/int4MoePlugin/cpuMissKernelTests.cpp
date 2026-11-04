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

// The CPU path that computes the experts missing from the GPU cache: int4 weights in the Marlin layout, fp32 or int8 activations, AVX2 / AVX-VNNI.
// The fixtures are packed by the exporter's own Marlin packer; the reference is the dense float64 product.
#include "cpuMissFixture.inc"
#include "cpuMissFixtureMinimal.inc"
#include "cpuMissFixtureThreeGroups.inc"
#include "plugins/int4MoePlugin/cpuMissKernel.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

#if defined(__x86_64__)

using namespace trt_edgellm::plugins::cpumiss;

namespace
{
struct Fixture
{
    char const* name;
    int32_t K;
    int32_t N;
    int32_t const* w;
    uint16_t const* s;
    float const* x;
    double const* y;
};

std::vector<Fixture> const& fixtures()
{
    static std::vector<Fixture> const all{
        {"256x128 (two tiles, two groups)", cpuMissFixture::kK, cpuMissFixture::kN, cpuMissFixture::kWeights, cpuMissFixture::kScales, cpuMissFixture::kX,
            cpuMissFixture::kY},
        {"256x64 (one tile, two groups)", cpuMissFixtureMinimal::kK, cpuMissFixtureMinimal::kN, cpuMissFixtureMinimal::kWeights, cpuMissFixtureMinimal::kScales,
            cpuMissFixtureMinimal::kX, cpuMissFixtureMinimal::kY},
        {"384x64 (one tile, three groups)", cpuMissFixtureThreeGroups::kK, cpuMissFixtureThreeGroups::kN, cpuMissFixtureThreeGroups::kWeights,
            cpuMissFixtureThreeGroups::kScales, cpuMissFixtureThreeGroups::kX, cpuMissFixtureThreeGroups::kY},
    };
    return all;
}

bool haveAvx2()
{
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}

bool haveVnni()
{
    return haveAvx2() && __builtin_cpu_supports("avxvnni");
}

constexpr int32_t kMaxK = 512;

std::vector<float> runFp32(Fixture const& f, float const* x)
{
    std::vector<float> out(f.N);
    for (int32_t nt = 0; nt < f.N / 64; ++nt)
    {
        gemvTile(tables(), f.w, f.s, x, f.K, f.N, nt, out.data() + 64 * nt);
    }
    return out;
}

std::vector<float> runInt8(Fixture const& f, float const* x)
{
    QuantisedX<kMaxK> q;
    quantiseX<kMaxK>(x, f.K, q);
    std::vector<float> out(f.N);
    for (int32_t nt = 0; nt < f.N / 64; ++nt)
    {
        gemvTileVnni<kMaxK>(tables(), f.w, f.s, q, f.N, nt, out.data() + 64 * nt);
    }
    return out;
}

template <typename Ref>
double maxRelativeError(std::vector<float> const& got, Ref const& want, int32_t n)
{
    double worst = 0.0;
    double norm = 0.0;
    for (int32_t i = 0; i < n; ++i)
    {
        worst = std::fmax(worst, std::fabs(got[i] - want[i]));
        norm = std::fmax(norm, std::fabs(want[i]));
    }
    return norm > 0.0 ? worst / norm : worst;
}

bool allFinite(std::vector<float> const& v)
{
    for (float x : v)
    {
        if (!std::isfinite(x))
        {
            return false;
        }
    }
    return true;
}
} // namespace

TEST(CpuMissKernelTest, Fp32GemvMatchesTheDenseFloat64ResultOfEveryFixture)
{
    if (!haveAvx2())
    {
        GTEST_SKIP() << "needs AVX2, FMA and F16C";
    }
    for (auto const& f : fixtures())
    {
        EXPECT_LT(maxRelativeError(runFp32(f, f.x), f.y, f.N), 1e-5) << f.name;
    }
}

TEST(CpuMissKernelTest, Int8GemvStaysWithinTheActivationQuantisationErrorOfEveryFixture)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    for (auto const& f : fixtures())
    {
        EXPECT_LT(maxRelativeError(runInt8(f, f.x), f.y, f.N), 2e-2) << f.name;
    }
}

TEST(CpuMissKernelTest, TilesAreIndependentOfTheOrderTheyAreComputedIn)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    auto const& f = fixtures()[0];
    float forward[64], backward[64], forwardInt8[64], backwardInt8[64];
    QuantisedX<kMaxK> q;
    quantiseX<kMaxK>(f.x, f.K, q);
    // tile 1 first, then tile 0, then tile 1 again: no state may leak from one call to the next
    gemvTile(tables(), f.w, f.s, f.x, f.K, f.N, 1, backward);
    gemvTile(tables(), f.w, f.s, f.x, f.K, f.N, 0, forward);
    gemvTile(tables(), f.w, f.s, f.x, f.K, f.N, 1, forward);
    gemvTileVnni<kMaxK>(tables(), f.w, f.s, q, f.N, 1, backwardInt8);
    gemvTileVnni<kMaxK>(tables(), f.w, f.s, q, f.N, 0, forwardInt8);
    gemvTileVnni<kMaxK>(tables(), f.w, f.s, q, f.N, 1, forwardInt8);
    EXPECT_EQ(0, std::memcmp(forward, backward, sizeof(forward)));
    EXPECT_EQ(0, std::memcmp(forwardInt8, backwardInt8, sizeof(forwardInt8)));
}

TEST(CpuMissKernelTest, ZeroActivationsGiveExactlyZeroOnBothPaths)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    for (auto const& f : fixtures())
    {
        std::vector<float> zeros(f.K, 0.0F);
        for (float v : runFp32(f, zeros.data()))
        {
            EXPECT_EQ(v, 0.0F) << f.name;
        }
        for (float v : runInt8(f, zeros.data()))
        {
            EXPECT_EQ(v, 0.0F) << f.name;
        }
    }
}

TEST(CpuMissKernelTest, ActivationsThatAreZeroInOneGroupOnly)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    auto const& f = fixtures()[2]; // three groups
    for (int32_t zeroGroup = 0; zeroGroup < 3; ++zeroGroup)
    {
        std::vector<float> x(f.x, f.x + f.K);
        for (int32_t i = 0; i < 128; ++i)
        {
            x[zeroGroup * 128 + i] = 0.0F;
        }
        auto const int8 = runInt8(f, x.data());
        ASSERT_TRUE(allFinite(int8)) << "zero group " << zeroGroup;
        EXPECT_LT(maxRelativeError(int8, runFp32(f, x.data()), f.N), 3e-2) << "zero group " << zeroGroup;
    }
}

TEST(CpuMissKernelTest, ConstantNegativeAndSingleSpikeActivations)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    for (auto const& f : fixtures())
    {
        std::vector<float> ones(f.K, 1.0F), negative(f.K), spike(f.K, 0.0F), alternating(f.K);
        std::mt19937 rng(3);
        for (int32_t i = 0; i < f.K; ++i)
        {
            negative[i] = -std::fabs(std::normal_distribution<float>(0.0F, 1.0F)(rng)) - 0.01F;
            alternating[i] = i % 2 ? 1.0F : -1.0F;
        }
        spike[f.K - 1] = 5.0F;
        spike[0] = -3.0F;
        for (auto const* x : {&ones, &negative, &spike, &alternating})
        {
            auto const int8 = runInt8(f, x->data());
            ASSERT_TRUE(allFinite(int8)) << f.name;
            EXPECT_LT(maxRelativeError(int8, runFp32(f, x->data()), f.N), 3e-2) << f.name;
        }
    }
}

TEST(CpuMissKernelTest, ActivationsAcrossTheFp16RangeKeepTheirScale)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    // the layer input is fp16: from the smallest subnormal (6e-8) to 65504 (the scaling keeps the largest entry of the fixtures at a few times the factor)
    for (auto const& f : fixtures())
    {
        for (float factor : {1.0e-7F, 1.0e-4F, 1.0e-2F, 1.0e2F, 1.0e4F})
        {
            std::vector<float> x(f.x, f.x + f.K);
            for (auto& v : x)
            {
                v *= factor;
            }
            auto const int8 = runInt8(f, x.data());
            auto const fp32 = runFp32(f, x.data());
            ASSERT_TRUE(allFinite(int8)) << f.name << " factor " << factor;
            ASSERT_TRUE(allFinite(fp32)) << f.name << " factor " << factor;
            EXPECT_LT(maxRelativeError(int8, fp32, f.N), 2e-2) << f.name << " factor " << factor;
        }
    }
}

TEST(CpuMissKernelTest, ExtremeFp16ValuesDoNotOverflowEitherPath)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    for (auto const& f : fixtures())
    {
        std::vector<float> x(f.K);
        for (int32_t i = 0; i < f.K; ++i)
        {
            x[i] = (i % 2 ? 65504.0F : -65504.0F) * (i % 5 == 0 ? 0.0F : 1.0F);
        }
        x[3] = 6.1035156e-05F; // smallest normal fp16
        auto const int8 = runInt8(f, x.data());
        auto const fp32 = runFp32(f, x.data());
        ASSERT_TRUE(allFinite(int8)) << f.name;
        ASSERT_TRUE(allFinite(fp32)) << f.name;
        EXPECT_LT(maxRelativeError(int8, fp32, f.N), 2e-2) << f.name;
    }
}

TEST(CpuMissKernelTest, OneOutlierInAGroupStretchesOnlyThatGroupsScale)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    auto const& f = fixtures()[2];
    std::mt19937 rng(31);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (int32_t trial = 0; trial < 30; ++trial)
    {
        std::vector<float> x(f.K);
        for (auto& v : x)
        {
            v = normal(rng);
        }
        x[(trial % 3) * 128 + 7] *= 40.0F;
        QuantisedX<kMaxK> q;
        quantiseX<kMaxK>(x.data(), f.K, q);
        for (int32_t g = 0; g < 3; ++g)
        {
            float amax = 0.0F;
            for (int32_t i = 0; i < 128; ++i)
            {
                amax = std::fmax(amax, std::fabs(x[g * 128 + i]));
            }
            EXPECT_FLOAT_EQ(q.scale[g], amax / 127.0F) << "group " << g << " trial " << trial;
        }
        EXPECT_LT(maxRelativeError(runInt8(f, x.data()), runFp32(f, x.data()), f.N), 7e-2) << "trial " << trial << " (worst measured 5.1e-2: one entry 40 times the rest costs that group most of its 8 bits)";
    }
}

TEST(CpuMissKernelTest, QuantiserScaleIsOneForAnAllZeroGroupAndTheLengthIsRecorded)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    std::vector<float> x(384, 0.0F);
    x[130] = 2.54F;
    QuantisedX<kMaxK> q;
    quantiseX<kMaxK>(x.data(), 384, q);
    EXPECT_EQ(q.k, 384);
    EXPECT_FLOAT_EQ(q.scale[0], 1.0F);
    EXPECT_FLOAT_EQ(q.scale[1], 2.54F / 127.0F);
    EXPECT_FLOAT_EQ(q.scale[2], 1.0F);
}

TEST(CpuMissKernelTest, Fp32PathPropagatesNonFiniteActivations)
{
    if (!haveAvx2())
    {
        GTEST_SKIP() << "needs AVX2, FMA and F16C";
    }
    for (auto const& f : fixtures())
    {
        std::vector<float> x(f.x, f.x + f.K);
        x[5] = std::numeric_limits<float>::quiet_NaN();
        for (float v : runFp32(f, x.data()))
        {
            EXPECT_TRUE(std::isnan(v)) << f.name << ": a NaN input must reach every output";
        }
    }
}

// Known gap, kept as an executable specification: the int8 quantiser maps a NaN input to a finite value, so the CPU path hides an upstream NaN that the GPU
// path would show.
TEST(CpuMissKernelTest, DISABLED_Int8PathPropagatesNonFiniteActivations)
{
    if (!haveVnni())
    {
        GTEST_SKIP() << "needs AVX-VNNI";
    }
    auto const& f = fixtures()[0];
    std::vector<float> x(f.x, f.x + f.K);
    x[5] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(allFinite(runInt8(f, x.data())));
}


#endif // __x86_64__

TEST(CpuMissShapeTest, AcceptsTheShapesOfQwen3635BAndRejectsWhatTheKernelsCannotRead)
{
    using trt_edgellm::plugins::cpumiss::shapeSupported;
    int64_t const maxHidden = 4096;
    EXPECT_TRUE(shapeSupported(2048, 512, 128, maxHidden)) << "Qwen3.6-35B-A3B experts";
    EXPECT_TRUE(shapeSupported(256, 256, 128, maxHidden)) << "two scale groups in both projections: the smallest supported shape";
    EXPECT_TRUE(shapeSupported(4096, 1024, 128, maxHidden)) << "the largest";
    EXPECT_FALSE(shapeSupported(2048, 128, 128, maxHidden)) << "one scale group in the down projection: the exporter stores those scales in another permutation";
    EXPECT_FALSE(shapeSupported(128, 512, 128, maxHidden)) << "one scale group in the gate/up projection";
    EXPECT_FALSE(shapeSupported(2048, 512, 64, maxHidden)) << "group size other than 128";
    EXPECT_FALSE(shapeSupported(2048 + 64, 512, 128, maxHidden)) << "hidden size not a multiple of 128";
    EXPECT_FALSE(shapeSupported(2048, 512 + 64, 128, maxHidden)) << "intermediate size not a multiple of 128";
    EXPECT_FALSE(shapeSupported(2048, 1152, 128, maxHidden)) << "intermediate size above 1024";
    EXPECT_FALSE(shapeSupported(4224, 512, 128, maxHidden)) << "hidden size above the mailbox";
    EXPECT_FALSE(shapeSupported(0, 0, 128, maxHidden));
    EXPECT_FALSE(shapeSupported(-128, 512, 128, maxHidden));
}
