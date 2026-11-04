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

#include <gtest/gtest.h>

#include "kernels/posEncoding/kvInt4Quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <limits>
#include <set>
#include <vector>

using namespace trt_edgellm::kernel;

namespace
{

//! Every finite fp16 value, in increasing order (-0 and +0 both present).
std::vector<float> allFiniteHalfValues()
{
    std::vector<float> values;
    values.reserve(65536);
    for (uint32_t bits = 0; bits < 65536; ++bits)
    {
        half const h = __ushort_as_half(static_cast<unsigned short>(bits));
        float const f = __half2float(h);
        if (std::isfinite(f))
        {
            values.push_back(f);
        }
    }
    std::sort(values.begin(), values.end());
    return values;
}

float halfBitsToFloat(uint32_t bits)
{
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

std::vector<float> const kScales{std::ldexp(1.0F, -24), std::ldexp(1.0F, -14), 0.001F, 0.045F, 0.5F, 1.0F, 3.3F,
    17.0F / 7.0F, 1000.0F, 65504.0F};

} // namespace

TEST(KvInt4Quant, ScaleOfZeroNonFiniteAndOutOfRangeGroupsIsZero)
{
    EXPECT_EQ(vInt4ScaleFromAbsMax(0.0F), 0.0F);
    EXPECT_EQ(vInt4ScaleFromAbsMax(std::numeric_limits<float>::quiet_NaN()), 0.0F);
    EXPECT_EQ(vInt4ScaleFromAbsMax(std::numeric_limits<float>::infinity()), 0.0F);
    // Anything above the largest fp16 value rounds up to infinity.
    EXPECT_EQ(vInt4ScaleFromAbsMax(7.0F * 65505.0F), 0.0F);
    EXPECT_EQ(vInt4ScaleFromAbsMax(7.0F * 1.0e6F), 0.0F);
    // The scale is rounded up, so any positive maximum, however small, gets at least the smallest fp16 subnormal.
    EXPECT_EQ(vInt4ScaleFromAbsMax(7.0F * std::ldexp(1.0F, -25)), std::ldexp(1.0F, -24));
    EXPECT_EQ(vInt4ScaleFromAbsMax(7.0F * std::ldexp(1.0F, -24)), std::ldexp(1.0F, -24));
    EXPECT_EQ(vInt4ScaleFromAbsMax(1.0e-30F), std::ldexp(1.0F, -24));
    // The largest representable scale.
    EXPECT_EQ(vInt4ScaleFromAbsMax(7.0F * 65504.0F), 65504.0F);
}

TEST(KvInt4Quant, ScaleIsStoredAsFp16)
{
    for (uint32_t bits = 1; bits < 0x7C00; ++bits)
    {
        float const absMax = halfBitsToFloat(bits);
        float const scale = vInt4ScaleFromAbsMax(absMax);
        if (scale == 0.0F)
        {
            continue;
        }
        EXPECT_EQ(__half2float(__float2half_rn(scale)), scale) << "absMax=" << absMax;
    }
}

TEST(KvInt4Quant, TheGroupMaximumQuantisesToTheTopCodeWhenTheScaleIsNormal)
{
    // The fp16 rounding of the scale may fall below absMax / 7, so absMax / scale can slightly exceed 7: the clamp has to
    // bring it back to exactly 7, for every finite fp16 maximum whose scale is a normal fp16 number.
    constexpr float kMinNormalHalf = 6.103515625e-05F;
    int32_t checked = 0;
    for (uint32_t bits = 1; bits < 0x7C00; ++bits)
    {
        float const absMax = halfBitsToFloat(bits);
        float const scale = vInt4ScaleFromAbsMax(absMax);
        if (scale < kMinNormalHalf)
        {
            continue;
        }
        ++checked;
        ASSERT_EQ(vInt4Code(absMax, scale), kVInt4MaxCode) << "absMax=" << absMax;
        ASSERT_EQ(vInt4Code(-absMax, scale), -kVInt4MaxCode) << "absMax=" << absMax;
    }
    EXPECT_GT(checked, 25000);
}

TEST(KvInt4Quant, TheGroupMaximumIsNeverClippedAndReconstructsWithinHalfAStep)
{
    // Holds for every finite fp16 maximum, including the ones whose scale is a coarse fp16 subnormal.
    int32_t checked = 0;
    for (uint32_t bits = 1; bits < 0x7C00; ++bits)
    {
        float const absMax = halfBitsToFloat(bits);
        float const scale = vInt4ScaleFromAbsMax(absMax);
        ASSERT_GT(scale, 0.0F) << "absMax=" << absMax;
        ASSERT_LE(absMax / scale, static_cast<float>(kVInt4MaxCode) * (1.0F + 1.0e-6F)) << "absMax=" << absMax;
        int32_t const code = vInt4Code(absMax, scale);
        ASSERT_GE(code, 0) << "absMax=" << absMax;
        ASSERT_LE(code, kVInt4MaxCode) << "absMax=" << absMax;
        ASSERT_LE(std::fabs(static_cast<float>(code) * scale - absMax), 0.5F * scale * (1.0F + 1.0e-5F))
            << "absMax=" << absMax;
        ++checked;
    }
    EXPECT_GT(checked, 30000);
}

TEST(KvInt4Quant, CodesStayInRangeAreAntisymmetricAndMonotone)
{
    auto const values = allFiniteHalfValues();
    for (float const scale : kScales)
    {
        int32_t previous = -kVInt4MaxCode - 1;
        for (float const x : values)
        {
            int32_t const code = vInt4Code(x, scale);
            ASSERT_GE(code, -kVInt4MaxCode) << "x=" << x << " scale=" << scale;
            ASSERT_LE(code, kVInt4MaxCode) << "x=" << x << " scale=" << scale;
            ASSERT_EQ(vInt4Code(-x, scale), -code) << "x=" << x << " scale=" << scale;
            ASSERT_GE(code, previous) << "not monotone at x=" << x << " scale=" << scale;
            previous = code;
        }
    }
}

TEST(KvInt4Quant, TiesRoundToEvenAndSaturateAtSeven)
{
    std::vector<std::pair<float, int32_t>> const cases{{0.0F, 0}, {0.49F, 0}, {0.5F, 0}, {0.51F, 1}, {1.5F, 2},
        {2.5F, 2}, {3.5F, 4}, {4.5F, 4}, {5.5F, 6}, {6.5F, 6}, {6.51F, 7}, {7.0F, 7}, {7.5F, 7}, {100.0F, 7}};
    for (auto const& [x, expected] : cases)
    {
        EXPECT_EQ(vInt4Code(x, 1.0F), expected) << "x=" << x;
        EXPECT_EQ(vInt4Code(-x, 1.0F), -expected) << "x=" << -x;
    }
}

TEST(KvInt4Quant, NonFiniteInputsStayInsideTheCodeRange)
{
    float const inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(vInt4Code(inf, 0.5F), kVInt4MaxCode);
    EXPECT_EQ(vInt4Code(-inf, 0.5F), -kVInt4MaxCode);
    // NaN is dropped by fmax/fmin and lands on the lower clamp, as in the int8 path (which gives -127).
    EXPECT_EQ(vInt4Code(std::numeric_limits<float>::quiet_NaN(), 0.5F), -kVInt4MaxCode);
}

TEST(KvInt4Quant, ReconstructionErrorIsHalfAStepInsideTheRange)
{
    auto const values = allFiniteHalfValues();
    for (float const scale : kScales)
    {
        for (float const x : values)
        {
            if (std::fabs(x) > static_cast<float>(kVInt4MaxCode) * scale)
            {
                continue;
            }
            float const reconstructed = static_cast<float>(vInt4Code(x, scale)) * scale;
            // The extra relative slack covers the float rounding of x * (1 / scale).
            ASSERT_LE(std::fabs(reconstructed - x), 0.5F * scale * (1.0F + 1.0e-5F) + 1.0e-12F)
                << "x=" << x << " scale=" << scale;
        }
    }
}

TEST(KvInt4Quant, StaticModeUsesAtMostFifteenInt8Levels)
{
    auto const values = allFiniteHalfValues();
    for (float const vScale : {0.001F, 0.045F, 0.1F, 0.7F, 2.0F})
    {
        float const scale4 = vInt4StaticScale(vScale);
        std::set<int32_t> expectedLevels;
        for (int32_t code = -kVInt4MaxCode; code <= kVInt4MaxCode; ++code)
        {
            expectedLevels.insert(vInt4EmulatedInt8(static_cast<float>(code) * scale4, scale4, vScale));
        }
        EXPECT_EQ(expectedLevels.size(), 15U) << "vScale=" << vScale;
        std::set<int32_t> seen;
        int32_t previous = -128;
        for (float const x : values)
        {
            int32_t const out = vInt4EmulatedInt8(x, scale4, vScale);
            ASSERT_TRUE(expectedLevels.count(out) == 1) << "x=" << x << " out=" << out << " vScale=" << vScale;
            ASSERT_GE(out, previous) << "not monotone at x=" << x;
            previous = out;
            seen.insert(out);
        }
        // The extremes reach the int8 limits: the 4-bit range spans the whole int8 range.
        EXPECT_EQ(*seen.begin(), -127);
        EXPECT_EQ(*seen.rbegin(), 127);
    }
}

TEST(KvInt4Quant, EmulatedInt8NeverUsesMinusOneHundredTwentyEight)
{
    auto const values = allFiniteHalfValues();
    for (float const vScale : kScales)
    {
        for (float const scale4 : {0.0F, vInt4StaticScale(vScale), 0.01F, 5.0F})
        {
            for (float const x : values)
            {
                int32_t const out = vInt4EmulatedInt8(x, scale4, vScale);
                ASSERT_GE(out, -127) << "x=" << x;
                ASSERT_LE(out, 127) << "x=" << x;
            }
        }
    }
}

TEST(KvInt4Quant, ZeroScaleGivesAllZeroOutput)
{
    for (float const x : allFiniteHalfValues())
    {
        ASSERT_EQ(vInt4EmulatedInt8(x, 0.0F, 0.045F), 0) << "x=" << x;
    }
}

TEST(KvInt4Quant, StaticScaleSpansTheInt8Range)
{
    for (float const vScale : {0.001F, 0.045F, 1.0F, 12.5F})
    {
        EXPECT_NEAR(vInt4StaticScale(vScale) * static_cast<float>(kVInt4MaxCode), 127.0F * vScale, 1.0e-4F * vScale);
    }
}

TEST(KvInt4Quant, GroupSizeDividesTheHeadDimension)
{
    EXPECT_EQ(256 % kVInt4GroupSize, 0);
    EXPECT_EQ(kVInt4GroupSize % 8, 0);
}
