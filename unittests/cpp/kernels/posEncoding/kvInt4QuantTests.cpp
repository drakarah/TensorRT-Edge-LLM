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

TEST(KvInt4Asym, DegenerateRangesGiveZeroGroups)
{
    float const inf = std::numeric_limits<float>::infinity();
    // No finite element at all (min stays +inf, max stays -inf), or an infinity in the group.
    for (auto const& [lo, hi] : std::vector<std::pair<float, float>>{{inf, -inf}, {-inf, 1.0F}, {0.0F, inf}, {-inf, inf},
             {-70000.0F, 1.0F}, {0.0F, 70000.0F}})
    {
        VInt4Asym const p = vInt4AsymParams(lo, hi);
        EXPECT_EQ(p.scale, 0.0F) << lo << " " << hi;
        EXPECT_EQ(p.zero, 0.0F) << lo << " " << hi;
        EXPECT_EQ(vInt4AsymCode(0.5F, p), 0);
        EXPECT_EQ(vInt4AsymDequant(0, p), 0.0F);
    }
    // An inverted range (min above max) is not a group.
    EXPECT_EQ(vInt4AsymParams(2.0F, 1.0F).scale, 0.0F);
}

TEST(KvInt4Asym, ConstantGroupReconstructsTheValue)
{
    for (float const v : {0.0F, 3.0F, -2.5F, 65504.0F, -65504.0F, 1.0e-7F})
    {
        VInt4Asym const p = vInt4AsymParams(v, v);
        int32_t const code = vInt4AsymCode(v, p);
        EXPECT_NEAR(vInt4AsymDequant(code, p), v, std::fabs(v) * 1.0e-3F + 1.0e-7F) << v;
    }
    // Values exactly representable in fp16 are reconstructed exactly.
    for (float const v : {0.0F, 3.0F, -2.5F, 65504.0F})
    {
        VInt4Asym const p = vInt4AsymParams(v, v);
        EXPECT_EQ(vInt4AsymDequant(vInt4AsymCode(v, p), p), v) << v;
    }
}

TEST(KvInt4Asym, NeitherEndOfTheRangeIsClippedAndTheErrorIsHalfAStep)
{
    // Deterministic pseudo-random ranges over the whole fp16 range, including subnormal and near-maximum values.
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33);
    };
    auto randomHalf = [&]() {
        float f;
        do
        {
            f = halfBitsToFloat(next() & 0xFFFFu);
        } while (!std::isfinite(f));
        return f;
    };
    int32_t checked = 0;
    for (int32_t trial = 0; trial < 200000; ++trial)
    {
        float lo = randomHalf();
        float hi = randomHalf();
        if (lo > hi)
        {
            std::swap(lo, hi);
        }
        VInt4Asym const p = vInt4AsymParams(lo, hi);
        if (!(p.scale > 0.0F))
        {
            continue;
        }
        ++checked;
        ASSERT_LE(p.zero, lo) << lo << " " << hi;
        ASSERT_LE((hi - p.zero) / p.scale, static_cast<float>(kVInt4AsymMaxCode) * (1.0F + 1.0e-6F)) << lo << " " << hi;
        for (float const x : {lo, hi, 0.5F * (lo + hi), lo + 0.25F * (hi - lo)})
        {
            int32_t const code = vInt4AsymCode(x, p);
            ASSERT_GE(code, 0);
            ASSERT_LE(code, kVInt4AsymMaxCode);
            float const err = std::fabs(vInt4AsymDequant(code, p) - x);
            ASSERT_LE(err, 0.5F * p.scale * (1.0F + 1.0e-5F) + 2.0e-7F * std::max(std::fabs(x), std::fabs(p.zero)) + 1.0e-12F)
                << "x=" << x << " lo=" << lo << " hi=" << hi << " scale=" << p.scale;
        }
    }
    EXPECT_GT(checked, 100000);
}

TEST(KvInt4Asym, CodesAreMonotoneInTheValue)
{
    VInt4Asym const p = vInt4AsymParams(-3.0F, 5.0F);
    int32_t previous = -1;
    for (float x = -4.0F; x <= 6.0F; x += 0.0173F)
    {
        int32_t const code = vInt4AsymCode(x, p);
        ASSERT_GE(code, previous) << x;
        previous = code;
    }
    EXPECT_EQ(vInt4AsymCode(-100.0F, p), 0);
    EXPECT_EQ(vInt4AsymCode(100.0F, p), kVInt4AsymMaxCode);
}

TEST(KvInt4Hadamard, MatchesTheNaiveMatrixAndIsOrthonormal)
{
    uint64_t state = 12345;
    auto uniform = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<int32_t>(state >> 40) - (1 << 23)) / static_cast<float>(1 << 23) * 5.0F;
    };
    for (int32_t trial = 0; trial < 50; ++trial)
    {
        std::vector<float> x(128);
        for (auto& v : x)
        {
            v = uniform();
        }
        std::vector<float> fast = x;
        vInt4Hadamard128Host(fast.data());
        double normIn = 0.0;
        double normOut = 0.0;
        for (int32_t i = 0; i < 128; ++i)
        {
            double naive = 0.0;
            for (int32_t j = 0; j < 128; ++j)
            {
                naive += ((__builtin_popcount(static_cast<unsigned>(i & j)) & 1) ? -1.0 : 1.0) * x[j];
            }
            naive /= std::sqrt(128.0);
            ASSERT_NEAR(fast[i], naive, 2.0e-5) << "i=" << i;
            normIn += static_cast<double>(x[i]) * x[i];
            normOut += static_cast<double>(fast[i]) * fast[i];
        }
        EXPECT_NEAR(normOut, normIn, 1.0e-4 * normIn);
        vInt4Hadamard128Host(fast.data());
        for (int32_t i = 0; i < 128; ++i)
        {
            ASSERT_NEAR(fast[i], x[i], 2.0e-5) << "round trip i=" << i;
        }
    }
}

TEST(KvInt4Hadamard, KnownVectors)
{
    std::vector<float> unit(128, 0.0F);
    unit[0] = 1.0F;
    vInt4Hadamard128Host(unit.data());
    for (float const v : unit)
    {
        EXPECT_NEAR(v, kVInt4HadamardScale, 1.0e-7F);
    }
    std::vector<float> ones(128, 1.0F);
    vInt4Hadamard128Host(ones.data());
    EXPECT_NEAR(ones[0], std::sqrt(128.0F), 1.0e-4F);
    for (int32_t i = 1; i < 128; ++i)
    {
        EXPECT_NEAR(ones[i], 0.0F, 1.0e-5F);
    }
    // An outlier is spread over the whole block: every output has the same magnitude.
    std::vector<float> spike(128, 0.0F);
    spike[77] = 50.0F;
    vInt4Hadamard128Host(spike.data());
    for (float const v : spike)
    {
        EXPECT_NEAR(std::fabs(v), 50.0F * kVInt4HadamardScale, 1.0e-4F);
    }
}

TEST(KvInt4Hadamard, ScaleConstantIsOneOverSqrtOf128)
{
    EXPECT_EQ(kVInt4HadamardScale, static_cast<float>(1.0 / std::sqrt(128.0)));
}

TEST(KvInt4Packed, LayoutConstantsAreConsistent)
{
    EXPECT_EQ(kVInt4PackedChannels, 256);
    EXPECT_EQ(kVInt4PackedNibbleBytes * 2, kVInt4PackedChannels);
    EXPECT_EQ(kVInt4PackedGroups * kVInt4GroupSize, kVInt4PackedChannels);
    EXPECT_EQ(kVInt4PackedMetaOffset, kVInt4PackedNibbleBytes);
    EXPECT_EQ(vInt4PackedScaleByte(0), 128);
    EXPECT_EQ(vInt4PackedZeroByte(0), 130);
    EXPECT_EQ(vInt4PackedScaleByte(7), 156);
    EXPECT_EQ(vInt4PackedZeroByte(7), 158);
    EXPECT_EQ(kVInt4PackedUsedBytes, vInt4PackedZeroByte(kVInt4PackedGroups - 1) + 2);
    EXPECT_LE(kVInt4PackedUsedBytes, kVInt4PackedChannels);
}

TEST(KvInt4Packed, NibbleOrderAndUntouchedTail)
{
    // Channel i is nibble i: channel 0 in the low nibble of byte 0, channel 1 in its high nibble.
    std::vector<float> values(256);
    for (int32_t i = 0; i < 256; ++i)
    {
        values[i] = static_cast<float>(i % 32); // 0..31 in every group: codes spread over 0..15
    }
    std::vector<uint8_t> row(256, 0xA5);
    vInt4PackRowHost(values.data(), row.data());
    for (int32_t i = kVInt4PackedUsedBytes; i < 256; ++i)
    {
        EXPECT_EQ(row[i], 0xA5) << "byte " << i << " must not be written";
    }
    EXPECT_EQ(row[0] & 0x0F, 0);  // channel 0 is the group minimum: code 0
    EXPECT_EQ(row[15] >> 4, 15);  // channel 31 is the group maximum: code 15
    std::vector<float> decoded(256);
    vInt4UnpackRowHost(row.data(), decoded.data());
    for (int32_t i = 0; i < 256; ++i)
    {
        EXPECT_NEAR(decoded[i], values[i], 31.0F / 15.0F * 0.5F + 1e-3F);
    }
}

TEST(KvInt4Packed, DegenerateGroupsRoundTripToTheirConstant)
{
    // Constant groups, an all-zero group, and a group with +inf / NaN: no crash, finite zero reconstruction for the last two.
    std::vector<float> values(256, 0.0F);
    for (int32_t i = 32; i < 64; ++i) values[i] = 3.5F;
    for (int32_t i = 64; i < 96; ++i) values[i] = -2.25F;
    values[96] = INFINITY;
    for (int32_t i = 128; i < 160; ++i) values[i] = std::numeric_limits<float>::quiet_NaN();
    for (int32_t i = 160; i < 192; ++i) values[i] = 70000.0F;
    std::vector<uint8_t> row(256, 0);
    vInt4PackRowHost(values.data(), row.data());
    std::vector<float> decoded(256);
    vInt4UnpackRowHost(row.data(), decoded.data());
    for (int32_t i = 0; i < 32; ++i) EXPECT_EQ(decoded[i], 0.0F);
    for (int32_t i = 32; i < 64; ++i) EXPECT_EQ(decoded[i], 3.5F);
    for (int32_t i = 64; i < 96; ++i) EXPECT_EQ(decoded[i], -2.25F);
    for (int32_t i = 96; i < 128; ++i) EXPECT_EQ(decoded[i], 0.0F);
    for (int32_t i = 128; i < 160; ++i) EXPECT_EQ(decoded[i], 0.0F);
    for (int32_t i = 160; i < 192; ++i) EXPECT_EQ(decoded[i], 0.0F);
}

TEST(KvInt4Packed, OverflowGuardAndDecodeAreExactFp16Fma)
{
    // A group spanning the whole fp16 range: the rounded-up scale (8736) would push code 15 to 65536 > 65504, so the group is stored
    // as degenerate (scale 0, zero 0) and decodes to zero instead of +inf.
    VInt4Asym const wide = vInt4AsymParams(-65504.0F, 65504.0F);
    EXPECT_EQ(wide.scale, 0.0F);
    EXPECT_EQ(wide.zero, 0.0F);
    // A group whose top code stays inside the range is kept.
    VInt4Asym const ok = vInt4AsymParams(-30000.0F, 30000.0F);
    EXPECT_GT(ok.scale, 0.0F);
    EXPECT_LE(15.0F * ok.scale + ok.zero, 65504.0F);
    // The decode is one rounding of the exact value, not two: pick values whose float product-plus-sum would round twice.
    __half const scale = __float2half_rn(0.3333F);
    __half const zero = __float2half_rn(1000.0F);
    for (int32_t code = 0; code < 16; ++code)
    {
        double const exact = static_cast<double>(code) * __half2float(scale) + __half2float(zero);
        EXPECT_EQ(vInt4PackedDecode(code, scale, zero), __half2float(__double2half(exact)));
    }
    // Overflow of the decode itself gives +inf (the writer never produces such metadata).
    EXPECT_TRUE(std::isinf(vInt4PackedDecode(15, __float2half_rn(65504.0F), __float2half_rn(65504.0F))));
}
