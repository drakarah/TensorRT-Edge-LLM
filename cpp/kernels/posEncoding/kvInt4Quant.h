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

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! How V is reduced to 4 bits when the int8 KV pool emulates a K8V4 cache.
//! The emulation quantises V to 4-bit codes, dequantises them and stores the result on the int8 grid, so the decode
//! kernels read an ordinary int8 pool. The numeric rules below are shared with the packed 4-bit format.
enum class VInt4Mode : int32_t
{
    kOff = 0,      //!< V is plain int8 with the per-layer static scale.
    kStatic = 1,   //!< One static scale for the whole layer: codes cover the same range as int8.
    kPerToken = 2, //!< One scale per (token, kv head) row, from the row's absolute maximum.
    kGroup32 = 3,  //!< One scale per 32 channels of a (token, kv head) row.
    kGroup32Asym = 4,     //!< Scale and zero point per 32 channels: 16 levels between the group minimum and maximum.
    kHadamard128 = 5,     //!< Rotate blocks of 128 channels with a Hadamard transform, one symmetric scale per block.
    kHadamard128Asym = 6, //!< Rotate blocks of 128 channels, scale and zero point per block.
    kPackedGroup32Asym = 7, //!< The packed K8V4 row: 4-bit codes and fp16 scale + zero point per 32 channels, see below.
};

//! Packed V row of 256 channels inside a 256-byte int8 row (the pool keeps its geometry; only the first 160 bytes are used):
//!   bytes   0..127  nibble i (low nibble of byte i/2 for even i) is the code 0..15 of channel i;
//!   bytes 128..159  eight {fp16 scale, fp16 zero point} pairs, one per group of 32 channels, group g at byte 128 + 4 g (scale
//!                   first); a pair is one 32-bit word so a reader fetches both with one load;
//!   bytes 160..255  never written or read.
//! Channel value = fp16 fused multiply-add code * scale + zero, rounded once to fp16 (what the decode kernel computes with one
//! half2 FMA). The groups and their parameters are those of VInt4Mode::kGroup32Asym.
constexpr int32_t kVInt4PackedChannels = 256;
constexpr int32_t kVInt4PackedNibbleBytes = 128;
constexpr int32_t kVInt4PackedMetaOffset = 128;
constexpr int32_t kVInt4PackedUsedBytes = 160;
constexpr int32_t vInt4PackedScaleByte(int32_t group)
{
    return kVInt4PackedMetaOffset + 4 * group;
}
constexpr int32_t vInt4PackedZeroByte(int32_t group)
{
    return kVInt4PackedMetaOffset + 4 * group + 2;
}
constexpr int32_t kVInt4PackedGroups = 8;

//! Element offset of a packed V row in the compact layout: the V plane starts after the full-size K plane
//! (@p numPages pages of @p kRowElems-wide rows) and stores kVInt4PackedUsedBytes-wide rows.
__host__ __device__ constexpr int64_t vPackedCompactRowOffset(int64_t vPage, int64_t inPage, int64_t numKVHead,
    int64_t kvHead, int64_t numPages, int64_t kRowElems, int64_t tokensPerPage)
{
    return numPages * tokensPerPage * numKVHead * kRowElems
        + ((vPage - numPages) * tokensPerPage + inPage) * numKVHead * kVInt4PackedUsedBytes
        + kvHead * kVInt4PackedUsedBytes;
}

//! Largest code magnitude. Codes lie in [-7, 7]; the code -8 is never produced so that negation stays exact.
constexpr int32_t kVInt4MaxCode = 7;
//! Channels that share a scale in VInt4Mode::kGroup32.
constexpr int32_t kVInt4GroupSize = 32;
//! Asymmetric codes lie in [0, 15].
constexpr int32_t kVInt4AsymMaxCode = 15;
//! Channels per Hadamard block and the factor that makes the transform orthonormal (1 / sqrt(128)).
constexpr int32_t kVInt4HadamardBlock = 128;
constexpr float kVInt4HadamardScale = 0.08838834764831845F;
//! Largest finite fp16 value; scales above it are not representable and quantise the group to zero.
constexpr float kVInt4MaxFp16 = 65504.0F;

//! Correctly rounded reciprocal, the same on the host and the device.
__host__ __device__ inline float vInt4Reciprocal(float x)
{
#ifdef __CUDA_ARCH__
    return __frcp_rn(x);
#else
    return 1.0F / x;
#endif
}

//! Correctly rounded division. The kernels are compiled with --use_fast_math, which turns a plain `/` into an
//! approximate division; the explicit intrinsic keeps the result bit-identical to the host.
__host__ __device__ inline float vInt4Divide(float numerator, float denominator)
{
#ifdef __CUDA_ARCH__
    return __fdiv_rn(numerator, denominator);
#else
    return numerator / denominator;
#endif
}

//! Round to nearest, ties to even.
__host__ __device__ inline int32_t vInt4RoundToInt(float x)
{
#ifdef __CUDA_ARCH__
    return __float2int_rn(x);
#else
    return static_cast<int32_t>(std::nearbyintf(x));
#endif
}

//! Scale of a group from its absolute maximum, as the fp16 value the packed format stores. The scale is rounded up, so
//! absMax / scale <= 7 holds for every group and the group maximum is never clipped; rounding to nearest could clip it by
//! several steps when the scale is a coarse fp16 subnormal.
//! Returns 0 when no code can be formed: an all-zero group, a non-finite maximum, or a scale above the fp16 range.
//! A zero scale means every code of the group is 0.
__host__ __device__ inline float vInt4ScaleFromAbsMax(float absMax)
{
    float const raw = vInt4Divide(absMax, static_cast<float>(kVInt4MaxCode));
    float const stored = __half2float(__float2half_ru(raw));
    return (stored > 0.0F && stored <= kVInt4MaxFp16) ? stored : 0.0F;
}

//! Code of @p x for a positive @p scale: round(x / scale) clamped to [-7, 7]. NaN maps to -7 (std fmax/fmin drop it),
//! matching what the int8 path does with NaN.
__host__ __device__ inline int32_t vInt4Code(float x, float scale)
{
    float const scaled = x * vInt4Reciprocal(scale);
    float const clamped = fminf(fmaxf(scaled, -static_cast<float>(kVInt4MaxCode)), static_cast<float>(kVInt4MaxCode));
    return vInt4RoundToInt(clamped);
}

//! The int8 value the emulation stores for @p x when the 4-bit scale of its group is @p scale4 (0 = all codes zero)
//! and the layer's int8 scale is @p vScaleQuantOrig.
__host__ __device__ inline int8_t vInt4EmulatedInt8(float x, float scale4, float vScaleQuantOrig)
{
    float const dequant = (scale4 > 0.0F) ? static_cast<float>(vInt4Code(x, scale4)) * scale4 : 0.0F;
    float const scaled = dequant * vInt4Reciprocal(vScaleQuantOrig);
    float const clamped = fminf(fmaxf(scaled, -127.0F), 127.0F);
    return static_cast<int8_t>(vInt4RoundToInt(clamped));
}

//! Explicit IEEE add, subtract and multiply: they cannot be contracted into fused multiply-adds, so the device result
//! stays bit-identical to the host.
__host__ __device__ inline float vInt4Add(float a, float b)
{
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

__host__ __device__ inline float vInt4Sub(float a, float b)
{
#ifdef __CUDA_ARCH__
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}

__host__ __device__ inline float vInt4Mul(float a, float b)
{
#ifdef __CUDA_ARCH__
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

//! Asymmetric 4-bit parameters of a group: value = code * scale + zero, code in [0, 15]. zero is the group minimum
//! rounded down to fp16 and scale is (max - zero) / 15 rounded up to fp16, so neither end of the group is clipped.
//! A zero scale means every code is 0 and the group is reconstructed as zero (the fp16-rounded minimum for a constant
//! group; exactly 0 when the group holds a non-finite value or a magnitude beyond fp16).
struct VInt4Asym
{
    float scale;
    float zero;
};

__host__ __device__ inline VInt4Asym vInt4AsymParams(float groupMin, float groupMax)
{
    if (!(groupMin <= groupMax))
    {
        return {0.0F, 0.0F};
    }
    float const zero = __half2float(__float2half_rd(groupMin));
    float const top = __half2float(__float2half_rn(groupMax));
    if (!(zero >= -kVInt4MaxFp16 && zero <= kVInt4MaxFp16 && top >= -kVInt4MaxFp16 && top <= kVInt4MaxFp16))
    {
        return {0.0F, 0.0F};
    }
    float const range = vInt4Sub(groupMax, zero);
    float const raw = vInt4Divide(range, static_cast<float>(kVInt4AsymMaxCode));
    float const scale = __half2float(__float2half_ru(raw));
    if (!(scale <= kVInt4MaxFp16))
    {
        return {0.0F, 0.0F};
    }
    if (!(scale > 0.0F))
    {
        return {0.0F, zero}; // constant group: every code is 0 and the group decodes to its (fp16-rounded) value
    }
    // The decoder computes code * scale + zero in fp16; the largest code must not reach beyond the fp16 range, which rounding the
    // scale up can cause for a group whose maximum is within one step of 65504.
    float const reach = vInt4Add(vInt4Mul(static_cast<float>(kVInt4AsymMaxCode), scale), zero);
    if (!(reach <= kVInt4MaxFp16))
    {
        return {0.0F, 0.0F};
    }
    return {scale, zero};
}

//! Asymmetric code of @p x: round((x - zero) / scale) clamped to [0, 15]; 0 for a zero scale.
__host__ __device__ inline int32_t vInt4AsymCode(float x, VInt4Asym params)
{
    if (!(params.scale > 0.0F))
    {
        return 0;
    }
    float const scaled = vInt4Mul(vInt4Sub(x, params.zero), vInt4Reciprocal(params.scale));
    float const clamped = fminf(fmaxf(scaled, 0.0F), static_cast<float>(kVInt4AsymMaxCode));
    return vInt4RoundToInt(clamped);
}

__host__ __device__ inline float vInt4AsymDequant(int32_t code, VInt4Asym params)
{
    return vInt4Add(vInt4Mul(static_cast<float>(code), params.scale), params.zero);
}

//! Symmetric dequantisation with the same explicit multiply.
__host__ __device__ inline float vInt4SymDequant(int32_t code, float scale)
{
    return vInt4Mul(static_cast<float>(code), scale);
}

//! The int8 value stored for a dequantised V value @p value under the layer's int8 scale.
__host__ __device__ inline int8_t vInt4Int8FromValue(float value, float vScaleQuantOrig)
{
    float const scaled = vInt4Mul(value, vInt4Reciprocal(vScaleQuantOrig));
    float const clamped = fminf(fmaxf(scaled, -127.0F), 127.0F);
    return static_cast<int8_t>(vInt4RoundToInt(clamped));
}

//! Host reference of the orthonormal 128-point Hadamard transform used by the rotated modes, in place on 128 values.
//! Stage order is fixed because float addition is not associative and the device kernel must match it bit for bit: index
//! bits 0 to 6 in turn, the lower element of each pair gets a + b and the upper a - b, then one multiply by 1 / sqrt(128).
inline void vInt4Hadamard128Host(float* x)
{
    for (int32_t bit = 0; bit < 7; ++bit)
    {
        for (int32_t i = 0; i < kVInt4HadamardBlock; ++i)
        {
            if (((i >> bit) & 1) == 0)
            {
                float const a = x[i];
                float const b = x[i | (1 << bit)];
                x[i] = a + b;
                x[i | (1 << bit)] = a - b;
            }
        }
    }
    for (int32_t i = 0; i < kVInt4HadamardBlock; ++i)
    {
        x[i] = x[i] * kVInt4HadamardScale;
    }
}

//! Host reference of the packed row writer: packs 256 channel values (as the kernel sees them, already fp16-exact floats) into
//! the first 160 bytes of @p row; the other bytes are left alone.
inline void vInt4PackRowHost(float const* values, uint8_t* row)
{
    for (int32_t g = 0; g < kVInt4PackedGroups; ++g)
    {
        float groupMin = INFINITY;
        float groupMax = -INFINITY;
        for (int32_t i = 0; i < kVInt4GroupSize; ++i)
        {
            float const x = values[g * kVInt4GroupSize + i];
            groupMin = fminf(groupMin, x);
            groupMax = fmaxf(groupMax, x);
        }
        VInt4Asym const params = vInt4AsymParams(groupMin, groupMax);
        for (int32_t i = 0; i < kVInt4GroupSize; ++i)
        {
            int32_t const channel = g * kVInt4GroupSize + i;
            uint32_t const code = static_cast<uint32_t>(vInt4AsymCode(values[channel], params));
            uint8_t& byte = row[channel / 2];
            byte = (channel % 2 == 0) ? static_cast<uint8_t>((byte & 0xF0U) | code) : static_cast<uint8_t>((byte & 0x0FU) | (code << 4));
        }
        __half const scale = __float2half_rn(params.scale);
        __half const zero = __float2half_rn(params.zero);
        std::memcpy(row + vInt4PackedScaleByte(g), &scale, 2);
        std::memcpy(row + vInt4PackedZeroByte(g), &zero, 2);
    }
}

//! The canonical decoded value: code * scale + zero evaluated exactly (the operands are fp16, so double holds the result
//! exactly) and rounded once to fp16, which is what a half-precision fused multiply-add gives.
inline float vInt4PackedDecode(int32_t code, __half scale, __half zero)
{
    double const exact = static_cast<double>(code) * static_cast<double>(__half2float(scale)) + static_cast<double>(__half2float(zero));
    return __half2float(__double2half(exact));
}

//! Host reference of the unpack: channel values of a packed row (see vInt4PackedDecode).
inline void vInt4UnpackRowHost(uint8_t const* row, float* values)
{
    for (int32_t channel = 0; channel < kVInt4PackedChannels; ++channel)
    {
        int32_t const g = channel / kVInt4GroupSize;
        uint8_t const byte = row[channel / 2];
        int32_t const code = (channel % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        __half scale;
        __half zero;
        std::memcpy(&scale, row + vInt4PackedScaleByte(g), 2);
        std::memcpy(&zero, row + vInt4PackedZeroByte(g), 2);
        values[channel] = vInt4PackedDecode(code, scale, zero);
    }
}

//! Scale used by VInt4Mode::kStatic: the 4-bit step that spans the same range as the int8 scale.
__host__ __device__ inline float vInt4StaticScale(float vScaleQuantOrig)
{
    return vScaleQuantOrig * (127.0F / static_cast<float>(kVInt4MaxCode));
}

} // namespace kernel
} // namespace trt_edgellm
