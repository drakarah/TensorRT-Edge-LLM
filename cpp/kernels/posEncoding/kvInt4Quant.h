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
};

//! Largest code magnitude. Codes lie in [-7, 7]; the code -8 is never produced so that negation stays exact.
constexpr int32_t kVInt4MaxCode = 7;
//! Channels that share a scale in VInt4Mode::kGroup32.
constexpr int32_t kVInt4GroupSize = 32;
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

//! Scale used by VInt4Mode::kStatic: the 4-bit step that spans the same range as the int8 scale.
__host__ __device__ inline float vInt4StaticScale(float vScaleQuantOrig)
{
    return vScaleQuantOrig * (127.0F / static_cast<float>(kVInt4MaxCode));
}

} // namespace kernel
} // namespace trt_edgellm
