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

#include "geluMulKernels.h"

#include <cstdint>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr float kSQRT_2_OVER_PI{0.7978845608028654F};
constexpr float kGELU_CUBIC{0.044715F};
constexpr float kCLAMP{2048.0F};
constexpr int32_t kTHREADS{256};

//! tanh without the hardware approximation (MUFU.TANH has ~2^-11 relative error, which is a large relative error in 1 + tanh for negative arguments):
//! a minimax polynomial for |x| < 0.6, 1 - 2 / (exp(2|x|) + 1) above, saturated at 9.0109 (the formulation of the TensorRT-generated kernel it replaces).
__device__ __forceinline__ float tanhAccurate(float x)
{
    float const ax = fabsf(x);
    float const x2 = x * x;
    float poly = fmaf(x2, 0.015739683F, -0.052303962409496307F);
    poly = fmaf(x2, poly, 0.13315297663211822F);
    poly = fmaf(x2, poly, -0.33332768082618713F);
    poly = x2 * poly;
    float const small = fmaf(x, poly, x);
    float e;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(ax * 2.8853900817779268F));
    float const large = fmaf(__fdividef(1.0F, e + 1.0F), -2.0F, 1.0F);
    float const t = ax >= 9.0109F ? 1.0F : large;
    return ax < 0.6F ? small : copysignf(t, x);
}

__device__ __forceinline__ float geluTanhMulOne(float gate, float up)
{
    float const inner = kSQRT_2_OVER_PI * (gate + kGELU_CUBIC * gate * gate * gate);
    float const act = 0.5F * gate * (1.0F + tanhAccurate(inner));
    return fminf(fmaxf(act * up, -kCLAMP), kCLAMP);
}

//! width % 8 == 0 and 16-byte aligned rows: one thread per 8 consecutive output elements; blockIdx.y strides over the rows (no division).
__global__ void geluTanhMulVecKernel(uint4 const* __restrict__ input, uint4* __restrict__ output, int32_t vecPerRow, int64_t rows)
{
    int32_t const col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= vecPerRow)
    {
        return;
    }
    for (int64_t row = blockIdx.y; row < rows; row += gridDim.y)
    {
        uint4 const gate = input[row * 2 * vecPerRow + col];
        uint4 const up = input[row * 2 * vecPerRow + vecPerRow + col];
        uint4 result;
        auto const* g = reinterpret_cast<__half2 const*>(&gate);
        auto const* u = reinterpret_cast<__half2 const*>(&up);
        auto* r = reinterpret_cast<__half2*>(&result);
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            float2 const gf = __half22float2(g[i]);
            float2 const uf = __half22float2(u[i]);
            r[i] = __floats2half2_rn(geluTanhMulOne(gf.x, uf.x), geluTanhMulOne(gf.y, uf.y));
        }
        output[row * vecPerRow + col] = result;
    }
}

__global__ void geluTanhMulScalarKernel(__half const* __restrict__ input, __half* __restrict__ output, int32_t width, int64_t rows)
{
    int32_t const col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width)
    {
        return;
    }
    for (int64_t row = blockIdx.y; row < rows; row += gridDim.y)
    {
        float const gate = __half2float(input[row * 2 * width + col]);
        float const up = __half2float(input[row * 2 * width + width + col]);
        output[row * width + col] = __float2half_rn(geluTanhMulOne(gate, up));
    }
}
} // namespace

void launchGeluTanhMul(void const* input, void* output, int64_t rows, int32_t width, cudaStream_t stream)
{
    if (rows <= 0 || width <= 0)
    {
        return;
    }
    bool const aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0) && (reinterpret_cast<uintptr_t>(output) % 16 == 0);
    unsigned const gridY = static_cast<unsigned>(rows < 65535 ? rows : 65535);
    if (width % 8 == 0 && aligned)
    {
        int32_t const vecPerRow = width / 8;
        dim3 const grid((vecPerRow + kTHREADS - 1) / kTHREADS, gridY);
        geluTanhMulVecKernel<<<grid, kTHREADS, 0, stream>>>(
            static_cast<uint4 const*>(input), static_cast<uint4*>(output), vecPerRow, rows);
    }
    else
    {
        dim3 const grid((width + kTHREADS - 1) / kTHREADS, gridY);
        geluTanhMulScalarKernel<<<grid, kTHREADS, 0, stream>>>(
            static_cast<__half const*>(input), static_cast<__half*>(output), width, rows);
    }
}

} // namespace kernel
} // namespace trt_edgellm
