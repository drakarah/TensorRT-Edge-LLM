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

__device__ __forceinline__ float geluTanhMulOne(float gate, float up)
{
    float const inner = kSQRT_2_OVER_PI * (gate + kGELU_CUBIC * gate * gate * gate);
    float const act = 0.5F * gate * (1.0F + tanhf(inner));
    return fminf(fmaxf(act * up, -kCLAMP), kCLAMP);
}

//! width % 8 == 0 and 16-byte aligned rows: one thread per 8 consecutive output elements.
__global__ void geluTanhMulVecKernel(uint4 const* __restrict__ input, uint4* __restrict__ output, int64_t vecPerRow, int64_t total)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int64_t const row = idx / vecPerRow;
    int64_t const col = idx - row * vecPerRow;
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
    output[idx] = result;
}

__global__ void geluTanhMulScalarKernel(__half const* __restrict__ input, __half* __restrict__ output, int32_t width, int64_t total)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int64_t const row = idx / width;
    int64_t const col = idx - row * width;
    float const gate = __half2float(input[row * 2 * width + col]);
    float const up = __half2float(input[row * 2 * width + width + col]);
    output[idx] = __float2half_rn(geluTanhMulOne(gate, up));
}
} // namespace

void launchGeluTanhMul(void const* input, void* output, int64_t rows, int32_t width, cudaStream_t stream)
{
    if (rows <= 0 || width <= 0)
    {
        return;
    }
    bool const aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0) && (reinterpret_cast<uintptr_t>(output) % 16 == 0);
    if (width % 8 == 0 && aligned)
    {
        int64_t const vecPerRow = width / 8;
        int64_t const total = rows * vecPerRow;
        int64_t const blocks = (total + kTHREADS - 1) / kTHREADS;
        geluTanhMulVecKernel<<<static_cast<unsigned>(blocks), kTHREADS, 0, stream>>>(
            static_cast<uint4 const*>(input), static_cast<uint4*>(output), vecPerRow, total);
    }
    else
    {
        int64_t const total = rows * width;
        int64_t const blocks = (total + kTHREADS - 1) / kTHREADS;
        geluTanhMulScalarKernel<<<static_cast<unsigned>(blocks), kTHREADS, 0, stream>>>(
            static_cast<__half const*>(input), static_cast<__half*>(output), width, total);
    }
}

} // namespace kernel
} // namespace trt_edgellm
