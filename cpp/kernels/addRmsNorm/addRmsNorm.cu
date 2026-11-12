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

#include "addRmsNorm.h"

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr int32_t kThreads = 256;

// One CTA per row. Pass 1 loads x (and residual) 16 bytes at a time, adds in FP16, stages the sum in shared memory, stores it and
// accumulates the squares in FP32; pass 2 scales the staged row. Every byte of the row crosses DRAM once per tensor.
__global__ void addRmsNormKernel(half const* __restrict__ x, half const* __restrict__ residual, half const* __restrict__ gamma,
    half* __restrict__ sumOut, half* __restrict__ normOut, int32_t hidden, float eps, int32_t weightBeforeCast)
{
    __shared__ half row[kAddRmsNormMaxHidden];
    __shared__ float partial[kThreads / 32];
    int64_t const base = static_cast<int64_t>(blockIdx.x) * hidden;

    float ss = 0.F;
    for (int32_t i = threadIdx.x * 8; i < hidden; i += kThreads * 8)
    {
        uint4 v = *reinterpret_cast<uint4 const*>(x + base + i);
        half2* h2 = reinterpret_cast<half2*>(&v);
        if (residual != nullptr)
        {
            uint4 const r = *reinterpret_cast<uint4 const*>(residual + base + i);
            half2 const* r2 = reinterpret_cast<half2 const*>(&r);
#pragma unroll
            for (int32_t e = 0; e < 4; ++e)
            {
                h2[e] = __hadd2(h2[e], r2[e]);
            }
            if (sumOut != nullptr)
            {
                *reinterpret_cast<uint4*>(sumOut + base + i) = v;
            }
        }
        *reinterpret_cast<uint4*>(row + i) = v;
#pragma unroll
        for (int32_t e = 0; e < 4; ++e)
        {
            float2 const f = __half22float2(h2[e]);
            ss += f.x * f.x + f.y * f.y;
        }
    }
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1)
    {
        ss += __shfl_xor_sync(0xFFFFFFFFu, ss, off);
    }
    if ((threadIdx.x & 31) == 0)
    {
        partial[threadIdx.x >> 5] = ss;
    }
    __syncthreads();
    ss = partial[0];
#pragma unroll
    for (int32_t w = 1; w < kThreads / 32; ++w)
    {
        ss += partial[w];
    }
    float const inv = 1.F / sqrtf(ss / static_cast<float>(hidden) + eps);

    for (int32_t i = threadIdx.x * 8; i < hidden; i += kThreads * 8)
    {
        uint4 const v = *reinterpret_cast<uint4 const*>(row + i);
        uint4 const g = *reinterpret_cast<uint4 const*>(gamma + i);
        half2 const* h2 = reinterpret_cast<half2 const*>(&v);
        half2 const* g2 = reinterpret_cast<half2 const*>(&g);
        uint4 out;
        half2* o2 = reinterpret_cast<half2*>(&out);
#pragma unroll
        for (int32_t e = 0; e < 4; ++e)
        {
            float2 f = __half22float2(h2[e]);
            f.x *= inv;
            f.y *= inv;
            if (weightBeforeCast != 0)
            {
                float2 const gf = __half22float2(g2[e]);
                f.x *= gf.x;
                f.y *= gf.y;
                o2[e] = __float22half2_rn(f);
            }
            else
            {
                o2[e] = __hmul2(__float22half2_rn(f), g2[e]);
            }
        }
        *reinterpret_cast<uint4*>(normOut + base + i) = out;
    }
}
} // namespace

void launchAddRmsNorm(half const* x, half const* residual, half const* gamma, half* sumOut, half* normOut, int32_t rows,
    int32_t hidden, float eps, int32_t weightBeforeCast, cudaStream_t stream)
{
    if (rows <= 0)
    {
        return;
    }
    addRmsNormKernel<<<rows, kThreads, 0, stream>>>(x, residual, gamma, sumOut, normOut, hidden, eps, weightBeforeCast);
}

} // namespace kernel
} // namespace trt_edgellm
