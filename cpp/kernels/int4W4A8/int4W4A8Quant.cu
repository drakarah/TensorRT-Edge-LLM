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

#include "int4W4A8Quant.h"

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr int32_t kThreads = 256;
constexpr int32_t kMaxK = 16384;

// One CTA per row. The row is staged in shared memory (FP16), reduced to its absmax, then written as INT8 four bytes at a time in the
// permuted column order: output quad i4 = 4 consecutive positions q..q+3 of 32-block c, q = 4 * (i4 % 8), half h = (i4 % 8) / 4,
// thread-quad t4 = i4 % 4 and byte j stand for column 32 c + 16 h + {2 t4, 2 t4 + 8, 2 t4 + 1, 2 t4 + 9}[j].
__global__ void quantizeRowsInt8PermutedKernel(
    half const* __restrict__ activations, int8_t* __restrict__ out, float* __restrict__ rowScale, int32_t K)
{
    __shared__ half row[kMaxK];
    __shared__ float partial[kThreads / 32];
    int32_t const m = blockIdx.x;
    half const* src = activations + static_cast<int64_t>(m) * K;

    float amax = 0.F;
    for (int32_t i = threadIdx.x * 8; i < K; i += kThreads * 8)
    {
        uint4 const v = *reinterpret_cast<uint4 const*>(src + i);
        *reinterpret_cast<uint4*>(row + i) = v;
        half2 const* h2 = reinterpret_cast<half2 const*>(&v);
#pragma unroll
        for (int32_t e = 0; e < 4; ++e)
        {
            float2 const f = __half22float2(h2[e]);
            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1)
    {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    }
    if ((threadIdx.x & 31) == 0)
    {
        partial[threadIdx.x >> 5] = amax;
    }
    __syncthreads();
    amax = partial[0];
#pragma unroll
    for (int32_t w = 1; w < kThreads / 32; ++w)
    {
        amax = fmaxf(amax, partial[w]);
    }
    float const inv = amax > 0.F ? 127.F / amax : 0.F;
    if (threadIdx.x == 0)
    {
        rowScale[m] = amax > 0.F ? amax / 127.F : 1.F;
    }

    int8_t* dst = out + static_cast<int64_t>(m) * K;
    for (int32_t i4 = threadIdx.x; i4 < K / 4; i4 += kThreads)
    {
        int32_t const c = i4 >> 3;
        int32_t const u = i4 & 7;
        int32_t const base = 32 * c + 16 * (u >> 2) + 2 * (u & 3);
        int32_t const cols[4] = {base, base + 8, base + 1, base + 9};
        uint32_t packed = 0;
#pragma unroll
        for (int32_t j = 0; j < 4; ++j)
        {
            int32_t const q = max(-127, min(127, __float2int_rn(__half2float(row[cols[j]]) * inv)));
            packed |= (static_cast<uint32_t>(q) & 0xFFu) << (8 * j);
        }
        *reinterpret_cast<uint32_t*>(dst + 4 * i4) = packed;
    }
}
} // namespace

void launchQuantizeRowsInt8Permuted(
    half const* activations, int8_t* out, float* rowScale, int32_t M, int32_t K, cudaStream_t stream)
{
    quantizeRowsInt8PermutedKernel<<<M, kThreads, 0, stream>>>(activations, out, rowScale, K);
}


namespace
{
__device__ void atomicMaxPos(float* addr, float value)
{
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value)); // non-negative floats order like their bit patterns
}

// One CTA per row: absmax, rms and the mean of per-128-group absmax / row absmax.
__global__ void probeRowsKernel(half const* __restrict__ x, int32_t K, float* stats)
{
    __shared__ float groupMax[kMaxK / 128];
    __shared__ float sums[3][kThreads / 32];
    int32_t const m = blockIdx.x;
    int32_t const groups = K / 128;
    for (int32_t g = threadIdx.x; g < groups; g += kThreads)
    {
        groupMax[g] = 0.F;
    }
    __syncthreads();
    half const* src = x + static_cast<int64_t>(m) * K;
    float amax = 0.F;
    float sq = 0.F;
    for (int32_t i = threadIdx.x; i < K; i += kThreads)
    {
        float const v = fabsf(__half2float(src[i]));
        amax = fmaxf(amax, v);
        sq += v * v;
        atomicMaxPos(&groupMax[i >> 7], v);
    }
    for (int32_t off = 16; off > 0; off >>= 1)
    {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        sq += __shfl_xor_sync(0xFFFFFFFFu, sq, off);
    }
    if ((threadIdx.x & 31) == 0)
    {
        sums[0][threadIdx.x >> 5] = amax;
        sums[1][threadIdx.x >> 5] = sq;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float rowMax = 0.F;
        float rowSq = 0.F;
        for (int32_t w = 0; w < kThreads / 32; ++w)
        {
            rowMax = fmaxf(rowMax, sums[0][w]);
            rowSq += sums[1][w];
        }
        float const rms = sqrtf(rowSq / static_cast<float>(K));
        float groupRatio = 0.F;
        for (int32_t g = 0; g < groups; ++g)
        {
            groupRatio += rowMax > 0.F ? groupMax[g] / rowMax : 1.F;
        }
        float const ratio = rms > 0.F ? rowMax / rms : 0.F;
        atomicAdd(&stats[3], ratio);
        atomicAdd(&stats[4], groupRatio / static_cast<float>(groups));
        atomicMaxPos(&stats[5], ratio);
        atomicAdd(&stats[6], 1.F);
    }
}

__global__ void probeOutputKernel(half const* __restrict__ y8, half const* __restrict__ y16, int64_t count, float* stats)
{
    float err2 = 0.F;
    float ref2 = 0.F;
    float maxErr = 0.F;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        float const a = __half2float(y8[i]);
        float const b = __half2float(y16[i]);
        err2 += (a - b) * (a - b);
        ref2 += b * b;
        maxErr = fmaxf(maxErr, fabsf(a - b));
    }
    atomicAdd(&stats[0], err2);
    atomicAdd(&stats[1], ref2);
    atomicMaxPos(&stats[2], maxErr);
}
} // namespace

void launchW4A8Probe(half const* x, half const* y8, half const* y16, int32_t M, int32_t K, int32_t N, float* stats,
    cudaStream_t stream)
{
    probeRowsKernel<<<M, kThreads, 0, stream>>>(x, K, stats);
    probeOutputKernel<<<256, kThreads, 0, stream>>>(y8, y16, static_cast<int64_t>(M) * N, stats);
}

} // namespace kernel
} // namespace trt_edgellm
