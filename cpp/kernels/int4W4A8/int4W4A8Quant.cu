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

} // namespace kernel
} // namespace trt_edgellm
