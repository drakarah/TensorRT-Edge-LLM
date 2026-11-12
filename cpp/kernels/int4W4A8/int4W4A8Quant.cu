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
constexpr int32_t kMaxK = 20480; // 40 KiB of row staging; the static shared-memory limit is 48 KiB
// Rows from which the SiLU-mul gather uses the row-staged kernel. Measured at 16 x 17,408: row-staged 54 us, element gather 102 us.
constexpr int32_t kRowStagedMinRows = 1;

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

// sigmoid(x) = 1 / (1 + 2^(-x log2 e)) on the approximate exp2 and reciprocal instructions (one MUFU each, a few FP32 ulp,
// relative accuracy kept in both tails) instead of the library exponential and an IEEE division. A tanh-based form
// (0.5 + 0.5 tanh(x / 2)) was tried and dropped: it cancels for negative x (0.7% relative error on the product).
__device__ __forceinline__ float fastSigmoid(float x)
{
    float e;
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(-1.4426950408889634F * x));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(1.F + e));
    return r;
}

// SiLU(gate) * up for eight FP16 elements with TensorRT's rounding points: sigmoid in FP16, gate * sigmoid in FP16, times up in FP16.
__device__ __forceinline__ uint4 siluMul8(uint4 gateBits, uint4 upBits)
{
    half2* g2 = reinterpret_cast<half2*>(&gateBits);
    half2 const* u2 = reinterpret_cast<half2 const*>(&upBits);
#pragma unroll
    for (int32_t e = 0; e < 4; ++e)
    {
        float2 const f = __half22float2(g2[e]);
        float2 sig;
        sig.x = fastSigmoid(f.x);
        sig.y = fastSigmoid(f.y);
        g2[e] = __hmul2(__hmul2(g2[e], __float22half2_rn(sig)), u2[e]);
    }
    return gateBits;
}

// Stage SiLU(gate) * up of one row in shared memory and return this thread's partial absmax.
__device__ __forceinline__ float stageSiluMulRow(
    half const* __restrict__ gate, half const* __restrict__ up, half* row, int64_t base, int32_t K)
{
    float amax = 0.F;
    for (int32_t i = threadIdx.x * 8; i < K; i += kThreads * 8)
    {
        uint4 const v = siluMul8(
            *reinterpret_cast<uint4 const*>(gate + base + i), *reinterpret_cast<uint4 const*>(up + base + i));
        *reinterpret_cast<uint4*>(row + i) = v;
        half2 const* h2 = reinterpret_cast<half2 const*>(&v);
#pragma unroll
        for (int32_t e = 0; e < 4; ++e)
        {
            float2 const f = __half22float2(h2[e]);
            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    return amax;
}

// quantizeRowsInt8PermutedKernel over the activation SiLU(gate) * up computed on the fly, with an optional column gather
// (activation[j] = product[perm[j]], the GPTQ activation permutation) folded into the permuted INT8 write.
__global__ void quantizeRowsInt8PermutedSiluMulKernel(half const* __restrict__ gate, half const* __restrict__ up,
    int32_t const* __restrict__ perm, int8_t* __restrict__ out, float* __restrict__ rowScale, int32_t K)
{
    __shared__ half row[kMaxK];
    __shared__ float partial[kThreads / 32];
    int32_t const m = blockIdx.x;
    float amax = stageSiluMulRow(gate, up, row, static_cast<int64_t>(m) * K, K);
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
            int32_t const src = perm != nullptr ? perm[cols[j]] : cols[j];
            int32_t const q = max(-127, min(127, __float2int_rn(__half2float(row[src]) * inv)));
            packed |= (static_cast<uint32_t>(q) & 0xFFu) << (8 * j);
        }
        *reinterpret_cast<uint32_t*>(dst + 4 * i4) = packed;
    }
}

// out[m, j] = (SiLU(gate) * up)[m, perm[j]] in FP16 (perm may be null): the FP16 path of a gated GEMM input. One thread per
// eight output elements; with a perm the two operands are gathered element by element (used only when K exceeds the staged
// row of siluMulGatherRowsKernel).
__global__ void siluMulGatherKernel(half const* __restrict__ gate, half const* __restrict__ up,
    int32_t const* __restrict__ perm, half* __restrict__ out, int32_t K)
{
    // Grid: x covers the K / 8 vectors of a row, y the rows (no 64-bit division per thread).
    int32_t const i = static_cast<int32_t>(blockIdx.x * kThreads + threadIdx.x) * 8;
    if (i >= K)
    {
        return;
    }
    int64_t const base = static_cast<int64_t>(blockIdx.y) * K;
    uint4 g;
    uint4 u;
    if (perm == nullptr)
    {
        g = *reinterpret_cast<uint4 const*>(gate + base + i);
        u = *reinterpret_cast<uint4 const*>(up + base + i);
    }
    else
    {
        uint4 const p0 = *reinterpret_cast<uint4 const*>(perm + i);
        uint4 const p1 = *reinterpret_cast<uint4 const*>(perm + i + 4);
        uint32_t const idx[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
        half* gh = reinterpret_cast<half*>(&g);
        half* uh = reinterpret_cast<half*>(&u);
#pragma unroll
        for (int32_t e = 0; e < 8; ++e)
        {
            gh[e] = gate[base + idx[e]];
            uh[e] = up[base + idx[e]];
        }
    }
    *reinterpret_cast<uint4*>(out + base + i) = siluMul8(g, u);
}

// Row-staged form of siluMulGatherKernel for a permutation: one block per row computes the product from coalesced loads into
// shared memory and gathers from there, so DRAM sees sequential traffic only (measured faster at 16 rows and at 1,024 rows).
__global__ void siluMulGatherRowsKernel(half const* __restrict__ gate, half const* __restrict__ up,
    int32_t const* __restrict__ perm, half* __restrict__ out, int32_t K)
{
    __shared__ half row[kMaxK];
    int64_t const base = static_cast<int64_t>(blockIdx.x) * K;
    static_cast<void>(stageSiluMulRow(gate, up, row, base, K));
    __syncthreads();
    for (int32_t i = threadIdx.x * 8; i < K; i += kThreads * 8)
    {
        uint4 const p0 = *reinterpret_cast<uint4 const*>(perm + i);
        uint4 const p1 = *reinterpret_cast<uint4 const*>(perm + i + 4);
        uint4 v;
        half* h = reinterpret_cast<half*>(&v);
        h[0] = row[p0.x];
        h[1] = row[p0.y];
        h[2] = row[p0.z];
        h[3] = row[p0.w];
        h[4] = row[p1.x];
        h[5] = row[p1.y];
        h[6] = row[p1.z];
        h[7] = row[p1.w];
        *reinterpret_cast<uint4*>(out + base + i) = v;
    }
}
} // namespace

void launchQuantizeRowsInt8PermutedSiluMul(half const* gate, half const* up, int32_t const* perm, int8_t* out,
    float* rowScale, int32_t M, int32_t K, cudaStream_t stream)
{
    quantizeRowsInt8PermutedSiluMulKernel<<<M, kThreads, 0, stream>>>(gate, up, perm, out, rowScale, K);
}

void launchSiluMulGather(
    half const* gate, half const* up, int32_t const* perm, half* out, int32_t M, int32_t K, cudaStream_t stream)
{
    if (M <= 0 || K <= 0)
    {
        return;
    }
    if (perm != nullptr && K <= kMaxK && M >= kRowStagedMinRows)
    {
        siluMulGatherRowsKernel<<<M, kThreads, 0, stream>>>(gate, up, perm, out, K);
        return;
    }
    dim3 const grid(static_cast<uint32_t>((K / 8 + kThreads - 1) / kThreads), static_cast<uint32_t>(M));
    siluMulGatherKernel<<<grid, kThreads, 0, stream>>>(gate, up, perm, out, K);
}

void launchQuantizeRowsInt8Permuted(
    half const* activations, int8_t* out, float* rowScale, int32_t M, int32_t K, cudaStream_t stream)
{
    quantizeRowsInt8PermutedKernel<<<M, kThreads, 0, stream>>>(activations, out, rowScale, K);
}

} // namespace kernel
} // namespace trt_edgellm
