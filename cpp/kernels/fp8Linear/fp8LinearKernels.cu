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

#include "fp8LinearKernels.h"

#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{
constexpr int32_t kGemvWarps = 8;

//! 16 weight bytes -> floats. kInt8 = false: E4M3 codes; true: two's-complement INT8 (the sign flip plus the 0x6400 half exponent turns each byte b into the half
//! 1024 + (b + 128), exact, and one subtraction of 1152 gives b).
template <bool kInt8>
__device__ __forceinline__ void decode16(uint4 const v, float (&w)[16])
{
    uint32_t const words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        if constexpr (kInt8)
        {
            uint32_t const flipped = words[i] ^ 0x80808080U;
            uint32_t const lo = __byte_perm(flipped, 0x64646464U, 0x5140);
            uint32_t const hi = __byte_perm(flipped, 0x64646464U, 0x7362);
            __half2 const bias = __float2half2_rn(1152.0F);
            float2 const a = __half22float2(__hsub2(*reinterpret_cast<__half2 const*>(&lo), bias));
            float2 const b = __half22float2(__hsub2(*reinterpret_cast<__half2 const*>(&hi), bias));
            w[4 * i + 0] = a.x;
            w[4 * i + 1] = a.y;
            w[4 * i + 2] = b.x;
            w[4 * i + 3] = b.y;
            continue;
        }
        __half2_raw const lo = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(words[i] & 0xFFFFu), __NV_E4M3);
        __half2_raw const hi = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(words[i] >> 16), __NV_E4M3);
        float2 const a = __half22float2(__half2(lo));
        float2 const b = __half22float2(__half2(hi));
        w[4 * i + 0] = a.x;
        w[4 * i + 1] = a.y;
        w[4 * i + 2] = b.x;
        w[4 * i + 3] = b.y;
    }
}

__device__ __forceinline__ float loadScale(void const* scale, int32_t n, bool fp32)
{
    return fp32 ? static_cast<float const*>(scale)[n] : __half2float(static_cast<__half const*>(scale)[n]);
}

//! One warp per output row, up to kM tokens: the activations of all tokens sit in shared memory as floats, each lane streams 16 weight bytes at a time and keeps kM
//! partial sums; the row result comes from a shuffle reduction. Rows of a block are consecutive, so a block reads a contiguous slab of weights.
template <int32_t kM, bool kInt8>
__global__ void fp8GemvKernel(__half const* __restrict__ x, uint4 const* __restrict__ w, void const* scale, __half* __restrict__ y, int32_t numTokens, int32_t n,
    int32_t k, bool scaleIsFp32)
{
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;
    int32_t const row = blockIdx.x * kGemvWarps + warp;
    if (row >= n)
    {
        return;
    }
    int32_t const chunks = k / 16;
    uint4 const* rowPtr = w + static_cast<size_t>(row) * chunks;
    float acc[kM];
#pragma unroll
    for (int32_t m = 0; m < kM; ++m)
    {
        acc[m] = 0.0F;
    }
    constexpr int32_t kUnroll = 3;
    for (int32_t base = lane; base < chunks; base += 32 * kUnroll)
    {
        uint4 v[kUnroll];
#pragma unroll
        for (int32_t u = 0; u < kUnroll; ++u)
        {
            int32_t const c = base + u * 32;
            v[u] = c < chunks ? __ldg(rowPtr + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int32_t u = 0; u < kUnroll; ++u)
        {
            int32_t const c = base + u * 32;
            if (c < chunks)
            {
                float wv[16];
                decode16<kInt8>(v[u], wv);
#pragma unroll
                for (int32_t m = 0; m < kM; ++m)
                {
                    // activations straight from global memory (L1/L2 resident, shared by the 8 rows of the block): no shared-memory staging, so the weight loads
                    // above are never behind a barrier
                    uint4 const* xp = reinterpret_cast<uint4 const*>(x + static_cast<size_t>(m) * k + c * 16);
                    uint4 const xa = __ldg(xp);
                    uint4 const xb = __ldg(xp + 1);
                    uint32_t const xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                    for (int32_t q = 0; q < 8; ++q)
                    {
                        float2 const xv = __half22float2(*reinterpret_cast<__half2 const*>(&xw[q]));
                        acc[m] = fmaf(wv[2 * q + 0], xv.x, acc[m]);
                        acc[m] = fmaf(wv[2 * q + 1], xv.y, acc[m]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int32_t m = 0; m < kM; ++m)
    {
#pragma unroll
        for (int32_t off = 16; off > 0; off >>= 1)
        {
            acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], off);
        }
    }
    if (lane == 0)
    {
        float const s = loadScale(scale, row, scaleIsFp32);
#pragma unroll
        for (int32_t m = 0; m < kM; ++m)
        {
            if (m < numTokens)
            {
                y[static_cast<size_t>(m) * n + row] = __float2half_rn(acc[m] * s);
            }
        }
    }
}

constexpr int32_t kBm = 64;
constexpr int32_t kBn = 64;
constexpr int32_t kBk = 32;
constexpr int32_t kPad = 8;

//! 64 x 64 output tile per block of 4 warps (32 x 32 each): the FP8 weight tile is dequantised to half in shared memory, the product runs as 16x16x16 half WMMA with
//! FP32 accumulation, the row scale is applied in the epilogue. Rows beyond numTokens are zero-filled on load and not stored.
template <bool kInt8>
__global__ void fp8GemmKernel(__half const* __restrict__ x, uint4 const* __restrict__ w, void const* scale, __half* __restrict__ y, int32_t numTokens, int32_t n,
    int32_t k, bool scaleIsFp32)
{
    using namespace nvcuda;
    __shared__ __align__(128) __half xs[kBm][kBk + kPad];
    __shared__ __align__(128) __half ws[kBn][kBk + kPad];
    __shared__ __align__(128) float cs[kBm][kBn + 4];
    int32_t const m0 = blockIdx.y * kBm;
    int32_t const n0 = blockIdx.x * kBn;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const wm = (warp >> 1) * 32;
    int32_t const wn = (warp & 1) * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int32_t i = 0; i < 2; ++i)
    {
#pragma unroll
        for (int32_t j = 0; j < 2; ++j)
        {
            wmma::fill_fragment(acc[i][j], 0.0F);
        }
    }
    int32_t const chunksPerRow = k / 16;
    for (int32_t k0 = 0; k0 < k; k0 += kBk)
    {
        // x tile: 64 rows x 32 halves = 4 halves... one uint4 (8 halves) per thread-slot, 256 slots
        for (int32_t s = threadIdx.x; s < kBm * kBk / 8; s += blockDim.x)
        {
            int32_t const r = s / (kBk / 8);
            int32_t const c = (s % (kBk / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m0 + r < numTokens)
            {
                v = *reinterpret_cast<uint4 const*>(x + static_cast<size_t>(m0 + r) * k + k0 + c);
            }
            *reinterpret_cast<uint4*>(&xs[r][c]) = v;
        }
        // weight tile: 64 rows x 32 fp8 = two uint4 chunks of 16 per row
        for (int32_t s = threadIdx.x; s < kBn * 2; s += blockDim.x)
        {
            int32_t const r = s >> 1;
            int32_t const c = (s & 1) * 16;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (n0 + r < n)
            {
                v = __ldg(w + static_cast<size_t>(n0 + r) * chunksPerRow + (k0 + c) / 16);
            }
            float wv[16];
            decode16<kInt8>(v, wv);
#pragma unroll
            for (int32_t i = 0; i < 16; i += 2)
            {
                *reinterpret_cast<__half2*>(&ws[r][c + i]) = __floats2half2_rn(wv[i], wv[i + 1]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int32_t kk = 0; kk < kBk; kk += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
            for (int32_t i = 0; i < 2; ++i)
            {
                wmma::load_matrix_sync(a[i], &xs[wm + 16 * i][kk], kBk + kPad);
                wmma::load_matrix_sync(b[i], &ws[wn + 16 * i][kk], kBk + kPad);
            }
#pragma unroll
            for (int32_t i = 0; i < 2; ++i)
            {
#pragma unroll
                for (int32_t j = 0; j < 2; ++j)
                {
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int32_t i = 0; i < 2; ++i)
    {
#pragma unroll
        for (int32_t j = 0; j < 2; ++j)
        {
            wmma::store_matrix_sync(&cs[wm + 16 * i][wn + 16 * j], acc[i][j], kBn + 4, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int32_t s = threadIdx.x; s < kBm * kBn; s += blockDim.x)
    {
        int32_t const r = s / kBn;
        int32_t const c = s % kBn;
        if (m0 + r < numTokens && n0 + c < n)
        {
            y[static_cast<size_t>(m0 + r) * n + n0 + c] = __float2half_rn(cs[r][c] * loadScale(scale, n0 + c, scaleIsFp32));
        }
    }
}

//! kRows output rows per warp (verify steps with 3 to 8 tokens): the activations of a chunk are read once and used for the weights of kRows rows, which cuts the
//! activation loads per weight byte by kRows. Per row and token the products are accumulated in the same order as fp8GemvKernel, so results are bit-identical.
template <int32_t kM, bool kInt8, int32_t kRows>
__global__ void fp8GemvRowsKernel(__half const* __restrict__ x, uint4 const* __restrict__ w, void const* scale, __half* __restrict__ y, int32_t numTokens,
    int32_t n, int32_t k, bool scaleIsFp32)
{
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;
    int32_t const row0 = (blockIdx.x * kGemvWarps + warp) * kRows;
    if (row0 >= n)
    {
        return;
    }
    int32_t const chunks = k / 16;
    float acc[kRows][kM];
#pragma unroll
    for (int32_t r = 0; r < kRows; ++r)
    {
#pragma unroll
        for (int32_t m = 0; m < kM; ++m)
        {
            acc[r][m] = 0.0F;
        }
    }
    constexpr int32_t kUnroll = 2;
    for (int32_t base = lane; base < chunks; base += 32 * kUnroll)
    {
        uint4 v[kUnroll][kRows];
#pragma unroll
        for (int32_t u = 0; u < kUnroll; ++u)
        {
            int32_t const c = base + u * 32;
#pragma unroll
            for (int32_t r = 0; r < kRows; ++r)
            {
                int32_t const row = min(row0 + r, n - 1);
                v[u][r] = c < chunks ? __ldg(w + static_cast<size_t>(row) * chunks + c) : make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int32_t u = 0; u < kUnroll; ++u)
        {
            int32_t const c = base + u * 32;
            if (c < chunks)
            {
                float wv[kRows][16];
#pragma unroll
                for (int32_t r = 0; r < kRows; ++r)
                {
                    decode16<kInt8>(v[u][r], wv[r]);
                }
#pragma unroll
                for (int32_t m = 0; m < kM; ++m)
                {
                    uint4 const* xp = reinterpret_cast<uint4 const*>(x + static_cast<size_t>(min(m, numTokens - 1)) * k + c * 16); // rows of the tile beyond numTokens re-read the last token (never stored): no read past the activations
                    uint4 const xa = __ldg(xp);
                    uint4 const xb = __ldg(xp + 1);
                    uint32_t const xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                    for (int32_t q = 0; q < 8; ++q)
                    {
                        float2 const xv = __half22float2(*reinterpret_cast<__half2 const*>(&xw[q]));
#pragma unroll
                        for (int32_t r = 0; r < kRows; ++r)
                        {
                            acc[r][m] = fmaf(wv[r][2 * q + 0], xv.x, acc[r][m]);
                            acc[r][m] = fmaf(wv[r][2 * q + 1], xv.y, acc[r][m]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int32_t r = 0; r < kRows; ++r)
    {
#pragma unroll
        for (int32_t m = 0; m < kM; ++m)
        {
#pragma unroll
            for (int32_t off = 16; off > 0; off >>= 1)
            {
                acc[r][m] += __shfl_xor_sync(0xffffffffu, acc[r][m], off);
            }
        }
    }
    if (lane == 0)
    {
#pragma unroll
        for (int32_t r = 0; r < kRows; ++r)
        {
            if (row0 + r < n)
            {
                float const s = loadScale(scale, row0 + r, scaleIsFp32);
#pragma unroll
                for (int32_t m = 0; m < kM; ++m)
                {
                    if (m < numTokens)
                    {
                        y[static_cast<size_t>(m) * n + row0 + r] = __float2half_rn(acc[r][m] * s);
                    }
                }
            }
        }
    }
}

template <int32_t kM, bool kInt8, int32_t kRows>
void launchGemvRows(void const* x, void const* w, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32, cudaStream_t stream)
{
    int32_t const perBlock = kGemvWarps * kRows;
    fp8GemvRowsKernel<kM, kInt8, kRows><<<(n + perBlock - 1) / perBlock, kGemvWarps * 32, 0, stream>>>(
        static_cast<__half const*>(x), static_cast<uint4 const*>(w), scale, static_cast<__half*>(y), numTokens, n, k, scaleIsFp32);
}

//! INT8 weights in mma-fragment order (kLinearWeightInt8Mma): per 8 output rows x 64 k slab, lane (g = lane / 4, t = lane % 4) of a warp holds 16 contiguous bytes: for
//! each of four k16 tiles j the bytes W[g][16 j + {2t, 2t + 1, 2t + 8, 2t + 9}], i.e. exactly its m16n8k16 B fragment of that tile. Tensor cores (activations are the A operand,
//! tokens the M dimension, 16 per group) read it once per token tile with one 16-byte load per lane and k64.
namespace i8mma
{
constexpr int32_t kWarps = 4;
constexpr int32_t kStageSmall = 1024; //!< k elements of the activation tile staged in shared memory at a time (up to 8 tokens)
constexpr int32_t kStageTile = 256;   //!< the same for 64-token tiles

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void int8Pairs(uint32_t w, uint32_t& b0, uint32_t& b1)
{
    uint32_t const flipped = w ^ 0x80808080U;
    uint32_t const lo = __byte_perm(flipped, 0x64646464U, 0x5140);
    uint32_t const hi = __byte_perm(flipped, 0x64646464U, 0x7362);
    __half2 const bias = __float2half2_rn(1152.0F);
    __half2 const l = __hsub2(*reinterpret_cast<__half2 const*>(&lo), bias);
    __half2 const h = __hsub2(*reinterpret_cast<__half2 const*>(&hi), bias);
    b0 = *reinterpret_cast<uint32_t const*>(&l);
    b1 = *reinterpret_cast<uint32_t const*>(&h);
}

//! One warp per group of 8 output rows, kWarps groups per block; a block covers kGroups x 16 tokens (kGroups = 1: the A rows 8..15 are zero for up to 8 tokens when kSmall).
template <int32_t kGroups, bool kSmall>
__global__ void __launch_bounds__(kWarps * 32) gemm(__half const* __restrict__ x, uint4 const* __restrict__ wShuf, void const* scale, __half* __restrict__ y,
    int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32)
{
    constexpr int32_t kTokens = kSmall ? 8 : 16 * kGroups;
    constexpr int32_t kStageK = kSmall ? kStageSmall : kStageTile;
    constexpr int32_t kStride = kStageK + 8;
    extern __shared__ __align__(16) __half xs[]; // [kTokens][kStride]
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const g = lane >> 2;
    int32_t const t = lane & 3;
    int32_t const group = blockIdx.x * kWarps + warp; // output group of 8 rows
    bool const active = group * 8 < n;
    int32_t const token0 = blockIdx.y * kTokens;
    int32_t const kb64 = k / 64;
    float c[kGroups][4];
#pragma unroll
    for (int32_t i = 0; i < kGroups; ++i)
    {
        c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.F;
    }
    uint4 const* wp = wShuf + static_cast<size_t>(active ? group : 0) * kb64 * 32 + lane;
    for (int32_t k0 = 0; k0 < k; k0 += kStageK)
    {
        int32_t const kc = min(kStageK, k - k0);
        __syncthreads();
        // flat (row, column) walk over all threads with incremental counters: one division per thread and chunk instead of one per element (METRICS 230), and every
        // thread loads (a row-outer loop leaves most of the block idle when a row has fewer 16-byte pieces than threads, which slowed 6-8 tokens, METRICS 241)
        {
            int32_t const cols = kc / 8;
            int32_t const total = kTokens * cols;
            int32_t r = threadIdx.x / cols;
            int32_t cc = threadIdx.x - r * cols;
            int32_t const stepR = blockDim.x / cols;
            int32_t const stepC = blockDim.x - stepR * cols;
            for (int32_t idx = threadIdx.x; idx < total; idx += blockDim.x)
            {
                uint4 v = make_uint4(0, 0, 0, 0);
                if (token0 + r < numTokens)
                {
                    v = *reinterpret_cast<uint4 const*>(x + static_cast<size_t>(token0 + r) * k + k0 + cc * 8);
                }
                *reinterpret_cast<uint4*>(&xs[r * kStride + cc * 8]) = v;
                r += stepR;
                cc += stepC;
                if (cc >= cols)
                {
                    cc -= cols;
                    ++r;
                }
            }
        }
        __syncthreads();
        if (active)
        {
            int32_t const nkb = kc / 64;
            int32_t const kbBase = k0 / 64;
#pragma unroll 1
            for (int32_t kb = 0; kb < nkb; ++kb)
            {
                uint4 const w = __ldg(wp + static_cast<size_t>(kbBase + kb) * 32);
                uint32_t const ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (int32_t j = 0; j < 4; ++j)
                {
                    int32_t const kk = kb * 64 + j * 16;
                    uint32_t b0, b1;
                    int8Pairs(ws[j], b0, b1);
#pragma unroll
                    for (int32_t i = 0; i < kGroups; ++i)
                    {
                        int32_t const r0 = (kSmall ? 0 : 16 * i) + g;
                        uint32_t const a0 = *reinterpret_cast<uint32_t const*>(&xs[r0 * kStride + kk + 2 * t]);
                        uint32_t const a2 = *reinterpret_cast<uint32_t const*>(&xs[r0 * kStride + kk + 8 + 2 * t]);
                        uint32_t a1 = 0;
                        uint32_t a3 = 0;
                        if constexpr (!kSmall)
                        {
                            a1 = *reinterpret_cast<uint32_t const*>(&xs[(r0 + 8) * kStride + kk + 2 * t]);
                            a3 = *reinterpret_cast<uint32_t const*>(&xs[(r0 + 8) * kStride + kk + 8 + 2 * t]);
                        }
                        mma16816(c[i], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }
    }
    if (!active)
    {
        return;
    }
    int32_t const n0 = group * 8 + 2 * t;
    float const s0 = loadScale(scale, n0, scaleIsFp32);
    float const s1 = loadScale(scale, n0 + 1, scaleIsFp32);
#pragma unroll
    for (int32_t i = 0; i < kGroups; ++i)
    {
        int32_t const row0 = token0 + (kSmall ? 0 : 16 * i) + g;
        if (row0 < numTokens)
        {
            y[static_cast<size_t>(row0) * n + n0] = __float2half_rn(c[i][0] * s0);
            y[static_cast<size_t>(row0) * n + n0 + 1] = __float2half_rn(c[i][1] * s1);
        }
        if (!kSmall && row0 + 8 < numTokens)
        {
            y[static_cast<size_t>(row0 + 8) * n + n0] = __float2half_rn(c[i][2] * s0);
            y[static_cast<size_t>(row0 + 8) * n + n0 + 1] = __float2half_rn(c[i][3] * s1);
        }
    }
}

//! Up to 8 tokens, small and medium output sizes: kSplit warps share one group of 8 output rows (split-K: warp s takes the k64 blocks s, s + kSplit, ...), so a block of kWarps warps covers
//! kWarps / kSplit groups and a small n still fills the GPU; four weight loads are issued before they are used. The partial sums are reduced through shared memory.
#ifndef EDGELLM_GEMV_LOAD_BATCH
#define EDGELLM_GEMV_LOAD_BATCH 4
#endif
constexpr int32_t kLoadBatch = EDGELLM_GEMV_LOAD_BATCH; //!< weight loads issued before they are used

template <int32_t kSplit>
__global__ void __launch_bounds__(kWarps * 32) gemmSplit(__half const* __restrict__ x, uint4 const* __restrict__ wShuf, void const* scale, __half* __restrict__ y,
    int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32)
{
    constexpr int32_t kStride = kStageSmall + 8;
    constexpr int32_t kGroupsPerBlock = kWarps / kSplit;
    extern __shared__ __align__(16) __half xs[]; // [8][kStride]
    __shared__ float red[kSplit > 1 ? kWarps : 1][32][4];
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const g = lane >> 2;
    int32_t const t = lane & 3;
    int32_t const split = warp % kSplit;
    int32_t const group = blockIdx.x * kGroupsPerBlock + warp / kSplit;
    bool const active = group * 8 < n;
    int32_t const kb64 = k / 64;
    float c[4] = {0.F, 0.F, 0.F, 0.F};
    uint4 const* wp = wShuf + static_cast<size_t>(active ? group : 0) * kb64 * 32 + lane;
    for (int32_t k0 = 0; k0 < k; k0 += kStageSmall)
    {
        int32_t const kc = min(kStageSmall, k - k0);
        __syncthreads();
        {
            // flat walk, see the gemm kernel above
            int32_t const cols = kc / 8;
            int32_t const total = 8 * cols;
            int32_t r = threadIdx.x / cols;
            int32_t cc = threadIdx.x - r * cols;
            int32_t const stepR = blockDim.x / cols;
            int32_t const stepC = blockDim.x - stepR * cols;
            for (int32_t idx = threadIdx.x; idx < total; idx += blockDim.x)
            {
                uint4 v = make_uint4(0, 0, 0, 0);
                if (r < numTokens)
                {
                    v = *reinterpret_cast<uint4 const*>(x + static_cast<size_t>(r) * k + k0 + cc * 8);
                }
                *reinterpret_cast<uint4*>(&xs[r * kStride + cc * 8]) = v;
                r += stepR;
                cc += stepC;
                if (cc >= cols)
                {
                    cc -= cols;
                    ++r;
                }
            }
        }
        __syncthreads();
        if (active)
        {
            int32_t const nkb = kc / 64;
            int32_t const kbBase = k0 / 64;
#pragma unroll 1
            for (int32_t b = split; b < nkb; b += kLoadBatch * kSplit)
            {
                uint4 w[kLoadBatch];
#pragma unroll
                for (int32_t u = 0; u < kLoadBatch; ++u)
                {
                    int32_t const kb = b + u * kSplit;
                    w[u] = kb < nkb ? __ldg(wp + static_cast<size_t>(kbBase + kb) * 32) : make_uint4(0, 0, 0, 0);
                }
#pragma unroll
                for (int32_t u = 0; u < kLoadBatch; ++u)
                {
                    int32_t const kb = b + u * kSplit;
                    if (kb < nkb)
                    {
                        uint32_t const ws[4] = {w[u].x, w[u].y, w[u].z, w[u].w};
#pragma unroll
                        for (int32_t j = 0; j < 4; ++j)
                        {
                            int32_t const kk = kb * 64 + j * 16;
                            uint32_t b0, b1;
                            int8Pairs(ws[j], b0, b1);
                            uint32_t const a0 = *reinterpret_cast<uint32_t const*>(&xs[g * kStride + kk + 2 * t]);
                            uint32_t const a2 = *reinterpret_cast<uint32_t const*>(&xs[g * kStride + kk + 8 + 2 * t]);
                            mma16816(c, a0, 0U, a2, 0U, b0, b1);
                        }
                    }
                }
            }
        }
    }
    if constexpr (kSplit > 1)
    {
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            red[warp][lane][i] = c[i];
        }
        __syncthreads();
        if (split != 0)
        {
            return;
        }
#pragma unroll
        for (int32_t s2 = 1; s2 < kSplit; ++s2)
        {
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
                c[i] += red[warp + s2][lane][i];
            }
        }
    }
    if (!active)
    {
        return;
    }
    int32_t const n0 = group * 8 + 2 * t;
    float const s0 = loadScale(scale, n0, scaleIsFp32);
    float const s1 = loadScale(scale, n0 + 1, scaleIsFp32);
    if (g < numTokens)
    {
        y[static_cast<size_t>(g) * n + n0] = __float2half_rn(c[0] * s0);
        y[static_cast<size_t>(g) * n + n0 + 1] = __float2half_rn(c[1] * s1);
    }
}

template <int32_t kSplit>
void launchSplit(void const* x, void const* w, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32, cudaStream_t stream)
{
    constexpr int32_t kStride = kStageSmall + 8;
    size_t const smem = static_cast<size_t>(8) * kStride * sizeof(__half);
    constexpr int32_t kGroupsPerBlock = kWarps / kSplit;
    dim3 const grid((n / 8 + kGroupsPerBlock - 1) / kGroupsPerBlock);
    gemmSplit<kSplit><<<grid, kWarps * 32, smem, stream>>>(
        static_cast<__half const*>(x), static_cast<uint4 const*>(w), scale, static_cast<__half*>(y), numTokens, n, k, scaleIsFp32);
}

template <int32_t kGroups, bool kSmall>
void launch(void const* x, void const* w, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32, cudaStream_t stream)
{
    constexpr int32_t kTokens = kSmall ? 8 : 16 * kGroups;
    constexpr int32_t kStride = (kSmall ? kStageSmall : kStageTile) + 8;
    size_t const smem = static_cast<size_t>(kTokens) * kStride * sizeof(__half);
    dim3 const grid((n / 8 + kWarps - 1) / kWarps, (numTokens + kTokens - 1) / kTokens);
    gemm<kGroups, kSmall><<<grid, kWarps * 32, smem, stream>>>(
        static_cast<__half const*>(x), static_cast<uint4 const*>(w), scale, static_cast<__half*>(y), numTokens, n, k, scaleIsFp32);
}
//! Prefill tiles of 64 tokens with kNTiles groups of 8 output rows per warp (kWarps x kNTiles x 8 rows per block): the activation fragments of a k16 step are loaded from shared memory
//! once and used for kNTiles weight tiles (the one-group kernel above needs 4 shared-memory loads per MMA, this one 1 per kNTiles MMAs); the weights are dequantised in registers as before.
#ifndef EDGELLM_WIDE_GROUP_Y
#define EDGELLM_WIDE_GROUP_Y 8
#endif
constexpr int32_t kWideStage = 128; //!< k per pipeline stage of gemmWide
constexpr int32_t kWideGroupY = EDGELLM_WIDE_GROUP_Y; //!< token blocks launched together per weight tile column (L2 reuse of the weights)

template <int32_t kNTiles>
__global__ void __launch_bounds__(kWarps * 32) gemmWide(__half const* __restrict__ x, uint4 const* __restrict__ wShuf, void const* scale, __half* __restrict__ y,
    int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32, int32_t numX, int32_t numY)
{
    constexpr int32_t kGroups = 4; // m16 groups: 64 tokens
    constexpr int32_t kTokens = 16 * kGroups;
    constexpr int32_t kStride = kWideStage + 8;
    extern __shared__ __align__(16) __half xs[]; // two stages of [kTokens][kStride]
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const g = lane >> 2;
    int32_t const t = lane & 3;
    // grouped launch order: kWideGroupY consecutive token blocks run together over the same weight tiles, so the weights are read from DRAM once per group and the activations
    // of the group stay in L2 (the plain x-fastest order re-reads every weight tile from DRAM for each token block when the weights exceed the L2)
    int32_t const linear = blockIdx.x;
    int32_t const perGroup = kWideGroupY * numX;
    int32_t const grp = linear / perGroup;
    int32_t const firstY = grp * kWideGroupY;
    int32_t const groupY = min(numY - firstY, kWideGroupY);
    int32_t const inGroup = linear % perGroup;
    int32_t const tileX = inGroup / groupY;
    int32_t const tileY = firstY + inGroup % groupY;
    int32_t const group0 = (tileX * kWarps + warp) * kNTiles;
    int32_t const token0 = tileY * kTokens;
    int32_t const kb64 = k / 64;
    float c[kGroups][kNTiles][4];
#pragma unroll
    for (int32_t i = 0; i < kGroups; ++i)
    {
#pragma unroll
        for (int32_t nt = 0; nt < kNTiles; ++nt)
        {
            c[i][nt][0] = c[i][nt][1] = c[i][nt][2] = c[i][nt][3] = 0.F;
        }
    }
    uint4 const* wp[kNTiles];
    bool active[kNTiles];
#pragma unroll
    for (int32_t nt = 0; nt < kNTiles; ++nt)
    {
        active[nt] = (group0 + nt) * 8 < n;
        wp[nt] = wShuf + static_cast<size_t>(active[nt] ? group0 + nt : 0) * kb64 * 32 + lane;
    }
    // two-stage cp.async pipeline of the activation tile: the chunk k0 + kWideStage loads while the MMAs of chunk k0 run
    auto issue = [&](int32_t buf, int32_t k0)
    {
        int32_t const kc = min(kWideStage, k - k0);
        int32_t const chunks = kc / 8;
        for (int32_t idx = threadIdx.x; idx < kTokens * chunks; idx += blockDim.x)
        {
            int32_t const r = idx / chunks;
            int32_t const cc = idx % chunks;
            bool const valid = token0 + r < numTokens;
            __half const* src = x + static_cast<size_t>(valid ? token0 + r : 0) * k + k0 + cc * 8;
            uint32_t const dst = static_cast<uint32_t>(__cvta_generic_to_shared(&xs[buf * kTokens * kStride + r * kStride + cc * 8]));
            int32_t const bytes = valid ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    issue(0, 0);
    for (int32_t k0 = 0, cur = 0; k0 < k; k0 += kWideStage, cur ^= 1)
    {
        int32_t const kc = min(kWideStage, k - k0);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        if (k0 + kWideStage < k)
        {
            issue(cur ^ 1, k0 + kWideStage);
        }
        __half const* xcur = xs + cur * kTokens * kStride;
        int32_t const nkb = kc / 64;
        int32_t const kbBase = k0 / 64;
#pragma unroll 1
        for (int32_t kb = 0; kb < nkb; ++kb)
        {
            uint4 w[kNTiles];
#pragma unroll
            for (int32_t nt = 0; nt < kNTiles; ++nt)
            {
                w[nt] = active[nt] ? __ldg(wp[nt] + static_cast<size_t>(kbBase + kb) * 32) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int32_t j = 0; j < 4; ++j)
            {
                int32_t const kk = kb * 64 + j * 16;
                uint32_t b0[kNTiles];
                uint32_t b1[kNTiles];
#pragma unroll
                for (int32_t nt = 0; nt < kNTiles; ++nt)
                {
                    uint32_t const word = j == 0 ? w[nt].x : (j == 1 ? w[nt].y : (j == 2 ? w[nt].z : w[nt].w));
                    int8Pairs(word, b0[nt], b1[nt]);
                }
#pragma unroll
                for (int32_t i = 0; i < kGroups; ++i)
                {
                    int32_t const r0 = 16 * i + g;
                    uint32_t const a0 = *reinterpret_cast<uint32_t const*>(&xcur[r0 * kStride + kk + 2 * t]);
                    uint32_t const a2 = *reinterpret_cast<uint32_t const*>(&xcur[r0 * kStride + kk + 8 + 2 * t]);
                    uint32_t const a1 = *reinterpret_cast<uint32_t const*>(&xcur[(r0 + 8) * kStride + kk + 2 * t]);
                    uint32_t const a3 = *reinterpret_cast<uint32_t const*>(&xcur[(r0 + 8) * kStride + kk + 8 + 2 * t]);
#pragma unroll
                    for (int32_t nt = 0; nt < kNTiles; ++nt)
                    {
                        mma16816(c[i][nt], a0, a1, a2, a3, b0[nt], b1[nt]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int32_t nt = 0; nt < kNTiles; ++nt)
    {
        if (!active[nt])
        {
            continue;
        }
        int32_t const n0 = (group0 + nt) * 8 + 2 * t;
        float const s0 = loadScale(scale, n0, scaleIsFp32);
        float const s1 = loadScale(scale, n0 + 1, scaleIsFp32);
#pragma unroll
        for (int32_t i = 0; i < kGroups; ++i)
        {
            int32_t const row0 = token0 + 16 * i + g;
            if (row0 < numTokens)
            {
                y[static_cast<size_t>(row0) * n + n0] = __float2half_rn(c[i][nt][0] * s0);
                y[static_cast<size_t>(row0) * n + n0 + 1] = __float2half_rn(c[i][nt][1] * s1);
            }
            if (row0 + 8 < numTokens)
            {
                y[static_cast<size_t>(row0 + 8) * n + n0] = __float2half_rn(c[i][nt][2] * s0);
                y[static_cast<size_t>(row0 + 8) * n + n0 + 1] = __float2half_rn(c[i][nt][3] * s1);
            }
        }
    }
}

template <int32_t kNTiles>
void launchWide(void const* x, void const* w, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32, cudaStream_t stream)
{
    constexpr int32_t kStride = kWideStage + 8;
    size_t const smem = static_cast<size_t>(2) * 64 * kStride * sizeof(__half);
    int32_t const groups = n / 8;
    int32_t const numX = (groups + kWarps * kNTiles - 1) / (kWarps * kNTiles);
    int32_t const numY = (numTokens + 63) / 64;
    gemmWide<kNTiles><<<numX * numY, kWarps * 32, smem, stream>>>(
        static_cast<__half const*>(x), static_cast<uint4 const*>(w), scale, static_cast<__half*>(y), numTokens, n, k, scaleIsFp32, numX, numY);
}
} // namespace i8mma

template <int32_t kM, bool kInt8>
void launchGemv(void const* x, void const* w, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32, cudaStream_t stream)
{
    fp8GemvKernel<kM, kInt8><<<(n + kGemvWarps - 1) / kGemvWarps, kGemvWarps * 32, 0, stream>>>(
        static_cast<__half const*>(x), static_cast<uint4 const*>(w), scale, static_cast<__half*>(y), numTokens, n, k, scaleIsFp32);
}
} // namespace

bool fp8LinearUsesGemv(int32_t numTokens)
{
    return numTokens <= kFp8LinearMaxGemvTokens;
}

template <bool kInt8>
void launchLinear(void const* x, void const* weightWords, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32,
    cudaStream_t stream)
{
    if (numTokens <= 0)
    {
        return;
    }
    if (fp8LinearUsesGemv(numTokens))
    {
        switch (numTokens)
        {
        case 1: launchGemv<1, kInt8>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream); break;
        case 2: launchGemv<2, kInt8>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream); break;
        case 3:
        case 4: launchGemvRows<4, kInt8, 2>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream); break;
        default: launchGemvRows<8, kInt8, 2>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream); break;
        }
        return;
    }
    dim3 const grid((n + kBn - 1) / kBn, (numTokens + kBm - 1) / kBm);
    fp8GemmKernel<kInt8><<<grid, 128, 0, stream>>>(
        static_cast<__half const*>(x), static_cast<uint4 const*>(weightWords), scale, static_cast<__half*>(y), numTokens, n, k, scaleIsFp32);
}

void launchFp8Linear(void const* x, void const* weightWords, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32,
    cudaStream_t stream, int32_t weightFormat)
{
    if (weightFormat == kLinearWeightInt8Mma)
    {
        if (numTokens <= 0)
        {
            return;
        }
        // tensor cores for every token count: up to 8 tokens one half-filled m16 group, then 64-token tiles
        if (numTokens <= 8)
        {
            // split-K over two warps for the small output sizes with a short k (k, v, o projections), one warp per group of 8 rows for the large ones (fused q, head) and for a long k
            // (the 6,912 / 9,216 wide projections of the DFlash2 drafter: split 1 is 20-25% faster there): METRICS 208, 209
            if (n < 3000 && k <= 4096)
            {
                i8mma::launchSplit<2>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream);
            }
            else
            {
                i8mma::launchSplit<1>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream);
            }
        }
        else
        {
            static bool const wideOff = std::getenv("EDGELLM_INT8_MMA_WIDE") != nullptr && std::getenv("EDGELLM_INT8_MMA_WIDE")[0] == '0';
            if (wideOff)
            {
                i8mma::launch<4, false>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream);
            }
            else
            {
                i8mma::launchWide<4>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream);
            }
        }
        return;
    }
    if (weightFormat == kLinearWeightInt8)
    {
        launchLinear<true>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream);
    }
    else
    {
        launchLinear<false>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream);
    }
}

} // namespace kernel
} // namespace trt_edgellm
