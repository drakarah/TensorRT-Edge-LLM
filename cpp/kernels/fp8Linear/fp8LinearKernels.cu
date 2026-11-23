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

__device__ __forceinline__ void decode16(uint4 const v, float (&w)[16])
{
    uint32_t const words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
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
template <int32_t kM>
__global__ void fp8GemvKernel(__half const* __restrict__ x, uint4 const* __restrict__ w, void const* scale, __half* __restrict__ y, int32_t numTokens, int32_t n,
    int32_t k, bool scaleIsFp32)
{
    extern __shared__ float xs[]; // [kM][k]
    for (int32_t i = threadIdx.x; i < numTokens * k; i += blockDim.x)
    {
        xs[i] = __half2float(x[i]);
    }
    __syncthreads();
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
                decode16(v[u], wv);
#pragma unroll
                for (int32_t m = 0; m < kM; ++m)
                {
                    float4 const* xp = reinterpret_cast<float4 const*>(xs + static_cast<size_t>(m) * k + c * 16);
#pragma unroll
                    for (int32_t q = 0; q < 4; ++q)
                    {
                        float4 const xv = xp[q];
                        acc[m] = fmaf(wv[4 * q + 0], xv.x, acc[m]);
                        acc[m] = fmaf(wv[4 * q + 1], xv.y, acc[m]);
                        acc[m] = fmaf(wv[4 * q + 2], xv.z, acc[m]);
                        acc[m] = fmaf(wv[4 * q + 3], xv.w, acc[m]);
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
            decode16(v, wv);
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

template <int32_t kM>
void launchGemv(void const* x, void const* w, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32, cudaStream_t stream)
{
    size_t const smem = static_cast<size_t>(kM) * k * sizeof(float);
    if (smem > 48 * 1024)
    {
        cudaFuncSetAttribute(fp8GemvKernel<kM>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    }
    fp8GemvKernel<kM><<<(n + kGemvWarps - 1) / kGemvWarps, kGemvWarps * 32, smem, stream>>>(
        static_cast<__half const*>(x), static_cast<uint4 const*>(w), scale, static_cast<__half*>(y), numTokens, n, k, scaleIsFp32);
}
} // namespace

bool fp8LinearUsesGemv(int32_t numTokens)
{
    return numTokens <= kFp8LinearMaxGemvTokens;
}

void launchFp8Linear(void const* x, void const* weightWords, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32,
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
        case 1: launchGemv<1>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream); break;
        case 2: launchGemv<2>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream); break;
        default: launchGemv<4>(x, weightWords, scale, y, numTokens, n, k, scaleIsFp32, stream); break;
        }
        return;
    }
    dim3 const grid((n + kBn - 1) / kBn, (numTokens + kBm - 1) / kBm);
    fp8GemmKernel<<<grid, 128, 0, stream>>>(
        static_cast<__half const*>(x), static_cast<uint4 const*>(weightWords), scale, static_cast<__half*>(y), numTokens, n, k, scaleIsFp32);
}

} // namespace kernel
} // namespace trt_edgellm
