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

#include "inkling2NormRouteKernels.h"

#include <cmath>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr int32_t kNormThreads = 128;
constexpr int32_t kMaxVecPerThread = kInkling2NormRouteMaxHidden / 4 / kNormThreads; // 8 float4 per thread
constexpr int32_t kSmallMaxTokens = 8;
constexpr int32_t kSmallVecPerLane = 20; // hidden <= 2560 keeps the router row in registers
constexpr int32_t kSmallMaxHidden = kSmallVecPerLane * 32 * 4;

__device__ __forceinline__ float warpSum(float v)
{
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffU, v, o);
    }
    return v;
}

__device__ __forceinline__ void storeHalf4(__half* dst, float4 v)
{
    __half2 const lo = __floats2half2_rn(v.x, v.y);
    __half2 const hi = __floats2half2_rn(v.z, v.w);
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t const*>(&lo);
    packed.y = *reinterpret_cast<uint32_t const*>(&hi);
    *reinterpret_cast<uint2*>(dst) = packed;
}

//! One block per token: y16 = half(x * inv * normW); inv is also stored for the batched router GEMM.
__global__ void __launch_bounds__(kNormThreads) normRowsKernel(float const* __restrict__ x, float const* __restrict__ normW, __half* __restrict__ y16,
    float* __restrict__ invOut, int32_t hiddenSize, float eps)
{
    __shared__ float red[kNormThreads / 32];
    int64_t const token = blockIdx.x;
    int32_t const vecs = hiddenSize / 4;
    float4 const* xr = reinterpret_cast<float4 const*>(x + token * hiddenSize);
    float4 const* wr = reinterpret_cast<float4 const*>(normW);
    float4 xv[kMaxVecPerThread];
    float sq = 0.F;
#pragma unroll
    for (int32_t i = 0; i < kMaxVecPerThread; ++i)
    {
        int32_t const k = threadIdx.x + kNormThreads * i;
        xv[i] = k < vecs ? __ldg(xr + k) : make_float4(0.F, 0.F, 0.F, 0.F);
        sq += xv[i].x * xv[i].x + xv[i].y * xv[i].y + xv[i].z * xv[i].z + xv[i].w * xv[i].w;
    }
    sq = warpSum(sq);
    if ((threadIdx.x & 31) == 0)
    {
        red[threadIdx.x >> 5] = sq;
    }
    __syncthreads();
    float const total = red[0] + red[1] + red[2] + red[3];
    float const inv = 1.F / sqrtf(total / static_cast<float>(hiddenSize) + eps);
#pragma unroll
    for (int32_t i = 0; i < kMaxVecPerThread; ++i)
    {
        int32_t const k = threadIdx.x + kNormThreads * i;
        if (k < vecs)
        {
            float4 const w = __ldg(wr + k);
            storeHalf4(y16 + token * hiddenSize + 4 * k, make_float4(xv[i].x * inv * w.x, xv[i].y * inv * w.y, xv[i].z * inv * w.z, xv[i].w * inv * w.w));
        }
    }
    if (invOut != nullptr && threadIdx.x == 0)
    {
        invOut[token] = inv;
    }
}

//! Up to eight tokens: block e (one warp) holds router row e in registers and, per token, reduces sum x^2 and the dot product together. The first blocks also write y16.
__global__ void __launch_bounds__(32) normRouteSmallKernel(float const* __restrict__ x, float const* __restrict__ normW, float const* __restrict__ routeW,
    __half* __restrict__ y16, float* __restrict__ logits, int32_t numTokens, int32_t hiddenSize, int32_t numExperts, float eps)
{
    int32_t const lane = threadIdx.x;
    int32_t const e = blockIdx.x;
    int32_t const vecs = hiddenSize / 4;
    float4 const* wr = reinterpret_cast<float4 const*>(routeW + static_cast<int64_t>(e) * hiddenSize);
    float4 rv[kSmallVecPerLane];
#pragma unroll
    for (int32_t i = 0; i < kSmallVecPerLane; ++i)
    {
        int32_t const k = lane + 32 * i;
        rv[i] = k < vecs ? __ldg(wr + k) : make_float4(0.F, 0.F, 0.F, 0.F);
    }
    int32_t const writeVec = blockIdx.x * 32 + lane; // blocks 0.. write the normed activations, one float4 per lane
    float4 const nw = writeVec < vecs ? __ldg(reinterpret_cast<float4 const*>(normW) + writeVec) : make_float4(0.F, 0.F, 0.F, 0.F);
    for (int32_t t = 0; t < numTokens; ++t)
    {
        float4 const* xr = reinterpret_cast<float4 const*>(x + static_cast<int64_t>(t) * hiddenSize);
        float sq = 0.F;
        float dot = 0.F;
#pragma unroll
        for (int32_t i = 0; i < kSmallVecPerLane; ++i)
        {
            int32_t const k = lane + 32 * i;
            if (k < vecs)
            {
                float4 const xv = __ldg(xr + k);
                sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
                dot += xv.x * rv[i].x + xv.y * rv[i].y + xv.z * rv[i].z + xv.w * rv[i].w;
            }
        }
        sq = warpSum(sq);
        dot = warpSum(dot);
        float const inv = 1.F / sqrtf(sq / static_cast<float>(hiddenSize) + eps);
        if (lane == 0)
        {
            logits[static_cast<int64_t>(t) * numExperts + e] = dot * inv;
        }
        if (writeVec < vecs)
        {
            float4 const xv = __ldg(xr + writeVec);
            storeHalf4(y16 + static_cast<int64_t>(t) * hiddenSize + 4 * writeVec, make_float4(xv.x * inv * nw.x, xv.y * inv * nw.y, xv.z * inv * nw.z, xv.w * inv * nw.w));
        }
    }
}

constexpr int32_t kTokTile = 64;
constexpr int32_t kExpTile = 32;
constexpr int32_t kNTiles = kExpTile / 8;
constexpr int32_t kKTile = 32;
constexpr int32_t kPad = kKTile + 4; // bank-conflict free fragment loads

__device__ __forceinline__ uint32_t toTf32(float x)
{
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mmaTf32(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void splitStore(uint32_t* hi, uint32_t* lo, float4 v)
{
    float const f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        uint32_t const h = toTf32(f[i]);
        hi[i] = h;
        lo[i] = toTf32(f[i] - __uint_as_float(h));
    }
}

//! logits[t, e] = inv[t] * sum_k x[t, k] * routeW[e, k] on tensor cores: a 64 x 32 output tile per block, four warps of 16 tokens x 32 experts. Each operand is split into a TF32 high
//! part and the TF32 rounding of the remainder, and three MMAs (hi*hi, lo*hi, hi*lo) restore FP32 accuracy (about 1e-6 relative); the TensorRT TF32 matmul this replaces kept 10 bits.
__global__ void __launch_bounds__(128) routeGemmKernel(float const* __restrict__ x, float const* __restrict__ routeW, float const* __restrict__ inv,
    float* __restrict__ logits, int64_t numTokens, int32_t hiddenSize, int32_t numExperts)
{
    __shared__ uint32_t xsHi[kTokTile][kPad], xsLo[kTokTile][kPad];
    __shared__ uint32_t wsHi[kExpTile][kPad], wsLo[kExpTile][kPad];
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;
    int32_t const g = lane >> 2;
    int32_t const t4 = lane & 3;
    int64_t const t0 = static_cast<int64_t>(blockIdx.y) * kTokTile;
    int32_t const e0 = blockIdx.x * kExpTile;
    float acc[kNTiles][4];
#pragma unroll
    for (int32_t j = 0; j < kNTiles; ++j)
    {
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            acc[j][i] = 0.F;
        }
    }
    constexpr int32_t kXLoads = kTokTile * kKTile / 4 / 128; // float4 per thread per chunk
    constexpr int32_t kWLoads = kExpTile * kKTile / 4 / 128;
    float4 xr[kXLoads], wr[kWLoads];
    auto fetch = [&](int32_t k0) {
#pragma unroll
        for (int32_t u = 0; u < kXLoads; ++u)
        {
            int32_t const i = threadIdx.x + 128 * u;
            int64_t const t = t0 + i / (kKTile / 4);
            int32_t const k = k0 + (i % (kKTile / 4)) * 4;
            xr[u] = (k < hiddenSize && t < numTokens) ? *reinterpret_cast<float4 const*>(x + t * hiddenSize + k) : make_float4(0.F, 0.F, 0.F, 0.F);
        }
#pragma unroll
        for (int32_t u = 0; u < kWLoads; ++u)
        {
            int32_t const i = threadIdx.x + 128 * u;
            int32_t const e = e0 + i / (kKTile / 4);
            int32_t const k = k0 + (i % (kKTile / 4)) * 4;
            wr[u] = (k < hiddenSize && e < numExperts) ? *reinterpret_cast<float4 const*>(routeW + static_cast<int64_t>(e) * hiddenSize + k) : make_float4(0.F, 0.F, 0.F, 0.F);
        }
    };
    fetch(0);
    for (int32_t k0 = 0; k0 < hiddenSize; k0 += kKTile)
    {
#pragma unroll
        for (int32_t u = 0; u < kXLoads; ++u)
        {
            int32_t const i = threadIdx.x + 128 * u;
            splitStore(&xsHi[i / (kKTile / 4)][(i % (kKTile / 4)) * 4], &xsLo[i / (kKTile / 4)][(i % (kKTile / 4)) * 4], xr[u]);
        }
#pragma unroll
        for (int32_t u = 0; u < kWLoads; ++u)
        {
            int32_t const i = threadIdx.x + 128 * u;
            splitStore(&wsHi[i / (kKTile / 4)][(i % (kKTile / 4)) * 4], &wsLo[i / (kKTile / 4)][(i % (kKTile / 4)) * 4], wr[u]);
        }
        __syncthreads();
        if (k0 + kKTile < hiddenSize)
        {
            fetch(k0 + kKTile); // the next chunk's loads fly while this one computes
        }
#pragma unroll
        for (int32_t ks = 0; ks < kKTile; ks += 8)
        {
            uint32_t aHi[4], aLo[4];
            aHi[0] = xsHi[warp * 16 + g][ks + t4];
            aHi[1] = xsHi[warp * 16 + g + 8][ks + t4];
            aHi[2] = xsHi[warp * 16 + g][ks + t4 + 4];
            aHi[3] = xsHi[warp * 16 + g + 8][ks + t4 + 4];
            aLo[0] = xsLo[warp * 16 + g][ks + t4];
            aLo[1] = xsLo[warp * 16 + g + 8][ks + t4];
            aLo[2] = xsLo[warp * 16 + g][ks + t4 + 4];
            aLo[3] = xsLo[warp * 16 + g + 8][ks + t4 + 4];
#pragma unroll
            for (int32_t j = 0; j < kNTiles; ++j)
            {
                uint32_t const b0Hi = wsHi[j * 8 + g][ks + t4], b1Hi = wsHi[j * 8 + g][ks + t4 + 4];
                uint32_t const b0Lo = wsLo[j * 8 + g][ks + t4], b1Lo = wsLo[j * 8 + g][ks + t4 + 4];
                mmaTf32(acc[j], aLo, b0Hi, b1Hi);
                mmaTf32(acc[j], aHi, b0Lo, b1Lo);
                mmaTf32(acc[j], aHi, b0Hi, b1Hi);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int32_t j = 0; j < kNTiles; ++j)
    {
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            int64_t const t = t0 + warp * 16 + g + (i >> 1) * 8;
            int32_t const e = e0 + j * 8 + 2 * t4 + (i & 1);
            if (t < numTokens && e < numExperts)
            {
                logits[t * numExperts + e] = acc[j][i] * inv[t];
            }
        }
    }
}
} // namespace

size_t inkling2NormRouteWorkspaceBytes(int64_t numTokens)
{
    return static_cast<size_t>(numTokens > 0 ? numTokens : 0) * sizeof(float);
}

void launchInkling2NormRoute(float const* x, float const* normW, float const* routeW, __half* y16, float* logits, int64_t numTokens, int32_t hiddenSize,
    int32_t numExperts, float eps, void* workspace, cudaStream_t stream)
{
    if (numTokens <= 0)
    {
        return;
    }
    bool const route = numExperts > 0;
    if (route && numTokens <= kSmallMaxTokens && hiddenSize <= kSmallMaxHidden && static_cast<int64_t>(numExperts) * 32 >= hiddenSize / 4)
    {
        normRouteSmallKernel<<<static_cast<unsigned int>(numExperts), 32, 0, stream>>>(
            x, normW, routeW, y16, logits, static_cast<int32_t>(numTokens), hiddenSize, numExperts, eps);
        return;
    }
    float* inv = route ? static_cast<float*>(workspace) : nullptr;
    normRowsKernel<<<static_cast<unsigned int>(numTokens), kNormThreads, 0, stream>>>(x, normW, y16, inv, hiddenSize, eps);
    if (route)
    {
        dim3 const grid(static_cast<unsigned int>((numExperts + kExpTile - 1) / kExpTile), static_cast<unsigned int>((numTokens + kTokTile - 1) / kTokTile));
        routeGemmKernel<<<grid, 128, 0, stream>>>(x, routeW, inv, logits, numTokens, hiddenSize, numExperts);
    }
}

} // namespace kernel
} // namespace trt_edgellm
