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

constexpr int32_t kTile = 32;
constexpr int32_t kKTile = 32;

//! logits[t, e] = inv[t] * sum_k x[t, k] * routeW[e, k]: 32 x 32 output tile per block, 16 x 16 threads each owning 2 x 2 outputs, FP32 throughout.
__global__ void __launch_bounds__(256) routeGemmKernel(float const* __restrict__ x, float const* __restrict__ routeW, float const* __restrict__ inv,
    float* __restrict__ logits, int64_t numTokens, int32_t hiddenSize, int32_t numExperts)
{
    __shared__ float xs[kKTile][kTile + 1];
    __shared__ float ws[kKTile][kTile + 1];
    int32_t const tx = threadIdx.x & 15;
    int32_t const ty = threadIdx.x >> 4;
    int64_t const t0 = static_cast<int64_t>(blockIdx.y) * kTile;
    int32_t const e0 = blockIdx.x * kTile;
    float acc[2][2] = {{0.F, 0.F}, {0.F, 0.F}};
    for (int32_t k0 = 0; k0 < hiddenSize; k0 += kKTile)
    {
        for (int32_t i = threadIdx.x; i < kTile * kKTile; i += 256)
        {
            int32_t const r = i / kKTile;
            int32_t const c = i % kKTile;
            int64_t const t = t0 + r;
            int32_t const e = e0 + r;
            int32_t const k = k0 + c;
            xs[c][r] = (t < numTokens && k < hiddenSize) ? x[t * hiddenSize + k] : 0.F;
            ws[c][r] = (e < numExperts && k < hiddenSize) ? routeW[static_cast<int64_t>(e) * hiddenSize + k] : 0.F;
        }
        __syncthreads();
#pragma unroll
        for (int32_t c = 0; c < kKTile; ++c)
        {
            float const a0 = xs[c][ty], a1 = xs[c][ty + 16];
            float const b0 = ws[c][tx], b1 = ws[c][tx + 16];
            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }
        __syncthreads();
    }
#pragma unroll
    for (int32_t i = 0; i < 2; ++i)
    {
        int64_t const t = t0 + ty + 16 * i;
        if (t >= numTokens)
        {
            continue;
        }
#pragma unroll
        for (int32_t j = 0; j < 2; ++j)
        {
            int32_t const e = e0 + tx + 16 * j;
            if (e < numExperts)
            {
                logits[t * numExperts + e] = acc[i][j] * inv[t];
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
        dim3 const grid(static_cast<unsigned int>((numExperts + kTile - 1) / kTile), static_cast<unsigned int>((numTokens + kTile - 1) / kTile));
        routeGemmKernel<<<grid, 256, 0, stream>>>(x, routeW, inv, logits, numTokens, hiddenSize, numExperts);
    }
}

} // namespace kernel
} // namespace trt_edgellm
