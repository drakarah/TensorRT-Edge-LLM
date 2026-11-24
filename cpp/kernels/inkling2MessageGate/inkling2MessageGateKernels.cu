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

#include "inkling2MessageGateKernels.h"

#include <cmath>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr int32_t kLowWarps = 4;  // rows of `down` per block in stage 1
constexpr int32_t kApplyWarps = 8; // output channels per block in stage 2

__device__ __forceinline__ float bf16Lo(uint32_t w)
{
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float bf16Hi(uint32_t w)
{
    return __uint_as_float(w & 0xFFFF0000U);
}

__device__ __forceinline__ float warpSum(float v)
{
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffU, v, off);
    }
    return v;
}

//! Stage 1, grid (rank / kLowWarps, tokens), kLowWarps warps per block: every block derives the inverse RMS of h and m for its token (4,608 floats from L1) and
//! each warp computes one low[j] as the dot product of its row of `down` (contiguous BF16) with the normalised concatenation.
__global__ void __launch_bounds__(kLowWarps * 32) gateLowKernel(
    float const* __restrict__ hidden, float const* __restrict__ message, uint32_t const* __restrict__ down, float* __restrict__ low, int32_t hiddenSize,
    int32_t rank, float eps)
{
    __shared__ float norms[2];
    __shared__ float partial[kLowWarps][2];
    int64_t const token = blockIdx.y;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;
    float const* h = hidden + token * hiddenSize;
    float const* m = message + token * hiddenSize;
    float sh = 0.F;
    float sm = 0.F;
    for (int32_t k = threadIdx.x * 4; k < hiddenSize; k += kLowWarps * 32 * 4)
    {
        float4 const a = *reinterpret_cast<float4 const*>(h + k);
        float4 const b = *reinterpret_cast<float4 const*>(m + k);
        sh += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        sm += b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
    }
    sh = warpSum(sh);
    sm = warpSum(sm);
    if (lane == 0)
    {
        partial[warp][0] = sh;
        partial[warp][1] = sm;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float th = 0.F;
        float tm = 0.F;
        for (int32_t w = 0; w < kLowWarps; ++w)
        {
            th += partial[w][0];
            tm += partial[w][1];
        }
        norms[0] = 1.F / sqrtf(th / static_cast<float>(hiddenSize) + eps);
        norms[1] = 1.F / sqrtf(tm / static_cast<float>(hiddenSize) + eps);
    }
    __syncthreads();
    float const invH = norms[0];
    float const invM = norms[1];
    int32_t const j = blockIdx.x * kLowWarps + warp;
    if (j >= rank)
    {
        return;
    }
    uint4 const* row = reinterpret_cast<uint4 const*>(down + static_cast<int64_t>(j) * hiddenSize); // 2 * hidden BF16 = hidden words = hidden / 4 uint4
    int32_t const chunks = hiddenSize / 4;                                                         // uint4 of 8 BF16 weights
    float acc = 0.F;
    for (int32_t c = lane; c < chunks; c += 32)
    {
        uint4 const w = row[c];
        int32_t const k = c * 8; // index into the 2 * hidden concatenation
        float const* src = k < hiddenSize ? h + k : m + (k - hiddenSize);
        float const inv = k < hiddenSize ? invH : invM;
        float4 const x0 = *reinterpret_cast<float4 const*>(src);
        float4 const x1 = *reinterpret_cast<float4 const*>(src + 4);
        acc += inv * (bf16Lo(w.x) * x0.x + bf16Hi(w.x) * x0.y + bf16Lo(w.y) * x0.z + bf16Hi(w.y) * x0.w + bf16Lo(w.z) * x1.x + bf16Hi(w.z) * x1.y
            + bf16Lo(w.w) * x1.z + bf16Hi(w.w) * x1.w);
    }
    acc = warpSum(acc);
    if (lane == 0)
    {
        low[token * rank + j] = acc;
    }
}

//! Stage 2, grid (hidden / kApplyWarps, tokens): one warp per output channel c; lane l holds low[2l], low[2l + 1] (rank 64) and the matching BF16 pair of the
//! forget and gain rows; shuffle reductions, tanh and the gated mix.
template <int32_t kPairsPerLane>
__global__ void __launch_bounds__(kApplyWarps * 32) gateApplyKernel(float const* __restrict__ hidden, float const* __restrict__ message,
    uint32_t const* __restrict__ forgetW, uint32_t const* __restrict__ gainW, float const* __restrict__ low, float* __restrict__ out, int32_t hiddenSize,
    int32_t rank)
{
    int64_t const token = blockIdx.y;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;
    float lows[2 * kPairsPerLane];
#pragma unroll
    for (int32_t p = 0; p < kPairsPerLane; ++p)
    {
        lows[2 * p] = low[token * rank + 2 * (lane + 32 * p)];
        lows[2 * p + 1] = low[token * rank + 2 * (lane + 32 * p) + 1];
    }
    int32_t const c = blockIdx.x * kApplyWarps + warp;
    if (c >= hiddenSize)
    {
        return;
    }
    int32_t const wordsPerRow = rank / 2;
    float f = 0.F;
    float g = 0.F;
#pragma unroll
    for (int32_t p = 0; p < kPairsPerLane; ++p)
    {
        uint32_t const fw = forgetW[static_cast<int64_t>(c) * wordsPerRow + lane + 32 * p];
        uint32_t const gw = gainW[static_cast<int64_t>(c) * wordsPerRow + lane + 32 * p];
        f += bf16Lo(fw) * lows[2 * p] + bf16Hi(fw) * lows[2 * p + 1];
        g += bf16Lo(gw) * lows[2 * p] + bf16Hi(gw) * lows[2 * p + 1];
    }
    f = warpSum(f);
    g = warpSum(g);
    if (lane == 0)
    {
        float const hv = hidden[token * hiddenSize + c];
        float const mv = message[token * hiddenSize + c];
        out[token * hiddenSize + c] = (1.F - tanhf(f)) * hv + tanhf(g) * mv;
    }
}
} // namespace

size_t inkling2MessageGateWorkspaceBytes(int64_t numTokens, int32_t rank)
{
    return static_cast<size_t>(numTokens) * rank * sizeof(float);
}

void launchInkling2MessageGate(float const* hidden, float const* message, uint32_t const* down, uint32_t const* forgetW, uint32_t const* gainW, float* out,
    int64_t numTokens, int32_t hiddenSize, int32_t rank, float eps, void* workspace, cudaStream_t stream)
{
    if (numTokens <= 0)
    {
        return;
    }
    float* low = static_cast<float*>(workspace);
    dim3 const lowGrid((rank + kLowWarps - 1) / kLowWarps, static_cast<unsigned int>(numTokens));
    gateLowKernel<<<lowGrid, kLowWarps * 32, 0, stream>>>(hidden, message, down, low, hiddenSize, rank, eps);
    dim3 const applyGrid((hiddenSize + kApplyWarps - 1) / kApplyWarps, static_cast<unsigned int>(numTokens));
    if (rank == 64)
    {
        gateApplyKernel<1><<<applyGrid, kApplyWarps * 32, 0, stream>>>(hidden, message, forgetW, gainW, low, out, hiddenSize, rank);
    }
    else
    {
        gateApplyKernel<2><<<applyGrid, kApplyWarps * 32, 0, stream>>>(hidden, message, forgetW, gainW, low, out, hiddenSize, rank);
    }
}

} // namespace kernel
} // namespace trt_edgellm
