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
#include <cstdlib>
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

//! Decode variant of stage 1 (EDGELLM_INKLING_GATE_LOW_ROWS=1, default): grid (rank, tokens), one row of `down` per block and all 128 threads share it, so 64 blocks and every
//! weight load of the row is in flight at once (the 16-block, one-warp-per-row version was latency-bound at about 47 GB/s, METRICS 225).
__global__ void __launch_bounds__(128) gateLowRowKernel(
    float const* __restrict__ hidden, float const* __restrict__ message, uint32_t const* __restrict__ down, float* __restrict__ low, int32_t hiddenSize,
    int32_t rank, float eps, uint32_t const* forgetW, uint32_t const* gainW)
{
    int64_t const token = blockIdx.y;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;
    int32_t const j = blockIdx.x;
    float const* h = hidden + token * hiddenSize;
    float const* m = message + token * hiddenSize;
    uint4 const* row = reinterpret_cast<uint4 const*>(down + static_cast<int64_t>(j) * hiddenSize);
    int32_t const chunks = hiddenSize / 4; // uint4 of 8 BF16 weights; the 2 * hidden concatenation is h then m
    constexpr int32_t kMaxPer = 8;         // hidden <= kInkling2GateMaxHidden = 2,304 + slack: chunks / 128 <= 8
    uint4 w[kMaxPer];
#pragma unroll
    for (int32_t i = 0; i < kMaxPer; ++i)
    {
        int32_t const c = threadIdx.x + 128 * i;
        if (c < chunks)
        {
            w[i] = row[c];
        }
    }
    // L2 prefetch of the stage 2 weights (forget and gain rows, 1.18 MB, otherwise cold DRAM reads of the next kernel): 64 blocks share it
    if (forgetW != nullptr)
    {
        int64_t const bytes = static_cast<int64_t>(hiddenSize) * (rank / 2) * 4; // one matrix: hidden rows of rank BF16
        int64_t const lines = bytes / 128;
        for (int64_t l = static_cast<int64_t>(blockIdx.x) * 128 + threadIdx.x; l < lines; l += static_cast<int64_t>(gridDim.x) * 128)
        {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<char const*>(forgetW) + l * 128));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<char const*>(gainW) + l * 128));
        }
    }
    float sh = 0.F;
    float sm = 0.F;
    for (int32_t k = threadIdx.x * 4; k < hiddenSize; k += 128 * 4)
    {
        float4 const a = *reinterpret_cast<float4 const*>(h + k);
        float4 const b = *reinterpret_cast<float4 const*>(m + k);
        sh += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        sm += b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
    }
    float dotH = 0.F;
    float dotM = 0.F;
#pragma unroll
    for (int32_t i = 0; i < kMaxPer; ++i)
    {
        int32_t const c = threadIdx.x + 128 * i;
        if (c < chunks)
        {
            int32_t const k = c * 8;
            float const* src = k < hiddenSize ? h + k : m + (k - hiddenSize);
            float4 const x0 = *reinterpret_cast<float4 const*>(src);
            float4 const x1 = *reinterpret_cast<float4 const*>(src + 4);
            float const d = bf16Lo(w[i].x) * x0.x + bf16Hi(w[i].x) * x0.y + bf16Lo(w[i].y) * x0.z + bf16Hi(w[i].y) * x0.w + bf16Lo(w[i].z) * x1.x
                + bf16Hi(w[i].z) * x1.y + bf16Lo(w[i].w) * x1.z + bf16Hi(w[i].w) * x1.w;
            if (k < hiddenSize)
            {
                dotH += d;
            }
            else
            {
                dotM += d;
            }
        }
    }
    sh = warpSum(sh);
    sm = warpSum(sm);
    dotH = warpSum(dotH);
    dotM = warpSum(dotM);
    __shared__ float red[4][4];
    if (lane == 0)
    {
        red[warp][0] = sh;
        red[warp][1] = sm;
        red[warp][2] = dotH;
        red[warp][3] = dotM;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float th = 0.F, tm = 0.F, dh = 0.F, dm = 0.F;
        for (int32_t w2 = 0; w2 < 4; ++w2)
        {
            th += red[w2][0];
            tm += red[w2][1];
            dh += red[w2][2];
            dm += red[w2][3];
        }
        float const invH = 1.F / sqrtf(th / static_cast<float>(hiddenSize) + eps);
        float const invM = 1.F / sqrtf(tm / static_cast<float>(hiddenSize) + eps);
        low[token * rank + j] = invH * dh + invM * dm;
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
// ---- prefill path (rank 64, 32 or more tokens): tensor cores. The BF16 weights are exact in TF32; the FP32 activations are split into a TF32 high part and a TF32 low part (two MMAs
// per product), so the result stays at FP32 accuracy (about 1e-6 relative) while every token no longer re-reads the 590 KB of gate weights from L2 (METRICS 211).
//! The k index inside an m16n8k8 step is permuted (slot t <-> k 2t, slot t + 4 <-> k 2t + 1) so that each lane loads one float2 of activations and one 32-bit word (two BF16 weights).
__device__ __forceinline__ uint32_t toTf32(float x)
{
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void splitTf32(float x, uint32_t& hi, uint32_t& lo)
{
    hi = toTf32(x);
    lo = toTf32(x - __uint_as_float(hi));
}

__device__ __forceinline__ void mmaTf32(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

//! Per token: 1 / sqrt(mean(h^2) + eps) and the same for m, as a float2. Grid (tokens), 256 threads.
__global__ void gateNormsKernel(float const* __restrict__ hidden, float const* __restrict__ message, float2* __restrict__ norms, int32_t hiddenSize, float eps)
{
    __shared__ float partial[8][2];
    int64_t const token = blockIdx.x;
    float const* h = hidden + token * hiddenSize;
    float const* m = message + token * hiddenSize;
    float sh = 0.F;
    float sm = 0.F;
    for (int32_t k = threadIdx.x * 4; k < hiddenSize; k += 256 * 4)
    {
        float4 const a = *reinterpret_cast<float4 const*>(h + k);
        float4 const b = *reinterpret_cast<float4 const*>(m + k);
        sh += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        sm += b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
    }
    sh = warpSum(sh);
    sm = warpSum(sm);
    if ((threadIdx.x & 31) == 0)
    {
        partial[threadIdx.x >> 5][0] = sh;
        partial[threadIdx.x >> 5][1] = sm;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float th = 0.F;
        float tm = 0.F;
        for (int32_t w = 0; w < 8; ++w)
        {
            th += partial[w][0];
            tm += partial[w][1];
        }
        norms[token] = make_float2(1.F / sqrtf(th / static_cast<float>(hiddenSize) + eps), 1.F / sqrtf(tm / static_cast<float>(hiddenSize) + eps));
    }
}

//! Stage 1 as a GEMM: low [tokens, 64] = [h * invH | m * invM] [tokens, 2 hidden] x down^T. Block = 4 warps = 64 tokens, each warp 16 tokens x all 64 rank columns.
__global__ void __launch_bounds__(128) gateLowGemmKernel(float const* __restrict__ hidden, float const* __restrict__ message, uint32_t const* __restrict__ down,
    float2 const* __restrict__ norms, float* __restrict__ low, int64_t numTokens, int32_t hiddenSize)
{
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const g = lane >> 2;
    int32_t const t = lane & 3;
    int64_t const row0 = static_cast<int64_t>(blockIdx.x) * 64 + warp * 16 + g;
    int64_t const row1 = row0 + 8;
    bool const v0 = row0 < numTokens;
    bool const v1 = row1 < numTokens;
    float2 const n0 = v0 ? norms[row0] : make_float2(0.F, 0.F);
    float2 const n1 = v1 ? norms[row1] : make_float2(0.F, 0.F);
    float c[8][4];
#pragma unroll
    for (int32_t i = 0; i < 8; ++i)
    {
        c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.F;
    }
#pragma unroll 1
    for (int32_t phase = 0; phase < 2; ++phase)
    {
        float const* a = phase == 0 ? hidden : message;
        float const s0 = phase == 0 ? n0.x : n0.y;
        float const s1 = phase == 0 ? n1.x : n1.y;
        int64_t const woff = static_cast<int64_t>(phase) * (hiddenSize / 2);
        // the tensor cores truncate while accumulating, so a long chain of MMAs builds a bias of about (number of MMAs) x 2^-24: every chunk of 64 k starts from zero accumulators and
        // is added to the running sums with round-to-nearest FP32 additions
#pragma unroll 1
        for (int32_t kc = 0; kc < hiddenSize; kc += 64)
        {
            float tmp[8][4];
#pragma unroll
            for (int32_t i = 0; i < 8; ++i)
            {
                tmp[i][0] = tmp[i][1] = tmp[i][2] = tmp[i][3] = 0.F;
            }
#pragma unroll 1
            for (int32_t kk = kc; kk < kc + 64; kk += 8)
            {
                float2 const x0 = v0 ? *reinterpret_cast<float2 const*>(a + row0 * hiddenSize + kk + 2 * t) : make_float2(0.F, 0.F);
                float2 const x1 = v1 ? *reinterpret_cast<float2 const*>(a + row1 * hiddenSize + kk + 2 * t) : make_float2(0.F, 0.F);
                uint32_t ah[4];
                uint32_t al[4];
                splitTf32(x0.x * s0, ah[0], al[0]);
                splitTf32(x1.x * s1, ah[1], al[1]);
                splitTf32(x0.y * s0, ah[2], al[2]);
                splitTf32(x1.y * s1, ah[3], al[3]);
#pragma unroll
                for (int32_t nt = 0; nt < 8; ++nt)
                {
                    uint32_t const w = __ldg(down + static_cast<int64_t>(nt * 8 + g) * hiddenSize + woff + (kk >> 1) + t);
                    uint32_t const b0 = w << 16;
                    uint32_t const b1 = w & 0xFFFF0000U;
                    mmaTf32(tmp[nt], ah[0], ah[1], ah[2], ah[3], b0, b1);
                    mmaTf32(tmp[nt], al[0], al[1], al[2], al[3], b0, b1);
                }
            }
#pragma unroll
            for (int32_t i = 0; i < 8; ++i)
            {
                c[i][0] += tmp[i][0];
                c[i][1] += tmp[i][1];
                c[i][2] += tmp[i][2];
                c[i][3] += tmp[i][3];
            }
        }
    }
#pragma unroll
    for (int32_t nt = 0; nt < 8; ++nt)
    {
        int32_t const j = nt * 8 + 2 * t;
        if (v0)
        {
            *reinterpret_cast<float2*>(low + row0 * 64 + j) = make_float2(c[nt][0], c[nt][1]);
        }
        if (v1)
        {
            *reinterpret_cast<float2*>(low + row1 * 64 + j) = make_float2(c[nt][2], c[nt][3]);
        }
    }
}

//! Stage 2 as a GEMM with the gate epilogue: forget/gain [tokens, 64 channels per block] = low [tokens, 64] x forgetW/gainW^T, out = (1 - tanh f) h + tanh g m.
//! Grid (hidden / 64, tokens / 64), 4 warps, each 16 tokens x 64 channels.
__global__ void __launch_bounds__(128) gateApplyGemmKernel(float const* __restrict__ hidden, float const* __restrict__ message, uint32_t const* __restrict__ forgetW,
    uint32_t const* __restrict__ gainW, float const* __restrict__ low, float* __restrict__ out, int64_t numTokens, int32_t hiddenSize)
{
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const g = lane >> 2;
    int32_t const t = lane & 3;
    int64_t const row0 = static_cast<int64_t>(blockIdx.y) * 64 + warp * 16 + g;
    int64_t const row1 = row0 + 8;
    bool const v0 = row0 < numTokens;
    bool const v1 = row1 < numTokens;
    int32_t const cbase = blockIdx.x * 64;
    float cf[8][4];
    float cg[8][4];
#pragma unroll
    for (int32_t i = 0; i < 8; ++i)
    {
        cf[i][0] = cf[i][1] = cf[i][2] = cf[i][3] = 0.F;
        cg[i][0] = cg[i][1] = cg[i][2] = cg[i][3] = 0.F;
    }
#pragma unroll
    for (int32_t k0 = 0; k0 < 64; k0 += 8)
    {
        float2 const x0 = v0 ? *reinterpret_cast<float2 const*>(low + row0 * 64 + k0 + 2 * t) : make_float2(0.F, 0.F);
        float2 const x1 = v1 ? *reinterpret_cast<float2 const*>(low + row1 * 64 + k0 + 2 * t) : make_float2(0.F, 0.F);
        uint32_t ah[4];
        uint32_t al[4];
        splitTf32(x0.x, ah[0], al[0]);
        splitTf32(x1.x, ah[1], al[1]);
        splitTf32(x0.y, ah[2], al[2]);
        splitTf32(x1.y, ah[3], al[3]);
#pragma unroll
        for (int32_t nt = 0; nt < 8; ++nt)
        {
            int64_t const wi = static_cast<int64_t>(cbase + nt * 8 + g) * 32 + (k0 >> 1) + t;
            uint32_t const wf = __ldg(forgetW + wi);
            uint32_t const wg = __ldg(gainW + wi);
            mmaTf32(cf[nt], ah[0], ah[1], ah[2], ah[3], wf << 16, wf & 0xFFFF0000U);
            mmaTf32(cf[nt], al[0], al[1], al[2], al[3], wf << 16, wf & 0xFFFF0000U);
            mmaTf32(cg[nt], ah[0], ah[1], ah[2], ah[3], wg << 16, wg & 0xFFFF0000U);
            mmaTf32(cg[nt], al[0], al[1], al[2], al[3], wg << 16, wg & 0xFFFF0000U);
        }
    }
#pragma unroll
    for (int32_t nt = 0; nt < 8; ++nt)
    {
        int32_t const col = cbase + nt * 8 + 2 * t;
#pragma unroll
        for (int32_t half = 0; half < 2; ++half)
        {
            int64_t const row = half == 0 ? row0 : row1;
            if (row < numTokens)
            {
                float2 const hv = *reinterpret_cast<float2 const*>(hidden + row * hiddenSize + col);
                float2 const mv = *reinterpret_cast<float2 const*>(message + row * hiddenSize + col);
                float2 r;
                r.x = (1.F - tanhf(cf[nt][2 * half])) * hv.x + tanhf(cg[nt][2 * half]) * mv.x;
                r.y = (1.F - tanhf(cf[nt][2 * half + 1])) * hv.y + tanhf(cg[nt][2 * half + 1]) * mv.y;
                *reinterpret_cast<float2*>(out + row * hiddenSize + col) = r;
            }
        }
    }
}

constexpr int64_t kGemmMinTokens = 32;

} // namespace

size_t inkling2MessageGateWorkspaceBytes(int64_t numTokens, int32_t rank)
{
    // low [tokens, rank] and the per-token inverse norms [tokens, 2] of the prefill path
    return static_cast<size_t>(numTokens) * (rank + 2) * sizeof(float);
}

void launchInkling2MessageGate(float const* hidden, float const* message, uint32_t const* down, uint32_t const* forgetW, uint32_t const* gainW, float* out,
    int64_t numTokens, int32_t hiddenSize, int32_t rank, float eps, void* workspace, cudaStream_t stream)
{
    if (numTokens <= 0)
    {
        return;
    }
    float* low = static_cast<float*>(workspace);
    static bool const gemmOff = std::getenv("EDGELLM_INKLING_GATE_GEMM") != nullptr && std::getenv("EDGELLM_INKLING_GATE_GEMM")[0] == '0';
    if (!gemmOff && numTokens >= kGemmMinTokens && rank == 64 && hiddenSize % 64 == 0 && hiddenSize <= kInkling2GateMaxHidden)
    {
        float2* norms = reinterpret_cast<float2*>(low + numTokens * rank);
        gateNormsKernel<<<static_cast<unsigned int>(numTokens), 256, 0, stream>>>(hidden, message, norms, hiddenSize, eps);
        gateLowGemmKernel<<<static_cast<unsigned int>((numTokens + 63) / 64), 128, 0, stream>>>(hidden, message, down, norms, low, numTokens, hiddenSize);
        dim3 const gemmGrid(hiddenSize / 64, static_cast<unsigned int>((numTokens + 63) / 64));
        gateApplyGemmKernel<<<gemmGrid, 128, 0, stream>>>(hidden, message, forgetW, gainW, low, out, numTokens, hiddenSize);
        return;
    }
    dim3 const lowGrid((rank + kLowWarps - 1) / kLowWarps, static_cast<unsigned int>(numTokens));
    static bool const rowsOff = std::getenv("EDGELLM_INKLING_GATE_LOW_ROWS") != nullptr && std::getenv("EDGELLM_INKLING_GATE_LOW_ROWS")[0] == '0';
    if (!rowsOff && hiddenSize / 4 <= 128 * 8)
    {
        static bool const prefetchOff = std::getenv("EDGELLM_INKLING_GATE_PREFETCH") != nullptr && std::getenv("EDGELLM_INKLING_GATE_PREFETCH")[0] == '0';
        gateLowRowKernel<<<dim3(static_cast<unsigned int>(rank), static_cast<unsigned int>(numTokens)), 128, 0, stream>>>(
            hidden, message, down, low, hiddenSize, rank, eps, prefetchOff ? nullptr : forgetW, prefetchOff ? nullptr : gainW);
    }
    else
    {
        gateLowKernel<<<lowGrid, kLowWarps * 32, 0, stream>>>(hidden, message, down, low, hiddenSize, rank, eps);
    }
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
