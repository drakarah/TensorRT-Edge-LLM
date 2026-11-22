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

#include "inkling2AttentionKernels.h"

#include <cmath>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr int32_t kWarps = 4;
constexpr int32_t kThreads = kWarps * 32;

__device__ __forceinline__ float warpSum(float v)
{
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ float halfWarpSum(float v)
{
#pragma unroll
    for (int32_t o = 8; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

//! Sequence and position of token slot t; false for padding slots. offsets has batchSize + 1 entries.
__device__ __forceinline__ bool locateToken(int32_t t, int32_t const* offsets, int32_t const* lengths,
    int32_t const* past, int32_t batchSize, int32_t& seq, int32_t& pos)
{
    seq = 0;
    while (seq + 1 < batchSize && t >= offsets[seq + 1])
    {
        ++seq;
    }
    int32_t const local = t - offsets[seq];
    pos = past[seq] + local;
    return local >= 0 && local < lengths[seq];
}

__global__ void appendKernel(Inkling2AttentionParams p, half const* __restrict__ kNew, half const* __restrict__ vNew,
    half* __restrict__ kvPool, int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths,
    int32_t const* __restrict__ past, int32_t const* __restrict__ pageTable)
{
    int32_t const t = blockIdx.x;
    int32_t seq, pos;
    if (!locateToken(t, offsets, lengths, past, p.batchSize, seq, pos))
    {
        return;
    }
    int64_t const rowElems = int64_t(p.numKVHeads) * kInkling2HeadDim;
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * rowElems;
    int32_t const* table = pageTable + int64_t(seq) * 2 * p.maxPagesPerSeq;
    int64_t const kDst = int64_t(table[pos / kInkling2TokensPerPage]) * pageElems + (pos % kInkling2TokensPerPage) * rowElems;
    int64_t const vDst
        = int64_t(table[p.maxPagesPerSeq + pos / kInkling2TokensPerPage]) * pageElems + (pos % kInkling2TokensPerPage) * rowElems;
    for (int64_t i = threadIdx.x; i < rowElems; i += blockDim.x)
    {
        kvPool[kDst + i] = kNew[int64_t(t) * rowElems + i];
        kvPool[vDst + i] = vNew[int64_t(t) * rowElems + i];
    }
}

//! One block (4 warps) per (query token, query head). Each warp walks a strided subset of the keys with an online
//! softmax over four head-dim elements per lane; the warps merge through shared memory and the zero-value sink joins
//! the denominator.
__global__ void attentionKernel(Inkling2AttentionParams p, half const* __restrict__ qLin, half const* __restrict__ relLin,
    half const* __restrict__ sinkLin, half const* __restrict__ gateLin, float const* __restrict__ qScale,
    float const* __restrict__ relScale, float const* __restrict__ relProjT, half const* __restrict__ kvPool,
    int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths, int32_t const* __restrict__ past,
    int32_t const* __restrict__ pageTable, half* __restrict__ out)
{
    int32_t const t = blockIdx.x, h = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int32_t seq, qpos;
    if (!locateToken(t, offsets, lengths, past, p.batchSize, seq, qpos))
    {
        for (int32_t d = threadIdx.x; d < kInkling2HeadDim; d += blockDim.x)
        {
            out[(int64_t(t) * p.numQHeads + h) * kInkling2HeadDim + d] = __float2half(0.F);
        }
        return;
    }
    int32_t const kvh = h / (p.numQHeads / p.numKVHeads);
    int64_t const rowElems = int64_t(p.numKVHeads) * kInkling2HeadDim;
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * rowElems;
    int32_t const* table = pageTable + int64_t(seq) * 2 * p.maxPagesPerSeq;
    float const logScale = p.window == 0 ? 1.F + p.logAlpha * logf(fmaxf((qpos + 1.F) / p.logNFloor, 1.F)) : 1.F;

    float q[4];
    float ss = 0.F;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        int32_t const d = lane * 4 + i;
        q[i] = __half2float(qLin[(int64_t(t) * p.numQHeads + h) * kInkling2HeadDim + d]) * qScale[h * kInkling2HeadDim + d];
        ss += q[i] * q[i];
    }
    ss = warpSum(ss);
    float const qFactor = fminf(p.qClip / (sqrtf(ss) + 1e-6F), 1.F) * logScale;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        q[i] *= qFactor;
    }

    int32_t const j = lane & 15;
    float rs = __half2float(relLin[(int64_t(t) * p.numQHeads + h) * kInkling2RelDim + j]) * relScale[h * kInkling2RelDim + j];
    float const rss = halfWarpSum(rs * rs);
    rs *= fminf(p.relClip / (sqrtf(rss) + 1e-6F), 1.F) * logScale;

    int32_t const lo = p.window > 0 ? max(0, qpos - p.window + 1) : 0;
    float m = -INFINITY, l = 0.F, acc[4] = {0.F, 0.F, 0.F, 0.F};
    for (int32_t s = lo + w; s <= qpos; s += kWarps)
    {
        int64_t const inPage = int64_t(s % kInkling2TokensPerPage) * rowElems + int64_t(kvh) * kInkling2HeadDim + lane * 4;
        uint2 const kr = *reinterpret_cast<uint2 const*>(
            kvPool + int64_t(table[s / kInkling2TokensPerPage]) * pageElems + inPage);
        half2 const* k2 = reinterpret_cast<half2 const*>(&kr);
        float dot = 0.F;
#pragma unroll
        for (int32_t i = 0; i < 2; ++i)
        {
            float2 const f = __half22float2(k2[i]);
            dot += q[2 * i] * f.x + q[2 * i + 1] * f.y;
        }
        dot = warpSum(dot);
        int32_t const dist = qpos - s;
        float bias = 0.F;
        if (dist < p.relExtent)
        {
            bias = halfWarpSum(rs * relProjT[int64_t(dist) * kInkling2RelDim + j]);
        }
        float const score = dot * p.scaling + bias;
        float const mNew = fmaxf(m, score);
        float const corr = __expf(m - mNew), e = __expf(score - mNew);
        uint2 const vr = *reinterpret_cast<uint2 const*>(
            kvPool + int64_t(table[p.maxPagesPerSeq + s / kInkling2TokensPerPage]) * pageElems + inPage);
        half2 const* v2 = reinterpret_cast<half2 const*>(&vr);
#pragma unroll
        for (int32_t i = 0; i < 2; ++i)
        {
            float2 const f = __half22float2(v2[i]);
            acc[2 * i] = acc[2 * i] * corr + e * f.x;
            acc[2 * i + 1] = acc[2 * i + 1] * corr + e * f.y;
        }
        l = l * corr + e;
        m = mNew;
    }

    __shared__ float sm[kWarps], sl[kWarps], sacc[kWarps][kInkling2HeadDim];
    if (lane == 0)
    {
        sm[w] = m;
        sl[w] = l;
    }
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        sacc[w][lane * 4 + i] = acc[i];
    }
    __syncthreads();
    if (w == 0)
    {
        float const sink = __half2float(sinkLin[int64_t(t) * p.numQHeads + h]);
        float mm = sink;
#pragma unroll
        for (int32_t i = 0; i < kWarps; ++i)
        {
            mm = fmaxf(mm, sm[i]);
        }
        float den = __expf(sink - mm);
        float o[4] = {0.F, 0.F, 0.F, 0.F};
#pragma unroll
        for (int32_t i = 0; i < kWarps; ++i)
        {
            float const c = sm[i] == -INFINITY ? 0.F : __expf(sm[i] - mm);
            den += sl[i] * c;
#pragma unroll
            for (int32_t k = 0; k < 4; ++k)
            {
                o[k] += sacc[i][lane * 4 + k] * c;
            }
        }
#pragma unroll
        for (int32_t k = 0; k < 4; ++k)
        {
            int32_t const d = lane * 4 + k;
            int64_t const idx = (int64_t(t) * p.numQHeads + h) * kInkling2HeadDim + d;
            float const g = 1.F / (1.F + __expf(-__half2float(gateLin[idx])));
            out[idx] = __float2half(o[k] / den * g);
        }
    }
}
} // namespace

void launchInkling2Attention(Inkling2AttentionParams const& params, half const* qLin, half const* relLin,
    half const* sinkLin, half const* gateLin, float const* qScale, float const* relScale, float const* relProjT,
    half const* kNew, half const* vNew, half* kvPool, int32_t const* queryStartOffsets, int32_t const* queryLengths,
    int32_t const* pastLengths, int32_t const* pageTable, half* out, cudaStream_t stream)
{
    if (params.numTokens <= 0)
    {
        return;
    }
    appendKernel<<<params.numTokens, 128, 0, stream>>>(
        params, kNew, vNew, kvPool, queryStartOffsets, queryLengths, pastLengths, pageTable);
    attentionKernel<<<dim3(params.numTokens, params.numQHeads), kThreads, 0, stream>>>(params, qLin, relLin, sinkLin,
        gateLin, qScale, relScale, relProjT, kvPool, queryStartOffsets, queryLengths, pastLengths, pageTable, out);
}

} // namespace kernel
} // namespace trt_edgellm
