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

#include <algorithm>
#include <cmath>
#include <cuda_fp8.h>
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

//! KV pool element access: FP16, or FP8 E4M3 (unit scale, saturating) for the long-context cache.
__device__ __forceinline__ void loadKv4(half const* p, float (&f)[4])
{
    uint2 const r = *reinterpret_cast<uint2 const*>(p);
    half2 const* h = reinterpret_cast<half2 const*>(&r);
    float2 const a = __half22float2(h[0]), b = __half22float2(h[1]);
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
}

__device__ __forceinline__ void loadKv4(__nv_fp8_e4m3 const* p, float (&f)[4])
{
    uint32_t const r = *reinterpret_cast<uint32_t const*>(p);
    __nv_fp8x2_e4m3 const* h = reinterpret_cast<__nv_fp8x2_e4m3 const*>(&r);
    float2 const a = static_cast<float2>(h[0]), b = static_cast<float2>(h[1]);
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
}

__device__ __forceinline__ void storeKv(half* p, half v)
{
    *p = v;
}

__device__ __forceinline__ void storeKv(__nv_fp8_e4m3* p, half v)
{
    *p = __nv_fp8_e4m3(__half2float(v));
}

template <typename KvT>
__global__ void appendKernel(Inkling2AttentionParams p, half const* __restrict__ kNew, half const* __restrict__ vNew,
    KvT* __restrict__ kvPool, int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths,
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
    // A negative page id is an unmapped logical page (bounded window storage, graph-capture warmup): nothing to write.
    int32_t const kPage = table[pos / kInkling2TokensPerPage], vPage = table[p.maxPagesPerSeq + pos / kInkling2TokensPerPage];
    if (kPage < 0 || vPage < 0)
    {
        return;
    }
    int64_t const kDst = int64_t(kPage) * pageElems + (pos % kInkling2TokensPerPage) * rowElems;
    int64_t const vDst = int64_t(vPage) * pageElems + (pos % kInkling2TokensPerPage) * rowElems;
    for (int64_t i = threadIdx.x; i < rowElems; i += blockDim.x)
    {
        storeKv(kvPool + kDst + i, kNew[int64_t(t) * rowElems + i]);
        storeKv(kvPool + vDst + i, vNew[int64_t(t) * rowElems + i]);
    }
}

//! One block (4 warps) per (query token, query head). Each warp walks a strided subset of the keys with an online
//! softmax over four head-dim elements per lane; the warps merge through shared memory and the zero-value sink joins
//! the denominator.
template <typename KvT>
__global__ void attentionKernel(Inkling2AttentionParams p, half const* __restrict__ qLin,
    half const* __restrict__ relLin, half const* __restrict__ sinkLin, half const* __restrict__ gateLin,
    float const* __restrict__ qScale, float const* __restrict__ relScale, float const* __restrict__ relProjT,
    KvT const* __restrict__ kvPool, int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths,
    int32_t const* __restrict__ past, int32_t const* __restrict__ pageTable, half* __restrict__ out)
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
        q[i] = __half2float(qLin[(int64_t(t) * p.numQHeads + h) * kInkling2HeadDim + d])
            * qScale[h * kInkling2HeadDim + d];
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
    float rs = __half2float(relLin[(int64_t(t) * p.numQHeads + h) * kInkling2RelDim + j])
        * relScale[h * kInkling2RelDim + j];
    float const rss = halfWarpSum(rs * rs);
    rs *= fminf(p.relClip / (sqrtf(rss) + 1e-6F), 1.F) * logScale;

    int32_t const lo = p.window > 0 ? max(0, qpos - p.window + 1) : 0;
    float m = -INFINITY, l = 0.F, acc[4] = {0.F, 0.F, 0.F, 0.F};
    for (int32_t s = lo + w; s <= qpos; s += kWarps)
    {
        int64_t const inPage
            = int64_t(s % kInkling2TokensPerPage) * rowElems + int64_t(kvh) * kInkling2HeadDim + lane * 4;
        int32_t const kPage = table[s / kInkling2TokensPerPage], vPage = table[p.maxPagesPerSeq + s / kInkling2TokensPerPage];
        if (kPage < 0 || vPage < 0)
        {
            continue; // unmapped logical page: no key (never inside a live window)
        }
        float kf[4];
        loadKv4(kvPool + int64_t(kPage) * pageElems + inPage, kf);
        float dot = 0.F;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            dot += q[i] * kf[i];
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
        float vf[4];
        loadKv4(kvPool + int64_t(vPage) * pageElems + inPage, vf);
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            acc[i] = acc[i] * corr + e * vf[i];
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

// ---- Tiled attention: every block serves one KV head and a tile of consecutive query slots of one sequence (GQA group
// x query tile = up to 24 query rows share each K/V tile staged in shared memory), optionally over one key split.
// Partial (max, sum, accumulator) triples per (slot, head, split) go to a workspace; mergeKernel combines the splits,
// adds the zero-value sink to the denominator and applies the output gate. ----
constexpr int32_t kKeyTile = 64;
constexpr int32_t kTileThreads = 256;
constexpr int32_t kTileWarps = kTileThreads / 32;
constexpr int32_t kMaxRows = 24;
constexpr int32_t kKPad = kInkling2HeadDim + 2; // odd 32-bit stride per key row: conflict-free score loops
constexpr int32_t kSPad = kKeyTile + 1;
constexpr int32_t kPartialStride = kInkling2HeadDim + 2; // m, l, accumulator[128]

constexpr size_t tiledSmemBytes()
{
    return sizeof(half) * (kKeyTile * kKPad + kKeyTile * kInkling2HeadDim)
        + sizeof(float) * (kMaxRows * kSPad + kMaxRows * kInkling2HeadDim + kMaxRows * kInkling2RelDim + 3 * kMaxRows)
        + sizeof(int32_t) * 2 * kMaxRows;
}

//! One key row (128 elements of the KV head) from the pool into shared memory as halves; zeros when `valid` is false.
__device__ __forceinline__ void stageRow(half* dst, half const* src, bool valid, int32_t part)
{
    // 16 parts of 8 halves
    half2* d2 = reinterpret_cast<half2*>(dst) + part * 4;
    if (!valid)
    {
        d2[0] = d2[1] = d2[2] = d2[3] = __floats2half2_rn(0.F, 0.F);
        return;
    }
    uint4 const v = *reinterpret_cast<uint4 const*>(src + part * 8);
    d2[0] = *reinterpret_cast<half2 const*>(&v.x);
    d2[1] = *reinterpret_cast<half2 const*>(&v.y);
    d2[2] = *reinterpret_cast<half2 const*>(&v.z);
    d2[3] = *reinterpret_cast<half2 const*>(&v.w);
}

__device__ __forceinline__ void stageRow(half* dst, __nv_fp8_e4m3 const* src, bool valid, int32_t part)
{
    // 16 parts of 8 values: half a 16-byte vector each
    half2* d2 = reinterpret_cast<half2*>(dst) + part * 4;
    if (!valid)
    {
        d2[0] = d2[1] = d2[2] = d2[3] = __floats2half2_rn(0.F, 0.F);
        return;
    }
    uint2 const v = *reinterpret_cast<uint2 const*>(src + part * 8);
    __nv_fp8x2_e4m3 const* f = reinterpret_cast<__nv_fp8x2_e4m3 const*>(&v);
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        d2[i] = static_cast<__half2>(f[i]);
    }
}

template <typename KvT>
__global__ void __launch_bounds__(kTileThreads) tiledAttentionKernel(Inkling2AttentionParams p,
    half const* __restrict__ qLin, half const* __restrict__ relLin, float const* __restrict__ qScale,
    float const* __restrict__ relScale, float const* __restrict__ relProjT, KvT const* __restrict__ kvPool,
    int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths, int32_t const* __restrict__ past,
    int32_t const* __restrict__ pageTable, float* __restrict__ partials, int32_t qTile, int32_t splits)
{
    extern __shared__ __align__(16) unsigned char smemRaw[];
    half* Ksm = reinterpret_cast<half*>(smemRaw);
    half* Vsm = Ksm + kKeyTile * kKPad;
    float* Ssm = reinterpret_cast<float*>(Vsm + kKeyTile * kInkling2HeadDim);
    float* Qsm = Ssm + kMaxRows * kSPad;
    float* RSsm = Qsm + kMaxRows * kInkling2HeadDim;
    float* rowM = RSsm + kMaxRows * kInkling2RelDim;
    float* rowL = rowM + kMaxRows;
    float* rowCorr = rowL + kMaxRows;
    int32_t* rowQpos = reinterpret_cast<int32_t*>(rowCorr + kMaxRows);
    int32_t* rowLo = rowQpos + kMaxRows;

    int32_t const tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    int32_t tile = blockIdx.x, seq = 0;
    for (; seq < p.batchSize; ++seq)
    {
        int32_t const n = (offsets[seq + 1] - offsets[seq] + qTile - 1) / qTile;
        if (tile < n)
        {
            break;
        }
        tile -= n;
    }
    if (seq >= p.batchSize)
    {
        return;
    }
    int32_t const slot0 = offsets[seq] + tile * qTile, slotsEnd = offsets[seq + 1];
    int32_t const group = p.numQHeads / p.numKVHeads, rows = qTile * group;
    int32_t const kvh = blockIdx.y, split = blockIdx.z;
    int64_t const rowElems = int64_t(p.numKVHeads) * kInkling2HeadDim;
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * rowElems;
    int32_t const* table = pageTable + int64_t(seq) * 2 * p.maxPagesPerSeq;

    // Row setup: one warp per row, three rows per warp at most.
    for (int32_t row = w; row < rows; row += kTileWarps)
    {
        int32_t const qi = row / group, h = kvh * group + row % group;
        int32_t const slot = slot0 + qi, local = slot - offsets[seq];
        bool const valid = slot < slotsEnd && local < lengths[seq];
        int32_t const qpos = valid ? past[seq] + local : -1;
        if (lane == 0)
        {
            rowQpos[row] = qpos;
            rowLo[row] = valid ? (p.window > 0 ? max(0, qpos - p.window + 1) : 0) : 0;
            rowM[row] = -INFINITY;
            rowL[row] = 0.F;
        }
        if (!valid)
        {
            continue;
        }
        float const logScale = p.window == 0 ? 1.F + p.logAlpha * logf(fmaxf((qpos + 1.F) / p.logNFloor, 1.F)) : 1.F;
        float q[4];
        float ss = 0.F;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            int32_t const d = lane * 4 + i;
            q[i] = __half2float(qLin[(int64_t(slot) * p.numQHeads + h) * kInkling2HeadDim + d])
                * qScale[h * kInkling2HeadDim + d];
            ss += q[i] * q[i];
        }
        ss = warpSum(ss);
        float const qFactor = fminf(p.qClip / (sqrtf(ss) + 1e-6F), 1.F) * logScale;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            Qsm[row * kInkling2HeadDim + lane * 4 + i] = q[i] * qFactor;
        }
        int32_t const j = lane & 15;
        float rs = __half2float(relLin[(int64_t(slot) * p.numQHeads + h) * kInkling2RelDim + j])
            * relScale[h * kInkling2RelDim + j];
        float const rss = halfWarpSum(rs * rs);
        rs *= fminf(p.relClip / (sqrtf(rss) + 1e-6F), 1.F) * logScale;
        if (lane < 16)
        {
            RSsm[row * kInkling2RelDim + j] = rs;
        }
    }
    __syncthreads();

    int32_t kLo = INT32_MAX, kHi = -1;
    for (int32_t row = 0; row < rows; ++row)
    {
        if (rowQpos[row] >= 0)
        {
            kLo = min(kLo, rowLo[row]);
            kHi = max(kHi, rowQpos[row]);
        }
    }
    int32_t const nTiles = kHi >= kLo ? (kHi - kLo + kKeyTile) / kKeyTile : 0;
    int32_t const perSplit = (nTiles + splits - 1) / splits;
    int32_t const tBegin = min(nTiles, split * perSplit), tEnd = min(nTiles, tBegin + perSplit);

    int32_t const rg = tid >> 6; // row group of the score and value phases (4 groups, up to 6 rows each)
    int32_t const kj = tid & 63; // key column of the score phase / value pair of the value phase
    float acc[6][2];
#pragma unroll
    for (int32_t r = 0; r < 6; ++r)
    {
        acc[r][0] = acc[r][1] = 0.F;
    }

    for (int32_t ti = tBegin; ti < tEnd; ++ti)
    {
        int32_t const keyBase = kLo + ti * kKeyTile;
        // Stage K and V of the tile (16 parts per key row, 64 key rows, two matrices).
        for (int32_t item = tid; item < kKeyTile * 16; item += kTileThreads)
        {
            int32_t const key = item >> 4, part = item & 15;
            int32_t const s = keyBase + key;
            bool valid = s <= kHi;
            int32_t kPage = 0, vPage = 0;
            if (valid)
            {
                kPage = table[s / kInkling2TokensPerPage];
                vPage = table[p.maxPagesPerSeq + s / kInkling2TokensPerPage];
                valid = kPage >= 0 && vPage >= 0;
            }
            int64_t const inPage = int64_t(s % kInkling2TokensPerPage) * rowElems + int64_t(kvh) * kInkling2HeadDim;
            stageRow(Ksm + key * kKPad, kvPool + int64_t(kPage) * pageElems + inPage, valid, part);
            stageRow(Vsm + key * kInkling2HeadDim, kvPool + int64_t(vPage) * pageElems + inPage, valid, part);
        }
        __syncthreads();

        // Scores: thread (rg, kj) owns key kj for rows rg, rg + 4, ...
        {
            float dots[6];
#pragma unroll
            for (int32_t r = 0; r < 6; ++r)
            {
                dots[r] = 0.F;
            }
            half2 const* kRow = reinterpret_cast<half2 const*>(Ksm + kj * kKPad);
#pragma unroll 4
            for (int32_t d2 = 0; d2 < kInkling2HeadDim / 2; ++d2)
            {
                float2 const k = __half22float2(kRow[d2]);
#pragma unroll
                for (int32_t r = 0; r < 6; ++r)
                {
                    int32_t const row = rg + 4 * r;
                    if (row < rows)
                    {
                        float2 const q = *reinterpret_cast<float2 const*>(&Qsm[row * kInkling2HeadDim + 2 * d2]);
                        dots[r] += q.x * k.x + q.y * k.y;
                    }
                }
            }
            int32_t const s = keyBase + kj;
            bool mapped = s <= kHi;
            if (mapped)
            {
                mapped = table[s / kInkling2TokensPerPage] >= 0 && table[p.maxPagesPerSeq + s / kInkling2TokensPerPage] >= 0;
            }
#pragma unroll
            for (int32_t r = 0; r < 6; ++r)
            {
                int32_t const row = rg + 4 * r;
                if (row >= rows)
                {
                    continue;
                }
                int32_t const qpos = rowQpos[row];
                float score = -INFINITY;
                if (mapped && qpos >= 0 && s <= qpos && s >= rowLo[row])
                {
                    int32_t const dist = qpos - s;
                    float bias = 0.F;
                    if (dist < p.relExtent)
                    {
                        float const* rp = relProjT + int64_t(dist) * kInkling2RelDim;
                        float const* rsr = RSsm + row * kInkling2RelDim;
#pragma unroll
                        for (int32_t jj = 0; jj < kInkling2RelDim; jj += 4)
                        {
                            float4 const a = *reinterpret_cast<float4 const*>(rp + jj);
                            bias += rsr[jj] * a.x + rsr[jj + 1] * a.y + rsr[jj + 2] * a.z + rsr[jj + 3] * a.w;
                        }
                    }
                    score = dots[r] * p.scaling + bias;
                }
                Ssm[row * kSPad + kj] = score;
            }
        }
        __syncthreads();

        // Online softmax: one warp per row (three rows per warp at most).
        for (int32_t row = w; row < rows; row += kTileWarps)
        {
            float v0 = Ssm[row * kSPad + lane], v1 = Ssm[row * kSPad + lane + 32];
            float tileMax = fmaxf(v0, v1);
#pragma unroll
            for (int32_t o = 16; o > 0; o >>= 1)
            {
                tileMax = fmaxf(tileMax, __shfl_xor_sync(0xffffffffu, tileMax, o));
            }
            float const mOld = rowM[row];
            float const mNew = fmaxf(mOld, tileMax);
            float const corr = mOld == -INFINITY ? 0.F : __expf(mOld - mNew);
            float const e0 = v0 == -INFINITY ? 0.F : __expf(v0 - mNew), e1 = v1 == -INFINITY ? 0.F : __expf(v1 - mNew);
            Ssm[row * kSPad + lane] = e0;
            Ssm[row * kSPad + lane + 32] = e1;
            float const sum = warpSum(e0 + e1);
            float const lOld = rowL[row];
            __syncwarp(); // every lane has read the row statistics before lane 0 replaces them
            if (lane == 0)
            {
                rowL[row] = lOld * corr + sum;
                rowM[row] = mNew;
                rowCorr[row] = corr;
            }
        }
        __syncthreads();

        // Values: thread (rg, pair kj) accumulates dims 2 * kj and 2 * kj + 1 for rows rg, rg + 4, ...
#pragma unroll
        for (int32_t r = 0; r < 6; ++r)
        {
            int32_t const row = rg + 4 * r;
            if (row < rows)
            {
                float const c = rowCorr[row];
                acc[r][0] *= c;
                acc[r][1] *= c;
            }
        }
        for (int32_t key = 0; key < kKeyTile; ++key)
        {
            float2 const v = __half22float2(reinterpret_cast<half2 const*>(Vsm + key * kInkling2HeadDim)[kj]);
#pragma unroll
            for (int32_t r = 0; r < 6; ++r)
            {
                int32_t const row = rg + 4 * r;
                if (row < rows)
                {
                    float const pr = Ssm[row * kSPad + key];
                    acc[r][0] += pr * v.x;
                    acc[r][1] += pr * v.y;
                }
            }
        }
        __syncthreads();
    }

    // Partial results of every row whose slot exists.
#pragma unroll
    for (int32_t r = 0; r < 6; ++r)
    {
        int32_t const row = rg + 4 * r;
        if (row >= rows)
        {
            continue;
        }
        int32_t const slot = slot0 + row / group;
        if (slot >= slotsEnd)
        {
            continue;
        }
        int32_t const h = kvh * group + row % group;
        float* out = partials + ((int64_t(slot) * p.numQHeads + h) * splits + split) * kPartialStride;
        out[2 + 2 * kj] = acc[r][0];
        out[3 + 2 * kj] = acc[r][1];
        if (kj == 0)
        {
            out[0] = rowM[row];
            out[1] = rowL[row];
        }
    }
}

//! One block per (token slot, query head): merges the key splits, adds the sink to the denominator, applies the gate.
__global__ void mergeKernel(Inkling2AttentionParams p, half const* __restrict__ sinkLin, half const* __restrict__ gateLin,
    int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths, int32_t const* __restrict__ past,
    float const* __restrict__ partials, int32_t splits, half* __restrict__ out)
{
    int32_t const t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int32_t seq, qpos;
    int64_t const idx = (int64_t(t) * p.numQHeads + h) * kInkling2HeadDim + d;
    if (!locateToken(t, offsets, lengths, past, p.batchSize, seq, qpos))
    {
        out[idx] = __float2half(0.F);
        return;
    }
    float const* base = partials + (int64_t(t) * p.numQHeads + h) * splits * kPartialStride;
    float const sink = __half2float(sinkLin[int64_t(t) * p.numQHeads + h]);
    float mm = sink;
    for (int32_t i = 0; i < splits; ++i)
    {
        mm = fmaxf(mm, base[i * kPartialStride]);
    }
    float den = __expf(sink - mm), o = 0.F;
    for (int32_t i = 0; i < splits; ++i)
    {
        float const m = base[i * kPartialStride];
        float const c = m == -INFINITY ? 0.F : __expf(m - mm);
        den += base[i * kPartialStride + 1] * c;
        o += base[i * kPartialStride + 2 + d] * c;
    }
    float const g = 1.F / (1.F + __expf(-__half2float(gateLin[idx])));
    out[idx] = __float2half(o / den * g);
}
} // namespace

template <typename KvT>
static void launchTyped(Inkling2AttentionParams const& params, half const* qLin, half const* relLin, half const* sinkLin,
    half const* gateLin, float const* qScale, float const* relScale, float const* relProjT, half const* kNew,
    half const* vNew, KvT* kvPool, int32_t const* queryStartOffsets, int32_t const* queryLengths,
    int32_t const* pastLengths, int32_t const* pageTable, half* out, void* workspace, size_t workspaceBytes,
    cudaStream_t stream)
{
    appendKernel<KvT><<<params.numTokens, 128, 0, stream>>>(
        params, kNew, vNew, kvPool, queryStartOffsets, queryLengths, pastLengths, pageTable);
    int32_t const group = params.numQHeads / params.numKVHeads;
    int64_t const rowCap = static_cast<int64_t>(workspaceBytes / sizeof(float) / kPartialStride);
    if (workspace == nullptr || group > kMaxRows || rowCap < static_cast<int64_t>(params.numTokens) * params.numQHeads)
    {
        attentionKernel<KvT><<<dim3(params.numTokens, params.numQHeads), kThreads, 0, stream>>>(params, qLin, relLin,
            sinkLin, gateLin, qScale, relScale, relProjT, kvPool, queryStartOffsets, queryLengths, pastLengths,
            pageTable, out);
        return;
    }
    static bool const configured = []()
    {
        cudaFuncSetAttribute(tiledAttentionKernel<half>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(tiledSmemBytes()));
        cudaFuncSetAttribute(tiledAttentionKernel<__nv_fp8_e4m3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(tiledSmemBytes()));
        return true;
    }();
    static_cast<void>(configured);
    int32_t const qTile = std::max(1, kMaxRows / group);
    int32_t const tiles = (params.numTokens + qTile - 1) / qTile + params.batchSize;
    static int32_t const numSms = []()
    {
        int32_t sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        return std::max(sms, 1);
    }();
    // Fill the device: about four blocks per SM, never more splits than the page table can hold key tiles for,
    // and never more partials than the workspace holds.
    int64_t splits = (4LL * numSms + int64_t(tiles) * params.numKVHeads - 1) / (int64_t(tiles) * params.numKVHeads);
    splits = std::min<int64_t>(splits, std::max<int64_t>(1, int64_t(params.maxPagesPerSeq) * kInkling2TokensPerPage / kKeyTile));
    splits = std::min<int64_t>(splits, 32);
    splits = std::max<int64_t>(1, std::min<int64_t>(splits, rowCap / (int64_t(params.numTokens) * params.numQHeads)));
    tiledAttentionKernel<KvT><<<dim3(tiles, params.numKVHeads, static_cast<unsigned>(splits)), kTileThreads,
        tiledSmemBytes(), stream>>>(params, qLin, relLin, qScale, relScale, relProjT, kvPool, queryStartOffsets,
        queryLengths, pastLengths, pageTable, static_cast<float*>(workspace), qTile, static_cast<int32_t>(splits));
    mergeKernel<<<dim3(params.numTokens, params.numQHeads), kInkling2HeadDim, 0, stream>>>(params, sinkLin, gateLin,
        queryStartOffsets, queryLengths, pastLengths, static_cast<float const*>(workspace),
        static_cast<int32_t>(splits), out);
}

size_t inkling2AttentionWorkspaceBytes(int64_t maxTokens, int32_t numQHeads)
{
    int64_t const rows = std::max<int64_t>(maxTokens * numQHeads, 1024);
    return static_cast<size_t>(rows) * kPartialStride * sizeof(float);
}

void launchInkling2Attention(Inkling2AttentionParams const& params, half const* qLin, half const* relLin,
    half const* sinkLin, half const* gateLin, float const* qScale, float const* relScale, float const* relProjT,
    half const* kNew, half const* vNew, void* kvPool, int32_t const* queryStartOffsets, int32_t const* queryLengths,
    int32_t const* pastLengths, int32_t const* pageTable, half* out, void* workspace, size_t workspaceBytes,
    cudaStream_t stream)
{
    if (params.numTokens <= 0)
    {
        return;
    }
    if (params.kvFp8 != 0)
    {
        launchTyped(params, qLin, relLin, sinkLin, gateLin, qScale, relScale, relProjT, kNew, vNew,
            static_cast<__nv_fp8_e4m3*>(kvPool), queryStartOffsets, queryLengths, pastLengths, pageTable, out,
            workspace, workspaceBytes, stream);
    }
    else
    {
        launchTyped(params, qLin, relLin, sinkLin, gateLin, qScale, relScale, relProjT, kNew, vNew,
            static_cast<half*>(kvPool), queryStartOffsets, queryLengths, pastLengths, pageTable, out, workspace,
            workspaceBytes, stream);
    }
}

} // namespace kernel
} // namespace trt_edgellm
