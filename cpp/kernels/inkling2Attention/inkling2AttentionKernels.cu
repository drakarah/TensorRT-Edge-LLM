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
#include <cstdlib>
#include <cuda_fp8.h>
#include <type_traits>
#include <mma.h>
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

//! KV pool element access: FP16, FP8 E4M3 (unit scale, saturating), or INT8 with one FP16 scale per (token, KV head) row stored behind the 128 values: a row is
//! 132 bytes = 128 INT8 values, the FP16 scale and 2 pad bytes (4-byte aligned), so every row address is `row * rowStride` instead of `row * 128`.
template <typename KvT>
struct KvTraits
{
    static constexpr int32_t rowStride = kInkling2HeadDim;
};

template <>
struct KvTraits<int8_t>
{
    static constexpr int32_t rowStride = kInkling2HeadDim + 4;
};

//! Four consecutive values of a row starting at element `off` as floats.
__device__ __forceinline__ void loadKv4(half const* row, int32_t off, float (&f)[4])
{
    uint2 const r = *reinterpret_cast<uint2 const*>(row + off);
    half2 const* h = reinterpret_cast<half2 const*>(&r);
    float2 const a = __half22float2(h[0]), b = __half22float2(h[1]);
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
}

__device__ __forceinline__ void loadKv4(__nv_fp8_e4m3 const* row, int32_t off, float (&f)[4])
{
    uint32_t const r = *reinterpret_cast<uint32_t const*>(row + off);
    __nv_fp8x2_e4m3 const* h = reinterpret_cast<__nv_fp8x2_e4m3 const*>(&r);
    float2 const a = static_cast<float2>(h[0]), b = static_cast<float2>(h[1]);
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
}

__device__ __forceinline__ void loadKv4(int8_t const* row, int32_t off, float (&f)[4])
{
    float const scale = __half2float(*reinterpret_cast<half const*>(row + kInkling2HeadDim));
    uint32_t const r = *reinterpret_cast<uint32_t const*>(row + off);
    f[0] = static_cast<float>(static_cast<int8_t>(r & 0xFFU)) * scale;
    f[1] = static_cast<float>(static_cast<int8_t>((r >> 8) & 0xFFU)) * scale;
    f[2] = static_cast<float>(static_cast<int8_t>((r >> 16) & 0xFFU)) * scale;
    f[3] = static_cast<float>(static_cast<int8_t>(r >> 24)) * scale;
}

__device__ __forceinline__ void storeKv(half* p, half v)
{
    *p = v;
}

__device__ __forceinline__ void storeKv(__nv_fp8_e4m3* p, half v)
{
    *p = __nv_fp8_e4m3(__half2float(v));
}

//! INT8 rows are written by appendInt8Kernel (scale per row); this overload only keeps the generic template compiling.
__device__ __forceinline__ void storeKv(int8_t* p, half v)
{
    *p = static_cast<int8_t>(__half2float(v));
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
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * int64_t(p.numKVHeads) * KvTraits<KvT>::rowStride;
    int32_t const* table = pageTable + int64_t(seq) * 2 * p.maxPagesPerSeq;
    // A negative page id is an unmapped logical page (bounded window storage, graph-capture warmup): nothing to write.
    int32_t const kPage = table[pos / kInkling2TokensPerPage], vPage = table[p.maxPagesPerSeq + pos / kInkling2TokensPerPage];
    if (kPage < 0 || vPage < 0)
    {
        return;
    }
    int64_t const kDst = int64_t(kPage) * pageElems + (pos % kInkling2TokensPerPage) * int64_t(p.numKVHeads) * KvTraits<KvT>::rowStride;
    int64_t const vDst = int64_t(vPage) * pageElems + (pos % kInkling2TokensPerPage) * int64_t(p.numKVHeads) * KvTraits<KvT>::rowStride;
    for (int64_t i = threadIdx.x; i < rowElems; i += blockDim.x)
    {
        // element i of the new row = head (i / 128), element (i % 128); the pool row of that head starts at head * rowStride
        int64_t const head = i / kInkling2HeadDim, d = i % kInkling2HeadDim;
        storeKv(kvPool + kDst + head * KvTraits<KvT>::rowStride + d, kNew[int64_t(t) * rowElems + i]);
        storeKv(kvPool + vDst + head * KvTraits<KvT>::rowStride + d, vNew[int64_t(t) * rowElems + i]);
    }
}

//! INT8 append: one warp per (plane, KV head) row of the new token: absmax over the 128 values, FP16 scale = absmax / 127 (1 for an all-zero row), values rounded
//! to nearest with that scale, scale and two zero pad bytes behind the values. 8 warps per block cover K and V of up to 8 heads per pass.
__global__ void appendInt8Kernel(Inkling2AttentionParams p, half const* __restrict__ kNew, half const* __restrict__ vNew, int8_t* __restrict__ kvPool,
    int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths, int32_t const* __restrict__ past, int32_t const* __restrict__ pageTable)
{
    int32_t const t = blockIdx.x;
    int32_t seq, pos;
    if (!locateToken(t, offsets, lengths, past, p.batchSize, seq, pos))
    {
        return;
    }
    constexpr int32_t kStride = KvTraits<int8_t>::rowStride;
    int64_t const rowElems = int64_t(p.numKVHeads) * kInkling2HeadDim;
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * int64_t(p.numKVHeads) * kStride;
    int32_t const* table = pageTable + int64_t(seq) * 2 * p.maxPagesPerSeq;
    int32_t const kPage = table[pos / kInkling2TokensPerPage], vPage = table[p.maxPagesPerSeq + pos / kInkling2TokensPerPage];
    if (kPage < 0 || vPage < 0)
    {
        return;
    }
    int32_t const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int64_t const inPageRow = (pos % kInkling2TokensPerPage) * int64_t(p.numKVHeads) * kStride;
    for (int32_t job = warp; job < 2 * p.numKVHeads; job += blockDim.x >> 5)
    {
        int32_t const plane = job / p.numKVHeads, head = job % p.numKVHeads;
        half const* src = (plane == 0 ? kNew : vNew) + int64_t(t) * rowElems + int64_t(head) * kInkling2HeadDim + lane * 4;
        int8_t* dst = kvPool + int64_t(plane == 0 ? kPage : vPage) * pageElems + inPageRow + int64_t(head) * kStride;
        float v[4];
        uint2 const raw = *reinterpret_cast<uint2 const*>(src);
        half2 const* h = reinterpret_cast<half2 const*>(&raw);
        float2 const a = __half22float2(h[0]), b = __half22float2(h[1]);
        v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y;
        float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
        for (int32_t off = 16; off > 0; off >>= 1)
        {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, off));
        }
        half const scaleH = __float2half(amax > 0.F ? amax / 127.F : 1.F);
        float const inv = 1.F / __half2float(scaleH);
        uint32_t packed = 0;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            int32_t const q = max(-127, min(127, __float2int_rn(v[i] * inv)));
            packed |= static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q))) << (8 * i);
        }
        *reinterpret_cast<uint32_t*>(dst + lane * 4) = packed;
        if (lane == 0)
        {
            *reinterpret_cast<uint32_t*>(dst + kInkling2HeadDim) = static_cast<uint32_t>(__half_as_ushort(scaleH)); // scale, then two zero pad bytes
        }
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
    int64_t const rowElems = int64_t(p.numKVHeads) * KvTraits<KvT>::rowStride;
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * rowElems;
    int32_t const* table = pageTable + int64_t(seq) * 2 * p.maxPagesPerSeq;
    float const logScale = p.window == 0 ? 1.F + p.logAlpha * logf(fmaxf((qpos + 1.F) / p.logNFloor, 1.F)) : 1.F;

    float q[4];
    float ss = 0.F;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        int32_t const d = lane * 4 + i;
        q[i] = __half2float(qLin[int64_t(t) * p.qLd + h * kInkling2HeadDim + d])
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
    float rs = __half2float(relLin[int64_t(t) * p.relLd + h * kInkling2RelDim + j])
        * relScale[h * kInkling2RelDim + j];
    float const rss = halfWarpSum(rs * rs);
    rs *= fminf(p.relClip / (sqrtf(rss) + 1e-6F), 1.F) * logScale;

    int32_t const lo = p.window > 0 ? max(0, qpos - p.window + 1) : 0;
    float m = -INFINITY, l = 0.F, acc[4] = {0.F, 0.F, 0.F, 0.F};
    for (int32_t s = lo + w; s <= qpos; s += kWarps)
    {
        int64_t const inPage = int64_t(s % kInkling2TokensPerPage) * rowElems + int64_t(kvh) * KvTraits<KvT>::rowStride;
        int32_t const kPage = table[s / kInkling2TokensPerPage], vPage = table[p.maxPagesPerSeq + s / kInkling2TokensPerPage];
        if (kPage < 0 || vPage < 0)
        {
            continue; // unmapped logical page: no key (never inside a live window)
        }
        float kf[4];
        loadKv4(kvPool + int64_t(kPage) * pageElems + inPage, lane * 4, kf);
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
        loadKv4(kvPool + int64_t(vPage) * pageElems + inPage, lane * 4, vf);
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
        float const sink = __half2float(sinkLin[int64_t(t) * p.sinkLd + h]);
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
            float const g = 1.F / (1.F + __expf(-__half2float(gateLin[int64_t(t) * p.gateLd + h * kInkling2HeadDim + d])));
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

__device__ __forceinline__ void stageRow(half* dst, int8_t const* src, bool valid, int32_t part)
{
    // 16 parts of 8 values; the FP16 scale of the row sits behind its 128 values
    half2* d2 = reinterpret_cast<half2*>(dst) + part * 4;
    if (!valid)
    {
        d2[0] = d2[1] = d2[2] = d2[3] = __floats2half2_rn(0.F, 0.F);
        return;
    }
    float const scale = __half2float(*reinterpret_cast<half const*>(src + kInkling2HeadDim));
    // rows are 132 bytes (4-byte aligned, not 8): two 32-bit loads
    uint32_t const w[2] = {*reinterpret_cast<uint32_t const*>(src + part * 8), *reinterpret_cast<uint32_t const*>(src + part * 8 + 4)};
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        uint32_t const x = w[i >> 1] >> (16 * (i & 1));
        float const lo = static_cast<float>(static_cast<int8_t>(x & 0xFFU)) * scale;
        float const hi = static_cast<float>(static_cast<int8_t>((x >> 8) & 0xFFU)) * scale;
        d2[i] = __floats2half2_rn(lo, hi);
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
    int64_t const rowElems = int64_t(p.numKVHeads) * KvTraits<KvT>::rowStride;
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
            q[i] = __half2float(qLin[int64_t(slot) * p.qLd + h * kInkling2HeadDim + d])
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
        float rs = __half2float(relLin[int64_t(slot) * p.relLd + h * kInkling2RelDim + j])
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
            int64_t const inPage = int64_t(s % kInkling2TokensPerPage) * rowElems + int64_t(kvh) * KvTraits<KvT>::rowStride;
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


// ---- Decode-shaped attention (few query slots, GQA group up to 4): a block of 128 threads serves one query slot, one KV head and one key split with
// small shared tiles (32 keys), so several blocks share an SM and their K/V loads overlap. Same partial-result contract as tiledAttentionKernel. ----
constexpr int32_t kDecKeys = 32;
constexpr int32_t kDecThreads = 128;
constexpr int32_t kDecRows = 4;

constexpr size_t decodeSmemBytes()
{
    return sizeof(half) * (kDecKeys * kKPad + kDecKeys * kInkling2HeadDim)
        + sizeof(float) * (kDecRows * (kDecKeys + 1) + kDecRows * kInkling2HeadDim + kDecRows * kInkling2RelDim + 3 * kDecRows)
        + sizeof(int32_t) * 2 * kDecRows;
}

template <typename KvT>
__global__ void __launch_bounds__(kDecThreads) decodeAttentionKernel(Inkling2AttentionParams p, half const* __restrict__ qLin,
    half const* __restrict__ relLin, float const* __restrict__ qScale, float const* __restrict__ relScale,
    float const* __restrict__ relProjT, KvT const* __restrict__ kvPool, int32_t const* __restrict__ offsets,
    int32_t const* __restrict__ lengths, int32_t const* __restrict__ past, int32_t const* __restrict__ pageTable,
    float* __restrict__ partials, int32_t splits)
{
    extern __shared__ __align__(16) unsigned char smemRaw[];
    half* Ksm = reinterpret_cast<half*>(smemRaw);
    half* Vsm = Ksm + kDecKeys * kKPad;
    float* Ssm = reinterpret_cast<float*>(Vsm + kDecKeys * kInkling2HeadDim);
    float* Qsm = Ssm + kDecRows * (kDecKeys + 1);
    float* RSsm = Qsm + kDecRows * kInkling2HeadDim;
    float* rowM = RSsm + kDecRows * kInkling2RelDim;
    float* rowL = rowM + kDecRows;
    float* rowCorr = rowL + kDecRows;
    int32_t* rowQpos = reinterpret_cast<int32_t*>(rowCorr + kDecRows);
    int32_t* rowLo = rowQpos + kDecRows;

    int32_t const tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    int32_t const slot = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z;
    int32_t seq = 0, qpos = 0;
    bool const valid = locateToken(slot, offsets, lengths, past, p.batchSize, seq, qpos);
    int32_t const group = p.numQHeads / p.numKVHeads;
    int64_t const rowElems = int64_t(p.numKVHeads) * KvTraits<KvT>::rowStride;
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * rowElems;
    int32_t const* table = pageTable + int64_t(seq) * 2 * p.maxPagesPerSeq;

    // Row setup: warp w prepares the row of head kvh * group + w (rows >= group stay unused).
    if (w < group)
    {
        int32_t const h = kvh * group + w;
        if (lane == 0)
        {
            rowQpos[w] = valid ? qpos : -1;
            rowLo[w] = valid ? (p.window > 0 ? max(0, qpos - p.window + 1) : 0) : 0;
            rowM[w] = -INFINITY;
            rowL[w] = 0.F;
        }
        if (valid)
        {
            float const logScale = p.window == 0 ? 1.F + p.logAlpha * logf(fmaxf((qpos + 1.F) / p.logNFloor, 1.F)) : 1.F;
            float q[4];
            float ss = 0.F;
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
                int32_t const d = lane * 4 + i;
                q[i] = __half2float(qLin[int64_t(slot) * p.qLd + h * kInkling2HeadDim + d]) * qScale[h * kInkling2HeadDim + d];
                ss += q[i] * q[i];
            }
            ss = warpSum(ss);
            float const qFactor = fminf(p.qClip / (sqrtf(ss) + 1e-6F), 1.F) * logScale;
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
                Qsm[w * kInkling2HeadDim + lane * 4 + i] = q[i] * qFactor;
            }
            int32_t const j = lane & 15;
            float rs = __half2float(relLin[int64_t(slot) * p.relLd + h * kInkling2RelDim + j]) * relScale[h * kInkling2RelDim + j];
            float const rss = halfWarpSum(rs * rs);
            rs *= fminf(p.relClip / (sqrtf(rss) + 1e-6F), 1.F) * logScale;
            if (lane < 16)
            {
                RSsm[w * kInkling2RelDim + j] = rs;
            }
        }
    }
    else if (lane == 0 && w < kDecRows)
    {
        rowQpos[w] = -1;
        rowLo[w] = 0;
        rowM[w] = -INFINITY;
        rowL[w] = 0.F;
    }
    __syncthreads();

    int32_t kLo = INT32_MAX, kHi = -1;
    for (int32_t row = 0; row < group; ++row)
    {
        if (rowQpos[row] >= 0)
        {
            kLo = min(kLo, rowLo[row]);
            kHi = max(kHi, rowQpos[row]);
        }
    }
    int32_t const nTiles = kHi >= kLo ? (kHi - kLo + kDecKeys) / kDecKeys : 0;
    int32_t const perSplit = (nTiles + splits - 1) / splits;
    int32_t const tBegin = min(nTiles, split * perSplit), tEnd = min(nTiles, tBegin + perSplit);

    int32_t const sKey = tid >> 2, sRow = tid & 3; // score phase: 32 keys x 4 rows
    int32_t const vPair = tid & 63, vGroup = tid >> 6; // value phase: dims 2 * vPair, rows vGroup and vGroup + 2
    float acc[2][2] = {{0.F, 0.F}, {0.F, 0.F}};

    for (int32_t ti = tBegin; ti < tEnd; ++ti)
    {
        int32_t const keyBase = kLo + ti * kDecKeys;
        for (int32_t item = tid; item < kDecKeys * 16; item += kDecThreads)
        {
            int32_t const key = item >> 4, part = item & 15;
            int32_t const s = keyBase + key;
            bool ok = s <= kHi;
            int32_t kPage = 0, vPage = 0;
            if (ok)
            {
                kPage = table[s / kInkling2TokensPerPage];
                vPage = table[p.maxPagesPerSeq + s / kInkling2TokensPerPage];
                ok = kPage >= 0 && vPage >= 0;
            }
            int64_t const inPage = int64_t(s % kInkling2TokensPerPage) * rowElems + int64_t(kvh) * KvTraits<KvT>::rowStride;
            stageRow(Ksm + key * kKPad, kvPool + int64_t(kPage) * pageElems + inPage, ok, part);
            stageRow(Vsm + key * kInkling2HeadDim, kvPool + int64_t(vPage) * pageElems + inPage, ok, part);
        }
        __syncthreads();
        {
            float dot = 0.F;
            half2 const* kRow = reinterpret_cast<half2 const*>(Ksm + sKey * kKPad);
            float const* qRow = Qsm + sRow * kInkling2HeadDim;
#pragma unroll 8
            for (int32_t d2 = 0; d2 < kInkling2HeadDim / 2; ++d2)
            {
                float2 const k = __half22float2(kRow[d2]);
                float2 const q = *reinterpret_cast<float2 const*>(qRow + 2 * d2);
                dot += q.x * k.x + q.y * k.y;
            }
            int32_t const s = keyBase + sKey;
            bool mapped = s <= kHi;
            if (mapped)
            {
                mapped = table[s / kInkling2TokensPerPage] >= 0 && table[p.maxPagesPerSeq + s / kInkling2TokensPerPage] >= 0;
            }
            float score = -INFINITY;
            int32_t const qp = sRow < group ? rowQpos[sRow] : -1;
            if (mapped && qp >= 0 && s <= qp && s >= rowLo[sRow])
            {
                int32_t const dist = qp - s;
                float bias = 0.F;
                if (dist < p.relExtent)
                {
                    float const* rp = relProjT + int64_t(dist) * kInkling2RelDim;
                    float const* rsr = RSsm + sRow * kInkling2RelDim;
#pragma unroll
                    for (int32_t jj = 0; jj < kInkling2RelDim; jj += 4)
                    {
                        float4 const a = *reinterpret_cast<float4 const*>(rp + jj);
                        bias += rsr[jj] * a.x + rsr[jj + 1] * a.y + rsr[jj + 2] * a.z + rsr[jj + 3] * a.w;
                    }
                }
                score = dot * p.scaling + bias;
            }
            Ssm[sRow * (kDecKeys + 1) + sKey] = score;
        }
        __syncthreads();
        if (w < kDecRows)
        {
            float const v0 = Ssm[w * (kDecKeys + 1) + lane];
            float tileMax = v0;
#pragma unroll
            for (int32_t o = 16; o > 0; o >>= 1)
            {
                tileMax = fmaxf(tileMax, __shfl_xor_sync(0xffffffffu, tileMax, o));
            }
            float const mOld = rowM[w], lOld = rowL[w];
            float const mNew = fmaxf(mOld, tileMax);
            float const corr = mOld == -INFINITY ? 0.F : __expf(mOld - mNew);
            float const e0 = v0 == -INFINITY ? 0.F : __expf(v0 - mNew);
            Ssm[w * (kDecKeys + 1) + lane] = e0;
            float const sum = warpSum(e0);
            __syncwarp();
            if (lane == 0)
            {
                rowL[w] = lOld * corr + sum;
                rowM[w] = mNew;
                rowCorr[w] = corr;
            }
        }
        __syncthreads();
#pragma unroll
        for (int32_t r = 0; r < 2; ++r)
        {
            float const c = rowCorr[vGroup + 2 * r];
            acc[r][0] *= c;
            acc[r][1] *= c;
        }
        for (int32_t key = 0; key < kDecKeys; ++key)
        {
            float2 const v = __half22float2(reinterpret_cast<half2 const*>(Vsm + key * kInkling2HeadDim)[vPair]);
#pragma unroll
            for (int32_t r = 0; r < 2; ++r)
            {
                float const pr = Ssm[(vGroup + 2 * r) * (kDecKeys + 1) + key];
                acc[r][0] += pr * v.x;
                acc[r][1] += pr * v.y;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int32_t r = 0; r < 2; ++r)
    {
        int32_t const row = vGroup + 2 * r;
        if (row >= group)
        {
            continue;
        }
        int32_t const h = kvh * group + row;
        float* out = partials + ((int64_t(slot) * p.numQHeads + h) * splits + split) * kPartialStride;
        out[2 + 2 * vPair] = acc[r][0];
        out[3 + 2 * vPair] = acc[r][1];
        if (vPair == 0)
        {
            out[0] = rowM[row];
            out[1] = rowL[row];
        }
    }
}


// ---- Tensor-core version of the tiled kernel for prefill-sized tiles: same structure, the score and value products run as 16x16x16 WMMA
// (half inputs, fp32 accumulation). Q and the probabilities are rounded to half for the products; rows are padded to 32 for the fragments. ----
constexpr int32_t kMmaRows = 32; // fragment rows (the 24 query rows padded)
constexpr int32_t kKLd = kInkling2HeadDim + 8; // K tile and Q leading dimension (multiple of 8 halves)
constexpr int32_t kPLd = kKeyTile + 8;
constexpr int32_t kSLd = kKeyTile + 4;
// relative-position bias of a key tile as one WMMA product: bias[row][dist - dmin] = rs[row] . relProj[dist] for the 80 distances (16-wide tiles) a tile of 64 keys and up to 10 query positions spans
constexpr int32_t kRsLd = 24;   // leading dimension of the half copy of the row vectors and of the half relative projections (16 used)
constexpr int32_t kRelN = 80;   // distances per tile (at least kKeyTile + qTile - 1 = 71)
constexpr int32_t kBiasLd = 84; // leading dimension of the FP32 bias matrix (a multiple of 4)

constexpr size_t mmaSmemBytes()
{
    return sizeof(half) * (kKeyTile * kKLd + kKeyTile * kInkling2HeadDim + kMmaRows * kKLd + kMmaRows * kPLd + kMmaRows * kRsLd + kRelN * kRsLd)
        + sizeof(float) * (kMmaRows * kBiasLd + kMmaRows * kSLd + kMmaRows * kInkling2HeadDim + kMaxRows * kInkling2HeadDim + kMaxRows * kInkling2RelDim + 3 * kMaxRows)
        + sizeof(int32_t) * 2 * kMaxRows;
}

template <typename KvT>
__global__ void __launch_bounds__(kTileThreads) tiledAttentionMmaKernel(Inkling2AttentionParams p, half const* __restrict__ qLin,
    half const* __restrict__ relLin, float const* __restrict__ qScale, float const* __restrict__ relScale,
    float const* __restrict__ relProjT, KvT const* __restrict__ kvPool, int32_t const* __restrict__ offsets,
    int32_t const* __restrict__ lengths, int32_t const* __restrict__ past, int32_t const* __restrict__ pageTable,
    float* __restrict__ partials, int32_t qTile, int32_t splits)
{
    namespace wmma = nvcuda::wmma;
    extern __shared__ __align__(16) unsigned char smemRaw[];
    half* Ksm = reinterpret_cast<half*>(smemRaw);
    half* Vsm = Ksm + kKeyTile * kKLd;
    half* Qh = Vsm + kKeyTile * kInkling2HeadDim;
    half* Ph = Qh + kMmaRows * kKLd;
    half* RSh = Ph + kMmaRows * kPLd;
    half* RelB = RSh + kMmaRows * kRsLd;
    float* BiasS = reinterpret_cast<float*>(RelB + kRelN * kRsLd);
    float* Ssm = BiasS + kMmaRows * kBiasLd;
    float* Tmp = Ssm + kMmaRows * kSLd;
    float* Oacc = Tmp + kMmaRows * kInkling2HeadDim;
    float* RSsm = Oacc + kMaxRows * kInkling2HeadDim;
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
    int64_t const rowElems = int64_t(p.numKVHeads) * KvTraits<KvT>::rowStride;
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * rowElems;
    int32_t const* table = pageTable + int64_t(seq) * 2 * p.maxPagesPerSeq;

    // zero the padded rows of Q and P, and the running output
    for (int32_t i = tid; i < kMmaRows * kKLd; i += kTileThreads)
    {
        Qh[i] = __float2half(0.F);
    }
    for (int32_t i = tid; i < kMmaRows * kPLd; i += kTileThreads)
    {
        Ph[i] = __float2half(0.F);
    }
    for (int32_t i = tid; i < kMaxRows * kInkling2HeadDim; i += kTileThreads)
    {
        Oacc[i] = 0.F;
    }
    __syncthreads();

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
            q[i] = __half2float(qLin[int64_t(slot) * p.qLd + h * kInkling2HeadDim + d]) * qScale[h * kInkling2HeadDim + d];
            ss += q[i] * q[i];
        }
        ss = warpSum(ss);
        float const qFactor = fminf(p.qClip / (sqrtf(ss) + 1e-6F), 1.F) * logScale;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            Qh[row * kKLd + lane * 4 + i] = __float2half(q[i] * qFactor);
        }
        int32_t const j = lane & 15;
        float rs = __half2float(relLin[int64_t(slot) * p.relLd + h * kInkling2RelDim + j]) * relScale[h * kInkling2RelDim + j];
        float const rss = halfWarpSum(rs * rs);
        rs *= fminf(p.relClip / (sqrtf(rss) + 1e-6F), 1.F) * logScale;
        if (lane < 16)
        {
            RSsm[row * kInkling2RelDim + j] = rs;
        }
    }
    __syncthreads();

    int32_t kLo = INT32_MAX, kHi = -1, minQ = INT32_MAX;
    for (int32_t row = 0; row < rows; ++row)
    {
        if (rowQpos[row] >= 0)
        {
            kLo = min(kLo, rowLo[row]);
            kHi = max(kHi, rowQpos[row]);
            minQ = min(minQ, rowQpos[row]);
        }
    }
    // half copy of the (clipped, scaled) relative row vectors for the bias product; padded rows and columns are zero
    for (int32_t i = tid; i < kMmaRows * kRsLd; i += kTileThreads)
    {
        int32_t const r = i / kRsLd, j = i % kRsLd;
        RSh[i] = (r < rows && j < kInkling2RelDim) ? __float2half(RSsm[r * kInkling2RelDim + j]) : __float2half(0.F);
    }
    __syncthreads();
    int32_t const nTiles = kHi >= kLo ? (kHi - kLo + kKeyTile) / kKeyTile : 0;
    int32_t const perSplit = (nTiles + splits - 1) / splits;
    int32_t const tBegin = min(nTiles, split * perSplit), tEnd = min(nTiles, tBegin + perSplit);

    for (int32_t ti = tBegin; ti < tEnd; ++ti)
    {
        int32_t const keyBase = kLo + ti * kKeyTile;
        int32_t const dmin = minQ - (keyBase + kKeyTile - 1);
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
            int64_t const inPage = int64_t(s % kInkling2TokensPerPage) * rowElems + int64_t(kvh) * KvTraits<KvT>::rowStride;
            stageRow(Ksm + key * kKLd, kvPool + int64_t(kPage) * pageElems + inPage, valid, part);
            stageRow(Vsm + key * kInkling2HeadDim, kvPool + int64_t(vPage) * pageElems + inPage, valid, part);
        }
        __syncthreads();

        // S = Q K^T: warp w owns the 16x16 tile (w / 4, w % 4)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.F);
            int32_t const mi = w >> 2, ni = w & 3;
#pragma unroll
            for (int32_t k = 0; k < kInkling2HeadDim / 16; ++k)
            {
                wmma::load_matrix_sync(a, Qh + mi * 16 * kKLd + k * 16, kKLd);
                wmma::load_matrix_sync(b, Ksm + ni * 16 * kKLd + k * 16, kKLd);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(Ssm + mi * 16 * kSLd + ni * 16, c, kSLd, wmma::mem_row_major);
        }
        __syncthreads();

        // relative-position bias of this tile: bias [row][dist - dmin] = rs [row] . relProj [dist] as 10 WMMA tiles (2 row tiles x 5 distance tiles)
        for (int32_t i = tid; i < kRelN * kInkling2RelDim; i += kTileThreads)
        {
            int32_t const n = i >> 4, jj = i & 15;
            int32_t const dist = dmin + n;
            RelB[n * kRsLd + jj] = __float2half((dist >= 0 && dist < p.relExtent) ? relProjT[int64_t(dist) * kInkling2RelDim + jj] : 0.F);
        }
        __syncthreads();
        for (int32_t tIdx = w; tIdx < 10; tIdx += kTileWarps)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.F);
            int32_t const mi = tIdx / 5, ni = tIdx % 5;
            wmma::load_matrix_sync(a, RSh + mi * 16 * kRsLd, kRsLd);
            wmma::load_matrix_sync(b, RelB + ni * 16 * kRsLd, kRsLd);
            wmma::mma_sync(c, a, b, c);
            wmma::store_matrix_sync(BiasS + mi * 16 * kBiasLd + ni * 16, c, kBiasLd, wmma::mem_row_major);
        }
        __syncthreads();

        // scale, relative bias and masks
        for (int32_t e = tid; e < rows * kKeyTile; e += kTileThreads)
        {
            int32_t const row = e / kKeyTile, kj = e % kKeyTile;
            int32_t const s = keyBase + kj, qpos = rowQpos[row];
            float score = -INFINITY;
            bool mapped = s <= kHi;
            if (mapped)
            {
                mapped = table[s / kInkling2TokensPerPage] >= 0 && table[p.maxPagesPerSeq + s / kInkling2TokensPerPage] >= 0;
            }
            if (mapped && qpos >= 0 && s <= qpos && s >= rowLo[row])
            {
                int32_t const dist = qpos - s;
                float const bias = dist < p.relExtent ? BiasS[row * kBiasLd + (dist - dmin)] : 0.F;
                score = Ssm[row * kSLd + kj] * p.scaling + bias;
            }
            Ssm[row * kSLd + kj] = score;
        }
        __syncthreads();

        // online softmax; probabilities go to Ph as half
        for (int32_t row = w; row < rows; row += kTileWarps)
        {
            float v0 = Ssm[row * kSLd + lane], v1 = Ssm[row * kSLd + lane + 32];
            float tileMax = fmaxf(v0, v1);
#pragma unroll
            for (int32_t o = 16; o > 0; o >>= 1)
            {
                tileMax = fmaxf(tileMax, __shfl_xor_sync(0xffffffffu, tileMax, o));
            }
            float const mOld = rowM[row], lOld = rowL[row];
            float const mNew = fmaxf(mOld, tileMax);
            float const corr = mOld == -INFINITY ? 0.F : __expf(mOld - mNew);
            float const e0 = v0 == -INFINITY ? 0.F : __expf(v0 - mNew), e1 = v1 == -INFINITY ? 0.F : __expf(v1 - mNew);
            Ph[row * kPLd + lane] = __float2half(e0);
            Ph[row * kPLd + lane + 32] = __float2half(e1);
            float const sum = warpSum(e0 + e1);
            __syncwarp();
            if (lane == 0)
            {
                rowL[row] = lOld * corr + sum;
                rowM[row] = mNew;
                rowCorr[row] = corr;
            }
        }
        __syncthreads();

        // Tmp = P V: warp w owns two of the 2 x 8 output tiles
        {
#pragma unroll
            for (int32_t t2 = 0; t2 < 2; ++t2)
            {
                int32_t const tileIdx = w * 2 + t2, mi = tileIdx >> 3, ni = tileIdx & 7;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.F);
#pragma unroll
                for (int32_t k = 0; k < kKeyTile / 16; ++k)
                {
                    wmma::load_matrix_sync(a, Ph + mi * 16 * kPLd + k * 16, kPLd);
                    wmma::load_matrix_sync(b, Vsm + k * 16 * kInkling2HeadDim + ni * 16, kInkling2HeadDim);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(Tmp + mi * 16 * kInkling2HeadDim + ni * 16, c, kInkling2HeadDim, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int32_t e = tid; e < rows * kInkling2HeadDim; e += kTileThreads)
        {
            int32_t const row = e / kInkling2HeadDim;
            Oacc[e] = Oacc[e] * rowCorr[row] + Tmp[e];
        }
        __syncthreads();
    }

    for (int32_t e = tid; e < rows * kInkling2HeadDim; e += kTileThreads)
    {
        int32_t const row = e / kInkling2HeadDim, d = e % kInkling2HeadDim;
        int32_t const slot = slot0 + row / group;
        if (slot >= slotsEnd)
        {
            continue;
        }
        int32_t const h = kvh * group + row % group;
        float* out = partials + ((int64_t(slot) * p.numQHeads + h) * splits + split) * kPartialStride;
        out[2 + d] = Oacc[e];
        if (d == 0)
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
    float const sink = __half2float(sinkLin[int64_t(t) * p.sinkLd + h]);
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
    float const g = 1.F / (1.F + __expf(-__half2float(gateLin[int64_t(t) * p.gateLd + h * kInkling2HeadDim + d])));
    out[idx] = __float2half(o / den * g);
}
} // namespace


// ---- Tensor-core decode kernel (FP16 and INT8 KV): S = Q K^T and O = P V on mma.m16n8k16 with the GQA rows padded to 16, warp-private online softmax, one block per
// (token slot, KV head, key split). Every warp streams its own 16 keys of each 64-key tile through a private double-buffered shared-memory slice (cp.async for FP16,
// register-staged dequantisation for INT8), so the main loop has no block barriers. INT8 rows are dequantised to FP16 without their scale; the K scale multiplies the
// score columns and the V scale is folded into P (fp32 -> fp16), which is the same product with one rounding fewer. The four warps are merged once at the end. ----
constexpr int32_t kMmaThreads = 128;
constexpr int32_t kMmaWarps = 4;
constexpr int32_t kMmaWarpKeys = 16;
constexpr int32_t kMmaTile = kMmaWarps * kMmaWarpKeys; // keys per block tile
constexpr int32_t kMmaKP = kInkling2HeadDim + 8;       // staged row of 136 halves = 272 bytes: 16-byte multiple and conflict-free for ldmatrix
constexpr int32_t kMmaSliceHalves = 2 * kMmaWarpKeys * kMmaKP; // K and V of one warp stage

//! INT8 rows are staged through registers, so one shared-memory stage per warp is enough (the next tile's loads are in flight while the current one is computed);
//! FP16 rows are copied with cp.async into the other of two stages.
__host__ __device__ constexpr int32_t decodeMmaStages(bool int8)
{
    return int8 ? 1 : 2;
}

constexpr size_t decodeMmaSmemBytes(bool int8)
{
    return sizeof(half) * kMmaSliceHalves * decodeMmaStages(int8) * kMmaWarps // [warp][stage] K and V slices
        + kMmaWarps * 2 * kMmaWarpKeys                                        // flags (two stages reserved)
        + sizeof(float) * kMmaWarps * 2 * kMmaWarpKeys * 2                    // K and V scales
        + sizeof(float) * kMmaWarps * kInkling2HeadDim                        // Q rows
        + sizeof(float) * kMmaWarps * kInkling2RelDim;                        // relative-state rows
}

__device__ __forceinline__ void ldmatrixX4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, void const* smemPtr)
{
    uint32_t const addr = static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrixX4Trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, void const* smemPtr)
{
    uint32_t const addr = static_cast<uint32_t>(__cvta_generic_to_shared(smemPtr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

//! (c0, c1) += rows 0-7 of A (16 x 16; a1 = a3 = 0) * B (16 x 8): FP16 inputs, FP32 accumulation. Only the first row group is live, so the second half of the
//! accumulator fragment (rows 8-15) is a discarded temporary instead of 2 persistent registers per tile.
__device__ __forceinline__ void mmaRows8(float& c0, float& c1, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1)
{
    float d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(c0), "=f"(c1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(0U), "r"(a2), "r"(0U), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(0.F), "f"(0.F));
}

__device__ __forceinline__ uint32_t packHalf2(float lo, float hi)
{
    half2 const h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t const*>(&h);
}

__device__ __forceinline__ void cpAsync16(void* dst, void const* src, bool valid)
{
    uint32_t const addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    int32_t const srcSize = valid ? 16 : 0; // 0 bytes copied: the destination is zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(addr), "l"(src), "r"(srcSize));
}

__device__ __forceinline__ void cpAsyncCommitWaitAll()
{
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
}

//! Two signed bytes of x (selected by sel: 0x5140 for the low pair, 0x7362 for the high pair) as exact half2 values: (b ^ 0x80) | 0x6400 is 1024 + b + 128, minus 1152.
__device__ __forceinline__ uint32_t int8PairToHalf2(uint32_t x, uint32_t sel)
{
    uint32_t const biased = __byte_perm(x ^ 0x80808080U, 0x64646464U, sel);
    half2 const v = __hsub2(*reinterpret_cast<half2 const*>(&biased), __float2half2_rn(1152.F));
    return *reinterpret_cast<uint32_t const*>(&v);
}

template <typename KvT>
__global__ void __launch_bounds__(kMmaThreads, std::is_same_v<KvT, int8_t> ? 2 : 3) decodeMmaKernel(Inkling2AttentionParams p, half const* __restrict__ qLin, half const* __restrict__ relLin,
    float const* __restrict__ qScale, float const* __restrict__ relScale, float const* __restrict__ relProjT, KvT const* __restrict__ kvPool,
    int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths, int32_t const* __restrict__ past, int32_t const* __restrict__ pageTable,
    float* __restrict__ partials, int32_t splits)
{
    constexpr bool kInt8 = std::is_same_v<KvT, int8_t>;
    constexpr int32_t kStages = decodeMmaStages(kInt8);
    extern __shared__ __align__(16) unsigned char smemRaw[];
    half* const slices = reinterpret_cast<half*>(smemRaw);
    uint8_t* const flags = reinterpret_cast<uint8_t*>(slices + kMmaSliceHalves * kStages * kMmaWarps);
    float* const kScales = reinterpret_cast<float*>(flags + kMmaWarps * 2 * kMmaWarpKeys);
    float* const vScales = kScales + kMmaWarps * 2 * kMmaWarpKeys;
    float* const Qsm = vScales + kMmaWarps * 2 * kMmaWarpKeys;
    float* const RSsm = Qsm + kMmaWarps * kInkling2HeadDim;

    int32_t const tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    int32_t const gid = lane >> 2, tig = lane & 3;
    int32_t const slot = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z;
    int32_t seq = 0, qpos = 0;
    bool const valid = locateToken(slot, offsets, lengths, past, p.batchSize, seq, qpos);
    int32_t const group = p.numQHeads / p.numKVHeads;
    int64_t const rowElems = int64_t(p.numKVHeads) * KvTraits<KvT>::rowStride;
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * rowElems;
    int32_t const* table = pageTable + int64_t(seq) * 2 * p.maxPagesPerSeq;

    // Query rows: warp w < group prepares head kvh * group + w (same scaling and clipping as the other decode kernels).
    if (w < group && valid)
    {
        int32_t const h = kvh * group + w;
        float const logScale = p.window == 0 ? 1.F + p.logAlpha * logf(fmaxf((qpos + 1.F) / p.logNFloor, 1.F)) : 1.F;
        float q[4];
        float ss = 0.F;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            int32_t const d = lane * 4 + i;
            q[i] = __half2float(qLin[int64_t(slot) * p.qLd + h * kInkling2HeadDim + d]) * qScale[h * kInkling2HeadDim + d];
            ss += q[i] * q[i];
        }
        ss = warpSum(ss);
        float const qFactor = fminf(p.qClip / (sqrtf(ss) + 1e-6F), 1.F) * logScale;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            Qsm[w * kInkling2HeadDim + lane * 4 + i] = q[i] * qFactor;
        }
        int32_t const j = lane & 15;
        float rs = __half2float(relLin[int64_t(slot) * p.relLd + h * kInkling2RelDim + j]) * relScale[h * kInkling2RelDim + j];
        float const rss = halfWarpSum(rs * rs);
        rs *= fminf(p.relClip / (sqrtf(rss) + 1e-6F), 1.F) * logScale;
        if (lane < 16)
        {
            RSsm[w * kInkling2RelDim + j] = rs;
        }
    }
    __syncthreads();

    int32_t const lo = valid ? (p.window > 0 ? max(0, qpos - p.window + 1) : 0) : 0;
    int32_t const hi = valid ? qpos : -1;
    int32_t const nTiles = hi >= lo ? (hi - lo + kMmaTile) / kMmaTile : 0;
    int32_t const perSplit = (nTiles + splits - 1) / splits;
    int32_t const tBegin = min(nTiles, split * perSplit), tEnd = min(nTiles, tBegin + perSplit);

    // Q as the A operand of the score product: row gid (head kvh * group + gid), dimensions 16 * ks + 2 * tig + {0, 1} and + 8; rows >= group are zero.
    uint32_t qa[8][2];
#pragma unroll
    for (int32_t ks = 0; ks < 8; ++ks)
    {
        float const* row = Qsm + gid * kInkling2HeadDim + ks * 16 + 2 * tig;
        bool const live = gid < group && valid;
        qa[ks][0] = live ? packHalf2(row[0], row[1]) : 0U;
        qa[ks][1] = live ? packHalf2(row[8], row[9]) : 0U;
    }

    float accO[16][2];
#pragma unroll
    for (int32_t i = 0; i < 16; ++i)
    {
        accO[i][0] = accO[i][1] = 0.F;
    }
    float rowM = -INFINITY, rowL = 0.F;

    half* const warpSlices = slices + int64_t(w) * kStages * kMmaSliceHalves;
    uint8_t* const warpFlags = flags + w * 2 * kMmaWarpKeys;
    float* const warpKS = kScales + w * 2 * kMmaWarpKeys;
    float* const warpVS = vScales + w * 2 * kMmaWarpKeys;

    uint32_t pfK[kInt8 ? kMmaWarpKeys : 1], pfV[kInt8 ? kMmaWarpKeys : 1];
    float pfScale = 0.F; // INT8: lanes 0-15 hold the K scale of key `lane`, lanes 16-31 the V scale of key `lane - 16`
    uint32_t pfOk = 0;

    auto keyBaseOf = [&](int32_t ti) { return lo + ti * kMmaTile + w * kMmaWarpKeys; };

    // Page-table entries of a tile's slice (the K and V page of its first key and of the next page when the slice straddles a boundary); fetched one tile ahead so the
    // data loads of the next tile do not wait for a dependent table load.
    struct TileTable
    {
        int32_t k0, v0, k1, v1;
    };
    auto fetchTable = [&](int32_t ti)
    {
        int32_t const base = keyBaseOf(ti);
        int32_t const nValid = max(0, min(kMmaWarpKeys, hi - base + 1));
        int32_t const page0 = base / kInkling2TokensPerPage;
        int32_t const n0 = min(kMmaWarpKeys, kInkling2TokensPerPage - (base - page0 * kInkling2TokensPerPage));
        bool const has0 = nValid > 0, has1 = nValid > n0;
        TileTable t;
        t.k0 = has0 ? table[page0] : -1;
        t.v0 = has0 ? table[p.maxPagesPerSeq + page0] : -1;
        t.k1 = has1 ? table[page0 + 1] : -1;
        t.v1 = has1 ? table[p.maxPagesPerSeq + page0 + 1] : -1;
        return t;
    };

    // Loads the 16 keys of this warp's slice of tile ti (FP16: cp.async into stage st; INT8: into registers, converted and stored by storeStage).
    auto issueLoads = [&](int32_t ti, int32_t st, TileTable const& tt)
    {
        int32_t const base = keyBaseOf(ti);
        int32_t const nValid = max(0, min(kMmaWarpKeys, hi - base + 1));
        int32_t const page0 = base / kInkling2TokensPerPage;
        int32_t const row0 = base - page0 * kInkling2TokensPerPage;
        int32_t const n0 = min(kMmaWarpKeys, kInkling2TokensPerPage - row0); // keys of this slice in the first page, the rest are in the next one
        int32_t const k0 = tt.k0, v0 = tt.v0, k1 = tt.k1, v1 = tt.v1;
        uint32_t const validMask = nValid >= kMmaWarpKeys ? 0xFFFFU : ((1U << nValid) - 1U);
        uint32_t const lowMask = (1U << n0) - 1U; // n0 <= 16; n0 == 16 gives 0xFFFF
        uint32_t const okMask = validMask & (((k0 >= 0 && v0 >= 0) ? lowMask : 0U) | ((k1 >= 0 && v1 >= 0) ? (0xFFFFU & ~lowMask) : 0U));
        int64_t const kvOff = int64_t(kvh) * KvTraits<KvT>::rowStride;
        KvT const* const kBase0 = kvPool + int64_t(k0 < 0 ? 0 : k0) * pageElems + kvOff;
        KvT const* const vBase0 = kvPool + int64_t(v0 < 0 ? 0 : v0) * pageElems + kvOff;
        KvT const* const kBase1 = kvPool + int64_t(k1 < 0 ? 0 : k1) * pageElems + kvOff;
        KvT const* const vBase1 = kvPool + int64_t(v1 < 0 ? 0 : v1) * pageElems + kvOff;
        if constexpr (kInt8)
        {
            pfOk = okMask;
#pragma unroll
            for (int32_t kk = 0; kk < kMmaWarpKeys; ++kk)
            {
                bool const second = kk >= n0;
                int64_t const idx = (second ? kk - n0 : row0 + kk) * rowElems;
                KvT const* const kRow = (second ? kBase1 : kBase0) + idx;
                KvT const* const vRow = (second ? vBase1 : vBase0) + idx;
                bool const ok = ((okMask >> kk) & 1U) != 0U;
                pfK[kk] = ok ? *reinterpret_cast<uint32_t const*>(kRow + lane * 4) : 0U;
                pfV[kk] = ok ? *reinterpret_cast<uint32_t const*>(vRow + lane * 4) : 0U;
            }
            // one scale per lane: key (lane & 15), K rows for lanes 0-15 and V rows for lanes 16-31
            {
                int32_t const kk = lane & 15;
                bool const second = kk >= n0;
                int64_t const idx = (second ? kk - n0 : row0 + kk) * rowElems;
                KvT const* const row = (lane < 16 ? (second ? kBase1 : kBase0) : (second ? vBase1 : vBase0)) + idx;
                pfScale = ((okMask >> kk) & 1U) != 0U ? __half2float(*reinterpret_cast<half const*>(row + kInkling2HeadDim)) : 0.F;
            }
        }
        else
        {
            half* const kDst = warpSlices + st * kMmaSliceHalves;
            half* const vDst = kDst + kMmaWarpKeys * kMmaKP;
            // One cp.async per key for the whole warp: lanes 0-15 copy the 16 chunks of the K row, lanes 16-31 those of the V row (the former two divergent half-warp branches
            // issued every copy twice and recomputed a 64-bit address per key: about 630 of the 1,040 instructions per tile, METRICS 230). Two row pointers, one per page: row kk is
            // row (row0 + kk) of the first page, or row (kk - n0) of the next one.
            bool const isV = lane >= 16;
            int32_t const chunk = (lane & 15) * 8;
            KvT const* const p0 = (isV ? vBase0 : kBase0) + int64_t(row0) * rowElems + chunk;
            KvT const* const p1 = (isV ? vBase1 : kBase1) - int64_t(n0) * rowElems + chunk;
            half* const dst = (isV ? vDst : kDst) + chunk;
#pragma unroll
            for (int32_t kk = 0; kk < kMmaWarpKeys; ++kk)
            {
                cpAsync16(dst + kk * kMmaKP, (kk >= n0 ? p1 : p0) + int64_t(kk) * rowElems, ((okMask >> kk) & 1U) != 0U);
            }
            if (lane < kMmaWarpKeys)
            {
                warpFlags[st * kMmaWarpKeys + lane] = (okMask >> lane) & 1U;
            }
        }
    };

    auto storeStage = [&](int32_t st)
    {
        if constexpr (kInt8)
        {
            half* const kDst = warpSlices + st * kMmaSliceHalves;
            half* const vDst = kDst + kMmaWarpKeys * kMmaKP;
#pragma unroll
            for (int32_t kk = 0; kk < kMmaWarpKeys; ++kk)
            {
                uint2 kv, vv;
                kv.x = int8PairToHalf2(pfK[kk], 0x5140U);
                kv.y = int8PairToHalf2(pfK[kk], 0x7362U);
                vv.x = int8PairToHalf2(pfV[kk], 0x5140U);
                vv.y = int8PairToHalf2(pfV[kk], 0x7362U);
                *reinterpret_cast<uint2*>(kDst + kk * kMmaKP + lane * 4) = kv;
                *reinterpret_cast<uint2*>(vDst + kk * kMmaKP + lane * 4) = vv;
            }
            if (lane < kMmaWarpKeys)
            {
                warpKS[st * kMmaWarpKeys + lane] = pfScale;
                warpFlags[st * kMmaWarpKeys + lane] = (pfOk >> lane) & 1U;
            }
            else
            {
                warpVS[st * kMmaWarpKeys + lane - kMmaWarpKeys] = pfScale;
            }
        }
    };

    TileTable nextTable{-1, -1, -1, -1};
    if (tBegin < tEnd)
    {
        issueLoads(tBegin, 0, fetchTable(tBegin));
        if (tBegin + 1 < tEnd)
        {
            nextTable = fetchTable(tBegin + 1);
        }
        if constexpr (kInt8)
        {
            storeStage(0);
        }
        else
        {
            cpAsyncCommitWaitAll();
        }
        __syncwarp();
    }

    for (int32_t ti = tBegin; ti < tEnd; ++ti)
    {
        int32_t const st = kStages == 1 ? 0 : ((ti - tBegin) & 1);
        int32_t const nst = kStages == 1 ? 0 : (st ^ 1);
        bool const hasNext = ti + 1 < tEnd;
        if (hasNext)
        {
            issueLoads(ti + 1, nst, nextTable);
            if (ti + 2 < tEnd)
            {
                nextTable = fetchTable(ti + 2);
            }
        }
        half const* const kS = warpSlices + st * kMmaSliceHalves;
        half const* const vS = kS + kMmaWarpKeys * kMmaKP;

        // scores: S[row gid][key] for the warp's 16 keys as two n8 tiles
        float c[2][2] = {{0.F, 0.F}, {0.F, 0.F}};
#pragma unroll
        for (int32_t ks = 0; ks < 8; ++ks)
        {
            uint32_t b0, b1, b2, b3;
            int32_t const key = ((lane >> 4) & 1) * 8 + (lane & 7);
            int32_t const col = ks * 16 + ((lane >> 3) & 1) * 8;
            ldmatrixX4(b0, b1, b2, b3, kS + key * kMmaKP + col);
            mmaRows8(c[0][0], c[0][1], qa[ks][0], qa[ks][1], b0, b1);
            mmaRows8(c[1][0], c[1][1], qa[ks][0], qa[ks][1], b2, b3);
        }

        int32_t const base = keyBaseOf(ti);
        float sc[2][2];
        float tileMax = -INFINITY;
#pragma unroll
        for (int32_t n = 0; n < 2; ++n)
        {
#pragma unroll
            for (int32_t j = 0; j < 2; ++j)
            {
                int32_t const kt = n * 8 + 2 * tig + j; // key within the warp's slice
                int32_t const sKey = base + kt;
                float score = -INFINITY;
                if (gid < group && warpFlags[st * kMmaWarpKeys + kt] != 0 && sKey <= hi && sKey >= lo)
                {
                    float const ksc = kInt8 ? warpKS[st * kMmaWarpKeys + kt] : 1.F;
                    score = c[n][j] * ksc * p.scaling;
                    int32_t const dist = hi - sKey;
                    if (dist < p.relExtent)
                    {
                        float const* rp = relProjT + int64_t(dist) * kInkling2RelDim;
                        float const* rsr = RSsm + gid * kInkling2RelDim;
                        float bias = 0.F;
#pragma unroll
                        for (int32_t jj = 0; jj < kInkling2RelDim; jj += 4)
                        {
                            float4 const a = *reinterpret_cast<float4 const*>(rp + jj);
                            bias += rsr[jj] * a.x + rsr[jj + 1] * a.y + rsr[jj + 2] * a.z + rsr[jj + 3] * a.w;
                        }
                        score += bias;
                    }
                }
                sc[n][j] = score;
                tileMax = fmaxf(tileMax, score);
            }
        }
        tileMax = fmaxf(tileMax, __shfl_xor_sync(0xffffffffu, tileMax, 1));
        tileMax = fmaxf(tileMax, __shfl_xor_sync(0xffffffffu, tileMax, 2));
        float const mNew = fmaxf(rowM, tileMax);
        float const corr = rowM == -INFINITY ? 0.F : __expf(rowM - mNew);
        float e[2][2];
        float sum = 0.F;
#pragma unroll
        for (int32_t n = 0; n < 2; ++n)
        {
#pragma unroll
            for (int32_t j = 0; j < 2; ++j)
            {
                e[n][j] = sc[n][j] == -INFINITY ? 0.F : __expf(sc[n][j] - mNew);
                sum += e[n][j];
            }
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        rowL = rowL * corr + sum;
        rowM = mNew;
#pragma unroll
        for (int32_t i = 0; i < 16; ++i)
        {
            accO[i][0] *= corr;
            accO[i][1] *= corr;
        }

        // P as the A operand (V scale folded in)
        uint32_t pa0, pa1;
        {
            float v0 = 1.F, v1 = 1.F, v2 = 1.F, v3 = 1.F;
            if constexpr (kInt8)
            {
                v0 = warpVS[st * kMmaWarpKeys + 2 * tig];
                v1 = warpVS[st * kMmaWarpKeys + 2 * tig + 1];
                v2 = warpVS[st * kMmaWarpKeys + 8 + 2 * tig];
                v3 = warpVS[st * kMmaWarpKeys + 8 + 2 * tig + 1];
            }
            pa0 = packHalf2(e[0][0] * v0, e[0][1] * v1);
            pa1 = packHalf2(e[1][0] * v2, e[1][1] * v3);
        }
#pragma unroll
        for (int32_t dt = 0; dt < 8; ++dt)
        {
            uint32_t b0, b1, b2, b3;
            int32_t const key = ((lane >> 3) & 1) * 8 + (lane & 7);
            int32_t const col = dt * 16 + ((lane >> 4) & 1) * 8;
            ldmatrixX4Trans(b0, b1, b2, b3, vS + key * kMmaKP + col);
            mmaRows8(accO[2 * dt][0], accO[2 * dt][1], pa0, pa1, b0, b1);
            mmaRows8(accO[2 * dt + 1][0], accO[2 * dt + 1][1], pa0, pa1, b2, b3);
        }

        if (hasNext)
        {
            __syncwarp(); // every lane is done reading this stage (INT8 reuses it for the next tile)
            if constexpr (kInt8)
            {
                storeStage(nst);
            }
            else
            {
                cpAsyncCommitWaitAll();
            }
            __syncwarp();
        }
    }

    // Merge the four warps: (m, l, acc) per row.
    __syncthreads();
    float* const accS = reinterpret_cast<float*>(smemRaw); // reuses the stage slices
    float* const mS = accS + kMmaWarps * 4 * kInkling2HeadDim;
    float* const lS = mS + kMmaWarps * 4;
    if (gid < group)
    {
#pragma unroll
        for (int32_t i = 0; i < 16; ++i)
        {
            accS[(w * 4 + gid) * kInkling2HeadDim + i * 8 + 2 * tig] = accO[i][0];
            accS[(w * 4 + gid) * kInkling2HeadDim + i * 8 + 2 * tig + 1] = accO[i][1];
        }
        if (tig == 0)
        {
            mS[w * 4 + gid] = rowM;
            lS[w * 4 + gid] = rowL;
        }
    }
    __syncthreads();
    int32_t const d = tid;
    for (int32_t row = 0; row < group; ++row)
    {
        float mm = -INFINITY;
#pragma unroll
        for (int32_t ww = 0; ww < kMmaWarps; ++ww)
        {
            mm = fmaxf(mm, mS[ww * 4 + row]);
        }
        // A warp without a finite maximum has weight 0 but is still multiplied in, so a NaN in its partials (a NaN query) stays in this token like in the other kernels.
        float num = 0.F, den = 0.F;
#pragma unroll
        for (int32_t ww = 0; ww < kMmaWarps; ++ww)
        {
            float const mw = mS[ww * 4 + row];
            float const cf = (mm == -INFINITY || mw == -INFINITY) ? 0.F : __expf(mw - mm);
            num += accS[(ww * 4 + row) * kInkling2HeadDim + d] * cf;
            den += lS[ww * 4 + row] * cf;
        }
        int32_t const h = kvh * group + row;
        float* out = partials + ((int64_t(slot) * p.numQHeads + h) * splits + split) * kPartialStride;
        out[2 + d] = num;
        if (d == 0)
        {
            out[0] = mm;
            out[1] = den;
        }
    }
}

// ---- Multi-token variant for verification windows (one sequence, several consecutive query tokens): the KV of a window was streamed once per query token (METRICS 230: a 5-token
// window took 170 us per launch against 63 us for one token). One block serves up to 16 / group tokens (5 x 3 GQA rows = 15 rows of the m16 tile), so each key tile is read once.
// Row r = token (r / group) x head (r % group); the second half of the m16 tile (rows 8-15) is a real accumulator here. Every row keeps its own causal limit and relative-bias
// distance; the per-(token, head) partial contract of the merge kernel is unchanged. ----
constexpr int32_t kMultiRows = 16;

constexpr size_t decodeMultiSmemBytes(bool int8)
{
    return sizeof(half) * kMmaSliceHalves * decodeMmaStages(int8) * kMmaWarps // [warp][stage] K and V slices
        + kMmaWarps * 2 * kMmaWarpKeys                                        // flags
        + sizeof(float) * kMmaWarps * 2 * kMmaWarpKeys * 2                    // K and V scales
        + sizeof(float) * kMultiRows * kInkling2HeadDim                       // Q rows
        + sizeof(float) * kMultiRows * kInkling2RelDim                        // relative-state rows
        + sizeof(int32_t) * kMultiRows * 3;                                   // per-row qpos, lo, hi (hi < lo: dead row)
}

__device__ __forceinline__ void mma16x8x16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <typename KvT>
__global__ void __launch_bounds__(kMmaThreads, 2) decodeMmaMultiKernel(Inkling2AttentionParams p, half const* __restrict__ qLin, half const* __restrict__ relLin,
    float const* __restrict__ qScale, float const* __restrict__ relScale, float const* __restrict__ relProjT, KvT const* __restrict__ kvPool,
    int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths, int32_t const* __restrict__ past, int32_t const* __restrict__ pageTable,
    float* __restrict__ partials, int32_t splits)
{
    constexpr bool kInt8 = std::is_same_v<KvT, int8_t>;
    constexpr int32_t kStages = decodeMmaStages(kInt8);
    extern __shared__ __align__(16) unsigned char smemRaw[];
    half* const slices = reinterpret_cast<half*>(smemRaw);
    uint8_t* const flags = reinterpret_cast<uint8_t*>(slices + kMmaSliceHalves * kStages * kMmaWarps);
    float* const kScales = reinterpret_cast<float*>(flags + kMmaWarps * 2 * kMmaWarpKeys);
    float* const vScales = kScales + kMmaWarps * 2 * kMmaWarpKeys;
    float* const Qsm = vScales + kMmaWarps * 2 * kMmaWarpKeys;
    float* const RSsm = Qsm + kMultiRows * kInkling2HeadDim;
    int32_t* const rowQpos = reinterpret_cast<int32_t*>(RSsm + kMultiRows * kInkling2RelDim);
    int32_t* const rowLo = rowQpos + kMultiRows;
    int32_t* const rowHi = rowLo + kMultiRows;

    int32_t const tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    int32_t const gid = lane >> 2, tig = lane & 3;
    int32_t const kvh = blockIdx.y, split = blockIdx.z;
    int32_t const group = p.numQHeads / p.numKVHeads;
    int32_t const tb = kMultiRows / group; // query tokens per block
    int32_t const slot0 = blockIdx.x * tb;
    int64_t const rowElems = int64_t(p.numKVHeads) * KvTraits<KvT>::rowStride;
    int64_t const pageElems = int64_t(kInkling2TokensPerPage) * rowElems;

    // Everything the kernel needs from global memory without depending on another load is requested at once: the batch-1 sequence descriptor, the query-row inputs
    // (only the token range is checked here, a row's liveness is decided after the descriptor arrives) -- the latency of these loads overlaps instead of adding up.
    // This kernel is launched for batchSize == 1 only.
    int32_t const off0 = offsets[0], len0 = lengths[0], past0 = past[0];
    constexpr int32_t kRowsPerWarp = kMultiRows / kMmaWarps;
    float qRaw[kRowsPerWarp][4], qSc[kRowsPerWarp][4], rRaw[kRowsPerWarp], rSc[kRowsPerWarp];
#pragma unroll
    for (int32_t ri = 0; ri < kRowsPerWarp; ++ri)
    {
        int32_t const r = w + ri * kMmaWarps;
        int32_t const tl = r / group, hh = r - tl * group;
        int32_t const slot = slot0 + tl;
        bool const live = tl < tb && slot < p.numTokens;
        int32_t const h = kvh * group + hh;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            int32_t const d = lane * 4 + i;
            qRaw[ri][i] = live ? __half2float(qLin[int64_t(slot) * p.qLd + h * kInkling2HeadDim + d]) : 0.F;
            qSc[ri][i] = live ? qScale[h * kInkling2HeadDim + d] : 0.F;
        }
        int32_t const j = lane & 15;
        rRaw[ri] = live ? __half2float(relLin[int64_t(slot) * p.relLd + h * kInkling2RelDim + j]) : 0.F;
        rSc[ri] = live ? relScale[h * kInkling2RelDim + j] : 0.F;
    }

    // Per-row causal limits (threads 0-15 each take a row); a row is dead when its token is outside the batch.
    int32_t const seq0 = 0;
    bool const valid0 = slot0 - off0 >= 0 && slot0 - off0 < len0;
    if (tid < kMultiRows)
    {
        int32_t const tl = tid / group;
        int32_t qp = -1, lo = 0, hi = -1;
        int32_t const local = slot0 + tl - off0;
        if (valid0 && tl < tb && slot0 + tl < p.numTokens && local >= 0 && local < len0)
        {
            qp = past0 + local;
            lo = p.window > 0 ? max(0, qp - p.window + 1) : 0;
            hi = qp;
        }
        rowQpos[tid] = qp;
        rowLo[tid] = lo;
        rowHi[tid] = hi;
    }
    __syncthreads();
    int32_t const* const table = pageTable + int64_t(seq0) * 2 * p.maxPagesPerSeq;

    // union of the rows' key ranges (consecutive tokens: the earliest window start to the latest position)
    int32_t lo = 0x7fffffff, hi = -1;
#pragma unroll
    for (int32_t r = 0; r < kMultiRows; ++r)
    {
        if (rowHi[r] >= rowLo[r])
        {
            lo = min(lo, rowLo[r]);
            hi = max(hi, rowHi[r]);
        }
    }
    if (hi < 0)
    {
        lo = 0;
    }
    int32_t const nTiles = hi >= lo ? (hi - lo + kMmaTile) / kMmaTile : 0;
    int32_t const perSplit = (nTiles + splits - 1) / splits;
    int32_t const tBegin = min(nTiles, split * perSplit), tEnd = min(nTiles, tBegin + perSplit);

    half* const warpSlices = slices + int64_t(w) * kStages * kMmaSliceHalves;
    uint8_t* const warpFlags = flags + w * 2 * kMmaWarpKeys;
    float* const warpKS = kScales + w * 2 * kMmaWarpKeys;
    float* const warpVS = vScales + w * 2 * kMmaWarpKeys;

    uint32_t pfK[kInt8 ? kMmaWarpKeys : 1], pfV[kInt8 ? kMmaWarpKeys : 1];
    float pfScale = 0.F;
    uint32_t pfOk = 0;

    auto keyBaseOf = [&](int32_t ti) { return lo + ti * kMmaTile + w * kMmaWarpKeys; };

    struct TileTable
    {
        int32_t k0, v0, k1, v1;
    };
    auto fetchTable = [&](int32_t ti)
    {
        int32_t const base = keyBaseOf(ti);
        int32_t const nValid = max(0, min(kMmaWarpKeys, hi - base + 1));
        int32_t const page0 = base / kInkling2TokensPerPage;
        int32_t const n0 = min(kMmaWarpKeys, kInkling2TokensPerPage - (base - page0 * kInkling2TokensPerPage));
        bool const has0 = nValid > 0, has1 = nValid > n0;
        TileTable t;
        t.k0 = has0 ? table[page0] : -1;
        t.v0 = has0 ? table[p.maxPagesPerSeq + page0] : -1;
        t.k1 = has1 ? table[page0 + 1] : -1;
        t.v1 = has1 ? table[p.maxPagesPerSeq + page0 + 1] : -1;
        return t;
    };

    auto issueLoads = [&](int32_t ti, int32_t st, TileTable const& tt)
    {
        int32_t const base = keyBaseOf(ti);
        int32_t const nValid = max(0, min(kMmaWarpKeys, hi - base + 1));
        int32_t const page0 = base / kInkling2TokensPerPage;
        int32_t const row0 = base - page0 * kInkling2TokensPerPage;
        int32_t const n0 = min(kMmaWarpKeys, kInkling2TokensPerPage - row0);
        int32_t const k0 = tt.k0, v0 = tt.v0, k1 = tt.k1, v1 = tt.v1;
        uint32_t const validMask = nValid >= kMmaWarpKeys ? 0xFFFFU : ((1U << nValid) - 1U);
        uint32_t const lowMask = (1U << n0) - 1U;
        uint32_t const okMask = validMask & (((k0 >= 0 && v0 >= 0) ? lowMask : 0U) | ((k1 >= 0 && v1 >= 0) ? (0xFFFFU & ~lowMask) : 0U));
        int64_t const kvOff = int64_t(kvh) * KvTraits<KvT>::rowStride;
        KvT const* const kBase0 = kvPool + int64_t(k0 < 0 ? 0 : k0) * pageElems + kvOff;
        KvT const* const vBase0 = kvPool + int64_t(v0 < 0 ? 0 : v0) * pageElems + kvOff;
        KvT const* const kBase1 = kvPool + int64_t(k1 < 0 ? 0 : k1) * pageElems + kvOff;
        KvT const* const vBase1 = kvPool + int64_t(v1 < 0 ? 0 : v1) * pageElems + kvOff;
        if constexpr (kInt8)
        {
            pfOk = okMask;
#pragma unroll
            for (int32_t kk = 0; kk < kMmaWarpKeys; ++kk)
            {
                bool const second = kk >= n0;
                int64_t const idx = (second ? kk - n0 : row0 + kk) * rowElems;
                KvT const* const kRow = (second ? kBase1 : kBase0) + idx;
                KvT const* const vRow = (second ? vBase1 : vBase0) + idx;
                bool const ok = ((okMask >> kk) & 1U) != 0U;
                pfK[kk] = ok ? *reinterpret_cast<uint32_t const*>(kRow + lane * 4) : 0U;
                pfV[kk] = ok ? *reinterpret_cast<uint32_t const*>(vRow + lane * 4) : 0U;
            }
            {
                int32_t const kk = lane & 15;
                bool const second = kk >= n0;
                int64_t const idx = (second ? kk - n0 : row0 + kk) * rowElems;
                KvT const* const row = (lane < 16 ? (second ? kBase1 : kBase0) : (second ? vBase1 : vBase0)) + idx;
                pfScale = ((okMask >> kk) & 1U) != 0U ? __half2float(*reinterpret_cast<half const*>(row + kInkling2HeadDim)) : 0.F;
            }
        }
        else
        {
            half* const kDst = warpSlices + st * kMmaSliceHalves;
            half* const vDst = kDst + kMmaWarpKeys * kMmaKP;
            bool const isV = lane >= 16;
            int32_t const chunk = (lane & 15) * 8;
            KvT const* const p0 = (isV ? vBase0 : kBase0) + int64_t(row0) * rowElems + chunk;
            KvT const* const p1 = (isV ? vBase1 : kBase1) - int64_t(n0) * rowElems + chunk;
            half* const dst = (isV ? vDst : kDst) + chunk;
#pragma unroll
            for (int32_t kk = 0; kk < kMmaWarpKeys; ++kk)
            {
                cpAsync16(dst + kk * kMmaKP, (kk >= n0 ? p1 : p0) + int64_t(kk) * rowElems, ((okMask >> kk) & 1U) != 0U);
            }
            if (lane < kMmaWarpKeys)
            {
                warpFlags[st * kMmaWarpKeys + lane] = (okMask >> lane) & 1U;
            }
        }
    };

    auto storeStage = [&](int32_t st)
    {
        if constexpr (kInt8)
        {
            half* const kDst = warpSlices + st * kMmaSliceHalves;
            half* const vDst = kDst + kMmaWarpKeys * kMmaKP;
#pragma unroll
            for (int32_t kk = 0; kk < kMmaWarpKeys; ++kk)
            {
                uint2 kv, vv;
                kv.x = int8PairToHalf2(pfK[kk], 0x5140U);
                kv.y = int8PairToHalf2(pfK[kk], 0x7362U);
                vv.x = int8PairToHalf2(pfV[kk], 0x5140U);
                vv.y = int8PairToHalf2(pfV[kk], 0x7362U);
                *reinterpret_cast<uint2*>(kDst + kk * kMmaKP + lane * 4) = kv;
                *reinterpret_cast<uint2*>(vDst + kk * kMmaKP + lane * 4) = vv;
            }
            if (lane < kMmaWarpKeys)
            {
                warpKS[st * kMmaWarpKeys + lane] = pfScale;
                warpFlags[st * kMmaWarpKeys + lane] = (pfOk >> lane) & 1U;
            }
            else
            {
                warpVS[st * kMmaWarpKeys + lane - kMmaWarpKeys] = pfScale;
            }
        }
    };

    // The first key tile's loads (FP16: cp.async, no register staging) go out after the query-row loads are issued (so those are not queued behind the tile), so their latency overlaps the prologue.
    TileTable nextTable{-1, -1, -1, -1};
    if constexpr (!kInt8)
    {
        if (tBegin < tEnd)
        {
            issueLoads(tBegin, 0, fetchTable(tBegin));
            if (tBegin + 1 < tEnd)
            {
                nextTable = fetchTable(tBegin + 1);
            }
        }
    }

    // Query rows: warp w prepares rows w, w + 4, ... (same scaling and clipping as the other decode kernels).
#pragma unroll
    for (int32_t ri = 0; ri < kRowsPerWarp; ++ri)
    {
        int32_t const r = w + ri * kMmaWarps;
        // dead rows are computed too (their inputs are zero and the result is never read), so the four rows form one straight-line block the scheduler can interleave
        int32_t const qpos = max(rowQpos[r], 0);
        float const logScale = p.window == 0 ? 1.F + p.logAlpha * logf(fmaxf((qpos + 1.F) / p.logNFloor, 1.F)) : 1.F;
        float q[4];
        float ss = 0.F;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            q[i] = qRaw[ri][i] * qSc[ri][i];
            ss += q[i] * q[i];
        }
        ss = warpSum(ss);
        float const qFactor = fminf(p.qClip / (sqrtf(ss) + 1e-6F), 1.F) * logScale;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            Qsm[r * kInkling2HeadDim + lane * 4 + i] = q[i] * qFactor;
        }
        int32_t const j = lane & 15;
        float rs = rRaw[ri] * rSc[ri];
        float const rss = halfWarpSum(rs * rs);
        rs *= fminf(p.relClip / (sqrtf(rss) + 1e-6F), 1.F) * logScale;
        if (lane < 16)
        {
            RSsm[r * kInkling2RelDim + j] = rs;
        }
    }
    __syncthreads();

    // this lane's two accumulator rows (gid and gid + 8)
    int32_t const myLo[2] = {rowLo[gid], rowLo[gid + 8]};
    int32_t const myHi[2] = {rowHi[gid], rowHi[gid + 8]};
    bool const live[2] = {myHi[0] >= myLo[0], myHi[1] >= myLo[1]};

    // Q as the A operand of the score product: a0 (row gid, k), a1 (row gid + 8, k), a2 (row gid, k + 8), a3 (row gid + 8, k + 8)
    uint32_t qa[8][4];
#pragma unroll
    for (int32_t ks = 0; ks < 8; ++ks)
    {
        float const* row0 = Qsm + gid * kInkling2HeadDim + ks * 16 + 2 * tig;
        float const* row1 = Qsm + (gid + 8) * kInkling2HeadDim + ks * 16 + 2 * tig;
        qa[ks][0] = live[0] ? packHalf2(row0[0], row0[1]) : 0U;
        qa[ks][1] = live[1] ? packHalf2(row1[0], row1[1]) : 0U;
        qa[ks][2] = live[0] ? packHalf2(row0[8], row0[9]) : 0U;
        qa[ks][3] = live[1] ? packHalf2(row1[8], row1[9]) : 0U;
    }

    float accO[16][4];
#pragma unroll
    for (int32_t i = 0; i < 16; ++i)
    {
        accO[i][0] = accO[i][1] = accO[i][2] = accO[i][3] = 0.F;
    }
    float rowM[2] = {-INFINITY, -INFINITY}, rowL[2] = {0.F, 0.F};

    if (tBegin < tEnd)
    {
        if constexpr (kInt8)
        {
            issueLoads(tBegin, 0, fetchTable(tBegin));
            if (tBegin + 1 < tEnd)
            {
                nextTable = fetchTable(tBegin + 1);
            }
            storeStage(0);
        }
        else
        {
            cpAsyncCommitWaitAll();
        }
        __syncwarp();
    }


    for (int32_t ti = tBegin; ti < tEnd; ++ti)
    {
        int32_t const st = kStages == 1 ? 0 : ((ti - tBegin) & 1);
        int32_t const nst = kStages == 1 ? 0 : (st ^ 1);
        bool const hasNext = ti + 1 < tEnd;
        if (hasNext)
        {
            issueLoads(ti + 1, nst, nextTable);
            if (ti + 2 < tEnd)
            {
                nextTable = fetchTable(ti + 2);
            }
        }
        half const* const kS = warpSlices + st * kMmaSliceHalves;
        half const* const vS = kS + kMmaWarpKeys * kMmaKP;

        // scores: S[row][key] for the warp's 16 keys as two n8 tiles; c[n][0..1] row gid, c[n][2..3] row gid + 8
        float c[2][4] = {{0.F, 0.F, 0.F, 0.F}, {0.F, 0.F, 0.F, 0.F}};
#pragma unroll
        for (int32_t ks = 0; ks < 8; ++ks)
        {
            uint32_t b0, b1, b2, b3;
            int32_t const key = ((lane >> 4) & 1) * 8 + (lane & 7);
            int32_t const col = ks * 16 + ((lane >> 3) & 1) * 8;
            ldmatrixX4(b0, b1, b2, b3, kS + key * kMmaKP + col);
            mma16x8x16(c[0], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
            mma16x8x16(c[1], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b2, b3);
        }

        int32_t const base = keyBaseOf(ti);
        float sc[2][4];
        float tileMax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int32_t n = 0; n < 2; ++n)
        {
#pragma unroll
            for (int32_t j = 0; j < 2; ++j)
            {
                int32_t const kt = n * 8 + 2 * tig + j;
                int32_t const sKey = base + kt;
                bool const keyOk = warpFlags[st * kMmaWarpKeys + kt] != 0;
#pragma unroll
                for (int32_t hh = 0; hh < 2; ++hh)
                {
                    float score = -INFINITY;
                    if (live[hh] && keyOk && sKey <= myHi[hh] && sKey >= myLo[hh])
                    {
                        float const ksc = kInt8 ? warpKS[st * kMmaWarpKeys + kt] : 1.F;
                        score = c[n][hh * 2 + j] * ksc * p.scaling;
                        int32_t const dist = myHi[hh] - sKey;
                        if (dist < p.relExtent)
                        {
                            float const* rp = relProjT + int64_t(dist) * kInkling2RelDim;
                            float const* rsr = RSsm + (gid + 8 * hh) * kInkling2RelDim;
                            float bias = 0.F;
#pragma unroll
                            for (int32_t jj = 0; jj < kInkling2RelDim; jj += 4)
                            {
                                float4 const a = *reinterpret_cast<float4 const*>(rp + jj);
                                bias += rsr[jj] * a.x + rsr[jj + 1] * a.y + rsr[jj + 2] * a.z + rsr[jj + 3] * a.w;
                            }
                            score += bias;
                        }
                    }
                    sc[n][hh * 2 + j] = score;
                    tileMax[hh] = fmaxf(tileMax[hh], score);
                }
            }
        }
        float corr[2], e[2][4];
#pragma unroll
        for (int32_t hh = 0; hh < 2; ++hh)
        {
            tileMax[hh] = fmaxf(tileMax[hh], __shfl_xor_sync(0xffffffffu, tileMax[hh], 1));
            tileMax[hh] = fmaxf(tileMax[hh], __shfl_xor_sync(0xffffffffu, tileMax[hh], 2));
            float const mNew = fmaxf(rowM[hh], tileMax[hh]);
            corr[hh] = rowM[hh] == -INFINITY ? 0.F : __expf(rowM[hh] - mNew);
            float sum = 0.F;
#pragma unroll
            for (int32_t n = 0; n < 2; ++n)
            {
#pragma unroll
                for (int32_t j = 0; j < 2; ++j)
                {
                    float const s = sc[n][hh * 2 + j];
                    e[n][hh * 2 + j] = s == -INFINITY ? 0.F : __expf(s - mNew);
                    sum += e[n][hh * 2 + j];
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            rowL[hh] = rowL[hh] * corr[hh] + sum;
            rowM[hh] = mNew;
        }
#pragma unroll
        for (int32_t i = 0; i < 16; ++i)
        {
            accO[i][0] *= corr[0];
            accO[i][1] *= corr[0];
            accO[i][2] *= corr[1];
            accO[i][3] *= corr[1];
        }

        // P as the A operand (V scale folded in): a0 (row gid, keys 2 tig, +1), a1 (row gid + 8, same keys), a2 (row gid, keys 8 + 2 tig, +1), a3 (row gid + 8, same keys)
        float v0 = 1.F, v1 = 1.F, v2 = 1.F, v3 = 1.F;
        if constexpr (kInt8)
        {
            v0 = warpVS[st * kMmaWarpKeys + 2 * tig];
            v1 = warpVS[st * kMmaWarpKeys + 2 * tig + 1];
            v2 = warpVS[st * kMmaWarpKeys + 8 + 2 * tig];
            v3 = warpVS[st * kMmaWarpKeys + 8 + 2 * tig + 1];
        }
        uint32_t const pa0 = packHalf2(e[0][0] * v0, e[0][1] * v1);
        uint32_t const pa1 = packHalf2(e[0][2] * v0, e[0][3] * v1);
        uint32_t const pa2 = packHalf2(e[1][0] * v2, e[1][1] * v3);
        uint32_t const pa3 = packHalf2(e[1][2] * v2, e[1][3] * v3);
#pragma unroll
        for (int32_t dt = 0; dt < 8; ++dt)
        {
            uint32_t b0, b1, b2, b3;
            int32_t const key = ((lane >> 3) & 1) * 8 + (lane & 7);
            int32_t const col = dt * 16 + ((lane >> 4) & 1) * 8;
            ldmatrixX4Trans(b0, b1, b2, b3, vS + key * kMmaKP + col);
            mma16x8x16(accO[2 * dt], pa0, pa1, pa2, pa3, b0, b1);
            mma16x8x16(accO[2 * dt + 1], pa0, pa1, pa2, pa3, b2, b3);
        }

        if (hasNext)
        {
            __syncwarp();
            if constexpr (kInt8)
            {
                storeStage(nst);
            }
            else
            {
                cpAsyncCommitWaitAll();
            }
            __syncwarp();
        }
    }

    // Merge the four warps per row: (m, l, acc).
    __syncthreads();
    float* const accS = reinterpret_cast<float*>(smemRaw); // reuses the stage slices
    constexpr int32_t kAccStride = kInkling2HeadDim + 8; // 136 floats: the 8-byte stores of the 8 accumulator rows of a half-warp hit distinct banks
    float* const mS = accS + kMmaWarps * kMultiRows * kAccStride;
    float* const lS = mS + kMmaWarps * kMultiRows;
#pragma unroll
    for (int32_t hh = 0; hh < 2; ++hh)
    {
        int32_t const r = gid + 8 * hh;
        if (live[hh])
        {
#pragma unroll
            for (int32_t i = 0; i < 16; ++i)
            {
                *reinterpret_cast<float2*>(&accS[(w * kMultiRows + r) * kAccStride + i * 8 + 2 * tig]) = make_float2(accO[i][hh * 2], accO[i][hh * 2 + 1]);
            }
            if (tig == 0)
            {
                mS[w * kMultiRows + r] = rowM[hh];
                lS[w * kMultiRows + r] = rowL[hh];
            }
        }
    }
    // The per-row merge factors are the same for all 128 channels: one thread per row computes them once (and the row's partial index), the channels then only multiply.
    float* const cfS = lS + kMmaWarps * kMultiRows;      // [row][warp] factors
    float* const mmS = cfS + kMultiRows * kMmaWarps;     // [row] merged max
    float* const denS = mmS + kMultiRows;                // [row] merged denominator
    int32_t* const outIdx = reinterpret_cast<int32_t*>(denS + kMultiRows); // [row] (slot * heads + head) of the partial row
    __syncthreads();
    if (tid < kMultiRows)
    {
        int32_t const row = tid;
        float mm = -INFINITY;
#pragma unroll
        for (int32_t ww = 0; ww < kMmaWarps; ++ww)
        {
            mm = fmaxf(mm, mS[ww * kMultiRows + row]);
        }
        float den = 0.F;
#pragma unroll
        for (int32_t ww = 0; ww < kMmaWarps; ++ww)
        {
            float const mw = mS[ww * kMultiRows + row];
            float const cf = (mm == -INFINITY || mw == -INFINITY) ? 0.F : __expf(mw - mm);
            cfS[row * kMmaWarps + ww] = cf;
            den += lS[ww * kMultiRows + row] * cf;
        }
        mmS[row] = mm;
        denS[row] = den;
        int32_t const tl = row / group, hh2 = row - tl * group;
        outIdx[row] = (slot0 + tl) * p.numQHeads + kvh * group + hh2;
    }
    __syncthreads();
    // Warp w merges rows w, w + 4, ...: a lane takes 4 consecutive channels (16-byte shared-memory reads, two 8-byte stores).
#pragma unroll
    for (int32_t ri = 0; ri < kMultiRows / kMmaWarps; ++ri)
    {
        int32_t const row = w + ri * kMmaWarps;
        if (rowHi[row] < rowLo[row])
        {
            continue;
        }
        float4 num = make_float4(0.F, 0.F, 0.F, 0.F);
#pragma unroll
        for (int32_t ww = 0; ww < kMmaWarps; ++ww)
        {
            float const cf = cfS[row * kMmaWarps + ww];
            float4 const a4 = *reinterpret_cast<float4 const*>(&accS[(ww * kMultiRows + row) * kAccStride + lane * 4]);
            num.x += a4.x * cf;
            num.y += a4.y * cf;
            num.z += a4.z * cf;
            num.w += a4.w * cf;
        }
        float* out = partials + (int64_t(outIdx[row]) * splits + split) * kPartialStride;
        *reinterpret_cast<float2*>(out + 2 + lane * 4) = make_float2(num.x, num.y);
        *reinterpret_cast<float2*>(out + 2 + lane * 4 + 2) = make_float2(num.z, num.w);
        if (lane == 0)
        {
            out[0] = mmS[row];
            out[1] = denS[row];
        }
    }
}

template <typename KvT>
static void launchTyped(Inkling2AttentionParams const& params, half const* qLin, half const* relLin, half const* sinkLin,
    half const* gateLin, float const* qScale, float const* relScale, float const* relProjT, half const* kNew,
    half const* vNew, KvT* kvPool, int32_t const* queryStartOffsets, int32_t const* queryLengths,
    int32_t const* pastLengths, int32_t const* pageTable, half* out, void* workspace, size_t workspaceBytes,
    cudaStream_t stream)
{
    if constexpr (std::is_same_v<KvT, int8_t>)
    {
        appendInt8Kernel<<<params.numTokens, 256, 0, stream>>>(params, kNew, vNew, kvPool, queryStartOffsets, queryLengths, pastLengths, pageTable);
    }
    else
    {
        appendKernel<KvT><<<params.numTokens, 128, 0, stream>>>(
            params, kNew, vNew, kvPool, queryStartOffsets, queryLengths, pastLengths, pageTable);
    }
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
        cudaFuncSetAttribute(tiledAttentionKernel<int8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(tiledSmemBytes()));
        return true;
    }();
    static_cast<void>(configured);
    if (group <= kDecRows && params.numTokens <= 8)
    {
        static bool const decodeConfigured = []()
        {
            cudaFuncSetAttribute(decodeAttentionKernel<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(decodeSmemBytes()));
            cudaFuncSetAttribute(decodeAttentionKernel<__nv_fp8_e4m3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(decodeSmemBytes()));
            cudaFuncSetAttribute(decodeAttentionKernel<int8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(decodeSmemBytes()));
            return true;
        }();
        static_cast<void>(decodeConfigured);
        static int32_t const sms = []()
        {
            int32_t n = 0;
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
            return std::max(n, 1);
        }();
        // enough blocks for about eight per SM, bounded by the key tiles the table can hold and by the workspace
        int64_t splits = (8LL * sms + int64_t(params.numTokens) * params.numKVHeads - 1) / (int64_t(params.numTokens) * params.numKVHeads);
        splits = std::min<int64_t>(splits, std::max<int64_t>(1, int64_t(params.maxPagesPerSeq) * kInkling2TokensPerPage / kDecKeys));
        splits = std::min<int64_t>(splits, 64);
        splits = std::max<int64_t>(1, std::min<int64_t>(splits, rowCap / (int64_t(params.numTokens) * params.numQHeads)));
        // Tensor-core kernel for FP16 and INT8 KV (INKLING2_NO_DECODE_MMA=1 selects the scalar kernel); FP8 pools keep the scalar kernel.
        static bool const decodeMma = std::getenv("INKLING2_NO_DECODE_MMA") == nullptr;
        if constexpr (!std::is_same_v<KvT, __nv_fp8_e4m3>)
        {
            if (decodeMma)
            {
                static bool const decodeMmaConfigured = []()
                {
                    cudaFuncSetAttribute(decodeMmaKernel<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(decodeMmaSmemBytes(false)));
                    cudaFuncSetAttribute(decodeMmaKernel<int8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(decodeMmaSmemBytes(true)));
                    return true;
                }();
                static_cast<void>(decodeMmaConfigured);
                // about `INKLING2_DECODE_BLOCKS_PER_SM` blocks per SM (two fit by shared memory), never more splits than 64-key tiles exist
                static int32_t const blocksPerSm = []()
                {
                    char const* v = std::getenv("INKLING2_DECODE_BLOCKS_PER_SM");
                    return v != nullptr ? std::max(1, std::atoi(v)) : 4;
                }();
                // Verification windows of one sequence (2 to 16 tokens, group rows x tokens <= 16): one block serves several query tokens, so each key tile is streamed once
                // (INKLING2_NO_DECODE_MULTI=1 turns it off for A/B).
                static bool const multiOff = std::getenv("INKLING2_NO_DECODE_MULTI") != nullptr;
                if (!multiOff && params.batchSize == 1 && params.numTokens >= 2 && group >= 1 && group <= kMultiRows / 2)
                {
                    static bool const multiConfigured = []()
                    {
                        cudaFuncSetAttribute(decodeMmaMultiKernel<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(decodeMultiSmemBytes(false)));
                        cudaFuncSetAttribute(decodeMmaMultiKernel<int8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(decodeMultiSmemBytes(true)));
                        return true;
                    }();
                    static_cast<void>(multiConfigured);
                    int64_t const tb = kMultiRows / group;
                    int64_t const gridX = (params.numTokens + tb - 1) / tb;
                    // Few splits suit the usual contexts (the fixed per-block work repeats for every split); the default keeps every block resident in one wave
                    // (2 resident blocks per SM), INKLING2_MULTI_SPLITS overrides it.
                    static int32_t const multiSplitsEnv = []()
                    {
                        char const* v = std::getenv("INKLING2_MULTI_SPLITS");
                        return v != nullptr ? std::max(0, std::atoi(v)) : 0;
                    }();
                    int64_t multiSplits = multiSplitsEnv > 0
                        ? multiSplitsEnv
                        : (int64_t(blocksPerSm) * sms + gridX * params.numKVHeads - 1) / (gridX * params.numKVHeads);
                    multiSplits = std::min<int64_t>(multiSplits, std::max<int64_t>(1, int64_t(params.maxPagesPerSeq) * kInkling2TokensPerPage / kMmaTile));
                    multiSplits = std::max<int64_t>(1, std::min<int64_t>(multiSplits, rowCap / (int64_t(params.numTokens) * params.numQHeads)));
                    decodeMmaMultiKernel<KvT><<<dim3(static_cast<unsigned>(gridX), params.numKVHeads, static_cast<unsigned>(multiSplits)), kMmaThreads,
                        decodeMultiSmemBytes(std::is_same_v<KvT, int8_t>), stream>>>(params, qLin, relLin, qScale, relScale, relProjT, kvPool, queryStartOffsets,
                        queryLengths, pastLengths, pageTable, static_cast<float*>(workspace), static_cast<int32_t>(multiSplits));
                    mergeKernel<<<dim3(params.numTokens, params.numQHeads), kInkling2HeadDim, 0, stream>>>(params, sinkLin, gateLin, queryStartOffsets,
                        queryLengths, pastLengths, static_cast<float const*>(workspace), static_cast<int32_t>(multiSplits), out);
                    return;
                }
                int64_t mmaSplits = (int64_t(blocksPerSm) * sms + int64_t(params.numTokens) * params.numKVHeads - 1) / (int64_t(params.numTokens) * params.numKVHeads);
                mmaSplits = std::min<int64_t>(mmaSplits, std::max<int64_t>(1, int64_t(params.maxPagesPerSeq) * kInkling2TokensPerPage / kMmaTile));
                mmaSplits = std::max<int64_t>(1, std::min<int64_t>(mmaSplits, rowCap / (int64_t(params.numTokens) * params.numQHeads)));
                decodeMmaKernel<KvT><<<dim3(params.numTokens, params.numKVHeads, static_cast<unsigned>(mmaSplits)), kMmaThreads, decodeMmaSmemBytes(std::is_same_v<KvT, int8_t>), stream>>>(
                    params, qLin, relLin, qScale, relScale, relProjT, kvPool, queryStartOffsets, queryLengths, pastLengths, pageTable,
                    static_cast<float*>(workspace), static_cast<int32_t>(mmaSplits));
                mergeKernel<<<dim3(params.numTokens, params.numQHeads), kInkling2HeadDim, 0, stream>>>(params, sinkLin, gateLin, queryStartOffsets,
                    queryLengths, pastLengths, static_cast<float const*>(workspace), static_cast<int32_t>(mmaSplits), out);
                return;
            }
        }
        decodeAttentionKernel<KvT><<<dim3(params.numTokens, params.numKVHeads, static_cast<unsigned>(splits)), kDecThreads, decodeSmemBytes(), stream>>>(
            params, qLin, relLin, qScale, relScale, relProjT, kvPool, queryStartOffsets, queryLengths, pastLengths, pageTable,
            static_cast<float*>(workspace), static_cast<int32_t>(splits));
        mergeKernel<<<dim3(params.numTokens, params.numQHeads), kInkling2HeadDim, 0, stream>>>(params, sinkLin, gateLin, queryStartOffsets,
            queryLengths, pastLengths, static_cast<float const*>(workspace), static_cast<int32_t>(splits), out);
        return;
    }
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
    static bool const useMma = std::getenv("INKLING2_NO_MMA") == nullptr;
    if (useMma)
    {
        static bool const mmaConfigured = []()
        {
            cudaFuncSetAttribute(tiledAttentionMmaKernel<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mmaSmemBytes()));
            cudaFuncSetAttribute(tiledAttentionMmaKernel<__nv_fp8_e4m3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(mmaSmemBytes()));
            cudaFuncSetAttribute(tiledAttentionMmaKernel<int8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mmaSmemBytes()));
            return true;
        }();
        static_cast<void>(mmaConfigured);
        tiledAttentionMmaKernel<KvT><<<dim3(tiles, params.numKVHeads, static_cast<unsigned>(splits)), kTileThreads, mmaSmemBytes(), stream>>>(
            params, qLin, relLin, qScale, relScale, relProjT, kvPool, queryStartOffsets, queryLengths, pastLengths, pageTable,
            static_cast<float*>(workspace), qTile, static_cast<int32_t>(splits));
    }
    else
    {
        tiledAttentionKernel<KvT><<<dim3(tiles, params.numKVHeads, static_cast<unsigned>(splits)), kTileThreads,
            tiledSmemBytes(), stream>>>(params, qLin, relLin, qScale, relScale, relProjT, kvPool, queryStartOffsets,
            queryLengths, pastLengths, pageTable, static_cast<float*>(workspace), qTile, static_cast<int32_t>(splits));
    }
    mergeKernel<<<dim3(params.numTokens, params.numQHeads), kInkling2HeadDim, 0, stream>>>(params, sinkLin, gateLin,
        queryStartOffsets, queryLengths, pastLengths, static_cast<float const*>(workspace),
        static_cast<int32_t>(splits), out);
}

size_t inkling2AttentionWorkspaceBytes(int64_t maxTokens, int32_t numQHeads)
{
    int64_t const rows = std::max<int64_t>(maxTokens * numQHeads, 1024);
    return static_cast<size_t>(rows) * kPartialStride * sizeof(float);
}

void launchInkling2Attention(Inkling2AttentionParams const& callerParams, half const* qLin, half const* relLin,
    half const* sinkLin, half const* gateLin, float const* qScale, float const* relScale, float const* relProjT,
    half const* kNew, half const* vNew, void* kvPool, int32_t const* queryStartOffsets, int32_t const* queryLengths,
    int32_t const* pastLengths, int32_t const* pageTable, half* out, void* workspace, size_t workspaceBytes,
    cudaStream_t stream)
{
    if (callerParams.numTokens <= 0)
    {
        return;
    }
    Inkling2AttentionParams params = callerParams;
    if (params.qLd == 0)
    {
        params.qLd = params.numQHeads * kInkling2HeadDim;
    }
    if (params.relLd == 0)
    {
        params.relLd = params.numQHeads * kInkling2RelDim;
    }
    if (params.sinkLd == 0)
    {
        params.sinkLd = params.numQHeads;
    }
    if (params.gateLd == 0)
    {
        params.gateLd = params.numQHeads * kInkling2HeadDim;
    }
    if (params.kvInt8 != 0)
    {
        launchTyped(params, qLin, relLin, sinkLin, gateLin, qScale, relScale, relProjT, kNew, vNew, static_cast<int8_t*>(kvPool), queryStartOffsets,
            queryLengths, pastLengths, pageTable, out, workspace, workspaceBytes, stream);
    }
    else if (params.kvFp8 != 0)
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
