/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "inklingDflash2Kernels.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr float kEps = 1e-6F;

template <int N>
__device__ __forceinline__ void blockSumN(float (&v)[N], float* smem)
{
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int n = 0; n < N; ++n)
    {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            v[n] += __shfl_xor_sync(0xffffffffu, v[n], off);
        }
    }
    __syncthreads();
    if (lane == 0)
    {
#pragma unroll
        for (int n = 0; n < N; ++n)
        {
            smem[n * 32 + warp] = v[n];
        }
    }
    __syncthreads();
#pragma unroll
    for (int n = 0; n < N; ++n)
    {
        float s = 0.F;
        for (int w = 0; w < warps; ++w)
        {
            s += smem[n * 32 + w];
        }
        v[n] = s;
    }
}

__device__ __forceinline__ float ropeFreq(int j, float theta)
{
    return exp2f(-(static_cast<float>(j) / 64.F) * log2f(theta));
}

__global__ void ctxPrepKernel(float const* taps, float const* finalNorm, __half* out)
{
    __shared__ float red[32];
    size_t const row = blockIdx.x;
    float const* in = taps + row * 4 * kDfl2Hidden;
    __half* o = out + row * 4 * kDfl2Hidden;
    for (int i = threadIdx.x; i < 3 * kDfl2Hidden; i += blockDim.x)
    {
        o[i] = __float2half(in[i]);
    }
    float ss[1] = {0.F};
    for (int i = threadIdx.x; i < kDfl2Hidden; i += blockDim.x)
    {
        float const v = in[3 * kDfl2Hidden + i];
        ss[0] += v * v;
    }
    blockSumN<1>(ss, red);
    float const inv = rsqrtf(ss[0] / kDfl2Hidden + kEps);
    for (int i = threadIdx.x; i < kDfl2Hidden; i += blockDim.x)
    {
        o[3 * kDfl2Hidden + i] = __float2half(in[3 * kDfl2Hidden + i] * inv * finalNorm[i]);
    }
}

__global__ void rmsHalfKernel(__half const* x, float const* w, __half* out)
{
    __shared__ float red[32];
    size_t const r = blockIdx.x;
    float ss[1] = {0.F};
    for (int i = threadIdx.x; i < kDfl2Hidden; i += blockDim.x)
    {
        float const v = __half2float(x[r * kDfl2Hidden + i]);
        ss[0] += v * v;
    }
    blockSumN<1>(ss, red);
    float const inv = rsqrtf(ss[0] / kDfl2Hidden + kEps);
    for (int i = threadIdx.x; i < kDfl2Hidden; i += blockDim.x)
    {
        out[r * kDfl2Hidden + i] = __float2half(__half2float(x[r * kDfl2Hidden + i]) * inv * w[i]);
    }
}

constexpr int kQStride = kDfl2Heads * kDfl2HeadDim;
constexpr int kKvStride = kDfl2KvHeads * kDfl2HeadDim;

__global__ void qkvPrepKernel(__half const* qRaw, __half const* kRaw, __half const* vRaw, float const* qNorm, float const* kNorm, int32_t pos0, float theta, __half* q,
    __half* kb, __half* vb)
{
    __shared__ float sm[kDfl2HeadDim];
    __shared__ float red[32];
    int const s = blockIdx.x;
    int const head = blockIdx.y;
    int const d = threadIdx.x;
    if (head >= kDfl2Heads + kDfl2KvHeads)
    {
        int const h = head - kDfl2Heads - kDfl2KvHeads;
        vb[static_cast<size_t>(s) * kKvStride + h * kDfl2HeadDim + d] = vRaw[static_cast<size_t>(s) * kKvStride + h * kDfl2HeadDim + d];
        return;
    }
    bool const isQ = head < kDfl2Heads;
    float const raw = isQ ? __half2float(qRaw[static_cast<size_t>(s) * kQStride + head * kDfl2HeadDim + d])
                          : __half2float(kRaw[static_cast<size_t>(s) * kKvStride + (head - kDfl2Heads) * kDfl2HeadDim + d]);
    float ss[1] = {raw * raw};
    blockSumN<1>(ss, red);
    float const x = raw * rsqrtf(ss[0] / kDfl2HeadDim + kEps) * (isQ ? qNorm[d] : kNorm[d]);
    sm[d] = x;
    __syncthreads();
    float const partner = d < 64 ? sm[d + 64] : sm[d - 64];
    float const ang = static_cast<float>(pos0 + s) * ropeFreq(d & 63, theta);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float const out = d < 64 ? x * cs - partner * sn : x * cs + partner * sn;
    if (isQ)
    {
        q[static_cast<size_t>(s) * kQStride + head * kDfl2HeadDim + d] = __float2half(out);
    }
    else
    {
        kb[static_cast<size_t>(s) * kKvStride + (head - kDfl2Heads) * kDfl2HeadDim + d] = __float2half(out);
    }
}

constexpr int kScoreThreads = 128;                  //!< 32 key groups of 4 keys x 4 query groups of 8 query vectors
constexpr int kKeysPerBlock = 128;
constexpr int kQsStride = kDfl2HeadDim + 1;         //!< odd: the four query groups of a warp read distinct banks

//! Register-tiled scores: a thread owns 4 keys x 8 query vectors, so each loaded K value feeds 8 FMAs and each shared-memory q value 4.
__global__ void attnScoresKernel(__half const* q, __half const* ctxK, __half const* kb, int32_t lo, int32_t nCtx, int32_t qpos0, int32_t window, float* scores, int32_t stride)
{
    __shared__ float qs[kDfl2QueryRows][kQsStride];
    int const g = blockIdx.y;
    for (int idx = threadIdx.x; idx < kDfl2QueryRows * kDfl2HeadDim; idx += blockDim.x)
    {
        int const qi = idx / kDfl2HeadDim;
        int const d = idx % kDfl2HeadDim;
        int const s = qi / kDfl2Group;
        int const hh = qi % kDfl2Group;
        qs[qi][d] = __half2float(q[static_cast<size_t>(s) * kQStride + (g * kDfl2Group + hh) * kDfl2HeadDim + d]) * 0.08838834764831845F;
    }
    __syncthreads();
    int const nk = nCtx + kDfl2Block;
    int const qg = threadIdx.x & 3;
    int const kk0 = blockIdx.x * kKeysPerBlock + (threadIdx.x >> 2) * 4;
    if (kk0 >= nk)
    {
        return;
    }
    __half const* krow[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        int const kk = min(kk0 + j, nk - 1);
        krow[j] = kk < nCtx ? ctxK + static_cast<size_t>(lo + kk) * kKvStride + g * kDfl2HeadDim : kb + static_cast<size_t>(kk - nCtx) * kKvStride + g * kDfl2HeadDim;
    }
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            acc[i][j] = 0.F;
        }
    }
#pragma unroll 1
    for (int dc = 0; dc < kDfl2HeadDim / 8; ++dc)
    {
        float kf[4][8];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint4 const raw = *reinterpret_cast<uint4 const*>(krow[j] + dc * 8);
            __half2 const* h2 = reinterpret_cast<__half2 const*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 const f = __half22float2(h2[i]);
                kf[j][2 * i] = f.x;
                kf[j][2 * i + 1] = f.y;
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
#pragma unroll
            for (int qq = 0; qq < 8; ++qq)
            {
                float const qv = qs[qg * 8 + qq][dc * 8 + i];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    acc[qq][j] = fmaf(qv, kf[j][i], acc[qq][j]);
                }
            }
        }
    }
#pragma unroll
    for (int qq = 0; qq < 8; ++qq)
    {
        int const qi = qg * 8 + qq;
        float out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int const kk = kk0 + j;
            bool const visible = kk >= nCtx || (qpos0 + qi / kDfl2Group - (lo + kk) < window);
            out[j] = visible ? acc[qq][j] : -INFINITY;
        }
        float* dst = scores + static_cast<size_t>(g * kDfl2QueryRows + qi) * stride + kk0;
        if (kk0 + 3 < nk)
        {
            *reinterpret_cast<float4*>(dst) = make_float4(out[0], out[1], out[2], out[3]);
        }
        else
        {
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                if (kk0 + j < nk)
                {
                    dst[j] = out[j];
                }
            }
        }
    }
}

__global__ void attnSoftmaxKernel(float* scores, int32_t stride, int32_t nk)
{
    __shared__ float red[64];
    float* row = scores + static_cast<size_t>(blockIdx.x) * stride;
    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < nk; i += blockDim.x)
    {
        mx = fmaxf(mx, row[i]);
    }
    for (int off = 16; off > 0; off >>= 1)
    {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    if ((threadIdx.x & 31) == 0)
    {
        red[threadIdx.x >> 5] = mx;
    }
    __syncthreads();
    mx = red[0];
    for (int w = 1; w < (blockDim.x >> 5); ++w)
    {
        mx = fmaxf(mx, red[w]);
    }
    __syncthreads();
    float sum[1] = {0.F};
    for (int i = threadIdx.x; i < nk; i += blockDim.x)
    {
        float const e = __expf(row[i] - mx);
        row[i] = e;
        sum[0] += e;
    }
    blockSumN<1>(sum, red);
    float const inv = 1.F / sum[0];
    for (int i = threadIdx.x; i < nk; i += blockDim.x)
    {
        row[i] *= inv;
    }
}

//! Register-tiled P x V: thread (tq, td) owns 4 query vectors (4 tq .. 4 tq + 3) x 8 head dimensions (8 td .. 8 td + 7): per key 4 probabilities and one 16-byte V load feed 32 FMAs.
__global__ void attnPvKernel(float const* scores, int32_t stride, __half const* ctxV, __half const* vb, int32_t lo, int32_t nCtx, int32_t splits, float* partial)
{
    int const g = blockIdx.x;
    int const split = blockIdx.y;
    int const tq = threadIdx.x >> 4;
    int const td = threadIdx.x & 15;
    int const nk = nCtx + kDfl2Block;
    int const chunk = (nk + splits - 1) / splits;
    int const k0 = split * chunk;
    int const k1 = min(nk, k0 + chunk);
    float acc[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            acc[i][j] = 0.F;
        }
    }
    float const* prow[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        prow[i] = scores + static_cast<size_t>(g * kDfl2QueryRows + tq * 4 + i) * stride;
    }
    for (int k = k0; k < k1; ++k)
    {
        __half const* vrow = k < nCtx ? ctxV + static_cast<size_t>(lo + k) * kKvStride + g * kDfl2HeadDim : vb + static_cast<size_t>(k - nCtx) * kKvStride + g * kDfl2HeadDim;
        uint4 const raw = *reinterpret_cast<uint4 const*>(vrow + td * 8);
        __half2 const* h2 = reinterpret_cast<__half2 const*>(&raw);
        float vf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 const f = __half22float2(h2[i]);
            vf[2 * i] = f.x;
            vf[2 * i + 1] = f.y;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float const pv = prow[i][k];
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                acc[i][j] = fmaf(pv, vf[j], acc[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float* dst = partial + ((static_cast<size_t>(split) * kDfl2KvHeads + g) * kDfl2QueryRows + tq * 4 + i) * kDfl2HeadDim + td * 8;
        *reinterpret_cast<float4*>(dst) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(dst + 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

__global__ void attnReduceKernel(float const* partial, int32_t splits, __half* out)
{
    int const g = blockIdx.x / kDfl2QueryRows;
    int const qi = blockIdx.x % kDfl2QueryRows;
    int const d = threadIdx.x;
    float s = 0.F;
    for (int sp = 0; sp < splits; ++sp)
    {
        s += partial[((static_cast<size_t>(sp) * kDfl2KvHeads + g) * kDfl2QueryRows + qi) * kDfl2HeadDim + d];
    }
    int const row = qi / kDfl2Group;
    int const hh = qi % kDfl2Group;
    out[static_cast<size_t>(row) * kQStride + (g * kDfl2Group + hh) * kDfl2HeadDim + d] = __float2half(s);
}


// ---- Fused tensor-core attention of the drafter block (replaces scores + softmax + P x V + reduce: four launches that were latency bound at the drafter's context sizes) ----
// One block per (KV head, key split), four warps: warp (rt, kh) owns the 16 query vectors 16 rt .. 16 rt + 15 of the head's 32 (8 block rows x 4 heads) and the keys of
// 16-key sub-tile kh of every 32-key tile. Each warp streams its keys through a private double-buffered shared-memory slice (cp.async), scores with mma.m16n8k16, keeps a
// warp-private online softmax and accumulates P x V with FP16 P; the two key halves of a row tile and the splits are merged exactly (m, l, acc) at the end.
constexpr int kAWarps = 4;
constexpr int kAThreads = 128;
constexpr int kAWarpKeys = 16;
constexpr int kATile = 32;                              //!< keys per block iteration
constexpr int kAKP = kDfl2HeadDim + 8;                  //!< staged row of 136 halves: conflict-free for ldmatrix
constexpr int kASlice = 2 * kAWarpKeys * kAKP;          //!< halves of one stage (K then V)
constexpr int kAPartialStride = kDfl2HeadDim + 2;       //!< m, l, acc[128] per (split, KV head, query vector)
constexpr int kAMaxSplits = 14;
constexpr size_t kASmemBytes = sizeof(__half) * kASlice * 2 * kAWarps;

__device__ __forceinline__ void aMma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void aLdmatrix(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, void const* p)
{
    uint32_t const addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void aLdmatrixTrans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, void const* p)
{
    uint32_t const addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void aCpAsync16(void* dst, void const* src, bool valid)
{
    uint32_t const addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    int32_t const srcSize = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(addr), "l"(src), "r"(srcSize));
}

__device__ __forceinline__ void aCpAsyncWaitAll()
{
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ uint32_t aPackHalf2(float lo, float hi)
{
    __half2 const h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t const*>(&h);
}

__global__ void __launch_bounds__(kAThreads, 2) attnFusedKernel(__half const* q, __half const* ctxK, __half const* ctxV, __half const* kb, __half const* vb, int32_t lo,
    int32_t nCtx, int32_t qpos0, int32_t window, int32_t splits, float* partial)
{
    extern __shared__ __align__(16) unsigned char smemRaw[];
    __half* const slices = reinterpret_cast<__half*>(smemRaw);
    int32_t const tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    int32_t const gid = lane >> 2, tig = lane & 3;
    int32_t const g = blockIdx.x, split = blockIdx.y;
    int32_t const rt = w & 1, kh = w >> 1;
    int32_t const nk = nCtx + kDfl2Block;
    int32_t const nTiles = (nk + kATile - 1) / kATile;
    int32_t const perSplit = (nTiles + splits - 1) / splits;
    int32_t const tBegin = min(nTiles, split * perSplit), tEnd = min(nTiles, tBegin + perSplit);
    __half* const warpSlices = slices + static_cast<size_t>(w) * 2 * kASlice;

    // Q fragments of this warp's 16 query vectors (A operand rows gid and gid + 8)
    uint32_t qa[8][4];
    {
        int32_t const qiA = rt * 16 + gid, qiB = qiA + 8;
        __half const* rowA = q + static_cast<size_t>(qiA / kDfl2Group) * kQStride + (g * kDfl2Group + qiA % kDfl2Group) * kDfl2HeadDim;
        __half const* rowB = q + static_cast<size_t>(qiB / kDfl2Group) * kQStride + (g * kDfl2Group + qiB % kDfl2Group) * kDfl2HeadDim;
#pragma unroll
        for (int32_t ks = 0; ks < 8; ++ks)
        {
            qa[ks][0] = *reinterpret_cast<uint32_t const*>(rowA + ks * 16 + 2 * tig);
            qa[ks][1] = *reinterpret_cast<uint32_t const*>(rowB + ks * 16 + 2 * tig);
            qa[ks][2] = *reinterpret_cast<uint32_t const*>(rowA + ks * 16 + 8 + 2 * tig);
            qa[ks][3] = *reinterpret_cast<uint32_t const*>(rowB + ks * 16 + 8 + 2 * tig);
        }
    }
    float accO[16][4];
#pragma unroll
    for (int32_t i = 0; i < 16; ++i)
    {
        accO[i][0] = accO[i][1] = accO[i][2] = accO[i][3] = 0.F;
    }
    float rowM[2] = {-INFINITY, -INFINITY}, rowL[2] = {0.F, 0.F};
    int32_t const sTok[2] = {(rt * 16 + gid) / kDfl2Group, (rt * 16 + gid + 8) / kDfl2Group}; // block row (token) of the two accumulator rows

    bool const isV = lane >= 16;
    int32_t const chunk = (lane & 15) * 8;
    auto issue = [&](int32_t t, int32_t st)
    {
        int32_t const base = t * kATile + kh * kAWarpKeys;
        __half* const dst = warpSlices + st * kASlice + (isV ? kAWarpKeys * kAKP : 0) + chunk;
#pragma unroll
        for (int32_t kk = 0; kk < kAWarpKeys; ++kk)
        {
            int32_t const key = base + kk;
            __half const* src = nullptr;
            if (key < nCtx)
            {
                src = (isV ? ctxV : ctxK) + static_cast<size_t>(lo + key) * kKvStride + g * kDfl2HeadDim + chunk;
            }
            else
            {
                src = (isV ? vb : kb) + static_cast<size_t>(min(key, nk - 1) - nCtx) * kKvStride + g * kDfl2HeadDim + chunk;
            }
            aCpAsync16(dst + kk * kAKP, src, key < nk);
        }
    };
    if (tBegin < tEnd)
    {
        issue(tBegin, 0);
        aCpAsyncWaitAll();
        __syncwarp();
    }
    for (int32_t t = tBegin; t < tEnd; ++t)
    {
        int32_t const st = (t - tBegin) & 1;
        bool const hasNext = t + 1 < tEnd;
        if (hasNext)
        {
            issue(t + 1, st ^ 1);
        }
        __half const* const kS = warpSlices + st * kASlice;
        __half const* const vS = kS + kAWarpKeys * kAKP;
        float c[2][4] = {{0.F, 0.F, 0.F, 0.F}, {0.F, 0.F, 0.F, 0.F}};
#pragma unroll
        for (int32_t ks = 0; ks < 8; ++ks)
        {
            uint32_t b0, b1, b2, b3;
            int32_t const key = ((lane >> 4) & 1) * 8 + (lane & 7);
            int32_t const col = ks * 16 + ((lane >> 3) & 1) * 8;
            aLdmatrix(b0, b1, b2, b3, kS + key * kAKP + col);
            aMma(c[0], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
            aMma(c[1], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b2, b3);
        }
        int32_t const base = t * kATile + kh * kAWarpKeys;
        float sc[2][4];
        float tileMax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int32_t n = 0; n < 2; ++n)
        {
#pragma unroll
            for (int32_t j = 0; j < 2; ++j)
            {
                int32_t const key = base + n * 8 + 2 * tig + j;
#pragma unroll
                for (int32_t hh = 0; hh < 2; ++hh)
                {
                    bool const visible = key < nk && (key >= nCtx || (qpos0 + sTok[hh] - (lo + key) < window));
                    float const score = visible ? c[n][hh * 2 + j] * 0.08838834764831845F : -INFINITY;
                    sc[n][hh * 2 + j] = score;
                    tileMax[hh] = fmaxf(tileMax[hh], score);
                }
            }
        }
        float corr[2], e[2][4];
#pragma unroll
        for (int32_t hh = 0; hh < 2; ++hh)
        {
            tileMax[hh] = fmaxf(tileMax[hh], __shfl_xor_sync(0xFFFFFFFFU, tileMax[hh], 1));
            tileMax[hh] = fmaxf(tileMax[hh], __shfl_xor_sync(0xFFFFFFFFU, tileMax[hh], 2));
            float const mNew = fmaxf(rowM[hh], tileMax[hh]);
            corr[hh] = rowM[hh] == -INFINITY ? 0.F : __expf(rowM[hh] - mNew);
            float sum = 0.F;
#pragma unroll
            for (int32_t n = 0; n < 2; ++n)
            {
#pragma unroll
                for (int32_t j = 0; j < 2; ++j)
                {
                    float const sv = sc[n][hh * 2 + j];
                    e[n][hh * 2 + j] = sv == -INFINITY ? 0.F : __expf(sv - mNew);
                    sum += e[n][hh * 2 + j];
                }
            }
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
            sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2);
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
        uint32_t const pa0 = aPackHalf2(e[0][0], e[0][1]);
        uint32_t const pa1 = aPackHalf2(e[0][2], e[0][3]);
        uint32_t const pa2 = aPackHalf2(e[1][0], e[1][1]);
        uint32_t const pa3 = aPackHalf2(e[1][2], e[1][3]);
#pragma unroll
        for (int32_t dt = 0; dt < 8; ++dt)
        {
            uint32_t b0, b1, b2, b3;
            int32_t const key = ((lane >> 3) & 1) * 8 + (lane & 7);
            int32_t const col = dt * 16 + ((lane >> 4) & 1) * 8;
            aLdmatrixTrans(b0, b1, b2, b3, vS + key * kAKP + col);
            aMma(accO[2 * dt], pa0, pa1, pa2, pa3, b0, b1);
            aMma(accO[2 * dt + 1], pa0, pa1, pa2, pa3, b2, b3);
        }
        if (hasNext)
        {
            aCpAsyncWaitAll();
            __syncwarp();
        }
    }

    // merge the two key halves of every row tile, then write one (m, l, acc) per (split, KV head, query vector)
    __syncthreads();
    float* const accS = reinterpret_cast<float*>(smemRaw);              // [warp][16 rows][128 + 8]
    constexpr int kAccStride = kDfl2HeadDim + 8;
    float* const mS = accS + kAWarps * 16 * kAccStride;                // [warp][16]
    float* const lS = mS + kAWarps * 16;
#pragma unroll
    for (int32_t hh = 0; hh < 2; ++hh)
    {
        int32_t const r = gid + 8 * hh;
#pragma unroll
        for (int32_t i = 0; i < 16; ++i)
        {
            *reinterpret_cast<float2*>(&accS[(w * 16 + r) * kAccStride + i * 8 + 2 * tig]) = make_float2(accO[i][hh * 2], accO[i][hh * 2 + 1]);
        }
        if (tig == 0)
        {
            mS[w * 16 + r] = rowM[hh];
            lS[w * 16 + r] = rowL[hh];
        }
    }
    __syncthreads();
    int32_t const d = tid;
    for (int32_t qi = 0; qi < kDfl2QueryRows; ++qi)
    {
        int32_t const w0 = qi / 16, w1 = w0 + 2, r = qi % 16;
        float const m0 = mS[w0 * 16 + r], m1 = mS[w1 * 16 + r];
        float const mm = fmaxf(m0, m1);
        float const c0 = m0 == -INFINITY ? 0.F : __expf(m0 - mm);
        float const c1 = m1 == -INFINITY ? 0.F : __expf(m1 - mm);
        float* const out = partial + (static_cast<size_t>(split * kDfl2KvHeads + g) * kDfl2QueryRows + qi) * kAPartialStride;
        out[2 + d] = accS[(w0 * 16 + r) * kAccStride + d] * c0 + accS[(w1 * 16 + r) * kAccStride + d] * c1;
        if (d == 0)
        {
            out[0] = mm;
            out[1] = lS[w0 * 16 + r] * c0 + lS[w1 * 16 + r] * c1;
        }
    }
}

//! Merge of the splits: grid (KV heads x query vectors), one thread per channel.
__global__ void attnFusedReduceKernel(float const* partial, int32_t splits, __half* out)
{
    int32_t const g = blockIdx.x / kDfl2QueryRows;
    int32_t const qi = blockIdx.x % kDfl2QueryRows;
    int32_t const d = threadIdx.x;
    float mm = -INFINITY;
    for (int32_t sp = 0; sp < splits; ++sp)
    {
        mm = fmaxf(mm, partial[(static_cast<size_t>(sp * kDfl2KvHeads + g) * kDfl2QueryRows + qi) * kAPartialStride]);
    }
    float num = 0.F, den = 0.F;
    for (int32_t sp = 0; sp < splits; ++sp)
    {
        float const* p = partial + (static_cast<size_t>(sp * kDfl2KvHeads + g) * kDfl2QueryRows + qi) * kAPartialStride;
        float const cf = p[0] == -INFINITY ? 0.F : __expf(p[0] - mm);
        num += p[2 + d] * cf;
        den += p[1] * cf;
    }
    int32_t const row = qi / kDfl2Group, hh = qi % kDfl2Group;
    out[static_cast<size_t>(row) * kQStride + (g * kDfl2Group + hh) * kDfl2HeadDim + d] = __float2half(num / den);
}

__global__ void logitsToFloatKernel(__half const* in, float* out, int32_t cols, int32_t validCols)
{
    size_t const row = blockIdx.y;
    for (int32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x)
    {
        out[row * cols + c] = c < validCols ? __half2float(in[row * cols + c]) : -1e30F;
    }
}

__global__ void gatherValuesKernel(float const* logits, int32_t const* indices, int32_t cols, int32_t k, float* values)
{
    int const row = blockIdx.x;
    int const j = threadIdx.x;
    if (j < k)
    {
        values[row * k + j] = logits[static_cast<size_t>(row) * cols + indices[row * k + j]];
    }
}

} // namespace

void launchDfl2CtxPrep(float const* taps, int32_t rows, float const* targetFinalNorm, __half* out, cudaStream_t stream)
{
    ctxPrepKernel<<<rows, 256, 0, stream>>>(taps, targetFinalNorm, out);
}

void launchDfl2RmsHalf(__half const* x, float const* w, __half* out, int32_t rows, cudaStream_t stream)
{
    rmsHalfKernel<<<rows, 256, 0, stream>>>(x, w, out);
}

void launchDfl2QkvPrep(__half const* qRaw, __half const* kRaw, __half const* vRaw, float const* qNorm, float const* kNorm, int32_t pos0, float theta, __half* q, __half* kb,
    __half* vb, cudaStream_t stream)
{
    qkvPrepKernel<<<dim3(kDfl2Block, kDfl2Heads + 2 * kDfl2KvHeads), kDfl2HeadDim, 0, stream>>>(qRaw, kRaw, vRaw, qNorm, kNorm, pos0, theta, q, kb, vb);
}

void launchDfl2Attention(__half const* q, __half const* ctxK, __half const* ctxV, __half const* kb, __half const* vb, int32_t lo, int32_t nCtx, int32_t qpos0, int32_t window,
    float* scores, int32_t scoreStride, float* partial, int32_t maxSplits, __half* out, cudaStream_t stream)
{
    static bool const oldPath = std::getenv("EDGELLM_INKLING_DRAFTER_ATTN_OLD") != nullptr;
    int32_t const nk = nCtx + kDfl2Block;
    if (!oldPath)
    {
        static bool const configured = []()
        {
            cudaFuncSetAttribute(attnFusedKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kASmemBytes));
            return true;
        }();
        static_cast<void>(configured);
        static int32_t const maxFused = []()
        {
            char const* v = std::getenv("EDGELLM_INKLING_DRAFTER_ATTN_SPLITS");
            return v != nullptr ? std::max(1, std::min(kAMaxSplits, std::atoi(v))) : 3;
        }();
        int32_t const tiles = (nk + kATile - 1) / kATile;
        int32_t const splits = std::max(1, std::min(maxFused, (tiles + 1) / 2));
        attnFusedKernel<<<dim3(kDfl2KvHeads, splits), kAThreads, kASmemBytes, stream>>>(q, ctxK, ctxV, kb, vb, lo, nCtx, qpos0, window, splits, partial);
        attnFusedReduceKernel<<<kDfl2KvHeads * kDfl2QueryRows, kDfl2HeadDim, 0, stream>>>(partial, splits, out);
        return;
    }
    attnScoresKernel<<<dim3((nk + kKeysPerBlock - 1) / kKeysPerBlock, kDfl2KvHeads), kScoreThreads, 0, stream>>>(
        q, ctxK, kb, lo, nCtx, qpos0, window, scores, scoreStride);
    attnSoftmaxKernel<<<kDfl2KvHeads * kDfl2QueryRows, 256, 0, stream>>>(scores, scoreStride, nk);
    int32_t const splits = std::max(1, std::min(maxSplits, nk / 32));
    attnPvKernel<<<dim3(kDfl2KvHeads, splits), kDfl2HeadDim, 0, stream>>>(scores, scoreStride, ctxV, vb, lo, nCtx, splits, partial);
    attnReduceKernel<<<kDfl2KvHeads * kDfl2QueryRows, kDfl2HeadDim, 0, stream>>>(partial, splits, out);
}

// ---- Top-16 of the rows of FP16 logits (replaces the float conversion, the generic top-k and the value gather: 0.66 ms -> a few tens of us per call) ----
// Ordering key = (order-preserving bits of the value) << 32 | (0xFFFFFFFF - index): the maximum key is the largest value, ties go to the smaller index.
constexpr int32_t kTop16 = 16;
constexpr int32_t kTop16Chunks = 32;
constexpr int32_t kTop16Threads = 256;
constexpr int32_t kTop16StageTwoThreads = kTop16Chunks * kTop16;

__device__ __forceinline__ uint64_t top16Key(float v, int32_t idx)
{
    if (v != v)
    {
        v = -INFINITY;
    }
    uint32_t const u = __float_as_uint(v);
    uint32_t const ordered = (u & 0x80000000U) != 0U ? ~u : (u | 0x80000000U);
    return (static_cast<uint64_t>(ordered) << 32) | static_cast<uint64_t>(0xFFFFFFFFU - static_cast<uint32_t>(idx));
}

__device__ __forceinline__ float top16Value(uint64_t key)
{
    uint32_t const ordered = static_cast<uint32_t>(key >> 32);
    uint32_t const u = (ordered & 0x80000000U) != 0U ? (ordered & 0x7FFFFFFFU) : ~ordered;
    return __uint_as_float(u);
}

__device__ __forceinline__ uint64_t warpMaxKey(uint64_t k)
{
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1)
    {
        uint64_t const o = __shfl_xor_sync(0xFFFFFFFFU, k, off);
        k = o > k ? o : k;
    }
    return k;
}

//! Stage 1: grid (chunks, rows). Each block takes a slice of a row (16-byte loads) and keeps one 32-bit key per element in registers: the order-preserving 16 bits of
//! the FP16 value above (0xFFFF - the index inside the slice), so the largest key is the largest value and ties go to the smaller index. It extracts its 16 largest
//! by 16 block-wide maximum rounds; every thread caches its own maximum and only the winner rescans. The 16 results are written as 64-bit stage-2 keys.
__global__ void __launch_bounds__(kTop16Threads) top16Stage1Kernel(__half const* logits, int32_t cols, int32_t validCols, int32_t perChunk, uint64_t* partial)
{
    constexpr int32_t kVecPerThread = 4; // 4 x 8 halves per thread: 8,192 columns per chunk at most
    __shared__ uint32_t warpBest[kTop16Threads / 32];
    int32_t const row = blockIdx.y, chunk = blockIdx.x;
    int32_t const begin = chunk * perChunk;
    int32_t const end = min(validCols, begin + perChunk);
    __half const* rowPtr = logits + static_cast<size_t>(row) * cols;
    uint32_t keys[kVecPerThread * 8];
#pragma unroll
    for (int32_t v = 0; v < kVecPerThread; ++v)
    {
        int32_t const local0 = (v * kTop16Threads + threadIdx.x) * 8;
        if (begin + local0 < end)
        {
            uint4 const raw = *reinterpret_cast<uint4 const*>(rowPtr + begin + local0);
            uint32_t const words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
            for (int32_t j = 0; j < 8; ++j)
            {
                uint32_t bits = (words[j >> 1] >> ((j & 1) * 16)) & 0xFFFFU;
                if ((bits & 0x7FFFU) > 0x7C00U)
                {
                    bits = 0xFC00U; // NaN counts as -infinity
                }
                uint32_t const ordered = (bits & 0x8000U) != 0U ? (~bits & 0xFFFFU) : (bits | 0x8000U);
                keys[v * 8 + j] = begin + local0 + j < end ? (ordered << 16) | (0xFFFFU - static_cast<uint32_t>(local0 + j)) : 0U;
            }
        }
        else
        {
#pragma unroll
            for (int32_t j = 0; j < 8; ++j)
            {
                keys[v * 8 + j] = 0U;
            }
        }
    }
    int32_t const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint32_t cur = 0U;
#pragma unroll
    for (int32_t i = 0; i < kVecPerThread * 8; ++i)
    {
        cur = max(cur, keys[i]);
    }
    for (int32_t r = 0; r < kTop16; ++r)
    {
        uint32_t best = cur;
#pragma unroll
        for (int32_t off = 16; off > 0; off >>= 1)
        {
            best = max(best, __shfl_xor_sync(0xFFFFFFFFU, best, off));
        }
        if (lane == 0)
        {
            warpBest[warp] = best;
        }
        __syncthreads();
        uint32_t blockBest = warpBest[0];
#pragma unroll
        for (int32_t w = 1; w < kTop16Threads / 32; ++w)
        {
            blockBest = max(blockBest, warpBest[w]);
        }
        if (threadIdx.x == 0)
        {
            // stage-2 key: float value of the FP16 pattern and the global index
            uint32_t const ordered = blockBest >> 16;
            uint32_t const bits = (ordered & 0x8000U) != 0U ? (ordered & 0x7FFFU) : (~ordered & 0xFFFFU);
            float const value = __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
            int32_t const globalIdx = begin + static_cast<int32_t>(0xFFFFU - (blockBest & 0xFFFFU));
            partial[(static_cast<size_t>(row) * kTop16Chunks + chunk) * kTop16 + r] = blockBest == 0U ? 0ULL : top16Key(value, globalIdx);
        }
        if (cur == blockBest)
        {
            cur = 0U;
#pragma unroll
            for (int32_t i = 0; i < kVecPerThread * 8; ++i)
            {
                keys[i] = keys[i] == blockBest ? 0U : keys[i];
                cur = max(cur, keys[i]);
            }
        }
        __syncthreads();
    }
}

//! Stage 2: one block per row merges the chunks' 16 candidates (one key per thread) into the sorted top 16: indices and the logit values as floats.
__global__ void __launch_bounds__(kTop16StageTwoThreads) top16Stage2Kernel(uint64_t const* partial, int32_t* indices, float* values)
{
    __shared__ uint64_t warpBest[kTop16StageTwoThreads / 32];
    int32_t const row = blockIdx.x;
    uint64_t key = partial[static_cast<size_t>(row) * kTop16StageTwoThreads + threadIdx.x];
    int32_t const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int32_t r = 0; r < kTop16; ++r)
    {
        uint64_t best = warpMaxKey(key);
        if (lane == 0)
        {
            warpBest[warp] = best;
        }
        __syncthreads();
        uint64_t blockBest = warpBest[0];
#pragma unroll
        for (int32_t w = 1; w < kTop16StageTwoThreads / 32; ++w)
        {
            blockBest = warpBest[w] > blockBest ? warpBest[w] : blockBest;
        }
        if (threadIdx.x == 0)
        {
            indices[row * kTop16 + r] = static_cast<int32_t>(0xFFFFFFFFU - static_cast<uint32_t>(blockBest & 0xFFFFFFFFULL));
            values[row * kTop16 + r] = top16Value(blockBest);
        }
        key = key == blockBest ? 0ULL : key;
        __syncthreads();
    }
}


void launchDfl2LogitsToFloat(__half const* in, float* out, int32_t rows, int32_t cols, int32_t validCols, cudaStream_t stream)
{
    logitsToFloatKernel<<<dim3(256, rows), 256, 0, stream>>>(in, out, cols, validCols);
}

void launchDfl2GatherValues(float const* logits, int32_t const* indices, int32_t rows, int32_t cols, int32_t k, float* values, cudaStream_t stream)
{
    gatherValuesKernel<<<rows, 32, 0, stream>>>(logits, indices, cols, k, values);
}


void launchDfl2Top16(__half const* logits, int32_t rows, int32_t cols, int32_t validCols, uint64_t* workspace, int32_t* indices, float* values, cudaStream_t stream)
{
    int32_t const perChunk = ((validCols + kTop16Chunks - 1) / kTop16Chunks + 7) / 8 * 8;
    top16Stage1Kernel<<<dim3(kTop16Chunks, rows), kTop16Threads, 0, stream>>>(logits, cols, validCols, perChunk, workspace);
    top16Stage2Kernel<<<rows, kTop16StageTwoThreads, 0, stream>>>(workspace, indices, values);
}

size_t dfl2Top16WorkspaceBytes(int32_t rows)
{
    return static_cast<size_t>(rows) * kTop16Chunks * kTop16 * sizeof(uint64_t);
}

} // namespace kernel
} // namespace trt_edgellm
