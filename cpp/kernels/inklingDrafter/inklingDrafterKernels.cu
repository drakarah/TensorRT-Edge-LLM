/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "inklingDrafterKernels.h"
#include "kernels/fp8Linear/fp8LinearKernels.h"

#include <cstdlib>

#include <cfloat>
#include <algorithm>
#include <cmath>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr float kEps = 1e-6F;

//! Sum over the block of N values per thread (N independent sums); smem has >= N * 32 floats. Result in all threads.
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

__global__ void ctxMixKernel(float const* taps, int32_t rows, int32_t consumerStride, float const* alpha, float const* dflare, float const* dflareM, float const* tfn, __half* ctxn,
    float* lastFeat)
{
    __shared__ float mixed[4][kDrafterHidden];
    __shared__ float red[4 * 32];
    int const r = blockIdx.x;
    float const* t0 = taps + static_cast<size_t>(r) * 4 * kDrafterHidden;
    float const* t1 = t0 + kDrafterHidden;
    float const* t2 = t1 + kDrafterHidden;
    float const* t3 = t2 + kDrafterHidden;
    float ss3[1] = {0.F};
    for (int i = threadIdx.x; i < kDrafterHidden; i += blockDim.x)
    {
        ss3[0] += t3[i] * t3[i];
    }
    blockSumN<1>(ss3, red);
    float const inv3 = rsqrtf(ss3[0] / kDrafterHidden + kEps);
    float ssc[4] = {0.F, 0.F, 0.F, 0.F};
    for (int i = threadIdx.x; i < kDrafterHidden; i += blockDim.x)
    {
        float const v0 = t0[i];
        float const v1 = t1[i];
        float const v2 = t2[i];
        float const v3 = t3[i] * inv3 * tfn[i];
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            float const m = alpha[c * 4 + 0] * v0 + alpha[c * 4 + 1] * v1 + alpha[c * 4 + 2] * v2 + alpha[c * 4 + 3] * v3;
            mixed[c][i] = m;
            ssc[c] += m * m;
        }
    }
    blockSumN<4>(ssc, red);
    float invc[4];
#pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        invc[c] = rsqrtf(ssc[c] / kDrafterHidden + kEps);
    }
    for (int i = threadIdx.x; i < kDrafterHidden; i += blockDim.x)
    {
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            ctxn[(static_cast<size_t>(c) * consumerStride + r) * kDrafterHidden + i] = __float2half(mixed[c][i] * invc[c] * dflare[c * kDrafterHidden + i]);
        }
        if (lastFeat != nullptr && r == rows - 1)
        {
            lastFeat[i] = mixed[3][i] * invc[3] * dflareM[i];
        }
    }
}

__global__ void ctxKvKernel(__half const* kRaw, __half const* vRaw, int32_t pos0, float const* kNorm, float theta, __half* ctxK, __half* ctxV, int32_t ctxCap)
{
    __shared__ float sm[kDrafterHeadDim];
    __shared__ float red[32];
    int const r = blockIdx.x;
    int const h = blockIdx.y;
    int const d = threadIdx.x;
    size_t const src = static_cast<size_t>(r) * kDrafterKvHeads * kDrafterHeadDim + h * kDrafterHeadDim + d;
    int32_t const dstRow = ctxCap > 0 ? (pos0 + r) % ctxCap : pos0 + r; // ring buffer when the cache holds fewer rows than positions
    size_t const dst = static_cast<size_t>(dstRow) * kDrafterKvHeads * kDrafterHeadDim + h * kDrafterHeadDim + d;
    float const kv = __half2float(kRaw[src]);
    float ss[1] = {kv * kv};
    blockSumN<1>(ss, red);
    float const x = kv * rsqrtf(ss[0] / kDrafterHeadDim + kEps) * kNorm[d];
    sm[d] = x;
    __syncthreads();
    float const partner = d < 64 ? sm[d + 64] : sm[d - 64];
    float const ang = static_cast<float>(pos0 + r) * ropeFreq(d & 63, theta);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float const out = d < 64 ? x * cs - partner * sn : x * cs + partner * sn;
    ctxK[dst] = __float2half(out);
    ctxV[dst] = vRaw[src];
}

__global__ void initBlockKernel(__half const* embedding, float const* embedNorm, float const* maskEmbed, float* x)
{
    __shared__ float red[32];
    int const row = blockIdx.x;
    if (row == 0)
    {
        float ss[1] = {0.F};
        for (int i = threadIdx.x; i < kDrafterHidden; i += blockDim.x)
        {
            float const v = __half2float(embedding[i]);
            ss[0] += v * v;
        }
        blockSumN<1>(ss, red);
        float const inv = rsqrtf(ss[0] / kDrafterHidden + kEps);
        for (int i = threadIdx.x; i < kDrafterHidden; i += blockDim.x)
        {
            x[i] = __half2float(embedding[i]) * inv * embedNorm[i];
        }
    }
    else
    {
        for (int i = threadIdx.x; i < kDrafterHidden; i += blockDim.x)
        {
            x[static_cast<size_t>(row) * kDrafterHidden + i] = maskEmbed[i];
        }
    }
}

__global__ void rmsKernel(float const* x, float const* w, __half* out, int32_t rowOffset)
{
    __shared__ float red[32];
    int const r = blockIdx.x + rowOffset;
    float ss[1] = {0.F};
    for (int i = threadIdx.x; i < kDrafterHidden; i += blockDim.x)
    {
        float const v = x[static_cast<size_t>(r) * kDrafterHidden + i];
        ss[0] += v * v;
    }
    blockSumN<1>(ss, red);
    float const inv = rsqrtf(ss[0] / kDrafterHidden + kEps);
    for (int i = threadIdx.x; i < kDrafterHidden; i += blockDim.x)
    {
        out[static_cast<size_t>(r) * kDrafterHidden + i] = __float2half(x[static_cast<size_t>(r) * kDrafterHidden + i] * inv * w[i]);
    }
}

__global__ void qkvPrepKernel(__half const* qRaw, __half const* kRaw, __half const* vRaw, float const* qNorm, float const* kNorm, int32_t pos0, float theta, __half* q,
    __half* kb, __half* vb)
{
    __shared__ float sm[kDrafterHeadDim];
    __shared__ float red[32];
    int const s = blockIdx.x;
    int const head = blockIdx.y;
    int const d = threadIdx.x;
    if (head >= kDrafterHeads + kDrafterKvHeads)
    {
        int const h = head - kDrafterHeads - kDrafterKvHeads;
        vb[static_cast<size_t>(s) * 1024 + h * kDrafterHeadDim + d] = vRaw[static_cast<size_t>(s) * 1024 + h * kDrafterHeadDim + d];
        return;
    }
    bool const isQ = head < kDrafterHeads;
    float const raw = isQ ? __half2float(qRaw[static_cast<size_t>(s) * 3072 + head * kDrafterHeadDim + d])
                          : __half2float(kRaw[static_cast<size_t>(s) * 1024 + (head - kDrafterHeads) * kDrafterHeadDim + d]);
    float ss[1] = {raw * raw};
    blockSumN<1>(ss, red);
    float const x = raw * rsqrtf(ss[0] / kDrafterHeadDim + kEps) * (isQ ? qNorm[d] : kNorm[d]);
    sm[d] = x;
    __syncthreads();
    float const partner = d < 64 ? sm[d + 64] : sm[d - 64];
    float const ang = static_cast<float>(pos0 + s) * ropeFreq(d & 63, theta);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float const out = d < 64 ? x * cs - partner * sn : x * cs + partner * sn;
    if (isQ)
    {
        q[static_cast<size_t>(s) * 3072 + head * kDrafterHeadDim + d] = __float2half(out);
    }
    else
    {
        kb[static_cast<size_t>(s) * 1024 + (head - kDrafterHeads) * kDrafterHeadDim + d] = __float2half(out);
    }
}

constexpr int kScoreThreads = 128;

__global__ void attnScoresKernel(__half const* q, __half const* ctxK, __half const* kb, int32_t lo, int32_t nCtx, float* scores, int32_t stride)
{
    __shared__ float qs[24][kDrafterHeadDim];
    int const g = blockIdx.y;
    for (int idx = threadIdx.x; idx < 24 * kDrafterHeadDim; idx += blockDim.x)
    {
        int const qi = idx / kDrafterHeadDim;
        int const d = idx % kDrafterHeadDim;
        int const s = qi / 3;
        int const hh = qi % 3;
        qs[qi][d] = __half2float(q[static_cast<size_t>(s) * 3072 + (g * 3 + hh) * kDrafterHeadDim + d]) * 0.08838834764831845F;
    }
    __syncthreads();
    int const nk = nCtx + kDrafterBlock;
    int const kk = blockIdx.x * kScoreThreads + threadIdx.x;
    if (kk >= nk)
    {
        return;
    }
    __half const* krow = kk < nCtx ? ctxK + static_cast<size_t>(lo + kk) * 1024 + g * kDrafterHeadDim : kb + static_cast<size_t>(kk - nCtx) * 1024 + g * kDrafterHeadDim;
    float acc[24];
#pragma unroll
    for (int qi = 0; qi < 24; ++qi)
    {
        acc[qi] = 0.F;
    }
#pragma unroll 1
    for (int dc = 0; dc < kDrafterHeadDim / 8; ++dc)
    {
        uint4 const raw = *reinterpret_cast<uint4 const*>(krow + dc * 8);
        __half2 const* h2 = reinterpret_cast<__half2 const*>(&raw);
        float kf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 const f = __half22float2(h2[i]);
            kf[2 * i] = f.x;
            kf[2 * i + 1] = f.y;
        }
#pragma unroll
        for (int qi = 0; qi < 24; ++qi)
        {
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                acc[qi] = fmaf(qs[qi][dc * 8 + i], kf[i], acc[qi]);
            }
        }
    }
#pragma unroll
    for (int qi = 0; qi < 24; ++qi)
    {
        scores[static_cast<size_t>(g * 24 + qi) * stride + kk] = acc[qi];
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

__global__ void attnPvKernel(float const* scores, int32_t stride, __half const* ctxV, __half const* vb, int32_t lo, int32_t nCtx, int32_t splits, float* partial)
{
    int const g = blockIdx.x;
    int const split = blockIdx.y;
    int const d = threadIdx.x;
    int const nk = nCtx + kDrafterBlock;
    int const chunk = (nk + splits - 1) / splits;
    int const k0 = split * chunk;
    int const k1 = min(nk, k0 + chunk);
    float acc[24];
#pragma unroll
    for (int qi = 0; qi < 24; ++qi)
    {
        acc[qi] = 0.F;
    }
    for (int k = k0; k < k1; ++k)
    {
        float const v = k < nCtx ? __half2float(ctxV[static_cast<size_t>(lo + k) * 1024 + g * kDrafterHeadDim + d])
                                 : __half2float(vb[static_cast<size_t>(k - nCtx) * 1024 + g * kDrafterHeadDim + d]);
#pragma unroll
        for (int qi = 0; qi < 24; ++qi)
        {
            acc[qi] = fmaf(scores[static_cast<size_t>(g * 24 + qi) * stride + k], v, acc[qi]);
        }
    }
#pragma unroll
    for (int qi = 0; qi < 24; ++qi)
    {
        partial[((static_cast<size_t>(split) * kDrafterKvHeads + g) * 24 + qi) * kDrafterHeadDim + d] = acc[qi];
    }
}

__global__ void attnReduceKernel(float const* partial, int32_t splits, __half* out)
{
    int const g = blockIdx.x / 24;
    int const qi = blockIdx.x % 24;
    int const d = threadIdx.x;
    float s = 0.F;
    for (int sp = 0; sp < splits; ++sp)
    {
        s += partial[((static_cast<size_t>(sp) * kDrafterKvHeads + g) * 24 + qi) * kDrafterHeadDim + d];
    }
    int const row = qi / 3;
    int const hh = qi % 3;
    out[static_cast<size_t>(row) * 3072 + (g * 3 + hh) * kDrafterHeadDim + d] = __float2half(s);
}

__global__ void addResidualKernel(float* x, __half const* y, int32_t n)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        x[i] += __half2float(y[i]);
    }
}

__global__ void swigluKernel(__half const* g, __half const* u, __half* out, int32_t n)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float const a = __half2float(g[i]);
        out[i] = __float2half(a / (1.F + __expf(-a)) * __half2float(u[i]));
    }
}

__global__ void buildInvKernel(__half const* hidNormed, float const* lastFeat, __half* xs)
{
    int const j = blockIdx.x;
    for (int i = threadIdx.x; i < kDrafterHidden; i += blockDim.x)
    {
        xs[static_cast<size_t>(j) * 2 * kDrafterHidden + i] = hidNormed[static_cast<size_t>(j + 1) * kDrafterHidden + i];
        xs[static_cast<size_t>(j) * 2 * kDrafterHidden + kDrafterHidden + i] = __float2half(lastFeat[i]);
    }
}

//! One warp per Markov index i (32 blocks x 8 warps): the 512-long dot products of the three gate rows are spread over the lanes (coalesced float4 rows).
__global__ void markovGateKernel(__half const* inv, float const* bias, float const* stateW, __half const* markovW1, int32_t const* prevToken, float const* stateIn,
    float* stateOut, __half* o, int32_t j)
{
    __shared__ float vec[2 * kDrafterMarkov];
    int const prev = *prevToken;
    for (int t = threadIdx.x; t < 2 * kDrafterMarkov; t += blockDim.x)
    {
        vec[t] = t < kDrafterMarkov ? stateIn[t] : __half2float(markovW1[static_cast<size_t>(prev) * kDrafterMarkov + (t - kDrafterMarkov)]);
    }
    __syncthreads();
    int const lane = threadIdx.x & 31;
    int const i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float dot[3];
#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        float4 const* w = reinterpret_cast<float4 const*>(stateW + static_cast<size_t>(c * kDrafterMarkov + i) * 2 * kDrafterMarkov);
        float a = 0.F;
#pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            float4 const wv = __ldg(w + r * 32 + lane);
            float4 const xv = *reinterpret_cast<float4 const*>(&vec[(r * 32 + lane) * 4]);
            a = fmaf(wv.x, xv.x, a);
            a = fmaf(wv.y, xv.y, a);
            a = fmaf(wv.z, xv.z, a);
            a = fmaf(wv.w, xv.w, a);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            a += __shfl_xor_sync(0xffffffffu, a, off);
        }
        dot[c] = a;
    }
    if (lane == 0)
    {
#pragma unroll
        for (int c = 0; c < 3; ++c)
        {
            dot[c] += __half2float(inv[static_cast<size_t>(j) * 768 + c * kDrafterMarkov + i]) + bias[c * kDrafterMarkov + i];
        }
        float const gate = 1.F / (1.F + __expf(-dot[0]));
        stateOut[i] = gate * stateIn[i] + (1.F - gate) * tanhf(dot[1]);
        o[i] = __float2half(tanhf(dot[2]));
    }
}

constexpr int kArgmaxBlocks = 128;

__global__ void argmaxPartialKernel(__half const* base, __half const* corr, float const* bU, int32_t vocab, int32_t j, float* scratchVal, int32_t* scratchIdx)
{
    __shared__ float sv[256];
    __shared__ int si[256];
    float best = -FLT_MAX;
    int bestIdx = 0x7fffffff;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < vocab; i += kArgmaxBlocks * 256)
    {
        float const v = __half2float(base[static_cast<size_t>(j) * vocab + i]) + __half2float(corr[i]) + bU[i];
        if (v > best || (v == best && i < bestIdx))
        {
            best = v;
            bestIdx = i;
        }
    }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bestIdx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            float const ov = sv[threadIdx.x + s];
            int const oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x]))
            {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        scratchVal[blockIdx.x] = sv[0];
        scratchIdx[blockIdx.x] = si[0];
    }
}

__global__ void argmaxFinalKernel(float const* scratchVal, int32_t const* scratchIdx, int32_t j, int32_t* draftTokens, int32_t* prevToken)
{
    __shared__ float sv[kArgmaxBlocks];
    __shared__ int si[kArgmaxBlocks];
    sv[threadIdx.x] = scratchVal[threadIdx.x];
    si[threadIdx.x] = scratchIdx[threadIdx.x];
    __syncthreads();
    for (int s = kArgmaxBlocks / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            float const ov = sv[threadIdx.x + s];
            int const oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x]))
            {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        draftTokens[j] = si[0];
        *prevToken = si[0];
    }
}
namespace i8mma
{
constexpr int kChunk = 1024;              // k elements of the activations staged in shared memory at a time
constexpr int kXStride = kChunk + 8;      // padded row stride (halves): rows map to distinct banks
constexpr int kWarps = 4;

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void i8pairs(uint32_t w, uint32_t& b0, uint32_t& b1) {
  uint32_t const flipped = w ^ 0x80808080u;
  uint32_t const lo = __byte_perm(flipped, 0x64646464u, 0x5140);
  uint32_t const hi = __byte_perm(flipped, 0x64646464u, 0x7362);
  __half2 const bias = __float2half2_rn(1152.f);
  __half2 const l = __hsub2(*reinterpret_cast<__half2 const*>(&lo), bias);
  __half2 const h = __hsub2(*reinterpret_cast<__half2 const*>(&hi), bias);
  b0 = *reinterpret_cast<uint32_t const*>(&l);
  b1 = *reinterpret_cast<uint32_t const*>(&h);
}

// x [T, K] fp16, wShuf int8 (see header), scale [N] fp32, y [T, N] fp16. N % 8 == 0, K % 64 == 0, T <= 8.
__global__ void __launch_bounds__(128) gemv(__half const* __restrict__ x, uint4 const* __restrict__ wShuf, float const* __restrict__ scale,
                                            __half* __restrict__ y, int T, int N, int K) {
  __shared__ __align__(16) __half xs[8][kXStride];
  int const lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int const g = lane >> 2, t = lane & 3;
  int const G = blockIdx.x * kWarps + warp;       // output group of 8 rows
  bool const active = G * 8 < N;
  int const kb64 = K / 64;
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  uint4 const* wp = wShuf + (size_t)(active ? G : 0) * kb64 * 32 + lane;
  for (int k0 = 0; k0 < K; k0 += kChunk) {
    int const kc = min(kChunk, K - k0);
    __syncthreads();
    for (int idx = threadIdx.x; idx < 8 * kc / 8; idx += blockDim.x) {   // 8 halves per uint4
      int const r = idx / (kc / 8), cc = idx % (kc / 8);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < T) v = *reinterpret_cast<uint4 const*>(x + (size_t)r * K + k0 + cc * 8);
      *reinterpret_cast<uint4*>(&xs[r][cc * 8]) = v;
    }
    __syncthreads();
    if (active) {
      int const nkb = kc / 64;
      int const kbBase = k0 / 64;
#pragma unroll 2
      for (int kb = 0; kb < nkb; ++kb) {
        uint4 const w = __ldg(wp + (size_t)(kbBase + kb) * 32);
        uint32_t const ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int const kk = kb * 64 + j * 16;
          uint32_t const a0 = *reinterpret_cast<uint32_t const*>(&xs[g][kk + 2 * t]);
          uint32_t const a2 = *reinterpret_cast<uint32_t const*>(&xs[g][kk + 8 + 2 * t]);
          uint32_t b0, b1;
          i8pairs(ws[j], b0, b1);
          mma16816(c, a0, a2, b0, b1);
        }
      }
    }
  }
  if (active && g < T) {
    int const n0 = G * 8 + 2 * t;
    y[(size_t)g * N + n0] = __float2half_rn(c[0] * scale[n0]);
    y[(size_t)g * N + n0 + 1] = __float2half_rn(c[1] * scale[n0 + 1]);
  }
}

} // namespace i8mma
} // namespace

void launchDrafterGemv(void const* x, void const* wShuf, void const* scale, void* y, int32_t tokens, int32_t n, int32_t k, cudaStream_t stream)
{
    // the fork's mma-ordered INT8 GEMV (split-K, batched weight loads; METRICS 208): same weight order and FP32 per-row scales as the drafter file
    static bool const legacy = std::getenv("EDGELLM_INKLING_DRAFTER_LEGACY_GEMV") != nullptr;
    if (!legacy)
    {
        launchFp8Linear(x, wShuf, scale, y, tokens, n, k, /*scaleIsFp32=*/true, stream, kLinearWeightInt8Mma);
        return;
    }
    i8mma::gemv<<<(n / 8 + i8mma::kWarps - 1) / i8mma::kWarps, 128, 0, stream>>>(
        static_cast<__half const*>(x), static_cast<uint4 const*>(wShuf), static_cast<float const*>(scale), static_cast<__half*>(y), tokens, n, k);
}

void launchDrafterCtxMix(float const* taps, int32_t rows, int32_t consumerStride, float const* alpha, float const* dflare, float const* dflareMarkov,
    float const* targetFinalNorm, __half* ctxn, float* lastFeat, cudaStream_t stream)
{
    ctxMixKernel<<<rows, 256, 0, stream>>>(taps, rows, consumerStride, alpha, dflare, dflareMarkov, targetFinalNorm, ctxn, lastFeat);
}

void launchDrafterCtxKv(__half const* kRaw, __half const* vRaw, int32_t rows, int32_t pos0, float const* kNormCtx, float theta, __half* ctxK, __half* ctxV,
    cudaStream_t stream, int32_t ctxCap)
{
    ctxKvKernel<<<dim3(rows, kDrafterKvHeads), kDrafterHeadDim, 0, stream>>>(kRaw, vRaw, pos0, kNormCtx, theta, ctxK, ctxV, ctxCap);
}

void launchDrafterInitBlock(__half const* embedding, float const* embedNorm, float const* maskEmbed, float* x, cudaStream_t stream)
{
    initBlockKernel<<<kDrafterBlock, 256, 0, stream>>>(embedding, embedNorm, maskEmbed, x);
}

void launchDrafterRms(float const* x, float const* w, __half* out, int32_t rows, int32_t rowOffset, cudaStream_t stream)
{
    rmsKernel<<<rows, 256, 0, stream>>>(x, w, out, rowOffset);
}

void launchDrafterQkvPrep(__half const* qRaw, __half const* kRaw, __half const* vRaw, float const* qNorm, float const* kNorm, int32_t pos0, float theta,
    __half* q, __half* kb, __half* vb, cudaStream_t stream)
{
    qkvPrepKernel<<<dim3(kDrafterBlock, kDrafterHeads + 2 * kDrafterKvHeads), kDrafterHeadDim, 0, stream>>>(qRaw, kRaw, vRaw, qNorm, kNorm, pos0, theta, q, kb, vb);
}

void launchDrafterAttention(__half const* q, __half const* ctxK, __half const* ctxV, __half const* kb, __half const* vb, int32_t lo, int32_t nCtx, float* scores,
    int32_t scoreStride, float* partial, int32_t maxSplits, __half* out, cudaStream_t stream)
{
    int32_t const nk = nCtx + kDrafterBlock;
    attnScoresKernel<<<dim3((nk + kScoreThreads - 1) / kScoreThreads, kDrafterKvHeads), kScoreThreads, 0, stream>>>(q, ctxK, kb, lo, nCtx, scores, scoreStride);
    attnSoftmaxKernel<<<kDrafterKvHeads * 24, 256, 0, stream>>>(scores, scoreStride, nk);
    int32_t const splits = std::max(1, std::min(maxSplits, nk / 128));
    attnPvKernel<<<dim3(kDrafterKvHeads, splits), kDrafterHeadDim, 0, stream>>>(scores, scoreStride, ctxV, vb, lo, nCtx, splits, partial);
    attnReduceKernel<<<kDrafterKvHeads * 24, kDrafterHeadDim, 0, stream>>>(partial, splits, out);
}

void launchDrafterAddResidual(float* x, __half const* y, int32_t n, cudaStream_t stream)
{
    addResidualKernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
}

void launchDrafterSwiglu(__half const* g, __half const* u, __half* out, int32_t n, cudaStream_t stream)
{
    swigluKernel<<<(n + 255) / 256, 256, 0, stream>>>(g, u, out, n);
}

void launchDrafterBuildInv(__half const* hidNormed, float const* lastFeat, __half* xs, cudaStream_t stream)
{
    buildInvKernel<<<kDrafterBlock - 1, 256, 0, stream>>>(hidNormed, lastFeat, xs);
}

void launchDrafterMarkovGate(__half const* inv, float const* bias, float const* stateW, __half const* markovW1, int32_t const* prevToken, float const* stateIn,
    float* stateOut, __half* o, int32_t j, cudaStream_t stream)
{
    markovGateKernel<<<32, 256, 0, stream>>>(inv, bias, stateW, markovW1, prevToken, stateIn, stateOut, o, j);
}

void launchDrafterArgmax(__half const* base, __half const* corr, float const* bU, int32_t vocabPadded, int32_t j, float* scratchVal, int32_t* scratchIdx,
    int32_t* draftTokens, int32_t* prevToken, cudaStream_t stream)
{
    argmaxPartialKernel<<<kArgmaxBlocks, 256, 0, stream>>>(base, corr, bU, vocabPadded, j, scratchVal, scratchIdx);
    argmaxFinalKernel<<<1, kArgmaxBlocks, 0, stream>>>(scratchVal, scratchIdx, j, draftTokens, prevToken);
}

} // namespace kernel
} // namespace trt_edgellm
