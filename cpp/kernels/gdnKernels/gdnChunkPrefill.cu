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

#include "gdnChunkPrefill.h"

#include <mma.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

using namespace nvcuda;

constexpr int32_t kC{kGDN_CHUNK};   // tokens per chunk
constexpr int32_t kD{128};          // dk = dv
constexpr int32_t kTV{32};          // V columns per state CTA
constexpr int32_t kPrepThreads{128}; // 4 warps
constexpr int32_t kStateThreads{256}; // 8 warps
constexpr int32_t kLdKQ{kD + 8};    // fp16 row stride of K / Q tiles in shared memory
constexpr int32_t kLdCC{kC + 4};    // fp32 row stride of C x C tiles
constexpr int32_t kLdCh{kC + 8};    // fp16 row stride of C x C tiles
constexpr int32_t kLdTVh{kTV + 8};  // fp16 row stride of C x 32 / 128 x 32 tiles
constexpr int32_t kLdTVf{kTV + 4};  // fp32 row stride of C x 32 / 128 x 32 tiles

// Per (sequence, v head, chunk), written by the prep kernel.
struct ChunkRecord
{
    __half T[kC * kC];    // (I + A)^-1, row-major
    __half attn[kC * kC]; // tril(Q K^T * D), row-major
    float G[kC];          // inclusive cumsum of log g (padded rows: unchanged)
    float beta[kC];       // sigmoid(b) (padded rows: 0)
};
// Per (sequence, k head, chunk): L2-normalised k and scaled q, fp16, row-major (padded rows zero).
struct ChunkQK
{
    __half K[kC * kD];
    __half Q[kC * kD];
};
static_assert(sizeof(ChunkRecord) % 32 == 0 && sizeof(ChunkQK) % 32 == 0, "records must stay 32-byte aligned");

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using FragACol = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
using FragBCol = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
using FragAcc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float softplus(float x)
{
    return x > 20.f ? x : log1pf(expf(x));
}

__device__ __forceinline__ int32_t sequenceLength(int32_t const* contextLengths, int32_t iN, int32_t seqLen)
{
    int32_t const len = contextLengths ? contextLengths[iN] : seqLen;
    return len < seqLen ? len : seqLen;
}

__device__ __forceinline__ void cpAsync16(void* dst, void const* src)
{
    uint32_t const d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src));
}

__device__ __forceinline__ void cpAsyncWaitAll()
{
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::);
}

// ---- Kernel 1: per (chunk, v head, sequence) build T = (I + A)^-1, Attn, the gates and the q / k tiles -----------
struct PrepSmem
{
    __half K[kC * kLdKQ];
    union
    {
        __half Q[kC * kLdKQ];
        float T[kC * kLdCC]; // after Q K^T
    };
    float F[kC * kLdCC]; // Q K^T, then K K^T -> A (lower) and block-solve temporaries (upper tiles)
    float G[kC];
    float beta[kC];
};

__global__ void __launch_bounds__(kPrepThreads) gdnChunkPrepKernel(__half const* __restrict__ q,
    __half const* __restrict__ k, __half const* __restrict__ a, __half const* __restrict__ b,
    float const* __restrict__ A_log, __half const* __restrict__ dt_bias, int32_t const* __restrict__ contextLengths,
    ChunkRecord* __restrict__ records, ChunkQK* __restrict__ qkTiles, int32_t segStart, int32_t chunksPerSeg,
    int32_t seqLen, int32_t h, int32_t hv)
{
    extern __shared__ __align__(128) char smemRaw[];
    PrepSmem& sm = *reinterpret_cast<PrepSmem*>(smemRaw);
    int32_t const c = blockIdx.x;
    int32_t const iHv = blockIdx.y;
    int32_t const iN = blockIdx.z;
    int32_t const ratio = hv / h;
    int32_t const iH = iHv / ratio;
    int32_t const t0 = segStart + c * kC;
    int32_t const nValid = min(kC, sequenceLength(contextLengths, iN, seqLen) - t0);
    if (nValid <= 0)
    {
        return;
    }
    int32_t const tid = threadIdx.x;
    int32_t const warp = tid / 32;
    int32_t const lane = tid % 32;
    ChunkRecord* rec = records + (static_cast<int64_t>(iN) * hv + iHv) * chunksPerSeg + c;

    // q / k rows: L2-normalise (q also scaled by 1/sqrt(dk)); each warp takes 16 rows, each lane 4 elements.
    float const qScale = rsqrtf(static_cast<float>(kD));
#pragma unroll 4
    for (int32_t r = warp; r < kC; r += kPrepThreads / 32)
    {
        float2 qa{0.f, 0.f}, qb{0.f, 0.f}, ka{0.f, 0.f}, kb{0.f, 0.f};
        if (r < nValid)
        {
            int64_t const off = ((static_cast<int64_t>(iN) * seqLen + t0 + r) * h + iH) * kD + lane * 4;
            qa = __half22float2(reinterpret_cast<__half2 const*>(q + off)[0]);
            qb = __half22float2(reinterpret_cast<__half2 const*>(q + off)[1]);
            ka = __half22float2(reinterpret_cast<__half2 const*>(k + off)[0]);
            kb = __half22float2(reinterpret_cast<__half2 const*>(k + off)[1]);
        }
        float sq = qa.x * qa.x + qa.y * qa.y + qb.x * qb.x + qb.y * qb.y;
        float sk = ka.x * ka.x + ka.y * ka.y + kb.x * kb.x + kb.y * kb.y;
#pragma unroll
        for (int32_t off = 16; off > 0; off >>= 1)
        {
            sq += __shfl_xor_sync(0xffffffffu, sq, off);
            sk += __shfl_xor_sync(0xffffffffu, sk, off);
        }
        float const fq = qScale * rsqrtf(sq + 1e-6f);
        float const fk = rsqrtf(sk + 1e-6f);
        __half2* dq = reinterpret_cast<__half2*>(sm.Q + r * kLdKQ + lane * 4);
        __half2* dk = reinterpret_cast<__half2*>(sm.K + r * kLdKQ + lane * 4);
        dq[0] = __floats2half2_rn(qa.x * fq, qa.y * fq);
        dq[1] = __floats2half2_rn(qb.x * fq, qb.y * fq);
        dk[0] = __floats2half2_rn(ka.x * fk, ka.y * fk);
        dk[1] = __floats2half2_rn(kb.x * fk, kb.y * fk);
    }
    if (tid < kC)
    {
        float logG = 0.f, beta = 0.f;
        if (tid < nValid)
        {
            int64_t const ab = (static_cast<int64_t>(iN) * seqLen + t0 + tid) * hv + iHv;
            float const x = __half2float(a[ab]) + __half2float(dt_bias[iHv]);
            logG = -expf(A_log[iHv]) * softplus(x);
            beta = 1.f / (1.f + expf(-__half2float(b[ab])));
        }
        sm.G[tid] = logG;
        sm.beta[tid] = beta;
    }
    __syncthreads();
    if (warp == 0)
    {
        // Inclusive scan of log g over 64 rows: two elements per lane.
        float const x0 = sm.G[2 * lane], x1 = sm.G[2 * lane + 1];
        float s = x0 + x1;
#pragma unroll
        for (int32_t off = 1; off < 32; off <<= 1)
        {
            float const y = __shfl_up_sync(0xffffffffu, s, off);
            s += lane >= off ? y : 0.f;
        }
        float const before = s - (x0 + x1);
        sm.G[2 * lane] = before + x0;
        sm.G[2 * lane + 1] = s;
    }
    if (iHv % ratio == 0)
    {
        // The k head's normalised tiles, once per k head.
        ChunkQK* qk = qkTiles + (static_cast<int64_t>(iN) * h + iH) * chunksPerSeg + c;
        for (int32_t idx = tid; idx < kC * kD / 8; idx += kPrepThreads)
        {
            int32_t const r = idx / (kD / 8);
            int32_t const col = (idx % (kD / 8)) * 8;
            *reinterpret_cast<uint4*>(qk->K + r * kD + col) = *reinterpret_cast<uint4 const*>(sm.K + r * kLdKQ + col);
            *reinterpret_cast<uint4*>(qk->Q + r * kD + col) = *reinterpret_cast<uint4 const*>(sm.Q + r * kLdKQ + col);
        }
    }

    // Q K^T then K K^T (lower-triangular 16 x 16 tiles only, 10 of 16), both through F.
    auto lowerGram = [&](__half const* lhs)
    {
        int32_t tile = 0;
        for (int32_t ti = 0; ti < kC / 16; ++ti)
        {
            for (int32_t tj = 0; tj <= ti; ++tj, ++tile)
            {
                if (tile % (kPrepThreads / 32) != warp)
                {
                    continue;
                }
                FragAcc acc;
                wmma::fill_fragment(acc, 0.f);
#pragma unroll
                for (int32_t kk = 0; kk < kD / 16; ++kk)
                {
                    FragA fa;
                    FragBCol fb; // B[d][j] = K[j][d]
                    wmma::load_matrix_sync(fb, sm.K + tj * 16 * kLdKQ + kk * 16, kLdKQ);
                    wmma::load_matrix_sync(fa, lhs + ti * 16 * kLdKQ + kk * 16, kLdKQ);
                    wmma::mma_sync(acc, fa, fb, acc);
                }
                wmma::store_matrix_sync(sm.F + ti * 16 * kLdCC + tj * 16, acc, kLdCC, wmma::mem_row_major);
            }
        }
    };
    lowerGram(sm.Q);
    __syncthreads();
    for (int32_t idx = tid; idx < kC * kC / 2; idx += kPrepThreads)
    {
        int32_t const i = (2 * idx) / kC;
        int32_t const j = (2 * idx) % kC;
        float const v0 = j <= i ? sm.F[i * kLdCC + j] * expf(sm.G[i] - sm.G[j]) : 0.f;
        float const v1 = j + 1 <= i ? sm.F[i * kLdCC + j + 1] * expf(sm.G[i] - sm.G[j + 1]) : 0.f;
        reinterpret_cast<__half2*>(rec->attn)[idx] = __floats2half2_rn(v0, v1);
    }
    __syncthreads();
    lowerGram(sm.K);
    __syncthreads();
    for (int32_t idx = tid; idx < kC * kC; idx += kPrepThreads)
    {
        int32_t const i = idx / kC;
        int32_t const j = idx % kC;
        if (j < i)
        {
            sm.F[i * kLdCC + j] *= sm.beta[i] * expf(sm.G[i] - sm.G[j]);
        }
        else if (j == i)
        {
            sm.F[i * kLdCC + j] = 0.f;
        }
    }
    __syncthreads();

    // T = (I + A)^-1, blocked on 16 x 16 tiles. Diagonal tiles: warp w inverts tile (w, w) by forward substitution,
    // lane = column. Then block rows I = 1..3: X_IJ = sum_{K=J}^{I-1} A_IK T_KJ (kept in the free upper tile (J, I)
    // of KK), T_IJ = -T_II X_IJ.
    float* Tm = sm.T;
    if (lane < 16)
    {
        int32_t const base = warp * 16;
        int32_t const j = lane;
        for (int32_t i = 0; i < 16; ++i)
        {
            float t = i == j ? 1.f : 0.f;
            if (i > j)
            {
                float s = 0.f;
                for (int32_t m = j; m < i; ++m)
                {
                    s += sm.F[(base + i) * kLdCC + base + m] * Tm[(base + m) * kLdCC + base + j];
                }
                t = -s;
            }
            Tm[(base + i) * kLdCC + base + j] = t;
        }
    }
    __syncthreads();
    for (int32_t I = 1; I < kC / 16; ++I)
    {
        for (int32_t idx = tid; idx < I * 256; idx += kPrepThreads)
        {
            int32_t const J = idx / 256;
            int32_t const i = (idx % 256) / 16;
            int32_t const j = idx % 16;
            float x = 0.f;
            for (int32_t K = J; K < I; ++K)
            {
                for (int32_t m = 0; m < 16; ++m)
                {
                    x += sm.F[(I * 16 + i) * kLdCC + K * 16 + m] * Tm[(K * 16 + m) * kLdCC + J * 16 + j];
                }
            }
            sm.F[(J * 16 + i) * kLdCC + I * 16 + j] = x;
        }
        __syncthreads();
        for (int32_t idx = tid; idx < I * 256; idx += kPrepThreads)
        {
            int32_t const J = idx / 256;
            int32_t const i = (idx % 256) / 16;
            int32_t const j = idx % 16;
            float t = 0.f;
            for (int32_t m = 0; m <= i; ++m)
            {
                t += Tm[(I * 16 + i) * kLdCC + I * 16 + m] * sm.F[(J * 16 + m) * kLdCC + I * 16 + j];
            }
            Tm[(I * 16 + i) * kLdCC + J * 16 + j] = -t;
        }
        __syncthreads();
    }
    for (int32_t idx = tid; idx < kC * kC / 2; idx += kPrepThreads)
    {
        int32_t const i = (2 * idx) / kC;
        int32_t const j = (2 * idx) % kC;
        reinterpret_cast<__half2*>(rec->T)[idx] = __floats2half2_rn(
            j <= i ? Tm[i * kLdCC + j] : 0.f, j + 1 <= i ? Tm[i * kLdCC + j + 1] : 0.f);
    }
    if (tid < kC)
    {
        rec->G[tid] = sm.G[tid];
        rec->beta[tid] = sm.beta[tid];
    }
}

// ---- Kernel 2: per (V tile, v head, sequence) walk the segment's chunks, state in WMMA accumulators -----------
// Buffers alias by lifetime so two CTAs fit per SM: TA holds T, then Attn (loaded after T R); S holds the state as
// fp16 until K S / Q S are done, then R (later the decayed new_v) and new_v.
struct StateSmem
{
    __half K[kC * kLdKQ];
    __half Q[kC * kLdKQ];
    __half TA[kC * kLdCh];
    __half V[kC * kLdTVh];
    float G[kC];
    float beta[kC];
    __half S[kD * kLdTVh];
    float scratch[kD * kLdTVf]; // state fp32 staging, then [K S | Q S] (2 x kC rows)
};

__global__ void __launch_bounds__(kStateThreads, 2) gdnChunkStateKernel(__half const* __restrict__ v,
    float* __restrict__ statePool, int32_t const* __restrict__ stateIndices, int32_t statePoolRows,
    int32_t const* __restrict__ contextLengths, ChunkRecord const* __restrict__ records,
    ChunkQK const* __restrict__ qkTiles, __half* __restrict__ o, int32_t segStart, int32_t segLen, int32_t chunksPerSeg,
    int32_t seqLen, int32_t h, int32_t hv)
{
    extern __shared__ __align__(128) char smemRaw[];
    StateSmem& sm = *reinterpret_cast<StateSmem*>(smemRaw);

    int32_t const vt = blockIdx.x;
    int32_t const iHv = blockIdx.y;
    int32_t const iN = blockIdx.z;
    int32_t const iH = iHv / (hv / h);
    int32_t const tid = threadIdx.x;
    int32_t const warp = tid / 32;
    int32_t const rt = warp % 4; // C x 32 output tiles: row tile
    int32_t const ct = warp / 4; //                      column tile
    int32_t const len = sequenceLength(contextLengths, iN, seqLen);
    int32_t const row = stateIndices ? stateIndices[iN] : iN;
    bool const validRow = row >= 0 && row < statePoolRows;
    float* state = validRow ? statePool + (static_cast<int64_t>(row) * hv + iHv) * kD * kD + vt * kTV : nullptr;
    float* sKS = sm.scratch;
    float* sQS = sm.scratch + kC * kLdTVf;
    __half* sR = sm.S;                  // R, then the decayed new_v
    __half* sNV = sm.S + kC * kLdTVh;   // new_v

    // Warp w owns state rows 16w .. 16w + 15 x the two 16-column tiles.
    FragAcc fS[2];
#pragma unroll
    for (int32_t j = 0; j < 2; ++j)
    {
        if (validRow)
        {
            wmma::load_matrix_sync(fS[j], state + warp * 16 * kD + j * 16, kD, wmma::mem_row_major);
        }
        else
        {
            wmma::fill_fragment(fS[j], 0.f);
        }
    }

    for (int32_t c = 0; c * kC < segLen; ++c)
    {
        int32_t const t0 = segStart + c * kC;
        int32_t const nValid = min(kC, len - t0);
        int32_t const nRows = min(kC, seqLen - t0);
        if (nValid <= 0)
        {
            for (int32_t idx = tid; idx < nRows * kTV; idx += kStateThreads)
            {
                int32_t const r = idx / kTV;
                o[((static_cast<int64_t>(iN) * seqLen + t0 + r) * hv + iHv) * kD + vt * kTV + idx % kTV]
                    = __float2half(0.f);
            }
            continue;
        }
        ChunkRecord const* rec = records + (static_cast<int64_t>(iN) * hv + iHv) * chunksPerSeg + c;
        ChunkQK const* qk = qkTiles + (static_cast<int64_t>(iN) * h + iH) * chunksPerSeg + c;

        // 1. Bulk async copy of the chunk's tiles (16-byte pieces), overlapped with staging the state as fp16.
        for (int32_t idx = tid; idx < kC * kD / 8; idx += kStateThreads)
        {
            int32_t const r = idx / (kD / 8);
            int32_t const col = (idx % (kD / 8)) * 8;
            cpAsync16(sm.K + r * kLdKQ + col, qk->K + r * kD + col);
            cpAsync16(sm.Q + r * kLdKQ + col, qk->Q + r * kD + col);
        }
        for (int32_t idx = tid; idx < kC * kC / 8; idx += kStateThreads)
        {
            int32_t const r = idx / (kC / 8);
            int32_t const col = (idx % (kC / 8)) * 8;
            cpAsync16(sm.TA + r * kLdCh + col, rec->T + r * kC + col);
        }
        if (tid < kC * kTV / 8)
        {
            int32_t const r = tid / (kTV / 8);
            int32_t const col = (tid % (kTV / 8)) * 8;
            if (r < nRows)
            {
                cpAsync16(sm.V + r * kLdTVh + col,
                    v + ((static_cast<int64_t>(iN) * seqLen + t0 + r) * hv + iHv) * kD + vt * kTV + col);
            }
        }
        if (tid < 2 * kC / 4)
        {
            float* dst = tid < kC / 4 ? sm.G + tid * 4 : sm.beta + (tid - kC / 4) * 4;
            float const* src = tid < kC / 4 ? rec->G + tid * 4 : rec->beta + (tid - kC / 4) * 4;
            cpAsync16(dst, src);
        }
#pragma unroll
        for (int32_t j = 0; j < 2; ++j)
        {
            wmma::store_matrix_sync(sm.scratch + warp * 16 * kLdTVf + j * 16, fS[j], kLdTVf, wmma::mem_row_major);
        }
        __syncwarp();
        // Each warp converts its own 16 state rows.
        for (int32_t idx = threadIdx.x % 32; idx < 16 * kTV; idx += 32)
        {
            int32_t const r = warp * 16 + idx / kTV;
            sm.S[r * kLdTVh + idx % kTV] = __float2half(sm.scratch[r * kLdTVf + idx % kTV]);
        }
        cpAsyncWaitAll();
        __syncthreads();

        // 2. K S and Q S (C x 32): warp -> (row tile rt, column tile ct).
        {
            FragAcc accK, accQ;
            wmma::fill_fragment(accK, 0.f);
            wmma::fill_fragment(accQ, 0.f);
#pragma unroll
            for (int32_t kk = 0; kk < kD / 16; ++kk)
            {
                FragA fa;
                FragB fb;
                wmma::load_matrix_sync(fb, sm.S + kk * 16 * kLdTVh + ct * 16, kLdTVh);
                wmma::load_matrix_sync(fa, sm.K + rt * 16 * kLdKQ + kk * 16, kLdKQ);
                wmma::mma_sync(accK, fa, fb, accK);
                wmma::load_matrix_sync(fa, sm.Q + rt * 16 * kLdKQ + kk * 16, kLdKQ);
                wmma::mma_sync(accQ, fa, fb, accQ);
            }
            __syncthreads(); // every warp is done reading the state staging that K S / Q S overwrite
            wmma::store_matrix_sync(sKS + rt * 16 * kLdTVf + ct * 16, accK, kLdTVf, wmma::mem_row_major);
            wmma::store_matrix_sync(sQS + rt * 16 * kLdTVf + ct * 16, accQ, kLdTVf, wmma::mem_row_major);
        }
        __syncthreads();

        // 3. R = beta * (V - exp(G) * K S); Q S scaled by exp(G) for the output.
        for (int32_t idx = tid; idx < kC * kTV; idx += kStateThreads)
        {
            int32_t const r = idx / kTV;
            int32_t const col = idx % kTV;
            float rv = 0.f;
            if (r < nValid)
            {
                float const eg = expf(sm.G[r]);
                rv = sm.beta[r] * (__half2float(sm.V[r * kLdTVh + col]) - eg * sKS[r * kLdTVf + col]);
                sQS[r * kLdTVf + col] *= eg;
            }
            sR[r * kLdTVh + col] = __float2half(rv);
        }
        __syncthreads();

        // 4. new_v = T R, into the K S scratch.
        {
            FragAcc acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int32_t kk = 0; kk < kC / 16; ++kk)
            {
                FragA fa;
                FragB fb;
                wmma::load_matrix_sync(fa, sm.TA + rt * 16 * kLdCh + kk * 16, kLdCh);
                wmma::load_matrix_sync(fb, sR + kk * 16 * kLdTVh + ct * 16, kLdTVh);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(sKS + rt * 16 * kLdTVf + ct * 16, acc, kLdTVf, wmma::mem_row_major);
        }
        __syncthreads();
        // T is done: fetch Attn into its buffer while new_v is converted.
        for (int32_t idx = tid; idx < kC * kC / 8; idx += kStateThreads)
        {
            int32_t const r = idx / (kC / 8);
            int32_t const col = (idx % (kC / 8)) * 8;
            cpAsync16(sm.TA + r * kLdCh + col, rec->attn + r * kC + col);
        }
        float const gLast = sm.G[kC - 1];
        for (int32_t idx = tid; idx < kC * kTV; idx += kStateThreads)
        {
            int32_t const r = idx / kTV;
            int32_t const col = idx % kTV;
            float const nv = sKS[r * kLdTVf + col];
            sNV[r * kLdTVh + col] = __float2half(nv);
            sR[r * kLdTVh + col] = __float2half(nv * expf(gLast - sm.G[r]));
        }
        cpAsyncWaitAll();
        __syncthreads();

        // 5. O = exp(G) Q S + Attn new_v.
        {
            FragAcc acc;
            wmma::load_matrix_sync(acc, sQS + rt * 16 * kLdTVf + ct * 16, kLdTVf, wmma::mem_row_major);
#pragma unroll
            for (int32_t kk = 0; kk < kC / 16; ++kk)
            {
                FragA fa;
                FragB fb;
                wmma::load_matrix_sync(fa, sm.TA + rt * 16 * kLdCh + kk * 16, kLdCh);
                wmma::load_matrix_sync(fb, sNV + kk * 16 * kLdTVh + ct * 16, kLdTVh);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(sQS + rt * 16 * kLdTVf + ct * 16, acc, kLdTVf, wmma::mem_row_major);
        }

        // 6. State: S = exp(G_last) S + K^T (exp(G_last - G) new_v); A = K^T is K read column-major.
        float const decay = expf(gLast);
#pragma unroll
        for (int32_t j = 0; j < 2; ++j)
        {
#pragma unroll
            for (int32_t e = 0; e < fS[j].num_elements; ++e)
            {
                fS[j].x[e] *= decay;
            }
#pragma unroll
            for (int32_t kk = 0; kk < kC / 16; ++kk)
            {
                FragACol fa;
                FragB fb;
                wmma::load_matrix_sync(fa, sm.K + kk * 16 * kLdKQ + warp * 16, kLdKQ);
                wmma::load_matrix_sync(fb, sR + kk * 16 * kLdTVh + j * 16, kLdTVh);
                wmma::mma_sync(fS[j], fa, fb, fS[j]);
            }
        }
        __syncthreads();
        for (int32_t idx = tid; idx < nRows * kTV / 2; idx += kStateThreads)
        {
            int32_t const r = idx / (kTV / 2);
            int32_t const col = (idx % (kTV / 2)) * 2;
            __half2 const val = r < nValid
                ? __floats2half2_rn(sQS[r * kLdTVf + col], sQS[r * kLdTVf + col + 1])
                : __floats2half2_rn(0.f, 0.f);
            *reinterpret_cast<__half2*>(o + ((static_cast<int64_t>(iN) * seqLen + t0 + r) * hv + iHv) * kD
                + vt * kTV + col)
                = val;
        }
        __syncthreads();
    }

    if (validRow)
    {
#pragma unroll
        for (int32_t j = 0; j < 2; ++j)
        {
            wmma::store_matrix_sync(state + warp * 16 * kD + j * 16, fS[j], kD, wmma::mem_row_major);
        }
    }
}

int32_t chunksPerSegment()
{
    return kGDN_CHUNK_SEGMENT / kC;
}

} // namespace

size_t gdnChunkPrefillWorkspaceBytes(int32_t n, int32_t h, int32_t hv)
{
    return static_cast<size_t>(n) * chunksPerSegment() * (hv * sizeof(ChunkRecord) + h * sizeof(ChunkQK));
}

cudaError_t gdnChunkPrefill(__half const* q, __half const* k, __half const* v, __half const* a, __half const* b,
    float const* A_log, __half const* dt_bias, float* statePool, int32_t const* stateIndices, int32_t statePoolRows,
    int32_t const* contextLengths, __half* o, int32_t n, int32_t seqLen, int32_t h, int32_t hv, void* workspace,
    cudaStream_t stream)
{
    if (h <= 0 || hv % h != 0 || workspace == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    static bool const kSmemSet = [] {
        return cudaFuncSetAttribute(gdnChunkStateKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int32_t>(sizeof(StateSmem)))
            == cudaSuccess
            && cudaFuncSetAttribute(gdnChunkStateKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared)
            == cudaSuccess
            && cudaFuncSetAttribute(gdnChunkPrepKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                   cudaSharedmemCarveoutMaxShared)
            == cudaSuccess
            && cudaFuncSetAttribute(gdnChunkPrepKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int32_t>(sizeof(PrepSmem)))
            == cudaSuccess;
    }();
    if (!kSmemSet)
    {
        return cudaErrorInvalidConfiguration;
    }
    int32_t const chunksPerSeg = chunksPerSegment();
    auto* records = static_cast<ChunkRecord*>(workspace);
    auto* qkTiles = reinterpret_cast<ChunkQK*>(records + static_cast<size_t>(n) * hv * chunksPerSeg);
    for (int32_t segStart = 0; segStart < seqLen; segStart += kGDN_CHUNK_SEGMENT)
    {
        int32_t const segLen = seqLen - segStart < kGDN_CHUNK_SEGMENT ? seqLen - segStart : kGDN_CHUNK_SEGMENT;
        int32_t const chunks = (segLen + kC - 1) / kC;
        gdnChunkPrepKernel<<<dim3(chunks, hv, n), kPrepThreads, sizeof(PrepSmem), stream>>>(
            q, k, a, b, A_log, dt_bias, contextLengths, records, qkTiles, segStart, chunksPerSeg, seqLen, h, hv);
        gdnChunkStateKernel<<<dim3(kD / kTV, hv, n), kStateThreads, sizeof(StateSmem), stream>>>(v, statePool,
            stateIndices, statePoolRows, contextLengths, records, qkTiles, o, segStart, segLen, chunksPerSeg, seqLen,
            h, hv);
        cudaError_t const err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    return cudaSuccess;
}

} // namespace kernel
} // namespace trt_edgellm
