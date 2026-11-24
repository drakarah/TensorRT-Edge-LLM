/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Kernels of the Inkling 2 DSpark drafter (3 layers, hidden 2304, 24 query / 8 KV heads of 128, block of 8 rows, rank-256 Markov head).
constexpr int32_t kDrafterHidden = 2304;
constexpr int32_t kDrafterHeads = 24;
constexpr int32_t kDrafterKvHeads = 8;
constexpr int32_t kDrafterHeadDim = 128;
constexpr int32_t kDrafterBlock = 8;
constexpr int32_t kDrafterMarkov = 256;

//! y [tokens, n] FP16 = scale[n] * (x [tokens, k] FP16 times W^T), W INT8 [n, k] pre-shuffled by the drafter converter (8 rows x 64 k slabs of 32 lanes x 16 bytes: each lane holds its
//! mma B fragments of four k16 tiles), tensor cores (m16n8k16, tokens <= 8, n % 8 == 0, k % 64 == 0).
void launchDrafterGemv(void const* x, void const* wShuf, void const* scale, void* y, int32_t tokens, int32_t n, int32_t k, cudaStream_t stream);

//! Context features of `rows` target positions (taps [rows, 4 * 2304] FP32: the stream entering visits 5, 23, 41 and after the last visit): mixes the four taps
//! with softmax(fuse_weight) once per consumer (three layers and the Markov head), RMS-normalises each mix with that consumer's dflare weights and writes the
//! three layer inputs ctxn [3, consumerStride, 2304] (FP16, rows used). The head feature of the last row (consumer 3) is written to lastFeat [2304] (FP32) when lastFeat is not null.
//! The fourth tap is normalised with the target's final norm weights first.
void launchDrafterCtxMix(float const* taps, int32_t rows, int32_t consumerStride, float const* alpha, float const* dflare, float const* dflareMarkov,
    float const* targetFinalNorm, __half* ctxn, float* lastFeat, cudaStream_t stream);

//! Context K/V of rows at positions pos0..: per KV head RMS norm (kNormCtx [128]) and RoPE (NeoX, theta) of kRaw, plain copy of vRaw, into the caches at
//! [position, 8, 128].
void launchDrafterCtxKv(__half const* kRaw, __half const* vRaw, int32_t rows, int32_t pos0, float const* kNormCtx, float theta, __half* ctxK, __half* ctxV,
    cudaStream_t stream);

//! x row 0 = RMS(embedding) * embedNorm, rows 1..7 = mask embedding (FP32 [8, 2304]).
void launchDrafterInitBlock(__half const* embedding, float const* embedNorm, float const* maskEmbed, float* x, cudaStream_t stream);

//! out [rows, 2304] FP16 = RMS(x [rows, 2304] FP32) * w.
void launchDrafterRms(float const* x, float const* w, __half* out, int32_t rows, int32_t rowOffset, cudaStream_t stream);

//! Block q/k/v: per head RMS norms (qNorm, kNorm [128]) and RoPE at positions pos0 + row for q and k; v unchanged. qRaw [8, 3072], kRaw/vRaw [8, 1024] FP16.
void launchDrafterQkvPrep(__half const* qRaw, __half const* kRaw, __half const* vRaw, float const* qNorm, float const* kNorm, int32_t pos0, float theta,
    __half* q, __half* kb, __half* vb, cudaStream_t stream);

//! Attention of the 8 block rows over keys [lo, lo + nCtx) of the context caches plus the 8 block keys (all rows see all block keys). Writes out [8, 3072] FP16.
//! scores/partial are scratch: scores [8 * 24 * (nCtx + 8)] FP32 at least, partial [maxSplits * 8 * 24 * 128] FP32.
void launchDrafterAttention(__half const* q, __half const* ctxK, __half const* ctxV, __half const* kb, __half const* vb, int32_t lo, int32_t nCtx, float* scores,
    int32_t scoreStride, float* partial, int32_t maxSplits, __half* out, cudaStream_t stream);

//! x [n] FP32 += y [n] FP16.
void launchDrafterAddResidual(float* x, __half const* y, int32_t n, cudaStream_t stream);

//! out = silu(g) * u (FP16).
void launchDrafterSwiglu(__half const* g, __half const* u, __half* out, int32_t n, cudaStream_t stream);

//! xs [7, 4608] FP16: row j = [hid_j (from hidNormed [8, 2304], rows 1..7), feat FP16].
void launchDrafterBuildInv(__half const* hidNormed, float const* lastFeat, __half* xs, cudaStream_t stream);

//! One Markov step j: proj = inv[j] + bias + W_state [768, 512] x [state, w1[prev]], gates, new state, o = tanh(outp) (FP16 [256]).
void launchDrafterMarkovGate(__half const* inv, float const* bias, float const* stateW, __half const* markovW1, int32_t const* prevToken, float const* stateIn,
    float* stateOut, __half* o, int32_t j, cudaStream_t stream);

//! Argmax over v of base[j, v] + corr[v] + bU[v]; writes the token to draftTokens[j] and prevToken. scratch: 2 * 128 values.
void launchDrafterArgmax(__half const* base, __half const* corr, float const* bU, int32_t vocabPadded, int32_t j, float* scratchVal, int32_t* scratchIdx,
    int32_t* draftTokens, int32_t* prevToken, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
