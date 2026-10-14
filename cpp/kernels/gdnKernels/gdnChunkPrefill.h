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

#pragma once

// Chunked GDN (gated delta rule) prefill for sm_80-class GPUs, tensor cores (WMMA fp16 -> fp32).
//
// The sequential prefill kernel updates the 128 x 128 state once per token on CUDA cores. In chunk form (WY
// representation, as flash-linear-attention's chunk_gated_delta_rule) a chunk of C = 64 tokens becomes small
// matrix products. With G the in-chunk inclusive cumsum of log g, D[i][j] = exp(G_i - G_j):
//   A = strict_tril(beta_i (K K^T) * D)            T = (I + A)^-1
//   Attn = tril(Q K^T * D)
//   R = beta * (V - exp(G) * (K S))                new_v = T R
//   O = exp(G) * (Q S) + Attn new_v
//   S <- exp(G_last) S + K^T (exp(G_last - G) * new_v)
// Kernel 1 (one CTA per chunk x v head x sequence, fully parallel) normalises q / k and builds T (blocked
// inverse on 16 x 16 tiles) and Attn; kernel 2 (one CTA of 8 warps per 32-column V tile x v head x sequence) walks
// the chunks with the state in WMMA accumulators, loading each chunk's tiles with one bulk cp.async. The prompt is
// processed in segments of up to kGDN_CHUNK_SEGMENT tokens so the workspace stays bounded.
//
// Shapes follow the GDN plugin prefill contract (dk = dv = 128, q/k L2-normalised in the kernel, q scaled by
// 1/sqrt(dk)): q, k [n, seqLen, h, 128] fp16; v, o [n, seqLen, hv, 128] fp16; a, b [n, seqLen, hv] fp16;
// A_log [hv] fp32; dt_bias [hv] fp16; statePool [poolRows, hv, 128, 128] fp32 updated in place for
// stateIndices[n] (an out-of-range row starts from a zero state and is not written); contextLengths [n]
// (positions past a sequence's length get o = 0 and do not touch the state; nullptr = all seqLen).
// Operands are rounded to fp16 for the tensor cores, so results differ from the fp32 sequential kernel at about
// fp16 output precision.

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kGDN_CHUNK{64};
constexpr int32_t kGDN_CHUNK_SEGMENT{2048};

//! Workspace bytes gdnChunkPrefill needs for n sequences, h key heads and hv value heads.
size_t gdnChunkPrefillWorkspaceBytes(int32_t n, int32_t h, int32_t hv);

//! Runs the chunked prefill. Returns the first launch error (cudaSuccess on success).
cudaError_t gdnChunkPrefill(__half const* q, __half const* k, __half const* v, __half const* a, __half const* b,
    float const* A_log, __half const* dt_bias, float* statePool, int32_t const* stateIndices, int32_t statePoolRows,
    int32_t const* contextLengths, __half* o, int32_t n, int32_t seqLen, int32_t h, int32_t hv, void* workspace,
    cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
