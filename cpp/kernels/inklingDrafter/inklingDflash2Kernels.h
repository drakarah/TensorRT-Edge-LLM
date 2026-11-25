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

//! Kernels of the Inkling 2 Mini DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2: 5 layers, hidden 2304, 32 query / 8 KV heads of 128, block of 8 rows, window 2048).
//! The GEMVs, the context K/V, the block initialisation and the swiglu come from inklingDrafterKernels.h; the dynamic convolutions and the candidate selector are the
//! fork's DFlash2 kernels (dflash2GroupedDynamicConv.h, dflash2CandidateSelector.h).
constexpr int32_t kDfl2Hidden = 2304;
constexpr int32_t kDfl2Heads = 32;
constexpr int32_t kDfl2KvHeads = 8;
constexpr int32_t kDfl2Group = 4; //!< query heads per KV head
constexpr int32_t kDfl2HeadDim = 128;
constexpr int32_t kDfl2Block = 8;
constexpr int32_t kDfl2QueryRows = kDfl2Block * kDfl2Group; //!< query vectors per KV head

//! taps [rows, 4 * 2304] FP32 (the stream entering visits 5, 23, 41 and after the last visit) -> out [rows, 4 * 2304] FP16: the first three slices converted, the fourth
//! RMS-normalised with the target's final norm weights.
void launchDfl2CtxPrep(float const* taps, int32_t rows, float const* targetFinalNorm, __half* out, cudaStream_t stream);

//! out [rows, 2304] FP16 = RMS(x [rows, 2304] FP16) * w.
void launchDfl2RmsHalf(__half const* x, float const* w, __half* out, int32_t rows, cudaStream_t stream);

//! Block q/k/v: per head RMS norms (qNorm, kNorm [128]) and RoPE (NeoX, theta) at positions pos0 + row for q and k; v unchanged. qRaw [8, 4096], kRaw/vRaw [8, 1024].
void launchDfl2QkvPrep(__half const* qRaw, __half const* kRaw, __half const* vRaw, float const* qNorm, float const* kNorm, int32_t pos0, float theta, __half* q, __half* kb,
    __half* vb, cudaStream_t stream);

//! Attention of the 8 block rows (positions qpos0 + row) over context keys [lo, lo + nCtx) plus the 8 block keys. A context key at position p is visible to row r when
//! qpos0 + r - p < window (the block keys are all visible). out [8, 4096] FP16. scores [256 * scoreStride] FP32 (scoreStride >= nCtx + 8), partial [maxSplits * 256 * 128] FP32.
void launchDfl2Attention(__half const* q, __half const* ctxK, __half const* ctxV, __half const* kb, __half const* vb, int32_t lo, int32_t nCtx, int32_t qpos0, int32_t window,
    float* scores, int32_t scoreStride, float* partial, int32_t maxSplits, __half* out, cudaStream_t stream);

//! out [rows, cols] FP32 = in [rows, cols] FP16 for the first validCols columns, -1e30 for the padding columns.
void launchDfl2LogitsToFloat(__half const* in, float* out, int32_t rows, int32_t cols, int32_t validCols, cudaStream_t stream);

//! values [rows, k] FP32 = logits[row, indices[row, j]] (the top-k values of rows whose indices selectAllTopK returned).
void launchDfl2GatherValues(float const* logits, int32_t const* indices, int32_t rows, int32_t cols, int32_t k, float* values, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
