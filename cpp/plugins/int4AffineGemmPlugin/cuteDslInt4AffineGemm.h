/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Adapter that lets Int4AffineGroupwiseGemmPlugin dispatch to the CuTe DSL affine INT4 (MLX style,
// w = scale * q + bias, group size 64) W4A16 AOT kernels: 16 GEMM variants (CTA tile x pipeline
// stages x serial split-K) and one decode GEMV per M in [1, 8]. Weights are consumed in the
// fragment layout (bN=128, bK=64) produced at export time; scales and the shifted bias
// b' = fp16(bias + 8 * scale) are fp16 [K/64, N]. Gated on the cuteDSL artifact carrying the
// int4_affine_gemm group.
#ifdef CUTE_DSL_INT4_AFFINE_GEMM_ENABLED

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace plugins
{

//! Quantisation group size baked into the affine kernels.
constexpr int32_t kInt4AffineGroupSize = 64;

//! Largest M the decode GEMV has a compiled variant for.
constexpr int32_t kInt4AffineGemvMaxM = 8;

//! One compiled GEMM variant (CTA tile + pipeline depth + serial split-K). bN is pinned to 128.
struct CuteDslInt4AffineVariant
{
    int32_t bM;
    int32_t bN;
    int32_t bK;
    int32_t stages;
    int32_t splitK;
};

//! Number of compiled GEMM variants (16).
int32_t cuteDslInt4AffineNumVariants();

//! GEMM variant at table index [0, cuteDslInt4AffineNumVariants()).
CuteDslInt4AffineVariant const& cuteDslInt4AffineVariantAt(int32_t index);

//! True if the variant can serve (N, K): N > 0, K % 64 == 0 and splitK divides K / 64.
bool cuteDslInt4AffineVariantValid(CuteDslInt4AffineVariant const& v, int32_t N, int32_t K);

//! Load all GEMM and GEMV modules once. Returns false on any load failure.
bool cuteDslInt4AffineLoadModules();

//! Fragment weight buffer rows for (N, K): ceil(N/128) * (K/64) * 8. Each row is 128 uint32 words.
int64_t cuteDslInt4AffineFragmentRows(int32_t N, int32_t K);

//! Number of int32 serial split-K lock entries for a launch grid.
int64_t cuteDslInt4AffineLockCount(int32_t M, int32_t N, int32_t bM, int32_t bN);

//! Launch a GEMM variant. A [M,K] fp16, fragW fragment-order weights, scales / bias [K/64,N] fp16,
//! C [M,N] fp16, locks zero-initialised int32 [LockCount] (only read when splitK > 1). Returns 0 on success.
int32_t cuteDslInt4AffineGemmLaunch(CuteDslInt4AffineVariant const& v, void const* A, void const* fragW,
    void const* scales, void const* bias, void* C, void* locks, int32_t M, int32_t N, int32_t K, cudaStream_t stream);

//! Launch the M-variant GEMV (1 <= M <= 8). Returns 0 on success, -1 on unsupported M or failed launch.
int32_t cuteDslInt4AffineGemvLaunch(int32_t M, void const* A, void const* fragW, void const* scales, void const* bias,
    void* Out, int32_t N, int32_t K, cudaStream_t stream);

} // namespace plugins
} // namespace trt_edgellm

#endif // CUTE_DSL_INT4_AFFINE_GEMM_ENABLED
