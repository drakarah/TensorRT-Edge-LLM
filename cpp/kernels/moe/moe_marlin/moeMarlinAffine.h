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

/*
 * MoE W4A16 grouped GEMM, Marlin-based, with an AFFINE int4 dequant (Flint / MLX affine INT4, group size 64).
 *
 * Dequantised weight: w = scale * q + bias, q in [0, 15], scale/bias continuous per (n, group of 64 k).
 * The kernel computes it as (q - 8) * scale + biasPrime with a single FP16 FMA, where
 *   biasPrime = bias + 8 * scale
 * is precomputed in FP32 and rounded to FP16 at weight-repack time. biasPrime uses the same Marlin
 * permutation and layout as the scales.
 */

#pragma once

#include "common/tensor.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @param input Input activations [numTokens, K] (FP16)
 * @param output [numTokens * topK, N] (FP16) (or [numTokens, N] when topK rows are already expanded)
 * @param weights Marlin-packed INT4 weights [numExperts, K/16, N*16/8] (INT32), kU4 nibble layout
 * @param scales Marlin-permuted FP16 scales [numExperts, K/64, N]
 * @param biasPrime Marlin-permuted FP16 b' = bias + 8*scale, same shape as scales
 * @param sortedTokenIds, expertIds, numTokensPostPadded, topkWeights: as for moeAwqW4A16MarlinGemm
 * @param workspace sized by getMoeMarlinWorkspaceSize()
 */
void moeAffineW4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& scales, rt::Tensor const& biasPrime, rt::Tensor const& sortedTokenIds,
    rt::Tensor const& expertIds, rt::Tensor const& numTokensPostPadded, rt::Tensor const& topkWeights,
    rt::Tensor& workspace, int64_t moeBlockSize, int64_t topK, bool mulTopkWeights, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
