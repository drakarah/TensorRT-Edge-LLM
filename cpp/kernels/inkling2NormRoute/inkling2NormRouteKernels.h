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

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kInkling2NormRouteMaxHidden = 4096;

//! Weighted RMS norm of the Inkling 2 layer inputs, optionally with the router matmul of the MoE input folded in:
//!   inv[t]    = 1 / sqrt(mean(x[t]^2) + eps)
//!   y16[t, k] = half(x[t, k] * inv[t] * normW[k])
//!   logits[t, e] = inv[t] * sum_k x[t, k] * routeW[e, k]         (only when numExperts > 0)
//! routeW is the router matrix with the norm weight already folded in (routeW[e, k] = router[e, k] * normW[k]), so logits equal the FP32 product of the normed activations with
//! the router, computed in FP32 (the TF32 matmul it replaces drops 13 mantissa bits). hidden must be a multiple of 4 and at most kInkling2NormRouteMaxHidden.
//! Up to 8 tokens with hidden <= 2560 (decode and speculative verification) run as one launch, a warp per router row; larger batches run a norm kernel and a tiled FP32 GEMM.
//! The workspace holds numTokens floats (inkling2NormRouteWorkspaceBytes), used by the batched path only.
void launchInkling2NormRoute(float const* x, float const* normW, float const* routeW, __half* y16, float* logits, int64_t numTokens, int32_t hiddenSize,
    int32_t numExperts, float eps, void* workspace, cudaStream_t stream);

size_t inkling2NormRouteWorkspaceBytes(int64_t numTokens);

} // namespace kernel
} // namespace trt_edgellm
