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
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kInkling2GateMaxHidden = 4096;
constexpr int32_t kInkling2GateMaxRank = 128;

//! Inkling 2 message gate per token (all FP32 activations, BF16 weights):
//!   gate_in = [h / sqrt(mean(h^2) + eps), m / sqrt(mean(m^2) + eps)]          (2 * hidden)
//!   low[j]  = sum_k down[j, k] * gate_in[k]                                    (rank)
//!   forget[c] = tanh(sum_j forgetW[c, j] * low[j]),  gain[c] = tanh(sum_j gainW[c, j] * low[j])
//!   out[c]  = (1 - forget[c]) * h[c] + gain[c] * m[c]
//! The weights arrive as BF16 bit pairs packed in 32-bit words: down [rank, 2 * hidden / 2], forgetW and gainW [hidden, rank / 2]. BF16 widens to FP32 exactly, so the
//! result equals an FP32 evaluation of the checkpoint's BF16 weights. hidden must be a multiple of 8 and rank a multiple of 64, rank <= kInkling2GateMaxRank. The
//! workspace holds numTokens * rank floats (inkling2MessageGateWorkspaceBytes).
void launchInkling2MessageGate(float const* hidden, float const* message, uint32_t const* down, uint32_t const* forgetW, uint32_t const* gainW, float* out,
    int64_t numTokens, int32_t hiddenSize, int32_t rank, float eps, void* workspace, cudaStream_t stream);

size_t inkling2MessageGateWorkspaceBytes(int64_t numTokens, int32_t rank);

} // namespace kernel
} // namespace trt_edgellm
