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

//! Largest hidden size of launchAddRmsNorm (one row is staged in shared memory, 2 bytes per element).
constexpr int32_t kAddRmsNormMaxHidden{8192};

//! Fused residual add and RMSNorm over rows of FP16 activations, one launch instead of TensorRT's two fused elementwise kernels:
//!   sum[m, :]  = x[m, :] + residual[m, :]                         (FP16 add, written when sumOut is not null)
//!   norm[m, :] = sum[m, :] / sqrt(mean(sum[m, :]^2) + eps) * gamma (mean and division in FP32)
//! weightBeforeCast != 0 multiplies by gamma in FP32 before the cast to FP16, 0 casts first and multiplies in FP16 (the two forms of
//! the builder's decomposed RMSNorm). residual may be null (plain RMSNorm). hidden must be a multiple of 8 and at most
//! kAddRmsNormMaxHidden; all buffers 16-byte aligned. sumOut may alias x or residual, normOut must not alias an input.
void launchAddRmsNorm(half const* x, half const* residual, half const* gamma, half* sumOut, half* normOut, int32_t rows,
    int32_t hidden, float eps, int32_t weightBeforeCast, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
