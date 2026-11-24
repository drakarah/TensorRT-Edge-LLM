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

namespace trt_edgellm::kernel
{
//! Calibration helper: hessian[K, K] += X^T X over the T rows of X (FP16 [T, K], accumulated in FP32; only tiles on or below the diagonal are written, the
//! caller mirrors them). K must be a multiple of 64.
void launchInkling2HessianAccumulate(
    void const* x, float* hessian, int64_t tokens, int32_t dim, cudaStream_t stream);
} // namespace trt_edgellm::kernel
