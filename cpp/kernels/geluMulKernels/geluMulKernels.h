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
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace kernel
{

//! Fused GELU(tanh) * up with clamp, for a fused gate/up projection output.
//! input  [rows, 2 * width] FP16: gate = [:, :width], up = [:, width:]
//! output [rows, width]     FP16 = clamp(gelu_tanh(float(gate)) * float(up), -2048, 2048)
//! The arithmetic is FP32 (the FP16 product can overflow before the clamp), the same as the exported graph (Cast to FP32, Gelu, Mul, Clip, Cast to FP16).
//! Eight elements per thread with 128-bit accesses when width % 8 == 0 and the buffers are 16-byte aligned, scalar otherwise.
void launchGeluTanhMul(void const* input, void* output, int64_t rows, int32_t width, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
