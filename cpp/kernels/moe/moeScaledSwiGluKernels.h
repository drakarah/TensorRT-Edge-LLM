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

/*!
 * SwiGLU with a power-of-two output scale, evaluated entirely in FP32 before the single FP16 rounding:
 *   out[r, i] = half(silu(float(gate[r, i])) * float(up[r, i]) * scale)
 * Input layout [rows, 2 * intermediateDim] (gate first, then up), output [rows, intermediateDim].
 *
 * Flint's massive-activation tokens make |silu(gate) * up| exceed the FP16 range (7.6e4 at layer 46); the caller passes
 * scale = 2^-k and folds 2^k into the routing weights of the down GEMM (exact, both are powers of two).
 * intermediateDim must be a multiple of 8 and both pointers 16-byte aligned.
 */
void moeScaledSwiGlu(
    half const* gateUp, half* output, int64_t numRows, int64_t intermediateDim, float scale, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
