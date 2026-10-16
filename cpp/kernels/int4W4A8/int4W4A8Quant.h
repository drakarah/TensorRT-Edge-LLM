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

//! Per-row symmetric INT8 quantisation of FP16 activations for the W4A8 GEMM (int4_int8_gemm_ampere.py).
//! out[m, 32 c + q] holds activation column 32 c + r(q), the column permutation that lets the INT8 A fragments
//! meet the weight nibbles of the repacked INT4 buffer; rowScale[m] = absmax / 127. K must be a multiple of 128
//! and at most 16384.
void launchQuantizeRowsInt8Permuted(
    half const* activations, int8_t* out, float* rowScale, int32_t M, int32_t K, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
