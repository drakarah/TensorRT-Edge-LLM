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
//! and at most 20480.
void launchQuantizeRowsInt8Permuted(
    half const* activations, int8_t* out, float* rowScale, int32_t M, int32_t K, cudaStream_t stream);

//! Largest K of the row kernels in this file (one FP16 row is staged in static shared memory).
constexpr int32_t kInt4W4A8MaxK{20480};

//! launchQuantizeRowsInt8Permuted over the activation SiLU(gate) * up computed on the fly (the SwiGLU product that feeds a down
//! projection), so the FP16 product never goes through memory. perm (int32 [K], may be null) is a column gather applied to the
//! product first: activation[m, j] = product[m, perm[j]]. Rounding points follow TensorRT's decomposed form (sigmoid, gate * sigmoid
//! and the product each rounded to FP16). K must be a multiple of 128 and at most kInt4W4A8MaxK.
void launchQuantizeRowsInt8PermutedSiluMul(half const* gate, half const* up, int32_t const* perm, int8_t* out,
    float* rowScale, int32_t M, int32_t K, cudaStream_t stream);

//! out[m, j] = (SiLU(gate) * up)[m, perm[j]] in FP16 (perm may be null); K a multiple of 8.
void launchSiluMulGather(
    half const* gate, half const* up, int32_t const* perm, half* out, int32_t M, int32_t K, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
