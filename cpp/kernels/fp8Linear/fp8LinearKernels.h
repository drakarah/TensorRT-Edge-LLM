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

constexpr int32_t kFp8LinearMaxGemvTokens = 4;

//! y[m, n] = scale[n] * sum_k x[m, k] * w[n, k]: FP8 (E4M3) weights [N, K] with one scale per output row, FP16 activations, FP32 accumulation, FP16 output.
//! The weights arrive as 4 bytes per int32 word ([N, K / 4]). K must be a multiple of 32, N of 8. Decode shapes (numTokens <= kFp8LinearMaxGemvTokens) run a
//! warp-per-row GEMV that streams the weights at memory bandwidth; longer inputs run a tensor-core GEMM that dequantises the weights tile by tile.
void launchFp8Linear(void const* x, void const* weightWords, void const* scale, void* y, int32_t numTokens, int32_t n, int32_t k, bool scaleIsFp32,
    cudaStream_t stream);

//! Which path launchFp8Linear takes (for tests).
bool fp8LinearUsesGemv(int32_t numTokens);

} // namespace kernel
} // namespace trt_edgellm
