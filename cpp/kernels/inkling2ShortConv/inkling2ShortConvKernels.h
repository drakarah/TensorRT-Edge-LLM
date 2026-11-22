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
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kInkling2MaxConvKernel = 8;

struct Inkling2ShortConvParams
{
    int32_t numTokens;   //!< Packed token slots over all sequences.
    int32_t batchSize;   //!< queryStartOffsets has batchSize + 1 entries.
    int32_t channels;    //!< C
    int32_t stateStride; //!< Channel stride of the state tensor (state is [rows, stateChannels, kernelSize]).
    int32_t stateRows;
    int32_t kernelSize;  //!< k, at most kInkling2MaxConvKernel.
    int32_t groups;      //!< 0: static taps only; otherwise channels / groups consecutive channels share dynamic taps.
    float scale;         //!< Applied to the sum of the convolution and the residual input.
};

//! Inkling 2 short convolution over each sequence's token history:
//!   y[t, c] = scale * (sum_{j<k} (taps[c, j] + dyn[t, c / (C / groups), j]) * x[t - j, c] + x[t, c])
//! with lag j = 0 the current token. Inputs before a sequence start come from its rolling state
//! state[stateIndices[b], c, :], which holds the last k - 1 inputs oldest first (slots 0..k-2; slot k-1 is unused).
//! Padding slots (local index >= queryLengths[b]) produce zero and leave the state untouched; the state is updated
//! in place after all outputs were computed. x, dyn, y, state FP16; taps FP32 [C, k].
void launchInkling2ShortConv(Inkling2ShortConvParams const& params, half const* x, float const* taps, half const* dyn,
    half* state, int32_t const* queryStartOffsets, int32_t const* queryLengths, int32_t const* stateIndices, half* y,
    cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
