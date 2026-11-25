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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kInkling2MaxResidualStates = 8;
constexpr int32_t kInkling2MaxResidualQueries = 4;
constexpr int32_t kInkling2MaxResidualHidden = 4096;

struct Inkling2BlockResidualParams
{
    int64_t numTokens;
    int32_t hidden;
    int32_t numStates;
    int32_t numQueries;
    float eps;
    float const* states[kInkling2MaxResidualStates];
    float* outputs[kInkling2MaxResidualQueries];
};

//! Per token: normalise each state row to unit RMS (x / sqrt(mean(x^2) + eps)), score it against every query row
//! (dot product scaled by hidden^-0.5), softmax over the states, and write the probability-weighted sum of the
//! normalised states, one output [T, hidden] per query. All FP32.
//! With a workspace (inkling2BlockResidualWorkspaceBytes(numTokens) bytes) the hidden dimension is split over kInkling2ResidualSplit blocks per token in two
//! kernels (partial reductions, then softmax and weighted sums), so decode no longer runs on a single SM; without one a single block per token is used.
void launchInkling2BlockResidual(
    Inkling2BlockResidualParams const& params, float const* queries, cudaStream_t stream, void* workspace = nullptr);

constexpr int32_t kInkling2ResidualSplit = 9; //!< 576 float4 of a hidden-2304 row = 9 slices of 64: every warp iteration is full (METRICS 217)

//! Debug capture of a [rows, hidden] tensor into a device ring (rows appended at the device-side counter; the step size is logged when stepLog is not null).
void launchTapCapture(float const* src, int64_t rows, int32_t hidden, float* dst, int32_t* counter, int32_t capacityRows, int32_t* stepLog, int32_t* stepCounter,
    int32_t capacitySteps, cudaStream_t stream);

size_t inkling2BlockResidualWorkspaceBytes(int64_t numTokens);

} // namespace kernel
} // namespace trt_edgellm
