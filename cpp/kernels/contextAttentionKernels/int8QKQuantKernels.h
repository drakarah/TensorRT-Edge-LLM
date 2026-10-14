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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

//! Keys per K quantisation tile: matches the INT8 Q·Kᵀ FMHA's key tile.
constexpr int32_t kINT8_QK_KEY_TILE = 32;

//! Symmetric per-row INT8 quantisation of Q for the INT8 Q·Kᵀ FMHA.
//! q [B, S, H, D] fp16 -> qOut [B, S, H, D] int8, qScale [B, H, S] fp32 (= amax / 127). D % 256 == 0.
void quantizeQPerRowInt8(half const* q, int8_t* qOut, float* qScale, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headDim, cudaStream_t stream);

//! Symmetric INT8 quantisation of K with one scale per 32-key tile and head.
//! k [B, S, H, D] fp16 -> kOut [B, S, H, D] int8, kScale [B, H, ceil(S / 32)] fp32. D % 256 == 0.
void quantizeKPerTileInt8(half const* k, int8_t* kOut, float* kScale, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headDim, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
