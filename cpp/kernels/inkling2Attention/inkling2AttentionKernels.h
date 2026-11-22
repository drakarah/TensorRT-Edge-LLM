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

constexpr int32_t kInkling2HeadDim = 128;
constexpr int32_t kInkling2RelDim = 16;
constexpr int32_t kInkling2TokensPerPage = 128;

struct Inkling2AttentionParams
{
    int32_t numTokens;      //!< Packed query tokens over all sequences.
    int32_t batchSize;      //!< Sequences; queryStartOffsets has batchSize + 1 entries.
    int32_t numQHeads;
    int32_t numKVHeads;     //!< Must divide numQHeads.
    int32_t relExtent;      //!< Keys at distance >= relExtent get no relative bias.
    int32_t window;         //!< Sliding window in tokens; 0 selects full attention (with log-length query scaling).
    int32_t maxPagesPerSeq;
    float scaling;          //!< Attention score scale (1 / sqrt(headDim)).
    float logAlpha;         //!< Full layers: q *= 1 + logAlpha * log(max((pos + 1) / logNFloor, 1)).
    float logNFloor;
    float qClip;            //!< L2 clip threshold for the scaled query.
    float relClip;          //!< L2 clip threshold for the relative state.
};

//! One Inkling 2 attention visit: appends the new K/V rows to the paged pool, then computes
//! out = sigmoid(gate) * softmax([q.k * scaling + rel_bias(distance), sink]) V per query token and head.
//!
//! qLin/gateLin [T, nh * 128], relLin [T, nh * 16], sinkLin [T, nh], kNew/vNew [T, nkv * 128], out [T, nh * 128], all FP16.
//! qScale [nh * 128] and relScale [nh * 16] FP32 per-channel scales; relProjT [relExtent, 16] FP32 (distance-major).
//! kvPool [2, numPages, 128, nkv, 128] FP16; pageTable [batch, 2, maxPagesPerSeq] holds pool page indices over the
//! flattened (plane, page) axis, so V entries already include the plane offset. Sequence b owns the token slots
//! [queryStartOffsets[b], queryStartOffsets[b + 1]); only the first queryLengths[b] are real (the rest is padding and
//! is neither cached nor attended; its output rows are zero). pastLengths [batch] is the number of tokens already
//! cached; real token i of sequence b is at position pastLengths[b] + i.
void launchInkling2Attention(Inkling2AttentionParams const& params, half const* qLin, half const* relLin,
    half const* sinkLin, half const* gateLin, float const* qScale, float const* relScale, float const* relProjT,
    half const* kNew, half const* vNew, half* kvPool, int32_t const* queryStartOffsets, int32_t const* queryLengths,
    int32_t const* pastLengths, int32_t const* pageTable, half* out, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
