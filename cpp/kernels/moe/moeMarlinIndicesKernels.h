/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cpuMissMailbox.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Compact copy of the Marlin routing arrays of the experts [e0, e1) (expert ids rebased to e - e0), for running the expert GEMMs chunk by chunk.
void launchBuildChunkMarlinIndicesKernel(int32_t const* paddedOffsets, int32_t const* numTokensPostPadded,
    int32_t const* sortedTokenIds, int32_t const* expertIds, int32_t e0, int32_t e1, int32_t numExperts,
    int32_t moeBlockSize, int32_t maxPaddedSlots, int32_t* outSortedTokenIds, int32_t* outExpertIds,
    int32_t* outNumTokensPostPadded, cudaStream_t stream);

/**
 * @brief Launch kernel to build Marlin indices from slot lists (per-expert).
 */
void launchBuildMarlinIndicesKernel(int32_t const* slotsByExpertWorkspace, int32_t const* slotsPerExpertWorkspace,
    int32_t const* paddedCounts, int32_t const* paddedOffsets, float const* topkWeights, int32_t* sortedTokenIds,
    float* topkWeightsFlat, int32_t* expertIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    int32_t moeBlockSize, cudaStream_t stream);

/**
 * @brief Build the trivial Marlin routing arrays for a dense (single-expert, topK=1) GEMM.
 *
 * The dense case degenerates the MoE routing: every one of the @p numTokens rows maps to expert 0 in order, so
 * sortedTokenIds is the identity 0..numTokens-1 with padded tail slots set to the out-of-range sentinel numTokens
 * (== numTokens*topK, matching marlin_template.h's `idx < prob_m*top_k` masking), expertIds is all zeros, and
 * numTokensPostPadded is the block-aligned row count. topkWeights is filled with 1.0f purely to keep the shared
 * Marlin call signature valid; it is unused because the dense wrapper passes mulTopkWeights=false.
 *
 * @param sortedTokenIds      [paddedRows] (INT32) output slot ids
 * @param expertIds           [paddedRows/moeBlockSize] (INT32) output, all zeros
 * @param numTokensPostPadded [1] (INT32) output, set to paddedRows
 * @param topkWeights         [paddedRows] (FP32) output, all 1.0f (unused by dense path)
 * @param numTokens           Number of real rows M
 * @param paddedRows          Block-aligned row count ceilDiv(M, moeBlockSize)*moeBlockSize
 * @param moeBlockSize        Marlin block size (8 for decode, 32 for prefill)
 */
void launchBuildDenseMarlinIndicesKernel(int32_t* sortedTokenIds, int32_t* expertIds, int32_t* numTokensPostPadded,
    float* topkWeights, int32_t numTokens, int32_t paddedRows, int32_t moeBlockSize, cudaStream_t stream);

/**
 * @brief Launch kernel to aggregate slot outputs back to tokens (sum over topK in slot order).
 */
void launchAggregateSlotOutputsKernel(void const* slotOutputs, void* aggregatedOutput, int32_t numTokens, int32_t topK,
    int32_t outDim, cudaStream_t stream);

/**
 * @brief Launch the BF16 slot-output aggregation kernel.
 *
 * Each output element is accumulated over topK slots in FP32 and converted to BF16 once at the final store.
 */
void launchAggregateSlotOutputsBf16Kernel(void const* slotOutputs, void* aggregatedOutput, int32_t numTokens,
    int32_t topK, int32_t outDim, cudaStream_t stream);


//! One staged tensor group: per-expert byte slices of up to four [numExperts, ...] tensors (each slice 16-byte multiple).
struct ExpertStagingCopy
{
    uint8_t const* src[4];
    uint8_t* dst[4];
    size_t bytesPerExpert[4];
    int32_t numTensors;
};

/**
 * @brief Copy the experts selected by @p topkIndices from @p copy.src (typically pinned host memory) into @p copy.dst
 * (device) at the same expert offsets; experts that no token selected are not copied. Graph-capture safe (no host sync).
 * @param neededWorkspace numExperts bytes of device scratch.
 * @param stats optional device counters [numExperts], incremented once per (token, selected expert).
 */
void launchStageRoutedExperts(int32_t const* topkIndices, int32_t count, int32_t numExperts, uint8_t* neededWorkspace,
    ExpertStagingCopy const& copy, cudaStream_t stream, int32_t* stats = nullptr);


//! Per-layer device-side LRU cache of experts: slotOf[numExperts] (-1 = not cached), expertOf[numSlots] (-1 = free),
//! lastUse[numSlots] (call stamp), counter[1], missList[1 + 2 * numSlots] = {count, (expert, slot)...}.
//! tally[3] (optional, may be null) accumulates {lookups, distinct experts needed, misses} over all calls.
struct ExpertCacheState
{
    int32_t* slotOf;
    int32_t* expertOf;
    int32_t* lastUse;
    int32_t* counter;
    int32_t* missList;
    int32_t numSlots;
    int32_t* tally{nullptr};
};

//! Look the selected experts up, assign slots to misses (LRU, never a slot used by this call) and rewrite topkIndices to slot ids.
void launchExpertCacheLookup(
    int32_t* topkIndices, int32_t count, int32_t numExperts, ExpertCacheState const& cache, cudaStream_t stream);

//! Copy the missed experts (copy.src = host expert arrays) into their slots of copy.dst (slot arrays, same per-expert sizes).
//! maxMisses = upper bound of the misses of one call (selected experts = tokens x top-k): the grid is sized by it, not by the number of slots.
void launchExpertCacheFill(ExpertCacheState const& cache, ExpertStagingCopy const& copy, int32_t maxMisses, cudaStream_t stream);

//! Cache-aware router for decode steps (<= a few tokens): selects top-k experts by logit + bias * [resident in the cache], weights = softmax of the
//! original logits over the selected experts. Replaces the top-k softmax kernel; writes expert ids (not slot ids) and fp32 weights.
//! tailSkip > 0: a non-resident expert at rank tailSkip+1 or lower gets weight 0 and the top-ranked expert's id (no fetch, no renormalisation).
void launchCacheAwareTopk(float const* logits, int32_t numTokens, int32_t numExperts, int32_t topK, float bias, int32_t tailSkip, ExpertCacheState const& cache,
    int32_t* topkIndices, float* topkWeights, cudaStream_t stream);

//! CPU miss path. Run after the router, before the cache lookup. A routed expert that is not in the cache is computed on the host unless its miss counter
//! (admit[expert], decayed every 256 calls; admit[numExperts] is the call counter) has reached admitThreshold, in which case the normal fill brings it in.
//! A host-computed slot is pointed at a resident expert with weight 0. The misses and the layer input are written to the mailbox, then its seq; nothing is
//! written when every expert is resident.
void launchCpuMissDoorbell(int32_t* topkIndices, float* topkWeights, int32_t numTokens, int32_t topK, int32_t numExperts, half const* hidden,
    int32_t hiddenSize, ExpertCacheState const& cache, int32_t* admit, int32_t admitThreshold, CpuMissMailbox* mailbox, CpuMissDeviceState* state,
    int32_t layer, cudaStream_t stream);

//! Wait for the host's answer to the last doorbell (no wait when it wrote none) and add it to output [numTokens, hiddenSize].
void launchCpuMissWaitAdd(half* output, int32_t numTokens, int32_t hiddenSize, CpuMissMailbox* mailbox, CpuMissDeviceState* state, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
