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

#include "common/cudaUtils.h"
#include "moeMarlinIndicesKernels.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

// Build Marlin indices from slot lists
__global__ void buildMarlinIndicesKernel(int32_t const* slotsByExpertWorkspace, int32_t const* slotsPerExpertWorkspace,
    int32_t const* paddedCounts, int32_t const* paddedOffsets, float const* topkWeights, int32_t* sortedTokenIds,
    float* topkWeightsFlat, int32_t* expertIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    int32_t moeBlockSize)
{
    int32_t expertId = blockIdx.x;
    if (expertId >= numExperts)
        return;

    int32_t count = slotsPerExpertWorkspace[expertId];
    int32_t paddedCount = paddedCounts[expertId];
    if (paddedCount <= 0)
        return;

    int32_t totalSlots = numTokens * topK;
    int32_t outStart = paddedOffsets[expertId];

    for (int32_t i = threadIdx.x; i < paddedCount; i += blockDim.x)
    {
        if (i < count)
        {
            int32_t slot = slotsByExpertWorkspace[expertId * totalSlots + i];
            sortedTokenIds[outStart + i] = slot;
            topkWeightsFlat[outStart + i] = topkWeights[slot];
        }
        else
        {
            sortedTokenIds[outStart + i] = totalSlots;
            topkWeightsFlat[outStart + i] = 0.0f;
        }
    }

    int32_t numBlocks = paddedCount / moeBlockSize;
    int32_t blockOffset = outStart / moeBlockSize;
    for (int32_t b = threadIdx.x; b < numBlocks; b += blockDim.x)
    {
        expertIds[blockOffset + b] = expertId;
    }
}

// Build the degenerate Marlin routing arrays for a dense (single-expert, topK=1) GEMM.
__global__ void buildDenseMarlinIndicesKernel(int32_t* sortedTokenIds, int32_t* expertIds, int32_t* numTokensPostPadded,
    float* topkWeights, int32_t numTokens, int32_t paddedRows, int32_t moeBlockSize)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const stride = gridDim.x * blockDim.x;

    // sortedTokenIds is the identity map with padded-tail slots masked by the out-of-range sentinel numTokens
    // (== numTokens*topK for topK=1). topkWeights is filled with 1.0f only to keep the shared Marlin signature valid.
    for (int32_t i = idx; i < paddedRows; i += stride)
    {
        sortedTokenIds[i] = i < numTokens ? i : numTokens;
        topkWeights[i] = 1.0f;
    }

    int32_t const numBlocks = paddedRows / moeBlockSize;
    for (int32_t b = idx; b < numBlocks; b += stride)
    {
        expertIds[b] = 0;
    }

    if (idx == 0)
    {
        numTokensPostPadded[0] = paddedRows;
    }
}

namespace
{

constexpr int32_t kAggregateThreadsPerBlock{256};

template <typename T>
struct AggregationTypeTraits;

template <>
struct AggregationTypeTraits<half>
{
    static __device__ float toFloat(half const value)
    {
        return __half2float(value);
    }

    static __device__ half fromFloat(float const value)
    {
        return __float2half(value);
    }
};

template <>
struct AggregationTypeTraits<__nv_bfloat16>
{
    static __device__ float toFloat(__nv_bfloat16 const value)
    {
        return __bfloat162float(value);
    }

    static __device__ __nv_bfloat16 fromFloat(float const value)
    {
        return __float2bfloat16_rn(value);
    }
};

// Aggregate slot outputs back to tokens: sum over topK in slot order.
template <typename T>
__global__ void aggregateSlotOutputsKernel(
    T const* slotOutputs, T* aggregatedOutput, int32_t numTokens, int32_t topK, int32_t outDim)
{
    int32_t const tokenId = blockIdx.x;
    int32_t const dimIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (tokenId >= numTokens || dimIdx >= outDim)
    {
        return;
    }

    float accum = 0.0F;
    int32_t const base = tokenId * topK;
    for (int32_t k = 0; k < topK; ++k)
    {
        int32_t const slot = base + k;
        accum += AggregationTypeTraits<T>::toFloat(slotOutputs[slot * outDim + dimIdx]);
    }

    aggregatedOutput[tokenId * outDim + dimIdx] = AggregationTypeTraits<T>::fromFloat(accum);
}

template <typename T>
void launchAggregateSlotOutputsKernelImpl(
    T const* slotOutputs, T* aggregatedOutput, int32_t numTokens, int32_t topK, int32_t outDim, cudaStream_t stream)
{
    dim3 const grid(numTokens, static_cast<uint32_t>(trt_edgellm::divUp(outDim, kAggregateThreadsPerBlock)));
    dim3 const block(kAggregateThreadsPerBlock);
    aggregateSlotOutputsKernel<T><<<grid, block, 0, stream>>>(slotOutputs, aggregatedOutput, numTokens, topK, outDim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launchBuildMarlinIndicesKernel(int32_t const* slotsByExpertWorkspace, int32_t const* slotsPerExpertWorkspace,
    int32_t const* paddedCounts, int32_t const* paddedOffsets, float const* topkWeights, int32_t* sortedTokenIds,
    float* topkWeightsFlat, int32_t* expertIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    int32_t moeBlockSize, cudaStream_t stream)
{
    buildMarlinIndicesKernel<<<numExperts, 256, 0, stream>>>(slotsByExpertWorkspace, slotsPerExpertWorkspace,
        paddedCounts, paddedOffsets, topkWeights, sortedTokenIds, topkWeightsFlat, expertIds, numTokens, topK,
        numExperts, moeBlockSize);
    CUDA_CHECK(cudaGetLastError());
}

namespace
{
//! Slice of the Marlin routing arrays that belongs to the experts [e0, e1): experts own contiguous padded blocks, so a chunk of experts is a contiguous
//! range of sorted token ids / block expert ids. The slice is copied to compact arrays (expert ids rebased to e - e0) so the Marlin GEMM can run on a
//! chunk of the expert weights without any change to its kernel.
__global__ void buildChunkMarlinIndicesKernel(int32_t const* paddedOffsets, int32_t const* numTokensPostPadded,
    int32_t const* sortedTokenIds, int32_t const* expertIds, int32_t e0, int32_t e1, int32_t numExperts,
    int32_t moeBlockSize, int32_t* outSortedTokenIds, int32_t* outExpertIds, int32_t* outNumTokensPostPadded)
{
    int32_t const start = paddedOffsets[e0];
    int32_t const end = e1 >= numExperts ? numTokensPostPadded[0] : paddedOffsets[e1];
    int32_t const n = end - start;
    int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < n; i += stride)
    {
        outSortedTokenIds[i] = sortedTokenIds[start + i];
    }
    int32_t const numBlocks = n / moeBlockSize;
    int32_t const blockStart = start / moeBlockSize;
    for (int32_t b = tid; b < numBlocks; b += stride)
    {
        outExpertIds[b] = expertIds[blockStart + b] - e0;
    }
    if (tid == 0)
    {
        outNumTokensPostPadded[0] = n;
    }
}
} // namespace

void launchBuildChunkMarlinIndicesKernel(int32_t const* paddedOffsets, int32_t const* numTokensPostPadded,
    int32_t const* sortedTokenIds, int32_t const* expertIds, int32_t e0, int32_t e1, int32_t numExperts,
    int32_t moeBlockSize, int32_t maxPaddedSlots, int32_t* outSortedTokenIds, int32_t* outExpertIds,
    int32_t* outNumTokensPostPadded, cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    int32_t const blocks = std::max(1, std::min(static_cast<int32_t>(trt_edgellm::divUp(maxPaddedSlots, kThreads)), 64));
    buildChunkMarlinIndicesKernel<<<blocks, kThreads, 0, stream>>>(paddedOffsets, numTokensPostPadded, sortedTokenIds,
        expertIds, e0, e1, numExperts, moeBlockSize, outSortedTokenIds, outExpertIds, outNumTokensPostPadded);
    CUDA_CHECK(cudaGetLastError());
}

void launchBuildDenseMarlinIndicesKernel(int32_t* sortedTokenIds, int32_t* expertIds, int32_t* numTokensPostPadded,
    float* topkWeights, int32_t numTokens, int32_t paddedRows, int32_t moeBlockSize, cudaStream_t stream)
{
    constexpr int32_t kThreadsPerBlock = 256;
    int32_t const grid = static_cast<int32_t>(trt_edgellm::divUp(std::max(paddedRows, 1), kThreadsPerBlock));
    buildDenseMarlinIndicesKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        sortedTokenIds, expertIds, numTokensPostPadded, topkWeights, numTokens, paddedRows, moeBlockSize);
    CUDA_CHECK(cudaGetLastError());
}

void launchAggregateSlotOutputsKernel(void const* slotOutputs, void* aggregatedOutput, int32_t numTokens, int32_t topK,
    int32_t outDim, cudaStream_t stream)
{
    launchAggregateSlotOutputsKernelImpl(
        static_cast<half const*>(slotOutputs), static_cast<half*>(aggregatedOutput), numTokens, topK, outDim, stream);
}

void launchAggregateSlotOutputsBf16Kernel(void const* slotOutputs, void* aggregatedOutput, int32_t numTokens,
    int32_t topK, int32_t outDim, cudaStream_t stream)
{
    launchAggregateSlotOutputsKernelImpl(static_cast<__nv_bfloat16 const*>(slotOutputs),
        static_cast<__nv_bfloat16*>(aggregatedOutput), numTokens, topK, outDim, stream);
}


namespace
{
__global__ void markRoutedExpertsKernel(
    int32_t const* topkIndices, int32_t count, int32_t numExperts, uint8_t* needed, int32_t* stats)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        int32_t const e = topkIndices[idx];
        if (e >= 0 && e < numExperts)
        {
            needed[e] = 1;
            if (stats != nullptr)
            {
                atomicAdd(&stats[e], 1);
            }
        }
    }
}

// grid = (experts, chunks, tensors); each block copies one chunk of one expert's slice of one tensor with 16-byte loads
// (host memory is read through its UVA address, so many loads must be in flight to reach PCIe line rate).
__global__ void stageRoutedExpertsKernel(uint8_t const* needed, ExpertStagingCopy copy)
{
    int32_t const expert = blockIdx.x;
    if (needed[expert] == 0)
    {
        return;
    }
    int32_t const t = blockIdx.z;
    size_t const bytes = copy.bytesPerExpert[t];
    size_t const chunk = (bytes / 16 + gridDim.y - 1) / gridDim.y;
    size_t const begin = chunk * blockIdx.y;
    size_t const end = min(begin + chunk, bytes / 16);
    uint4 const* src = reinterpret_cast<uint4 const*>(copy.src[t] + bytes * expert);
    uint4* dst = reinterpret_cast<uint4*>(copy.dst[t] + bytes * expert);
    // kUnroll independent loads per thread before the stores: reads through PCIe are latency-bound, so the bytes in flight
    // (blocks x threads x kUnroll x 16 B) set the bandwidth.
    constexpr int32_t kUnroll = 8;
    size_t const stride = static_cast<size_t>(blockDim.x) * kUnroll;
    for (size_t base = begin; base < end; base += stride)
    {
        uint4 v[kUnroll];
#pragma unroll
        for (int32_t u = 0; u < kUnroll; ++u)
        {
            size_t const i = base + threadIdx.x + static_cast<size_t>(u) * blockDim.x;
            if (i < end)
            {
                v[u] = src[i];
            }
        }
#pragma unroll
        for (int32_t u = 0; u < kUnroll; ++u)
        {
            size_t const i = base + threadIdx.x + static_cast<size_t>(u) * blockDim.x;
            if (i < end)
            {
                dst[i] = v[u];
            }
        }
    }
}
} // namespace

void launchStageRoutedExperts(int32_t const* topkIndices, int32_t count, int32_t numExperts, uint8_t* neededWorkspace,
    ExpertStagingCopy const& copy, cudaStream_t stream, int32_t* stats)
{
    cudaMemsetAsync(neededWorkspace, 0, numExperts, stream);
    markRoutedExpertsKernel<<<(count + 255) / 256, 256, 0, stream>>>(topkIndices, count, numExperts, neededWorkspace, stats);
    dim3 const grid(numExperts, 16, copy.numTensors);
    stageRoutedExpertsKernel<<<grid, 256, 0, stream>>>(neededWorkspace, copy);
}


namespace
{
// One block: looks the experts selected by the (<= a few) tokens up in a per-layer slot cache, assigns the least recently used
// slot (not used by this call) to every miss, records the misses and rewrites the top-k expert ids to slot ids.
__global__ void expertCacheLookupKernel(int32_t* topkIndices, int32_t count, int32_t numExperts, ExpertCacheState cache)
{
    __shared__ int32_t stamp;
    extern __shared__ uint8_t needed[]; // [numExperts]
    for (int32_t e = threadIdx.x; e < numExperts; e += blockDim.x)
    {
        needed[e] = 0;
    }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < count; i += blockDim.x)
    {
        int32_t const e = topkIndices[i];
        if (e >= 0 && e < numExperts)
        {
            needed[e] = 1;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        stamp = ++cache.counter[0];
        int32_t misses = 0;
        int32_t distinct = 0;
        for (int32_t e = 0; e < numExperts; ++e)
        {
            if (needed[e] == 0)
            {
                continue;
            }
            ++distinct;
            int32_t slot = cache.slotOf[e];
            if (slot < 0)
            {
                int32_t best = -1;
                int32_t bestUse = 0x7fffffff;
                for (int32_t s = 0; s < cache.numSlots; ++s)
                {
                    int32_t const use = cache.lastUse[s];
                    if (use != stamp && use < bestUse)
                    {
                        bestUse = use;
                        best = s;
                    }
                }
                if (best < 0)
                {
                    continue; // more distinct experts than slots: the caller keeps this path to <= numSlots / topK tokens
                }
                int32_t const old = cache.expertOf[best];
                if (old >= 0)
                {
                    cache.slotOf[old] = -1;
                }
                cache.slotOf[e] = best;
                cache.expertOf[best] = e;
                cache.missList[1 + 2 * misses] = e;
                cache.missList[2 + 2 * misses] = best;
                ++misses;
                slot = best;
            }
            cache.lastUse[slot] = stamp;
        }
        cache.missList[0] = misses;
        if (cache.tally != nullptr)
        {
            cache.tally[0] += 1;
            cache.tally[1] += distinct;
            cache.tally[2] += misses;
        }
    }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < count; i += blockDim.x)
    {
        int32_t const e = topkIndices[i];
        if (e >= 0 && e < numExperts)
        {
            int32_t const slot = cache.slotOf[e];
            topkIndices[i] = slot >= 0 ? slot : 0;
        }
    }
}

// grid = (numSlots, chunks, tensors): copies the experts listed as misses from host memory into their cache slots.
__global__ void expertCacheFillKernel(ExpertCacheState cache, ExpertStagingCopy copy)
{
    if (static_cast<int32_t>(blockIdx.x) >= cache.missList[0])
    {
        return;
    }
    int32_t const expert = cache.missList[1 + 2 * blockIdx.x];
    int32_t const slot = cache.missList[2 + 2 * blockIdx.x];
    int32_t const t = blockIdx.z;
    size_t const bytes = copy.bytesPerExpert[t];
    size_t const chunk = (bytes / 16 + gridDim.y - 1) / gridDim.y;
    size_t const begin = chunk * blockIdx.y;
    size_t const end = min(begin + chunk, bytes / 16);
    uint4 const* src = reinterpret_cast<uint4 const*>(copy.src[t] + bytes * expert);
    uint4* dst = reinterpret_cast<uint4*>(copy.dst[t] + bytes * slot);
    constexpr int32_t kUnroll = 8;
    size_t const stride = static_cast<size_t>(blockDim.x) * kUnroll;
    for (size_t base = begin; base < end; base += stride)
    {
        uint4 v[kUnroll];
#pragma unroll
        for (int32_t u = 0; u < kUnroll; ++u)
        {
            size_t const i = base + threadIdx.x + static_cast<size_t>(u) * blockDim.x;
            if (i < end)
            {
                v[u] = src[i];
            }
        }
#pragma unroll
        for (int32_t u = 0; u < kUnroll; ++u)
        {
            size_t const i = base + threadIdx.x + static_cast<size_t>(u) * blockDim.x;
            if (i < end)
            {
                dst[i] = v[u];
            }
        }
    }
}
} // namespace

void launchExpertCacheLookup(
    int32_t* topkIndices, int32_t count, int32_t numExperts, ExpertCacheState const& cache, cudaStream_t stream)
{
    expertCacheLookupKernel<<<1, 128, numExperts, stream>>>(topkIndices, count, numExperts, cache);
}

void launchExpertCacheFill(ExpertCacheState const& cache, ExpertStagingCopy const& copy, int32_t maxMisses, cudaStream_t stream)
{
    // at most one miss per selected expert: blocks beyond that exit at once but still cost a launch wave each (137 us per call with 60 slots, METRICS R25ax)
    dim3 const grid(std::min(cache.numSlots, std::max(maxMisses, 1)), 16, copy.numTensors);
    expertCacheFillKernel<<<grid, 256, 0, stream>>>(cache, copy);
}


namespace
{
// Cache-aware top-k routing for decode steps (rtx4060/moe-cache-aware-routing): one block per token. The experts are selected by
// logit + bias * [resident in this layer's cache], the combine weights are the softmax of the ORIGINAL logits over the selected experts (what an
// unbiased router with the same selection would give), so only WHICH experts run changes. bias = 0 reproduces the exact router.
__global__ void cacheAwareTopkKernel(float const* logits, int32_t numExperts, int32_t topK, float bias, int32_t tailSkip, int32_t const* slotOf,
    int32_t* topkIndices, float* topkWeights)
{
    extern __shared__ float scoreSmem[]; // [numExperts] biased scores
    __shared__ float warpBest[32];
    __shared__ int32_t warpIdx[32];
    __shared__ int32_t chosen[16];
    __shared__ float chosenLogit[16];
    float const* row = logits + static_cast<size_t>(blockIdx.x) * numExperts;
    for (int32_t e = threadIdx.x; e < numExperts; e += blockDim.x)
    {
        float const l = row[e];
        // a NaN logit (e.g. garbage activations in a draft/warm-up pass) would never win a comparison and leave bestIdx out of range
        scoreSmem[e] = (l == l ? l : -3.0e38f) + (slotOf[e] >= 0 ? bias : 0.0f);
    }
    __syncthreads();
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    for (int32_t k = 0; k < topK; ++k)
    {
        float best = -3.0e38f;
        int32_t bestIdx = 0x7fffffff;
        for (int32_t e = threadIdx.x; e < numExperts; e += blockDim.x)
        {
            float const v = scoreSmem[e];
            if (bestIdx == 0x7fffffff || v > best || (v == best && e < bestIdx))
            {
                best = v;
                bestIdx = e;
            }
        }
        for (int32_t off = 16; off > 0; off >>= 1)
        {
            float const ov = __shfl_xor_sync(0xffffffffu, best, off);
            int32_t const oi = __shfl_xor_sync(0xffffffffu, bestIdx, off);
            if (ov > best || (ov == best && oi < bestIdx))
            {
                best = ov;
                bestIdx = oi;
            }
        }
        if (lane == 0)
        {
            warpBest[warp] = best;
            warpIdx[warp] = bestIdx;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            float b = warpBest[0];
            int32_t bi = warpIdx[0];
            for (int32_t w = 1; w < (blockDim.x >> 5); ++w)
            {
                if (warpBest[w] > b || (warpBest[w] == b && warpIdx[w] < bi))
                {
                    b = warpBest[w];
                    bi = warpIdx[w];
                }
            }
            bi = bi < numExperts ? bi : 0;
            chosen[k] = bi;
            chosenLogit[k] = row[bi] == row[bi] ? row[bi] : -3.0e38f;
            scoreSmem[bi] = -3.4e38f;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        float mx = chosenLogit[0];
        for (int32_t k = 1; k < topK; ++k)
        {
            mx = fmaxf(mx, chosenLogit[k]);
        }
        float sum = 0.0f;
        for (int32_t k = 0; k < topK; ++k)
        {
            chosenLogit[k] = __expf(chosenLogit[k] - mx);
            sum += chosenLogit[k];
        }
        for (int32_t k = 0; k < topK; ++k)
        {
            int32_t id = chosen[k];
            float w = chosenLogit[k] / sum;
            // Tail skip (lossy, opt-in): a non-resident expert ranked at place tailSkip+1 or lower is not fetched; its slot is pointed at the
            // top-ranked expert with weight 0 (nothing is renormalised), so the cache lookup sees no miss for it.
            if (tailSkip > 0 && k >= tailSkip && slotOf[id] < 0)
            {
                id = chosen[0];
                w = 0.0f;
            }
            topkIndices[static_cast<size_t>(blockIdx.x) * topK + k] = id;
            topkWeights[static_cast<size_t>(blockIdx.x) * topK + k] = w;
        }
    }
}
} // namespace

void launchCacheAwareTopk(float const* logits, int32_t numTokens, int32_t numExperts, int32_t topK, float bias, int32_t tailSkip, ExpertCacheState const& cache,
    int32_t* topkIndices, float* topkWeights, cudaStream_t stream)
{
    cacheAwareTopkKernel<<<numTokens, 256, numExperts * sizeof(float), stream>>>(logits, numExperts, topK, bias, tailSkip, cache.slotOf, topkIndices, topkWeights);
}

} // namespace kernel
} // namespace trt_edgellm
