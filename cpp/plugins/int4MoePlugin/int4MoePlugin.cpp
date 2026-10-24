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

#include "int4MoePlugin.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/tensor.h"
#include "kernels/moe/moeActivationKernels.h"
#include "kernels/moe/moeAlignSumKernels.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include "kernels/moe/moeTopkSoftmaxKernels.h"
#include "kernels/moe/moe_marlin/moeMarlin.h"
#include "plugins/utils/pluginUtils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kINT4_MOE_PLUGIN_VERSION{"1"};
constexpr char const* kINT4_MOE_PLUGIN_NAME{"Int4MoePlugin"};

// Supported plugin-level activation types.
constexpr auto kACTIVATION_SILU = static_cast<ActivationType>(0);
constexpr auto kACTIVATION_GEGLU = static_cast<ActivationType>(5);
constexpr std::array<ActivationType, 2> kSUPPORTED_ACTIVATION_TYPES{kACTIVATION_SILU, kACTIVATION_GEGLU};
constexpr int32_t kSMALL_TOKEN_BLOCK_SIZE{8};
constexpr int32_t kLARGE_TOKEN_BLOCK_SIZE{32};
constexpr int32_t kTOKEN_MAJOR_SMALL_TOKEN_THRESHOLD{8};

bool getTokenCount(Dims const& hidden, int32_t hiddenSize, int64_t& tokenCount) noexcept
{
    if (hidden.nbDims != 2 || hidden.d[1] != hiddenSize || hidden.d[0] <= 0)
    {
        return false;
    }
    tokenCount = hidden.d[0];
    return true;
}

bool hasSameDims(Dims const& lhs, Dims const& rhs) noexcept
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t index = 0; index < lhs.nbDims; ++index)
    {
        if (lhs.d[index] != rhs.d[index])
        {
            return false;
        }
    }
    return true;
}

//! Selects a Marlin tile by aggregate token count. This is a performance heuristic; both tiles preserve the same
//! routing semantics and do not identify the execution phase.
int32_t getTunedMoeBlockSize(int64_t numTokens) noexcept
{
    //! EDGELLM_MOE_SMALL_THRESHOLD (read once at run time): token count up to which the 8-row block is used (default 8). With a
    //! verify tree of 16 the MoE sees 16 tokens x top-k rows spread over all experts, about one row per expert, so the 8-row
    //! block (smaller activation tile, no padding to 32 rows) is faster: 26B-A4B on the Orin decode +13-17% (METRICS 112).
    static int64_t const threshold = []() {
        char const* const env = std::getenv("EDGELLM_MOE_SMALL_THRESHOLD");
        return static_cast<int64_t>(env != nullptr ? std::atoi(env) : kTOKEN_MAJOR_SMALL_TOKEN_THRESHOLD);
    }();
    return numTokens <= threshold ? kSMALL_TOKEN_BLOCK_SIZE : kLARGE_TOKEN_BLOCK_SIZE;
}

bool hasValidRoutedRowCapacity(int64_t numTokens, int32_t numExperts, int32_t topK) noexcept
{
    int64_t const padding = static_cast<int64_t>(numExperts) * kLARGE_TOKEN_BLOCK_SIZE;
    return numTokens > 0 && numExperts > 0 && topK > 0 && padding < std::numeric_limits<int32_t>::max()
        && numTokens <= (std::numeric_limits<int32_t>::max() - padding) / topK;
}

// Workspace size for Int4 MoE plugin using accumulateWorkspaceSize (same order as assignTensorFromWorkspace in
// enqueue).
//! EDGELLM_MOE_STAGE_EXPERTS=1 at engine build: experts that live in pinned host memory (EDGELLM_HOST_EXTERNAL_WEIGHTS) are
//! copied to a device staging area, only the ones the routing selected, instead of being read by the GEMM over PCIe.
//! 0 = off, 1 = stage into the head of the TRT workspace, 2 = stage into private device buffers (no workspace reservation; the buffers also serve the
//! prefill prefetch, EDGELLM_MOE_PREFILL_OVERLAP).
int32_t stageExpertsFromEnv()
{
    char const* const value = std::getenv("EDGELLM_MOE_STAGE_EXPERTS");
    int32_t const mode = value != nullptr ? std::atoi(value) : 0;
    return mode < 0 ? 0 : (mode > 2 ? 2 : mode);
}

//! EDGELLM_MOE_TOKEN_TILE=<tokens> at engine build: the expert GEMM pipeline of a prefill pass runs over tiles of that many tokens (the experts are
//! staged once per layer for all tokens), so the workspace holds the intermediates of one tile instead of the whole prompt. 0 = off (default).
int32_t tokenTileFromEnv()
{
    char const* const value = std::getenv("EDGELLM_MOE_TOKEN_TILE");
    int32_t const tile = value != nullptr ? std::atoi(value) : 0;
    return tile < 16 ? 0 : tile;
}

//! Bytes of one staged layer: the four per-expert weight / scale tensors (inputs 2..5), 256-byte aligned each.
size_t stagedExpertBytes(nvinfer1::Dims const* const dims[4], nvinfer1::DataType const types[4]) noexcept
{
    size_t total = 0;
    for (int32_t i = 0; i < 4; ++i)
    {
        size_t bytes = 1;
        for (int32_t d = 0; d < dims[i]->nbDims; ++d)
        {
            bytes *= static_cast<size_t>(dims[i]->d[d]);
        }
        bytes *= (types[i] == nvinfer1::DataType::kHALF || types[i] == nvinfer1::DataType::kBF16)
            ? 2
            : (types[i] == nvinfer1::DataType::kINT32 || types[i] == nvinfer1::DataType::kFLOAT) ? 4 : 1;
        total += (bytes + 255) & ~static_cast<size_t>(255);
    }
    return total + 256; // + the routed-expert bitmap
}

//! EDGELLM_MOE_PREFILL_OVERLAP=1 (runtime): during prefill passes (>= kPrefetchMinTokens tokens) the experts of layer L+1 are copied by DMA on a
//! side stream into one of two private device buffers while layer L computes, so the PCIe copy that the routed-expert staging puts in front of every
//! layer's GEMMs (about 40% of a 3k-token prefill, METRICS R25m) is hidden behind compute. With that many tokens nearly every expert is routed in
//! every layer, so copying the whole layer is not wasteful. The first layer of a pass, chunks below the threshold and decode steps keep the staging
//! kernel / the expert cache. With stage mode 2 (engine built with EDGELLM_MOE_STAGE_EXPERTS=2) the staging kernel writes into buffer 0 of the same
//! private pair instead of the TRT workspace, so the pair replaces the workspace head (net cost: one layer of INT4 experts, ~386 MB for the 26B).
constexpr int32_t kPrefetchMinTokens = 128;

bool prefillOverlapFromEnv()
{
    char const* const value = std::getenv("EDGELLM_MOE_PREFILL_OVERLAP");
    return value != nullptr && std::atoi(value) != 0;
}

//! EDGELLM_MOE_CACHE_BIAS=<beta> (lossy, decode only; default 0.5, measured on Qwen3.6 only, set 0 for exact routing): the router prefers experts that are resident in the per-layer GPU expert cache by adding beta to their
//! logits before the top-k selection (combine weights stay the softmax of the original logits over the selected experts). Default 0.5 when unset; EDGELLM_MOE_CACHE_BIAS=0 gives exact routing.
//! beta 0.5: ~65% fewer cache misses for +0.09% perplexity, beta 1: 85% fewer for +1.5% (METRICS R25ar/R25av, Qwen3.6).
float cacheBiasFromEnv()
{
    static float const beta = []() {
        char const* const v = std::getenv("EDGELLM_MOE_CACHE_BIAS");
        return v != nullptr ? std::max(0.0f, static_cast<float>(std::atof(v))) : 0.5f;
    }();
    return beta;
}

//! Layer position inside the engine: the trailing integer of the plugin node name ("n0_18" -> 18); layers run in ascending order.
int32_t layerIdFromName(std::string const& name) noexcept
{
    size_t const pos = name.find_last_of('_');
    return pos == std::string::npos ? -1 : std::atoi(name.c_str() + pos + 1);
}

//! Layer ids of the MoE plugins of this process (for the automatic cache sizing). The node name carries a " [profile N]" suffix per optimisation
//! profile, so the layers are counted by their id, not by the name string.
std::set<int32_t>& moeLayerNames()
{
    static std::set<int32_t> names;
    return names;
}

std::mutex& moeLayerNamesMutex()
{
    static std::mutex mutex;
    return mutex;
}

void registerMoeLayerName(std::string const& name)
{
    std::lock_guard<std::mutex> lock(moeLayerNamesMutex());
    moeLayerNames().insert(layerIdFromName(name));
}

size_t moeLayerCount()
{
    std::lock_guard<std::mutex> lock(moeLayerNamesMutex());
    return moeLayerNames().size();
}

class PrefillExpertPrefetcher
{
public:
    enum class Result
    {
        kNone,       //!< no private buffer available
        kPrefetched, //!< ptrs hold this layer's experts, copied during the previous layer: no staging needed
        kTarget      //!< ptrs is an empty private buffer the staging kernel must fill (mode 2 only)
    };

    static PrefillExpertPrefetcher& get()
    {
        static PrefillExpertPrefetcher instance;
        return instance;
    }

    //! Allocate the private buffers (one, or two when prefetching) for layers of `bytes` bytes; false when the device is full.
    bool reserve(size_t bytes, bool two) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return ensure(bytes, two);
    }

    //! Called at the start of a layer's enqueue that needs staged experts (main stream `stream`). `prefetch` = this pass is long enough to prefetch the
    //! next layer; `wantTarget` = stage mode 2, a not-prefetched layer is staged into buffer 0 by the staging kernel.
    Result begin(int32_t layerId, void const* const src[4], size_t const bytes[4], cudaStream_t stream, bool prefetch, bool wantTarget,
        uint8_t* outPtrs[4]) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (layerId < 0)
        {
            return Result::kNone;
        }
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
        {
            cudaGetLastError();
        }
        bool const capturing = capture != cudaStreamCaptureStatusNone; // graph capture: no cross-stream events, no prefetch
        if (capturing)
        {
            prefetch = false;
        }
        Layer& layer = mLayers[layerId];
        size_t total = 0;
        for (int32_t i = 0; i < 4; ++i)
        {
            layer.src[i] = src[i];
            layer.bytes[i] = bytes[i];
            layer.offset[i] = total;
            total += (bytes[i] + 255) & ~static_cast<size_t>(255);
        }
        layer.total = total;
        if (!ensure(total, prefetch || mTwo))
        {
            return Result::kNone;
        }
        int32_t current = -1;
        if (prefetch && mPendingLayer == layerId && mPendingBuf >= 0)
        {
            cudaStreamWaitEvent(stream, mCopyDone[mPendingBuf], 0);
            current = mPendingBuf;
        }
        else if (wantTarget && !capturing)
        {
            // a stale prefetch into buffer 0 may still be in flight: the staging kernel must not overwrite it before it lands
            for (int32_t b = 0; b < 2; ++b)
            {
                if (mCopyRecorded[b])
                {
                    cudaStreamWaitEvent(stream, mCopyDone[b], 0);
                }
            }
        }
        mPendingLayer = -1;
        mPendingBuf = -1;
        int32_t const used = current >= 0 ? current : 0;
        if (current >= 0 || wantTarget)
        {
            for (int32_t i = 0; i < 4; ++i)
            {
                outPtrs[i] = mBuf[used] + layer.offset[i];
            }
        }
        auto next = mLayers.upper_bound(layerId);
        if (prefetch && mTwo && next != mLayers.end() && next->second.total <= mBufBytes)
        {
            int32_t const target = 1 - used;
            // everything queued on the main stream so far (the previous layer's GEMMs, which may still read this buffer) must finish first
            cudaEventRecord(mBufFree, stream);
            cudaStreamWaitEvent(mSide, mBufFree, 0);
            bool ok = true;
            for (int32_t i = 0; i < 4 && ok; ++i)
            {
                ok = cudaMemcpyAsync(mBuf[target] + next->second.offset[i], next->second.src[i], next->second.bytes[i], cudaMemcpyHostToDevice, mSide)
                    == cudaSuccess;
            }
            if (ok && cudaEventRecord(mCopyDone[target], mSide) == cudaSuccess)
            {
                mPendingLayer = next->first;
                mPendingBuf = target;
                mCopyRecorded[target] = true;
            }
            else
            {
                cudaGetLastError();
            }
        }
        if (current >= 0)
        {
            return Result::kPrefetched;
        }
        return wantTarget ? Result::kTarget : Result::kNone;
    }

    //! A decode step (or any non-prefill call) breaks the layer sequence: a prefetch issued for a layer that now will not run must not be consumed.
    void invalidate() noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mPendingLayer = -1;
        mPendingBuf = -1;
    }

private:
    struct Layer
    {
        void const* src[4]{};
        size_t bytes[4]{};
        size_t offset[4]{};
        size_t total{0};
    };

    bool ensure(size_t needed, bool two) noexcept
    {
        if (mFailed)
        {
            return false;
        }
        if (mReady && (mTwo || !two))
        {
            return needed <= mBufBytes;
        }
        // layer sizes are equal across the layers of this model; sized for the first layer seen
        size_t const size = mReady ? mBufBytes : needed;
        if (mBuf[0] == nullptr && cudaMalloc(reinterpret_cast<void**>(&mBuf[0]), size) != cudaSuccess)
        {
            cudaGetLastError();
            mFailed = true;
            LOG_WARNING("Int4MoePlugin: private expert buffer (%zu bytes) not allocated", size);
            return false;
        }
        if (two)
        {
            if (cudaMalloc(reinterpret_cast<void**>(&mBuf[1]), size) != cudaSuccess || cudaStreamCreateWithFlags(&mSide, cudaStreamNonBlocking) != cudaSuccess
                || cudaEventCreateWithFlags(&mCopyDone[0], cudaEventDisableTiming) != cudaSuccess
                || cudaEventCreateWithFlags(&mCopyDone[1], cudaEventDisableTiming) != cudaSuccess
                || cudaEventCreateWithFlags(&mBufFree, cudaEventDisableTiming) != cudaSuccess)
            {
                cudaGetLastError();
                if (mBuf[1] != nullptr)
                {
                    cudaFree(mBuf[1]);
                    mBuf[1] = nullptr;
                }
                LOG_WARNING("Int4MoePlugin: prefill prefetch buffer (%zu bytes) not allocated; staging without overlap", size);
                two = false;
            }
            else
            {
                mTwo = true;
            }
        }
        if (!mReady)
        {
            mBufBytes = size;
            mReady = true;
            LOG_INFO("Int4MoePlugin: private expert buffers %d x %.1f MiB (prefill overlap %s)", mTwo ? 2 : 1, size / 1048576.0, mTwo ? "on" : "off");
        }
        return needed <= mBufBytes;
    }

    std::mutex mMutex;
    std::map<int32_t, Layer> mLayers;
    uint8_t* mBuf[2]{nullptr, nullptr};
    size_t mBufBytes{0};
    cudaStream_t mSide{nullptr};
    cudaEvent_t mCopyDone[2]{nullptr, nullptr};
    cudaEvent_t mBufFree{nullptr};
    bool mCopyRecorded[2]{false, false};
    int32_t mPendingLayer{-1};
    int32_t mPendingBuf{-1};
    bool mReady{false};
    bool mTwo{false};
    bool mFailed{false};
};

//! EDGELLM_MOE_PREFILL_CHUNK=<experts> (serve time, needs an engine built with EDGELLM_MOE_STAGE_EXPERTS=2): the prefill expert GEMMs run in chunks of that many
//! experts. The experts of every chunk (gate/up tensors, then down tensors) are copied by DMA on a side stream into a small ring of private device buffers
//! (EDGELLM_MOE_PREFILL_RING slots, default 3) while the previous chunk's GEMM runs, across the layer boundary as well, so the whole layer never has to
//! be resident: the cost is RING x one chunk instead of one or two whole layers, and it does not grow with the number of experts. The Marlin GEMM is run once
//! per chunk on a compact copy of the routing arrays of that chunk (no kernel change).
int32_t prefillChunkFromEnv()
{
    char const* const value = std::getenv("EDGELLM_MOE_PREFILL_CHUNK");
    int32_t const chunk = value != nullptr ? std::atoi(value) : 0;
    return chunk > 0 ? chunk : 0;
}

class ExpertChunkRing
{
public:
    static ExpertChunkRing& get()
    {
        static ExpertChunkRing instance;
        return instance;
    }

    //! Allocate the ring for experts with `bpe[i]` bytes per expert in tensor i (0/1 gate-up weights/scales, 2/3 down weights/scales).
    bool reserve(size_t const bpe[4], int32_t numExperts, int32_t chunkExperts) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return ensure(bpe, numExperts, chunkExperts);
    }

    size_t ringBytes(size_t const bpe[4], int32_t chunkExperts) const noexcept
    {
        return static_cast<size_t>(slotCount()) * slotBytesFor(bpe, chunkExperts);
    }

    void registerLayer(int32_t layerId, void const* const src[4], size_t const bpe[4])
    {
        std::lock_guard<std::mutex> lock(mMutex);
        LayerSrc& layer = mLayers[layerId];
        for (int32_t i = 0; i < 4; ++i)
        {
            layer.src[i] = src[i];
            layer.bpe[i] = bpe[i];
        }
    }

    //! Wait for the copy of task (layerId, phase, chunk) (phase 0 = gate-up, 1 = down) and return its slot base pointer; the copies of the next tasks are
    //! issued first. Call release() after the GEMM of the task has been enqueued.
    bool acquire(int32_t layerId, int32_t phase, int32_t chunk, cudaStream_t stream, uint8_t** base, int32_t* weightsOffset, int32_t* scalesOffset) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mReady)
        {
            return false;
        }
        if (mQueue.empty() || mQueue.front().layerId != layerId || mQueue.front().phase != phase || mQueue.front().chunk != chunk)
        {
            // a new pass (or a break in the sequence): start the pipeline at this task; stale copies already queued on the side stream finish first
            // (single side stream, in order), so reusing their slots is safe
            mQueue.clear();
            mCursorValid = false;
            issue(layerId, phase, chunk, stream);
        }
        issueAhead(stream);
        if (mQueue.empty())
        {
            return false;
        }
        Issued const& task = mQueue.front();
        cudaStreamWaitEvent(stream, mCopyDone[task.slot], 0);
        *base = mBuf[task.slot];
        *weightsOffset = 0;
        *scalesOffset = static_cast<int32_t>(mScalesOffset[phase]);
        return true;
    }

    void release(cudaStream_t stream) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mQueue.empty())
        {
            return;
        }
        int32_t const slot = mQueue.front().slot;
        cudaEventRecord(mSlotFree[slot], stream);
        mSlotUsed[slot] = true;
        mQueue.pop_front();
        issueAhead(stream);
    }

    int32_t chunkExperts() const noexcept
    {
        return mChunk;
    }

private:
    struct LayerSrc
    {
        void const* src[4]{};
        size_t bpe[4]{};
    };
    struct Issued
    {
        int32_t layerId;
        int32_t phase;
        int32_t chunk;
        int32_t slot;
    };

    static int32_t slotCount() noexcept
    {
        char const* const value = std::getenv("EDGELLM_MOE_PREFILL_RING");
        int32_t const slots = value != nullptr ? std::atoi(value) : 3;
        return slots < 2 ? 2 : (slots > 8 ? 8 : slots);
    }

    static size_t align256(size_t bytes) noexcept
    {
        return (bytes + 255) & ~static_cast<size_t>(255);
    }

    //! Bytes of one ring slot: the larger of a gate-up chunk (tensors 0 + 1) and a down chunk (tensors 2 + 3), each tensor 256-byte aligned.
    static size_t slotBytesFor(size_t const bpe[4], int32_t chunkExperts) noexcept
    {
        size_t const e = static_cast<size_t>(chunkExperts);
        size_t const gateUp = align256(e * bpe[0]) + align256(e * bpe[1]);
        size_t const down = align256(e * bpe[2]) + align256(e * bpe[3]);
        return std::max(gateUp, down);
    }

    bool ensure(size_t const bpe[4], int32_t numExperts, int32_t chunkExperts) noexcept
    {
        if (mFailed)
        {
            return false;
        }
        if (mReady)
        {
            return true;
        }
        mNumExperts = numExperts;
        mChunk = std::min(chunkExperts, numExperts);
        mRing = slotCount();
        for (int32_t i = 0; i < 4; ++i)
        {
            mBpe[i] = bpe[i];
        }
        mScalesOffset[0] = align256(static_cast<size_t>(mChunk) * bpe[0]);
        mScalesOffset[1] = align256(static_cast<size_t>(mChunk) * bpe[2]);
        mSlotBytes = slotBytesFor(bpe, mChunk);
        bool ok = cudaStreamCreateWithFlags(&mSide, cudaStreamNonBlocking) == cudaSuccess;
        for (int32_t i = 0; i < mRing && ok; ++i)
        {
            ok = cudaMalloc(reinterpret_cast<void**>(&mBuf[i]), mSlotBytes) == cudaSuccess
                && cudaEventCreateWithFlags(&mCopyDone[i], cudaEventDisableTiming) == cudaSuccess
                && cudaEventCreateWithFlags(&mSlotFree[i], cudaEventDisableTiming) == cudaSuccess;
        }
        if (!ok)
        {
            cudaGetLastError();
            for (int32_t i = 0; i < mRing; ++i)
            {
                if (mBuf[i] != nullptr)
                {
                    cudaFree(mBuf[i]);
                    mBuf[i] = nullptr;
                }
            }
            mFailed = true;
            LOG_ERROR("Int4MoePlugin: prefill chunk ring (%d x %zu bytes) not allocated", mRing, mSlotBytes);
            return false;
        }
        mReady = true;
        LOG_INFO("Int4MoePlugin: prefill chunk ring %d x %.1f MiB (%d experts per chunk)", mRing, mSlotBytes / 1048576.0, mChunk);
        return true;
    }

    int32_t chunksPerPhase() const noexcept
    {
        return (mNumExperts + mChunk - 1) / mChunk;
    }

    //! The task that follows (layer, phase, chunk) in execution order, false when the next layer is not known yet.
    bool nextTask(int32_t& layer, int32_t& phase, int32_t& chunk) const
    {
        if (chunk + 1 < chunksPerPhase())
        {
            ++chunk;
            return true;
        }
        if (phase == 0)
        {
            phase = 1;
            chunk = 0;
            return true;
        }
        auto it = mLayers.upper_bound(layer);
        if (it == mLayers.end())
        {
            return false;
        }
        layer = it->first;
        phase = 0;
        chunk = 0;
        return true;
    }

    void issue(int32_t layerId, int32_t phase, int32_t chunk, cudaStream_t mainStream) noexcept
    {
        auto it = mLayers.find(layerId);
        if (it == mLayers.end())
        {
            return;
        }
        int32_t const slot = mIssueCounter % mRing;
        ++mIssueCounter;
        if (mSlotUsed[slot])
        {
            cudaStreamWaitEvent(mSide, mSlotFree[slot], 0); // the GEMM that last read this slot has been enqueued on the main stream
        }
        (void) mainStream;
        int32_t const e0 = chunk * mChunk;
        int32_t const count = std::min(mChunk, mNumExperts - e0);
        int32_t const t0 = phase == 0 ? 0 : 2;
        size_t offset = 0;
        for (int32_t t = 0; t < 2; ++t)
        {
            size_t const bytes = static_cast<size_t>(count) * it->second.bpe[t0 + t];
            cudaMemcpyAsync(mBuf[slot] + offset, static_cast<uint8_t const*>(it->second.src[t0 + t]) + static_cast<size_t>(e0) * it->second.bpe[t0 + t], bytes,
                cudaMemcpyHostToDevice, mSide);
            offset = align256(static_cast<size_t>(mChunk) * it->second.bpe[t0]);
        }
        cudaEventRecord(mCopyDone[slot], mSide);
        mQueue.push_back({layerId, phase, chunk, slot});
        mCursorLayer = layerId;
        mCursorPhase = phase;
        mCursorChunk = chunk;
        mCursorValid = true;
    }

    void issueAhead(cudaStream_t mainStream) noexcept
    {
        while (static_cast<int32_t>(mQueue.size()) < mRing && mCursorValid)
        {
            int32_t layer = mCursorLayer;
            int32_t phase = mCursorPhase;
            int32_t chunk = mCursorChunk;
            if (!nextTask(layer, phase, chunk))
            {
                break;
            }
            issue(layer, phase, chunk, mainStream);
        }
    }

    std::mutex mMutex;
    std::map<int32_t, LayerSrc> mLayers;
    std::deque<Issued> mQueue;
    uint8_t* mBuf[8]{};
    cudaEvent_t mCopyDone[8]{};
    cudaEvent_t mSlotFree[8]{};
    bool mSlotUsed[8]{};
    cudaStream_t mSide{nullptr};
    size_t mBpe[4]{};
    size_t mScalesOffset[2]{};
    size_t mSlotBytes{0};
    int32_t mNumExperts{0};
    int32_t mChunk{0};
    int32_t mRing{0};
    int32_t mIssueCounter{0};
    int32_t mCursorLayer{-1};
    int32_t mCursorPhase{0};
    int32_t mCursorChunk{0};
    bool mCursorValid{false};
    bool mReady{false};
    bool mFailed{false};
};

//! Workspace bytes reserved for staging: the whole layer in mode 1, only the routed-expert bitmap in mode 2.
size_t stagedWorkspaceBytes(int32_t mode, nvinfer1::Dims const* const dims[4], nvinfer1::DataType const types[4]) noexcept
{
    return mode == 1 ? stagedExpertBytes(dims, types) : (mode == 2 ? 256 : 0);
}

size_t computeInt4MoeWorkspaceSize(
    int64_t numTokens, int32_t numExperts, int32_t topK, int32_t hiddenSize, int32_t moeInterSize, int32_t tokenTile) noexcept
{
    try
    {
        if (!hasValidRoutedRowCapacity(numTokens, numExperts, topK))
        {
            return 0;
        }

        int32_t dev = 0;
        int32_t sms = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

        int32_t const tileTokens = tokenTile > 0 ? static_cast<int32_t>(std::min<int64_t>(numTokens, tokenTile)) : static_cast<int32_t>(numTokens);
        int32_t const totalSlots = tileTokens * topK;
        int32_t const maxPaddedSlots = totalSlots + numExperts * kLARGE_TOKEN_BLOCK_SIZE;
        int32_t const maxPaddedBlocks = std::max(
            static_cast<int32_t>(divUp(totalSlots + numExperts * kSMALL_TOKEN_BLOCK_SIZE, kSMALL_TOKEN_BLOCK_SIZE)),
            static_cast<int32_t>(divUp(maxPaddedSlots, kLARGE_TOKEN_BLOCK_SIZE)));
        size_t softmaxWorkspaceSizeBytes
            = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(static_cast<int32_t>(numTokens), numExperts);
        int64_t marlinWorkspaceSize{};
        for (int32_t const blockSize : {kSMALL_TOKEN_BLOCK_SIZE, kLARGE_TOKEN_BLOCK_SIZE})
        {
            marlinWorkspaceSize = std::max(marlinWorkspaceSize,
                trt_edgellm::kernel::getMoeMarlinWorkspaceSize(maxPaddedSlots, 2 * moeInterSize, blockSize, sms));
            marlinWorkspaceSize = std::max(marlinWorkspaceSize,
                trt_edgellm::kernel::getMoeMarlinWorkspaceSize(maxPaddedSlots, hiddenSize, blockSize, sms));
        }

        size_t size = 0;
        // TopK softmax outputs: selected expert weights and indices per token [numTokens, topK]
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kFLOAT);
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kINT32);
        // Optional softmax fallback workspace when numExperts is not a power of 2 (cub/radix sort etc.)
        if (softmaxWorkspaceSizeBytes > 0)
        {
            size = accumulateWorkspaceSize(
                size, rt::Coords{static_cast<int64_t>(softmaxWorkspaceSizeBytes)}, DataType::kINT8);
        }
        // Marlin GEMM indices: token order and expert id per block after padding for coalesced access
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedSlots}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedBlocks}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{1}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedSlots}, DataType::kFLOAT);
        // Per-expert padded slot counts/offsets used to build sorted indices
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts}, DataType::kINT32);
        // Slot lists per expert and slot count per expert (buildMarlinIndices / countSlotsPerExpert / buildSlotLists)
        size = accumulateWorkspaceSize(
            size, rt::Coords{static_cast<int64_t>(numExperts) * totalSlots}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{numExperts}, DataType::kINT32);
        // Expert GEMM intermediates: gate-up output, post-activation, down projection output
        size = accumulateWorkspaceSize(size, rt::Coords{totalSlots, 2 * moeInterSize}, DataType::kHALF);
        size = accumulateWorkspaceSize(size, rt::Coords{totalSlots, moeInterSize}, DataType::kHALF);
        size = accumulateWorkspaceSize(size, rt::Coords{totalSlots, hiddenSize}, DataType::kHALF);
        // Marlin W4A16 GEMM workspace (locks + FP32 reduction buffer)
        size = accumulateWorkspaceSize(size, rt::Coords{marlinWorkspaceSize}, DataType::kINT32);
        // Routing arrays of one expert chunk (chunked prefill GEMMs, EDGELLM_MOE_PREFILL_CHUNK)
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedSlots}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedBlocks}, DataType::kINT32);
        size = accumulateWorkspaceSize(size, rt::Coords{1}, DataType::kINT32);
        // Routing weights in padded slot order for down GEMM (mulTopkWeights=true)
        size = accumulateWorkspaceSize(size, rt::Coords{maxPaddedSlots}, DataType::kFLOAT);
        return size;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to compute Int4MoePlugin workspace size: %s", e.what());
        return 0;
    }
}
} // namespace

// Static class fields initialization
PluginFieldCollection Int4MoePluginCreator::mFieldCollection{};
std::vector<PluginField> Int4MoePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Int4MoePluginCreator);

Int4MoePlugin::Int4MoePlugin(std::string const& name, int32_t const numExperts, int32_t const topK,
    int32_t const hiddenSize, int32_t const moeInterSize, ActivationType const activationType,
    int32_t const quantizationGroupSize)
    : mLayerName(name)
    , mNumExperts(numExperts)
    , mTopK(topK)
    , mHiddenSize(hiddenSize)
    , mMoeInterSize(moeInterSize)
    , mActivationType(activationType)
    , mQuantizationGroupSize(quantizationGroupSize)
{
    registerMoeLayerName(mLayerName);
    if (std::find(kSUPPORTED_ACTIVATION_TYPES.begin(), kSUPPORTED_ACTIVATION_TYPES.end(), mActivationType)
        == kSUPPORTED_ACTIVATION_TYPES.end())
    {
        LOG_ERROR("Int4MoePlugin supports SiLU (type 0) and GeGLU (type 5), got type %d",
            static_cast<int32_t>(mActivationType));
    }
}

Int4MoePlugin::Int4MoePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    registerMoeLayerName(mLayerName);
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "num_experts")
        {
            mNumExperts = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "top_k")
        {
            mTopK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "hidden_size")
        {
            mHiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "moe_inter_size")
        {
            mMoeInterSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "activation_type")
        {
            mActivationType = *static_cast<nvinfer1::ActivationType const*>(fc->fields[i].data);
        }
        else if (fieldName == "quantization_group_size")
        {
            mQuantizationGroupSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "stage_experts")
        {
            mStageExperts = *static_cast<int32_t const*>(fc->fields[i].data);
            mStageExpertsFieldSeen = true;
        }
        else if (fieldName == "token_tile")
        {
            mTokenTile = *static_cast<int32_t const*>(fc->fields[i].data);
            mTokenTileFieldSeen = true;
        }
    }

    if (std::find(kSUPPORTED_ACTIVATION_TYPES.begin(), kSUPPORTED_ACTIVATION_TYPES.end(), mActivationType)
        == kSUPPORTED_ACTIVATION_TYPES.end())
    {
        LOG_ERROR("Int4MoePlugin supports SiLU (type 0) and GeGLU (type 5), got type %d",
            static_cast<int32_t>(mActivationType));
    }
}

namespace
{
//! EDGELLM_MOE_HOSTFUNC_PROBE=1: add a device-to-host copy of the router ids and an empty host node to every cached decode layer (cost probe for a CPU miss path).
bool hostRoundTripProbeFromEnv()
{
    static bool const on = []() { char const* v = std::getenv("EDGELLM_MOE_HOSTFUNC_PROBE"); return v != nullptr && v[0] == '1'; }();
    return on;
}
} // namespace

namespace
{
//! EDGELLM_MOE_STATS=<prefix>: every 15 s (and at destruction) the selection counters of a layer instance are written
//! to <prefix>.<layer>.json: {"decode":[per expert], "prefill":[per expert], "cache":{lookups, needed, misses}} (decode = passes of <= 16 rows;
//! cache = device expert cache lookups, distinct experts needed and misses summed over all calls, present once the cache is allocated).
void writeMoeStats(std::string const& prefix, std::string const& layer, int32_t const* device, int32_t numExperts,
    int32_t const* cacheTally)
{
    std::vector<int32_t> host(2 * static_cast<size_t>(numExperts));
    if (cudaMemcpy(host.data(), device, host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaGetLastError();
        return;
    }
    std::string text = "{\"decode\":[";
    for (int32_t e = 0; e < numExperts; ++e)
    {
        text += (e ? "," : "") + std::to_string(host[e]);
    }
    text += "],\"prefill\":[";
    for (int32_t e = 0; e < numExperts; ++e)
    {
        text += (e ? "," : "") + std::to_string(host[numExperts + e]);
    }
    text += "]";
    int32_t tally[3]{};
    if (cacheTally != nullptr && cudaMemcpy(tally, cacheTally, sizeof(tally), cudaMemcpyDeviceToHost) == cudaSuccess)
    {
        text += ",\"cache\":{\"lookups\":" + std::to_string(tally[0]) + ",\"needed\":" + std::to_string(tally[1])
            + ",\"misses\":" + std::to_string(tally[2]) + "}";
    }
    text += "}\n";
    std::string safe = layer;
    for (char& c : safe)
    {
        if (c == '/' || c == ' ')
        {
            c = '_';
        }
    }
    FILE* file = std::fopen((prefix + "." + safe + ".json").c_str(), "w");
    if (file != nullptr)
    {
        std::fputs(text.c_str(), file);
        std::fclose(file);
    }
}
} // namespace

Int4MoePlugin::~Int4MoePlugin() noexcept
{
    if (mStatsThread.joinable())
    {
        mStatsStop = true;
        mStatsThread.join();
    }
    if (mStats != nullptr)
    {
        cudaFree(mStats);
    }
    if (mCacheMem != nullptr)
    {
        cudaFree(mCacheMem);
    }
    if (mCacheState != nullptr)
    {
        cudaFree(mCacheState);
    }
}

bool Int4MoePlugin::ensureExpertCache(nvinfer1::PluginTensorDesc const* inputDesc) noexcept
{
    if (mCacheSlots < 0)
    {
        char const* value = std::getenv("EDGELLM_MOE_CACHE_SLOTS");
        if (value != nullptr && std::string(value) == "auto")
        {
            // EDGELLM_MOE_CACHE_SLOTS=auto: as many slots per layer as the free device memory holds, decided once (first decode step: the engine,
            // KV and every plugin workspace exist by then), after the private staging / prefetch buffers are reserved. EDGELLM_MOE_CACHE_RESERVE_MB
            // (default 250) stays free for transient allocations.
            static std::mutex autoMutex;
            static int32_t autoSlots = -1;
            std::lock_guard<std::mutex> lock(autoMutex);
            if (autoSlots < 0)
            {
                size_t perSlot = 0;
                size_t layerBytes = 0;
                for (int32_t i = 0; i < 4; ++i)
                {
                    size_t bytes = 1;
                    for (int32_t d = 0; d < inputDesc[2 + i].dims.nbDims; ++d)
                    {
                        bytes *= static_cast<size_t>(inputDesc[2 + i].dims.d[d]);
                    }
                    bytes *= inputDesc[2 + i].type == DataType::kHALF ? 2 : 1;
                    perSlot += bytes / mNumExperts + 256;
                    layerBytes += (bytes + 255) & ~static_cast<size_t>(255);
                }
                if (mStageExperts == 2 && prefillChunkFromEnv() > 0)
                {
                    size_t bpe[4];
                    for (int32_t i = 0; i < 4; ++i)
                    {
                        size_t bytes = 1;
                        for (int32_t d = 0; d < inputDesc[2 + i].dims.nbDims; ++d)
                        {
                            bytes *= static_cast<size_t>(inputDesc[2 + i].dims.d[d]);
                        }
                        bpe[i] = bytes * (inputDesc[2 + i].type == DataType::kHALF ? 2 : 1) / mNumExperts;
                    }
                    ExpertChunkRing::get().reserve(bpe, mNumExperts, prefillChunkFromEnv());
                }
                else if (mStageExperts == 2 || prefillOverlapFromEnv())
                {
                    PrefillExpertPrefetcher::get().reserve(layerBytes, prefillOverlapFromEnv());
                }
                size_t freeBytes = 0;
                size_t totalBytes = 0;
                cudaMemGetInfo(&freeBytes, &totalBytes);
                // EDGELLM_MOE_MAX_VRAM_MB: total device memory the server may occupy (what is in use now, the engine, KV, workspaces and the private
                // buffers, counts against it); makes the slot count independent of other GPU tenants and emulates a smaller card.
                if (char const* maxEnv = std::getenv("EDGELLM_MOE_MAX_VRAM_MB"))
                {
                    size_t const maxBytes = static_cast<size_t>(std::atoll(maxEnv)) << 20;
                    size_t const usedBytes = totalBytes - freeBytes;
                    size_t const capFree = maxBytes > usedBytes ? maxBytes - usedBytes : 0;
                    freeBytes = std::min(freeBytes, capFree);
                }
                char const* reserveEnv = std::getenv("EDGELLM_MOE_CACHE_RESERVE_MB");
                size_t const reserve = (reserveEnv != nullptr ? static_cast<size_t>(std::atoi(reserveEnv)) : 250) << 20;
                size_t const layers = std::max<size_t>(1, moeLayerCount());
                size_t const fit = freeBytes > reserve ? (freeBytes - reserve) / (layers * perSlot) : 0;
                autoSlots = static_cast<int32_t>(std::min<size_t>(fit, static_cast<size_t>(mNumExperts)));
                if (autoSlots < static_cast<int32_t>(mTopK))
                {
                    autoSlots = 0;
                }
                LOG_INFO("Int4MoePlugin: automatic expert cache size %d slots per layer x %zu layers (usable free %.0f MiB, reserve %.0f MiB, %.2f MiB per slot-layer)",
                    autoSlots, layers, freeBytes / 1048576.0, reserve / 1048576.0, perSlot / 1048576.0);
            }
            mCacheSlots = autoSlots;
        }
        else
        {
            mCacheSlots = value != nullptr ? std::atoi(value) : 0;
        }
    }
    if (mCacheSlots <= 0)
    {
        return false;
    }
    if (mCacheReady)
    {
        return true;
    }
    size_t total = 0;
    for (int32_t i = 0; i < 4; ++i)
    {
        size_t bytes = 1;
        for (int32_t d = 0; d < inputDesc[2 + i].dims.nbDims; ++d)
        {
            bytes *= static_cast<size_t>(inputDesc[2 + i].dims.d[d]);
        }
        bytes *= inputDesc[2 + i].type == DataType::kHALF ? 2 : 1;
        mCacheBytesPerExpert[i] = bytes / mNumExperts;
        total += (mCacheBytesPerExpert[i] * mCacheSlots + 255) & ~static_cast<size_t>(255);
    }
    size_t const stateInts = mNumExperts + 2 * static_cast<size_t>(mCacheSlots) + 1 + 1 + 2 * mCacheSlots + 3;
    if (cudaMalloc(&mCacheMem, total) != cudaSuccess || cudaMalloc(reinterpret_cast<void**>(&mCacheState), stateInts * 4) != cudaSuccess)
    {
        cudaGetLastError();
        LOG_WARNING("Int4MoePlugin %s: expert cache of %d slots not allocated (%zu bytes); using staging", mLayerName.c_str(),
            mCacheSlots, total);
        mCacheSlots = 0;
        return false;
    }
    uint8_t* cursor = static_cast<uint8_t*>(mCacheMem);
    for (int32_t i = 0; i < 4; ++i)
    {
        mCacheTensors[i] = cursor;
        cursor += (mCacheBytesPerExpert[i] * mCacheSlots + 255) & ~static_cast<size_t>(255);
    }
    cudaMemset(mCacheState, 0xFF, (static_cast<size_t>(mNumExperts) + mCacheSlots) * 4); // slotOf, expertOf = -1
    cudaMemset(mCacheState + mNumExperts + mCacheSlots, 0, (stateInts - mNumExperts - mCacheSlots) * 4);
    mCacheTally = mCacheState + stateInts - 3;
    mCacheReady = true;
    LOG_INFO("Int4MoePlugin %s: expert cache %d slots, %.1f MiB", mLayerName.c_str(), mCacheSlots, total / 1048576.0);
    return true;
}

void Int4MoePlugin::startStats(std::string const& prefix)
{
    size_t const bytes = 2 * static_cast<size_t>(mNumExperts) * sizeof(int32_t);
    if (cudaMalloc(reinterpret_cast<void**>(&mStats), bytes) != cudaSuccess)
    {
        mStats = nullptr;
        return;
    }
    cudaMemset(mStats, 0, bytes);
    mStatsThread = std::thread([this, prefix]() {
        int32_t waited = 0;
        while (!mStatsStop)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (++waited % 30 == 0 || mStatsStop)
            {
                writeMoeStats(prefix, mLayerName, mStats, mNumExperts, mCacheTally.load());
            }
        }
        writeMoeStats(prefix, mLayerName, mStats, mNumExperts, mCacheTally.load());
    });
}

IPluginCapability* Int4MoePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* Int4MoePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Int4MoePlugin(
            mLayerName, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, mActivationType, mQuantizationGroupSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->setStageExperts(mStageExperts);
        plugin->setTokenTile(mTokenTile);
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to clone Int4MoePlugin: %s", e.what());
        return nullptr;
    }
}

char const* Int4MoePlugin::getPluginName() const noexcept
{
    return kINT4_MOE_PLUGIN_NAME;
}

char const* Int4MoePlugin::getPluginVersion() const noexcept
{
    return kINT4_MOE_PLUGIN_VERSION;
}

char const* Int4MoePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Int4MoePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

int32_t Int4MoePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Int4MoePlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
    [[maybe_unused]] DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    assert(nbOutputs == 1);
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Int4MoePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    [[maybe_unused]] DimsExprs const* shapeInputs, [[maybe_unused]] int32_t nbShapeInputs, DimsExprs* outputs,
    int32_t nbOutputs, [[maybe_unused]] IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 6 || nbOutputs != 1 || inputs[1].nbDims != 2)
    {
        return -1;
    }
    outputs[0].nbDims = inputs[1].nbDims;
    for (int32_t index = 0; index < outputs[0].nbDims; ++index)
    {
        outputs[0].d[index] = inputs[1].d[index];
    }
    return 0;
}

bool Int4MoePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    auto checkHiddenStates = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[1] == mHiddenSize;
        }
        return status;
    };

    // router_logits (numTokens, numExperts) FP32; d[0] may be -1 (dynamic)
    auto checkRouterLogits = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kFLOAT;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 2;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[1] == mNumExperts;
        }
        return status;
    };

    // Marlin packed format: int8 view of int32-packed weights [E, K//16, 2*N]
    // gate_up: N=2*moeInterSize, so int32 last dim = 2*N = 4*moeInterSize; int8 = 16*moeInterSize
    auto checkFcGateUpQWeights = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT8;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        int32_t const expectedLastDim = (2 * 2 * mMoeInterSize) * 4; // 2*N int32 -> int8: 16*moeInterSize
        if (status)
        {
            status &= tensorDim.d[0] == mNumExperts;
            status &= tensorDim.d[1] == mHiddenSize / 16;
            status &= tensorDim.d[2] == expectedLastDim;
        }
        return status;
    };

    // gate_up scales: [E, hiddenSize/quantization_group_size, 2*moeInterSize]
    auto checkFcGateUpScales = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[0] == mNumExperts;
            status &= tensorDim.d[1] == mHiddenSize / mQuantizationGroupSize;
            status &= tensorDim.d[2] == 2 * mMoeInterSize; // Fused gate+up
        }
        return status;
    };

    // down: [E, K//16, 2*N*4] for GEMM input K=moeInterSize, output N=hiddenSize
    auto checkFcDownQWeights = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT8;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[0] == mNumExperts;
            status &= tensorDim.d[1] == mMoeInterSize / 16;
            status &= tensorDim.d[2] == (2 * mHiddenSize) * 4; // int8 view of int32
        }
        return status;
    };

    // down scales: [E, moeInterSize/quantization_group_size, hiddenSize]
    auto checkFcDownScales = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[0] == mNumExperts;
            status &= tensorDim.d[1] == mMoeInterSize / mQuantizationGroupSize;
            status &= tensorDim.d[2] == mHiddenSize;
        }
        return status;
    };

    assert(nbInputs == 6 && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));

    auto const& tensorDesc = inOut[pos].desc;

    switch (pos)
    {
    case 0: return checkRouterLogits(tensorDesc);
    case 1: return checkHiddenStates(tensorDesc);
    case 2: return checkFcGateUpQWeights(tensorDesc);
    case 3: return checkFcGateUpScales(tensorDesc);
    case 4: return checkFcDownQWeights(tensorDesc);
    case 5: return checkFcDownScales(tensorDesc);
    case 6: return checkHiddenStates(tensorDesc);
    default: return false;
    }
}

int32_t Int4MoePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != 6 || nbOutputs != 1)
    {
        return -1;
    }
    auto validateEndpoint = [this](Dims const& hidden, Dims const& router, Dims const& output) {
        int64_t numTokens{};
        return getTokenCount(hidden, mHiddenSize, numTokens) && router.nbDims == 2 && router.d[0] == numTokens
            && router.d[1] == mNumExperts && hasSameDims(output, hidden);
    };
    if (!validateEndpoint(in[1].min, in[0].min, out[0].min) || !validateEndpoint(in[1].opt, in[0].opt, out[0].opt)
        || !validateEndpoint(in[1].max, in[0].max, out[0].max))
    {
        LOG_ERROR("Int4MoePlugin: profile router/output shapes must match rank-2 hidden_states");
        return -1;
    }
    int64_t maxTokens{};
    if (!getTokenCount(in[1].max, mHiddenSize, maxTokens) || !hasValidRoutedRowCapacity(maxTokens, mNumExperts, mTopK))
    {
        LOG_ERROR("Int4MoePlugin: profile maximum exceeds the routed-row index capacity");
        return -1;
    }
    return 0;
}

size_t Int4MoePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* outputs, [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 6 || nbOutputs != 1)
    {
        return 0;
    }
    auto const& hiddenStatesMaxDims = inputs[1].max;
    int64_t maxTokens{};
    if (!getTokenCount(hiddenStatesMaxDims, mHiddenSize, maxTokens) || inputs[0].max.nbDims != 2
        || inputs[0].max.d[0] != maxTokens || inputs[0].max.d[1] != mNumExperts
        || !hasSameDims(outputs[0].max, hiddenStatesMaxDims))
    {
        LOG_ERROR("Int4MoePlugin: invalid profile maximum shapes while computing workspace");
        return 0;
    }
    size_t staged = 0;
    if (mStageExperts != 0)
    {
        nvinfer1::Dims const* const dims[4] = {&inputs[2].desc.dims, &inputs[3].desc.dims, &inputs[4].desc.dims, &inputs[5].desc.dims};
        nvinfer1::DataType const types[4] = {inputs[2].desc.type, inputs[3].desc.type, inputs[4].desc.type, inputs[5].desc.type};
        staged = stagedWorkspaceBytes(mStageExperts, dims, types);
    }
    return computeInt4MoeWorkspaceSize(maxTokens, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, mTokenTile) + staged;
}

int32_t Int4MoePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        using namespace trt_edgellm::kernel;
        using namespace trt_edgellm::rt;

        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || workspace == nullptr || outputs[0] == nullptr)
        {
            LOG_ERROR("Int4MoePlugin: enqueue received a null descriptor or buffer");
            return -1;
        }
        for (int32_t index = 0; index < 6; ++index)
        {
            if (inputs[index] == nullptr)
            {
                LOG_ERROR("Int4MoePlugin: null runtime input at position %d", index);
                return -1;
            }
        }

        // inputs[0] = router_logits (numTokens, numExperts), inputs[1] = hidden_states, then expert weights
        PluginTensorDesc const& hiddenStatesDesc = inputDesc[1];
        int64_t numTokens64{};
        if (!getTokenCount(hiddenStatesDesc.dims, mHiddenSize, numTokens64)
            || !hasValidRoutedRowCapacity(numTokens64, mNumExperts, mTopK) || inputDesc[0].dims.nbDims != 2
            || inputDesc[0].dims.d[0] != numTokens64 || inputDesc[0].dims.d[1] != mNumExperts
            || !hasSameDims(outputDesc[0].dims, hiddenStatesDesc.dims))
        {
            LOG_ERROR("Int4MoePlugin: runtime router/output shapes must match rank-2 hidden_states");
            return -1;
        }
        int32_t const numTokens = static_cast<int32_t>(numTokens64);
        int32_t const tileTokens = mTokenTile > 0 ? std::min(numTokens, mTokenTile) : numTokens;
        int32_t const moeBlockSize = getTunedMoeBlockSize(tileTokens);

        rt::Tensor hiddenStatesTensor(const_cast<void*>(inputs[1]), rt::Coords{hiddenStatesDesc.dims},
            rt::DeviceType::kGPU, hiddenStatesDesc.type);

        // With staging on, expert weights that live in pinned host memory are copied (routed experts only) into the head of the
        // workspace and the GEMMs read that copy; otherwise the GEMMs read the engine input directly.
        void const* weightPtrs[4] = {inputs[2], inputs[3], inputs[4], inputs[5]};
        std::byte* alignedWorkspacePtr = static_cast<std::byte*>(workspace);
        bool stageNow = false;
        trt_edgellm::kernel::ExpertStagingCopy stagingCopy{};
        uint8_t* neededPtr = nullptr;
        if (mStageExperts != 0)
        {
            cudaPointerAttributes attributes{};
            if (cudaPointerGetAttributes(&attributes, inputs[2]) != cudaSuccess)
            {
                cudaGetLastError();
            }
            else
            {
                stageNow = attributes.type == cudaMemoryTypeHost;
            }
        }
        bool useCache = false;
        static bool const prefillOverlap = prefillOverlapFromEnv();
        nvinfer1::Dims const* const stageDims[4] = {&inputDesc[2].dims, &inputDesc[3].dims, &inputDesc[4].dims, &inputDesc[5].dims};
        nvinfer1::DataType const stageTypes[4] = {inputDesc[2].type, inputDesc[3].type, inputDesc[4].type, inputDesc[5].type};
        size_t tensorBytes[4];
        for (int32_t i = 0; i < 4; ++i)
        {
            size_t bytes = 1;
            for (int32_t d = 0; d < inputDesc[2 + i].dims.nbDims; ++d)
            {
                bytes *= static_cast<size_t>(inputDesc[2 + i].dims.d[d]);
            }
            tensorBytes[i] = bytes * (inputDesc[2 + i].type == DataType::kHALF ? 2 : 1);
        }
        // decode steps served by the expert cache never stage
        bool const cacheStep = stageNow && numTokens <= 4 && ensureExpertCache(inputDesc) && numTokens * mTopK <= mCacheSlots;
        auto pfResult = PrefillExpertPrefetcher::Result::kNone;
        uint8_t* privatePtrs[4]{};
        static int32_t const chunkExperts = prefillChunkFromEnv();
        bool chunkMode = false;
        if (chunkExperts > 0 && mStageExperts == 2 && stageNow && !cacheStep && tileTokens == numTokens)
        {
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
            {
                cudaGetLastError();
            }
            if (capture == cudaStreamCaptureStatusNone)
            {
                size_t bpe[4];
                for (int32_t i = 0; i < 4; ++i)
                {
                    bpe[i] = tensorBytes[i] / mNumExperts;
                }
                void const* srcPtrs[4] = {inputs[2], inputs[3], inputs[4], inputs[5]};
                if (ExpertChunkRing::get().reserve(bpe, mNumExperts, chunkExperts))
                {
                    ExpertChunkRing::get().registerLayer(layerIdFromName(mLayerName), srcPtrs, bpe);
                    chunkMode = true;
                }
            }
        }
        if (stageNow && !cacheStep && !chunkMode && (prefillOverlap || mStageExperts == 2))
        {
            void const* srcPtrs[4] = {inputs[2], inputs[3], inputs[4], inputs[5]};
            pfResult = PrefillExpertPrefetcher::get().begin(layerIdFromName(mLayerName), srcPtrs, tensorBytes, stream,
                prefillOverlap && numTokens >= kPrefetchMinTokens, mStageExperts == 2, privatePtrs);
            if (mStageExperts == 2 && pfResult == PrefillExpertPrefetcher::Result::kNone)
            {
                LOG_ERROR("Int4MoePlugin %s: private staging buffer unavailable", mLayerName.c_str());
                return 1;
            }
        }
        else if (stageNow && prefillOverlap)
        {
            PrefillExpertPrefetcher::get().invalidate();
        }
        if (chunkMode)
        {
            // the GEMMs run chunk by chunk on the ring buffers: no layer-wide staging
            alignedWorkspacePtr += stagedWorkspaceBytes(mStageExperts, stageDims, stageTypes);
            stageNow = false;
        }
        else if (pfResult == PrefillExpertPrefetcher::Result::kPrefetched)
        {
            // this layer's experts were copied during the previous layer: the GEMMs read the private buffer, the staging kernel is skipped
            for (int32_t i = 0; i < 4; ++i)
            {
                weightPtrs[i] = privatePtrs[i];
            }
            alignedWorkspacePtr += stagedWorkspaceBytes(mStageExperts, stageDims, stageTypes); // the staging region stays unused
            stageNow = false;
        }
        else if (cacheStep)
        {
            useCache = true;
            for (int32_t i = 0; i < 4; ++i)
            {
                stagingCopy.src[i] = static_cast<uint8_t const*>(inputs[2 + i]);
                stagingCopy.dst[i] = mCacheTensors[i];
                stagingCopy.bytesPerExpert[i] = mCacheBytesPerExpert[i];
                weightPtrs[i] = mCacheTensors[i];
            }
            stagingCopy.numTensors = 4;
            alignedWorkspacePtr += stagedWorkspaceBytes(mStageExperts, stageDims, stageTypes); // the staging region stays unused
        }
        else if (stageNow)
        {
            for (int32_t i = 0; i < 4; ++i)
            {
                size_t const bytes = tensorBytes[i];
                stagingCopy.src[i] = static_cast<uint8_t const*>(inputs[2 + i]);
                stagingCopy.bytesPerExpert[i] = bytes / mNumExperts;
                if (mStageExperts == 2)
                {
                    stagingCopy.dst[i] = privatePtrs[i];
                    weightPtrs[i] = privatePtrs[i];
                }
                else
                {
                    stagingCopy.dst[i] = reinterpret_cast<uint8_t*>(alignedWorkspacePtr);
                    weightPtrs[i] = alignedWorkspacePtr;
                    alignedWorkspacePtr += (bytes + 255) & ~static_cast<size_t>(255);
                }
            }
            stagingCopy.numTensors = 4;
            neededPtr = reinterpret_cast<uint8_t*>(alignedWorkspacePtr);
            alignedWorkspacePtr += 256;
        }
        else if (mStageExperts != 0)
        {
            // staging space is reserved but unused; skip it so the later carving matches the size computation
            alignedWorkspacePtr += stagedWorkspaceBytes(mStageExperts, stageDims, stageTypes);
        }

        // Weights arrive as INT8 (view of Marlin int32-packed); Marlin expects INT32
        rt::Tensor gateUpQWeightsTensor(const_cast<void*>(weightPtrs[0]),
            rt::Coords{mNumExperts, mHiddenSize / 16, 2 * mMoeInterSize}, rt::DeviceType::kGPU, DataType::kINT32);

        rt::Tensor gateUpScalesTensor(
            const_cast<void*>(weightPtrs[1]), rt::Coords{inputDesc[3].dims}, rt::DeviceType::kGPU, inputDesc[3].type);

        rt::Tensor downQWeightsTensor(const_cast<void*>(weightPtrs[2]),
            rt::Coords{mNumExperts, mMoeInterSize / 16, 2 * mHiddenSize}, rt::DeviceType::kGPU, DataType::kINT32);

        rt::Tensor downScalesTensor(
            const_cast<void*>(weightPtrs[3]), rt::Coords{inputDesc[5].dims}, rt::DeviceType::kGPU, inputDesc[5].type);

        rt::Tensor outputTensor(outputs[0], rt::Coords{outputDesc[0].dims}, rt::DeviceType::kGPU, outputDesc[0].type);

        // Get GPU info
        int32_t dev = 0;
        int32_t sms = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

        // ==================== Workspace Allocation ====================
        int32_t totalSlots = tileTokens * mTopK;
        int32_t maxPaddedSlots = totalSlots + mNumExperts * moeBlockSize;
        int32_t maxPaddedBlocks = divUp(maxPaddedSlots, moeBlockSize);

        // Step 1: TopK Softmax outputs (router_logits from inputs[0] FP32, cast done in Python)
        rt::Tensor routerLogitsTensor(const_cast<void*>(inputs[0]), rt::Coords{inputDesc[0].dims}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT);
        // Selected expert weights and indices per token [numTokens, topK] for routing and aggregation
        float* topkWeightsPtr = static_cast<float*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {numTokens, mTopK}, DataType::kFLOAT).rawPointer());
        int32_t* topkIndicesPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {numTokens, mTopK}, DataType::kINT32).rawPointer());
        // Optional workspace for MoE top-k softmax when numExperts is not a power of 2 (cub/radix sort fallback)
        size_t softmaxWorkspaceSizeBytes = getMoeTopkSoftmaxWorkspaceSize(numTokens, mNumExperts);
        void* softmaxWorkspacePtr = (softmaxWorkspaceSizeBytes > 0)
            ? assignTensorFromWorkspace(
                  alignedWorkspacePtr, {static_cast<int64_t>(softmaxWorkspaceSizeBytes)}, DataType::kINT8)
                  .rawPointer()
            : nullptr;

        // Step 2: Marlin GEMM indices — token order and expert id per block after padding for coalesced GEMM
        int32_t* sortedTokenIdsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedSlots}, DataType::kINT32).rawPointer());
        int32_t* expertIdsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedBlocks}, DataType::kINT32).rawPointer());
        int32_t* numTokensPostPaddedPtr
            = static_cast<int32_t*>(assignTensorFromWorkspace(alignedWorkspacePtr, {1}, DataType::kINT32).rawPointer());
        float* topkWeightsFlatPtr = static_cast<float*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedSlots}, DataType::kFLOAT).rawPointer());
        // Per-expert padded slot counts and offsets used to build sorted indices
        int32_t* paddedCountsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {mNumExperts}, DataType::kINT32).rawPointer());
        int32_t* paddedOffsetsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {mNumExperts}, DataType::kINT32).rawPointer());

        // Slot lists per expert and slot count per expert (buildMarlinIndices / countSlotsPerExpert / buildSlotLists)
        int32_t* slotsByExpertWorkspace = static_cast<int32_t*>(assignTensorFromWorkspace(
            alignedWorkspacePtr, {static_cast<int64_t>(mNumExperts) * totalSlots}, DataType::kINT32)
                .rawPointer());
        int32_t* slotsPerExpertWorkspace = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {mNumExperts}, DataType::kINT32).rawPointer());

        // Step 3: Expert GEMM intermediates — gate-up output, post–SwiGLU, down projection output
        half* gateUpOutputPtr = static_cast<half*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {totalSlots, 2 * mMoeInterSize}, DataType::kHALF)
                .rawPointer());
        half* activationOutputPtr = static_cast<half*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {totalSlots, mMoeInterSize}, DataType::kHALF).rawPointer());
        half* downOutputPtr = static_cast<half*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {totalSlots, mHiddenSize}, DataType::kHALF).rawPointer());

        // Marlin W4A16 GEMM workspace (locks + FP32 reduction buffer)
        int64_t marlinWorkspaceSize = std::max(
            trt_edgellm::kernel::getMoeMarlinWorkspaceSize(maxPaddedSlots, 2 * mMoeInterSize, moeBlockSize, sms),
            trt_edgellm::kernel::getMoeMarlinWorkspaceSize(maxPaddedSlots, mHiddenSize, moeBlockSize, sms));
        int32_t* marlinWorkspacePtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {marlinWorkspaceSize}, DataType::kINT32).rawPointer());
        int32_t* chunkSortedIdsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedSlots}, DataType::kINT32).rawPointer());
        int32_t* chunkExpertIdsPtr = static_cast<int32_t*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedBlocks}, DataType::kINT32).rawPointer());
        int32_t* chunkNumPostPtr
            = static_cast<int32_t*>(assignTensorFromWorkspace(alignedWorkspacePtr, {1}, DataType::kINT32).rawPointer());

        // ==================== Step 1: TopK Softmax on router_logits ====================
        rt::Tensor topkWeightsTensor(
            topkWeightsPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor topkIndicesTensor(
            topkIndicesPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        if (useCache && cacheBiasFromEnv() > 0.0f && inputDesc[0].type == nvinfer1::DataType::kFLOAT)
        {
            kernel::ExpertCacheState const biasState{mCacheState, mCacheState + mNumExperts, mCacheState + mNumExperts + mCacheSlots,
                mCacheState + mNumExperts + 2 * mCacheSlots, mCacheState + mNumExperts + 2 * mCacheSlots + 1, mCacheSlots, mCacheTally.load()};
            kernel::launchCacheAwareTopk(static_cast<float const*>(inputs[0]), numTokens, mNumExperts, mTopK, cacheBiasFromEnv(), biasState,
                topkIndicesPtr, topkWeightsPtr, stream);
        }
        else
        {
            moeTopkSoftmax(routerLogitsTensor, topkWeightsTensor, topkIndicesTensor, mTopK, softmaxWorkspacePtr,
                softmaxWorkspaceSizeBytes, stream, true, 0.0f);
        }
        CUDA_CHECK(cudaGetLastError());

        if (useCache && hostRoundTripProbeFromEnv())
        {
            // rtx4060/moe-cpu-miss step 1: cost of the host round trip a CPU miss path needs (router ids to pinned host memory + a stream-blocking host node), no compute
            static int32_t* const probeHost = []() { void* p = nullptr; cudaMallocHost(&p, 4096); return static_cast<int32_t*>(p); }();
            cudaMemcpyAsync(probeHost, topkIndicesPtr, static_cast<size_t>(numTokens) * mTopK * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaLaunchHostFunc(stream, [](void*) {}, nullptr);
        }
        if (useCache)
        {
            kernel::ExpertCacheState state{mCacheState, mCacheState + mNumExperts, mCacheState + mNumExperts + mCacheSlots,
                mCacheState + mNumExperts + 2 * mCacheSlots, mCacheState + mNumExperts + 2 * mCacheSlots + 1, mCacheSlots, mCacheTally.load()};
            kernel::launchExpertCacheLookup(topkIndicesPtr, numTokens * mTopK, mNumExperts, state, stream);
            kernel::launchExpertCacheFill(state, stagingCopy, numTokens * mTopK, stream);
            CUDA_CHECK(cudaGetLastError());
        }
        else if (stageNow)
        {
            kernel::launchStageRoutedExperts(topkIndicesPtr, numTokens * mTopK, mNumExperts, neededPtr, stagingCopy, stream,
                mStats == nullptr ? nullptr : mStats + (numTokens > 16 ? mNumExperts : 0));
            CUDA_CHECK(cudaGetLastError());
        }

        float* const topkWeightsPaddedPtr = static_cast<float*>(
            assignTensorFromWorkspace(alignedWorkspacePtr, {maxPaddedSlots}, DataType::kFLOAT).rawPointer());

        // Everything below runs once per tile of tokens: the workspace holds the intermediates of one tile.
        auto runTile = [&](int32_t t0, int32_t tokens)
        {
            int32_t const tileSlots = tokens * mTopK;
            int32_t const tilePadded = tileSlots + mNumExperts * moeBlockSize;
            int32_t const tileBlocks = divUp(tilePadded, moeBlockSize);
            int32_t* const tileTopkIdx = topkIndicesPtr + static_cast<int64_t>(t0) * mTopK;
            float* const tileTopkW = topkWeightsPtr + static_cast<int64_t>(t0) * mTopK;
        // Reshape hidden_states to 2D for expert GEMMs
        rt::Tensor hiddenStates2D(const_cast<half*>(static_cast<half const*>(hiddenStatesTensor.rawPointer())) + static_cast<int64_t>(t0) * mHiddenSize,
            {tokens, mHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // ==================== Step 2: Prepare Marlin GEMM Indices ====================
        CUDA_CHECK(cudaMemsetAsync(slotsPerExpertWorkspace, 0, mNumExperts * sizeof(int32_t), stream));
        kernel::launchCountSlotsPerExpertKernel(
            tileTopkIdx, slotsPerExpertWorkspace, tokens, mTopK, mNumExperts, stream);
        CUDA_CHECK(cudaGetLastError());

        kernel::launchComputePaddedOffsetsKernel(slotsPerExpertWorkspace, paddedCountsPtr, paddedOffsetsPtr,
            numTokensPostPaddedPtr, mNumExperts, moeBlockSize, stream);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(slotsPerExpertWorkspace, 0, mNumExperts * sizeof(int32_t), stream));
        kernel::launchBuildSlotListsKernel(
            tileTopkIdx, slotsByExpertWorkspace, slotsPerExpertWorkspace, tokens, mTopK, mNumExperts, stream);
        CUDA_CHECK(cudaGetLastError());

        rt::Tensor sortedTokenIdsTensor(
            sortedTokenIdsPtr, {tilePadded}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor expertIdsTensor(expertIdsPtr, {tileBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor numTokensPostPaddedTensor(
            numTokensPostPaddedPtr, {1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        rt::Tensor topkWeightsFlatTensor(
            topkWeightsFlatPtr, {tilePadded}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor marlinWorkspaceTensor(
            marlinWorkspacePtr, {marlinWorkspaceSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

        kernel::launchBuildMarlinIndicesKernel(slotsByExpertWorkspace, slotsPerExpertWorkspace, paddedCountsPtr,
            paddedOffsetsPtr, tileTopkW, sortedTokenIdsPtr, topkWeightsFlatPtr, expertIdsPtr, tokens, mTopK,
            mNumExperts, moeBlockSize, stream);

        // ==================== Step 3: Gate-Up Projection (Marlin INT4 GEMM) ====================
        rt::Tensor gateUpOutputTensor(
            gateUpOutputPtr, {tileSlots, 2 * mMoeInterSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Chunked prefill (EDGELLM_MOE_PREFILL_CHUNK): the same GEMM once per chunk of experts, weights from the ring, routing arrays sliced per chunk.
        // phase 0 = gate-up, 1 = down; the weight tensors are the chunk's slice with the chunk's expert count as leading dimension.
        auto runChunkedGemm = [&](int32_t phase, rt::Tensor const& gemmInput, rt::Tensor& gemmOutput, int64_t kTile, int64_t nOut,
                                  nvinfer1::PluginTensorDesc const& scalesDesc, rt::Tensor const& gemmTopkWeights, int64_t gemmTopK,
                                  bool mulTopkWeights)
        {
            ExpertChunkRing& ring = ExpertChunkRing::get();
            int32_t const chunk = ring.chunkExperts();
            for (int32_t e0 = 0, c = 0; e0 < mNumExperts; e0 += chunk, ++c)
            {
                int32_t const e1 = std::min(e0 + chunk, mNumExperts);
                kernel::launchBuildChunkMarlinIndicesKernel(paddedOffsetsPtr, numTokensPostPaddedPtr, sortedTokenIdsPtr, expertIdsPtr, e0, e1,
                    mNumExperts, moeBlockSize, tilePadded, chunkSortedIdsPtr, chunkExpertIdsPtr, chunkNumPostPtr, stream);
                uint8_t* base = nullptr;
                int32_t weightsOffset = 0;
                int32_t scalesOffset = 0;
                if (!ring.acquire(layerIdFromName(mLayerName), phase, c, stream, &base, &weightsOffset, &scalesOffset))
                {
                    throw std::runtime_error("expert chunk ring: task not available");
                }
                int64_t const count = e1 - e0;
                rt::Tensor chunkWeights(base + weightsOffset, rt::Coords{count, kTile, nOut}, rt::DeviceType::kGPU, DataType::kINT32);
                rt::Tensor chunkScales(base + scalesOffset, rt::Coords{count, scalesDesc.dims.d[1], scalesDesc.dims.d[2]}, rt::DeviceType::kGPU,
                    scalesDesc.type);
                rt::Tensor chunkSortedIds(chunkSortedIdsPtr, {tilePadded}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
                rt::Tensor chunkExpertIds(chunkExpertIdsPtr, {tileBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
                rt::Tensor chunkNumPost(chunkNumPostPtr, {1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
                moeAwqW4A16MarlinGemm(gemmInput, gemmOutput, chunkWeights, chunkScales, chunkSortedIds, chunkExpertIds, chunkNumPost,
                    gemmTopkWeights, marlinWorkspaceTensor, moeBlockSize, gemmTopK, mulTopkWeights, stream);
                CUDA_CHECK(cudaGetLastError());
                ring.release(stream);
            }
        };

        if (chunkMode)
        {
            runChunkedGemm(0, hiddenStates2D, gateUpOutputTensor, mHiddenSize / 16, 2 * mMoeInterSize, inputDesc[3], topkWeightsFlatTensor, mTopK, false);
        }
        else
        {
            moeAwqW4A16MarlinGemm(hiddenStates2D, gateUpOutputTensor, gateUpQWeightsTensor, gateUpScalesTensor,
                sortedTokenIdsTensor, expertIdsTensor, numTokensPostPaddedTensor, topkWeightsFlatTensor,
                marlinWorkspaceTensor, moeBlockSize, mTopK, false, stream);
        }
        CUDA_CHECK(cudaGetLastError());

        // ==================== Step 4: Gated Activation ====================
        rt::Tensor activationOutputTensor(
            activationOutputPtr, {tileSlots, mMoeInterSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        int32_t const kernelActivationType = (mActivationType == kACTIVATION_GEGLU)
            ? trt_edgellm::kernel::MoeActivationType::kMoeGeGlu
            : trt_edgellm::kernel::MoeActivationType::kMoeSwiGlu;
        moeActivation(
            gateUpOutputTensor, activationOutputTensor, tileSlots, mMoeInterSize, kernelActivationType, stream);
        CUDA_CHECK(cudaGetLastError());

        // ==================== Step 5: Down Projection (Marlin INT4 GEMM) ====================
        rt::Tensor downOutputTensor(
            downOutputPtr, {tileSlots, mHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        // Routing weights in padded slot order for down GEMM (mulTopkWeights=true)
        cudaMemsetAsync(topkWeightsPaddedPtr, 0, tilePadded * sizeof(float), stream);
        cudaMemcpyAsync(
            topkWeightsPaddedPtr, tileTopkW, tileSlots * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        rt::Tensor topkWeightsPaddedTensor(
            topkWeightsPaddedPtr, {tilePadded}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

        if (chunkMode)
        {
            runChunkedGemm(1, activationOutputTensor, downOutputTensor, mMoeInterSize / 16, 2 * mHiddenSize, inputDesc[5], topkWeightsPaddedTensor, 1, true);
        }
        else
        {
            moeAwqW4A16MarlinGemm(activationOutputTensor, downOutputTensor, downQWeightsTensor, downScalesTensor,
                sortedTokenIdsTensor, expertIdsTensor, numTokensPostPaddedTensor, topkWeightsPaddedTensor,
                marlinWorkspaceTensor, moeBlockSize, 1, true, stream);
        }
        CUDA_CHECK(cudaGetLastError());

        // ==================== Step 6: Aggregate Slots to Tokens ====================
        kernel::launchAggregateSlotOutputsKernel(
            downOutputPtr, static_cast<half*>(outputTensor.rawPointer()) + static_cast<int64_t>(t0) * mHiddenSize, tokens, mTopK, mHiddenSize, stream);
        };
        for (int32_t t0 = 0; t0 < numTokens; t0 += tileTokens)
        {
            runTile(t0, std::min(tileTokens, numTokens - t0));
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Int4MoePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t Int4MoePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != 6 || nbOutputs != 1)
    {
        return -1;
    }
    int64_t numTokens{};
    if (!getTokenCount(in[1].dims, mHiddenSize, numTokens) || !hasValidRoutedRowCapacity(numTokens, mNumExperts, mTopK)
        || in[0].dims.nbDims != 2 || in[0].dims.d[0] != numTokens || in[0].dims.d[1] != mNumExperts
        || !hasSameDims(out[0].dims, in[1].dims))
    {
        LOG_ERROR("Int4MoePlugin: runtime router/output shapes must match rank-2 hidden_states");
        return -1;
    }
    return 0;
}

IPluginV3* Int4MoePlugin::attachToContext([[maybe_unused]] IPluginResourceContext* context) noexcept
{
    auto* plugin = static_cast<Int4MoePlugin*>(clone());
    if (plugin != nullptr && mStageExperts != 0 && std::getenv("EDGELLM_MOE_STATS") != nullptr)
    {
        plugin->startStats(std::getenv("EDGELLM_MOE_STATS"));
    }
    return plugin;
}

PluginFieldCollection const* Int4MoePlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("num_experts", &mNumExperts, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("top_k", &mTopK, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("moe_inter_size", &mMoeInterSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("activation_type", &mActivationType, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("quantization_group_size", &mQuantizationGroupSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("stage_experts", &mStageExperts, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("token_tile", &mTokenTile, PluginFieldType::kINT32, 1);

        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to serialize Int4MoePlugin fields: %s", e.what());
        return nullptr;
    }
}

Int4MoePluginCreator::Int4MoePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_experts", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("top_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("moe_inter_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("activation_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("quantization_group_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stage_experts", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("token_tile", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Int4MoePluginCreator::getPluginName() const noexcept
{
    return kINT4_MOE_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* Int4MoePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Int4MoePluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Int4MoePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Int4MoePluginCreator::getPluginVersion() const noexcept
{
    return kINT4_MOE_PLUGIN_VERSION;
}

IPluginV3* Int4MoePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        Int4MoePlugin* plugin = new Int4MoePlugin(std::string(name), fc);
        if (!plugin->hasStageExpertsField() && phase == TensorRTPhase::kBUILD)
        {
            plugin->setStageExperts(stageExpertsFromEnv());
        }
        if (!plugin->hasTokenTileField() && phase == TensorRTPhase::kBUILD)
        {
            plugin->setTokenTile(tokenTileFromEnv());
        }
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create Int4MoePlugin: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
