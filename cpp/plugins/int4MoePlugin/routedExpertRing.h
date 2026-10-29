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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

namespace trt_edgellm
{
namespace routedring
{

//! Prefill expert staging that overlaps the copy with the expert GEMMs and moves only the experts the router selected.
//!
//! Per layer: the routed experts (ascending id) are listed on the device (launchBuildRoutedList) and read back to the host once (readRouted); the experts
//! are then copied by DMA on a side stream, chunk by chunk (gate-up tensors of every chunk, then down tensors), into a small ring of device buffers while the GEMM of
//! the previous chunk runs on the main stream. A chunk is `chunkExperts` consecutive routed experts, stored compactly (expert j of the chunk at position j).
//! Slot layout, per phase: weights of the chunk at offset 0, scales at scalesOffset(phase).
class RoutedExpertRing
{
public:
    static RoutedExpertRing& get()
    {
        static RoutedExpertRing instance;
        return instance;
    }

    ~RoutedExpertRing()
    {
        release();
    }

    //! Allocate the ring (`slots` buffers of one chunk) and the routing scratch for layers of `numExperts` experts with `bpe[i]` bytes per expert in
    //! tensor i (0/1 = gate-up weights/scales, 2/3 = down weights/scales). The first call fixes the geometry; later calls report whether it fits.
    bool reserve(size_t const bpe[4], int32_t numExperts, int32_t chunkExperts, int32_t slots) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mFailed || numExperts <= 0 || numExperts > 1024 || chunkExperts <= 0 || slots < 2 || slots > kMaxSlots)
        {
            return false;
        }
        if (mReady)
        {
            for (int32_t i = 0; i < 4; ++i)
            {
                if (bpe[i] != mBpe[i])
                {
                    return false;
                }
            }
            return numExperts == mNumExperts;
        }
        mNumExperts = numExperts;
        mChunk = std::min(chunkExperts, numExperts);
        mSlots = slots;
        for (int32_t i = 0; i < 4; ++i)
        {
            mBpe[i] = bpe[i];
        }
        for (int32_t p = 0; p < 2; ++p)
        {
            mScalesOffset[p] = align256(static_cast<size_t>(mChunk) * bpe[2 * p]);
        }
        mSlotBytes = std::max(mScalesOffset[0] + static_cast<size_t>(mChunk) * bpe[1], mScalesOffset[1] + static_cast<size_t>(mChunk) * bpe[3]);
        bool ok = cudaStreamCreateWithFlags(&mSide, cudaStreamNonBlocking) == cudaSuccess
            && cudaMalloc(reinterpret_cast<void**>(&mScratch), (1 + 2 * static_cast<size_t>(numExperts)) * sizeof(int32_t)) == cudaSuccess
            && cudaMallocHost(reinterpret_cast<void**>(&mHostList), (1 + static_cast<size_t>(numExperts)) * sizeof(int32_t)) == cudaSuccess
            && cudaEventCreateWithFlags(&mListDone, cudaEventDisableTiming) == cudaSuccess;
        for (int32_t i = 0; i < slots && ok; ++i)
        {
            ok = cudaMalloc(reinterpret_cast<void**>(&mBuf[i]), mSlotBytes) == cudaSuccess
                && cudaEventCreateWithFlags(&mCopyDone[i], cudaEventDisableTiming) == cudaSuccess
                && cudaEventCreateWithFlags(&mSlotFree[i], cudaEventDisableTiming) == cudaSuccess;
        }
        if (!ok)
        {
            cudaGetLastError();
            releaseLocked();
            mFailed = true;
            return false;
        }
        mReady = true;
        return true;
    }

    bool ready() const noexcept
    {
        return mReady;
    }

    size_t slotBytes() const noexcept
    {
        return mSlotBytes;
    }

    int32_t chunkExperts() const noexcept
    {
        return mChunk;
    }

    //! Device scratch of the routed list (see launchBuildRoutedList): count, list[numExperts], rank[numExperts].
    int32_t* numRoutedDevice() const noexcept
    {
        return mScratch;
    }

    int32_t* routedListDevice() const noexcept
    {
        return mScratch + 1;
    }

    int32_t* rankOfDevice() const noexcept
    {
        return mScratch + 1 + mNumExperts;
    }

    //! Copy the routed list built on `stream` to the host and wait for it. Returns the number of routed experts, -1 on failure.
    int32_t readRouted(cudaStream_t stream) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mReady)
        {
            return -1;
        }
        if (cudaMemcpyAsync(mHostList, mScratch, (1 + static_cast<size_t>(mNumExperts)) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess
            || cudaEventRecord(mListDone, stream) != cudaSuccess || cudaEventSynchronize(mListDone) != cudaSuccess)
        {
            cudaGetLastError();
            return -1;
        }
        int32_t const n = mHostList[0];
        if (n < 1 || n > mNumExperts)
        {
            return -1;
        }
        mNumRouted = n;
        return n;
    }

    int32_t chunksPerPhase() const noexcept
    {
        return (mNumRouted + mChunk - 1) / mChunk;
    }

    //! Number of experts in chunk `chunk` of the routed list read last.
    int32_t chunkCount(int32_t chunk) const noexcept
    {
        return std::min(mChunk, mNumRouted - chunk * mChunk);
    }

    //! Start the copies of the layer whose routed list was read last; `src[i]` = host (pinned) tensor i of the layer, `numExperts` rows of bpe[i] bytes.
    //! The ring is filled as far as the previous layer's releases allow.
    bool beginLayer(void const* const src[4], cudaStream_t) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mReady || mNumRouted < 1 || mReleased != mIssued)
        {
            return false;
        }
        for (int32_t i = 0; i < 4; ++i)
        {
            mSrc[i] = static_cast<uint8_t const*>(src[i]);
        }
        mTasks = 2 * chunksPerPhase();
        mFirstTask = mTaskCounter;
        mIssued = 0;
        mReleased = 0;
        return issueAheadLocked();
    }

    //! Wait (on `stream`) for the copy of the next task (tasks run in order: phase 0 chunk 0..m-1, then phase 1 chunk 0..m-1) and return its slot.
    bool acquire(int32_t phase, int32_t chunk, cudaStream_t stream, uint8_t** base, int32_t* weightsOffset, int32_t* scalesOffset) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mReady || mReleased >= mIssued || phase != mReleased / chunksPerPhase() || chunk != mReleased % chunksPerPhase())
        {
            return false;
        }
        int32_t const slot = (mFirstTask + mReleased) % mSlots;
        if (cudaStreamWaitEvent(stream, mCopyDone[slot], 0) != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }
        *base = mBuf[slot];
        *weightsOffset = 0;
        *scalesOffset = static_cast<int32_t>(mScalesOffset[phase]);
        return true;
    }

    //! The GEMM that reads the slot of the acquired task has been enqueued on `stream`: the slot may be refilled.
    void release(cudaStream_t stream) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mReady || mReleased >= mIssued)
        {
            return;
        }
        int32_t const slot = (mFirstTask + mReleased) % mSlots;
        cudaEventRecord(mSlotFree[slot], stream);
        mSlotUsed[slot] = true;
        ++mReleased;
        if (mReleased == mTasks)
        {
            mTaskCounter = mFirstTask + mTasks;
        }
        issueAheadLocked();
    }

private:
    static constexpr int32_t kMaxSlots = 16;

    static size_t align256(size_t v) noexcept
    {
        return (v + 255) & ~static_cast<size_t>(255);
    }

    bool issueAheadLocked() noexcept
    {
        while (mIssued < mTasks && mIssued - mReleased < mSlots)
        {
            if (!issueLocked(mIssued))
            {
                return false;
            }
            ++mIssued;
        }
        return true;
    }

    bool issueLocked(int32_t task) noexcept
    {
        int32_t const perPhase = chunksPerPhase();
        int32_t const phase = task / perPhase;
        int32_t const chunk = task % perPhase;
        int32_t const slot = (mFirstTask + task) % mSlots;
        if (mSlotUsed[slot] && cudaStreamWaitEvent(mSide, mSlotFree[slot], 0) != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }
        int32_t const t0 = phase == 0 ? 0 : 2;
        int32_t const first = chunk * mChunk;
        int32_t const count = chunkCount(chunk);
        mDst.clear();
        mSrcRuns.clear();
        mSize.clear();
        for (int32_t t = 0; t < 2; ++t)
        {
            size_t const bpe = mBpe[t0 + t];
            uint8_t* const dst = mBuf[slot] + (t == 0 ? 0 : mScalesOffset[phase]);
            for (int32_t j = 0; j < count;)
            {
                int32_t const e = mHostList[1 + first + j];
                int32_t run = 1;
                while (j + run < count && mHostList[1 + first + j + run] == e + run)
                {
                    ++run;
                }
                mDst.push_back(dst + static_cast<size_t>(j) * bpe);
                mSrcRuns.push_back(mSrc[t0 + t] + static_cast<size_t>(e) * bpe);
                mSize.push_back(static_cast<size_t>(run) * bpe);
                j += run;
            }
        }
        cudaMemcpyAttributes attrs{};
        attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attrsIdx = 0;
        if (cudaMemcpyBatchAsync(mDst.data(), mSrcRuns.data(), mSize.data(), mDst.size(), &attrs, &attrsIdx, 1, mSide) != cudaSuccess)
        {
            cudaGetLastError();
            for (size_t k = 0; k < mDst.size(); ++k)
            {
                if (cudaMemcpyAsync(mDst[k], mSrcRuns[k], mSize[k], cudaMemcpyDefault, mSide) != cudaSuccess)
                {
                    cudaGetLastError();
                    return false;
                }
            }
        }
        return cudaEventRecord(mCopyDone[slot], mSide) == cudaSuccess;
    }

    void release()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        releaseLocked();
    }

    void releaseLocked() noexcept
    {
        if (mSide != nullptr)
        {
            cudaStreamSynchronize(mSide);
            cudaStreamDestroy(mSide);
            mSide = nullptr;
        }
        for (int32_t i = 0; i < kMaxSlots; ++i)
        {
            if (mBuf[i] != nullptr)
            {
                cudaFree(mBuf[i]);
                mBuf[i] = nullptr;
            }
            if (mCopyDone[i] != nullptr)
            {
                cudaEventDestroy(mCopyDone[i]);
                mCopyDone[i] = nullptr;
            }
            if (mSlotFree[i] != nullptr)
            {
                cudaEventDestroy(mSlotFree[i]);
                mSlotFree[i] = nullptr;
            }
        }
        if (mListDone != nullptr)
        {
            cudaEventDestroy(mListDone);
            mListDone = nullptr;
        }
        if (mScratch != nullptr)
        {
            cudaFree(mScratch);
            mScratch = nullptr;
        }
        if (mHostList != nullptr)
        {
            cudaFreeHost(mHostList);
            mHostList = nullptr;
        }
        mReady = false;
    }

    std::mutex mMutex;
    bool mReady{false};
    bool mFailed{false};
    int32_t mNumExperts{0};
    int32_t mChunk{0};
    int32_t mSlots{0};
    size_t mBpe[4]{};
    size_t mScalesOffset[2]{};
    size_t mSlotBytes{0};
    uint8_t* mBuf[kMaxSlots]{};
    cudaEvent_t mCopyDone[kMaxSlots]{};
    cudaEvent_t mSlotFree[kMaxSlots]{};
    bool mSlotUsed[kMaxSlots]{};
    cudaEvent_t mListDone{nullptr};
    cudaStream_t mSide{nullptr};
    int32_t* mScratch{nullptr};
    int32_t* mHostList{nullptr};
    int32_t mNumRouted{0};
    uint8_t const* mSrc[4]{};
    int32_t mTasks{0};
    int32_t mFirstTask{0};
    int32_t mTaskCounter{0};
    int32_t mIssued{0};
    int32_t mReleased{0};
    std::vector<void*> mDst;
    std::vector<void const*> mSrcRuns;
    std::vector<size_t> mSize;
};

} // namespace routedring
} // namespace trt_edgellm
