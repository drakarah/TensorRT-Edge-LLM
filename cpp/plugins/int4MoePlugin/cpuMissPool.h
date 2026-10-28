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

#include "cpuMissKernel.h"
#include "kernels/moe/cpuMissMailbox.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime.h>
#include <mutex>
#include <thread>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{
namespace cpumiss
{

//! True when the first `bytes` bytes at `p` can be read by this process: a pipe write of the range fails with EFAULT instead of faulting.
inline bool readableByCpu(void const* p, size_t bytes)
{
    int fds[2];
    if (pipe(fds) != 0)
    {
        return false;
    }
    bool ok = true;
    size_t done = 0;
    char sink[4096];
    while (done < bytes && ok)
    {
        size_t const chunk = std::min<size_t>(4096, bytes - done);
        ssize_t const n = write(fds[1], static_cast<char const*>(p) + done, chunk);
        if (n <= 0)
        {
            ok = false;
        }
        else
        {
            done += static_cast<size_t>(n);
            if (read(fds[0], sink, static_cast<size_t>(n)) != n)
            {
                ok = false;
            }
        }
    }
    close(fds[0]);
    close(fds[1]);
    return ok;
}

//! Host pointers of one layer's experts (the plugin's pinned Marlin arrays) and their per-expert byte strides.
struct LayerWeights
{
    uint8_t const* guW{nullptr};
    uint8_t const* guS{nullptr};
    uint8_t const* dnW{nullptr};
    uint8_t const* dnS{nullptr};
    size_t guWBytes{0}, guSBytes{0}, dnWBytes{0}, dnSBytes{0};
    int32_t hidden{0};
    int32_t inter{0};
};

//! Threads that spin on the mailbox's seq. One doorbell is answered by all of them: gate_up tiles, SwiGLU, down tiles, with a barrier in between, then
//! the first thread publishes done. Idle for 20 ms: the threads poll every 50 us until the next doorbell.
class Pool
{
public:
    static Pool& get()
    {
        static Pool pool;
        return pool;
    }

    bool init(int32_t threads)
    {
        std::lock_guard<std::mutex> lock(mInitMutex);
        if (mReady)
        {
            return true;
        }
        if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") || !__builtin_cpu_supports("f16c"))
        {
            return false;
        }
        void* host = nullptr;
        if (cudaHostAlloc(&host, sizeof(kernel::CpuMissMailbox), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }
        std::memset(host, 0, sizeof(kernel::CpuMissMailbox));
        mMailbox = static_cast<kernel::CpuMissMailbox*>(host);
        void* dev = nullptr;
        if (cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess || cudaMalloc(reinterpret_cast<void**>(&mStateDev), sizeof(kernel::CpuMissDeviceState)) != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }
        mMailboxDev = static_cast<kernel::CpuMissMailbox*>(dev);
        cudaMemset(mStateDev, 0, sizeof(kernel::CpuMissDeviceState));
        mThreads = std::max(1, threads);
        mBarrierThreads = mThreads;
        for (int32_t i = 0; i < mThreads; ++i)
        {
            mWorkers.emplace_back([this, i]() { run(i); });
        }
        mReady = true;
        return true;
    }

    kernel::CpuMissMailbox* mailboxDevice() const
    {
        return mMailboxDev;
    }

    kernel::CpuMissDeviceState* stateDevice() const
    {
        return mStateDev;
    }

    void registerLayer(int32_t layer, LayerWeights const& weights)
    {
        if (layer >= 0 && layer < kMaxLayers)
        {
            mLayers[layer] = weights;
        }
    }

    static constexpr int32_t kMaxLayers = 128;

    ~Pool()
    {
        mStop = true;
        for (auto& worker : mWorkers)
        {
            worker.join();
        }
    }

private:
    void barrier()
    {
        int32_t const generation = mGeneration.load(std::memory_order_acquire);
        if (mArrived.fetch_add(1, std::memory_order_acq_rel) == mBarrierThreads - 1)
        {
            mArrived.store(0, std::memory_order_relaxed);
            mGeneration.fetch_add(1, std::memory_order_release);
        }
        else
        {
            while (mGeneration.load(std::memory_order_acquire) == generation)
            {
                _mm_pause();
            }
        }
    }

    void run(int32_t tid)
    {
        uint64_t last = 0;
        auto lastWork = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (!mStop.load(std::memory_order_relaxed))
        {
            uint64_t const seq = mMailbox->seq;
            if (seq == last)
            {
                _mm_pause();
                if ((++spins & 4095) == 0 && std::chrono::steady_clock::now() - lastWork > std::chrono::milliseconds(20))
                {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            last = seq;
            process(tid, seq);
            lastWork = std::chrono::steady_clock::now();
        }
    }

    __attribute__((target("avx2,fma,f16c"))) void process(int32_t tid, uint64_t seq)
    {
        kernel::CpuMissMailbox& mb = *mMailbox;
        auto const startTime = std::chrono::steady_clock::now();
        int32_t const numMiss = mb.numMiss;
        int32_t const tokens = mb.numTokens;
        int32_t const hidden = mb.hidden;
        LayerWeights const& layer = (mb.layer >= 0 && mb.layer < kMaxLayers) ? mLayers[mb.layer] : mLayers[0];
        int32_t const inter = layer.inter;
        Tables const& tab = tables();
        thread_local std::vector<float> xf(static_cast<size_t>(kernel::kCpuMissMaxTokens) * kernel::kCpuMissMaxHidden);
        for (int32_t i = 0; i < tokens * hidden; i += 8)
        {
            _mm256_storeu_ps(xf.data() + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(mb.x + i))));
        }
        int32_t const guN = 2 * inter;
        for (int32_t tile = tid; tile < guN / 64; tile += mThreads)
        {
            for (int32_t m = 0; m < numMiss; ++m)
            {
                size_t const e = static_cast<size_t>(mb.expert[m]);
                gemvTile(tab, reinterpret_cast<int32_t const*>(layer.guW + e * layer.guWBytes), reinterpret_cast<uint16_t const*>(layer.guS + e * layer.guSBytes),
                    xf.data() + static_cast<size_t>(mb.token[m]) * hidden, hidden, guN, tile, &mGateUp[m][tile * 64]);
            }
        }
        barrier();
        int32_t const total = numMiss * inter;
        for (int32_t idx = tid; idx < total; idx += mThreads)
        {
            int32_t const m = idx / inter;
            int32_t const i = idx % inter;
            float const g = mGateUp[m][i];
            mAct[m][i] = g / (1.0F + std::exp(-g)) * mGateUp[m][inter + i];
        }
        barrier();
        for (int32_t tile = tid; tile < hidden / 64; tile += mThreads)
        {
            for (int32_t tok = 0; tok < tokens; ++tok)
            {
                std::memset(mb.y + static_cast<size_t>(tok) * hidden + tile * 64, 0, 64 * sizeof(float));
            }
            for (int32_t m = 0; m < numMiss; ++m)
            {
                size_t const e = static_cast<size_t>(mb.expert[m]);
                float out[64];
                gemvTile(tab, reinterpret_cast<int32_t const*>(layer.dnW + e * layer.dnWBytes), reinterpret_cast<uint16_t const*>(layer.dnS + e * layer.dnSBytes),
                    mAct[m], inter, hidden, tile, out);
                float* y = mb.y + static_cast<size_t>(mb.token[m]) * hidden + tile * 64;
                float const w = mb.weight[m];
                for (int32_t c = 0; c < 64; ++c)
                {
                    y[c] += w * out[c];
                }
            }
        }
        barrier();
        if (tid == 0)
        {
            std::atomic_thread_fence(std::memory_order_release);
            mb.done = seq;
            mCalls += 1;
            mMissSum += static_cast<uint64_t>(numMiss);
            mBusyNs += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTime).count());
            static bool const stats = std::getenv("EDGELLM_MOE_CPU_STATS") != nullptr;
            if (stats && mCalls % 4000 == 0)
            {
                double const doorbells = static_cast<double>(mb.doorbells);
                std::fprintf(stderr, "cpu miss: %llu layer calls, %llu doorbells (%.1f%%), %.2f misses per doorbell, host busy %.1f us per doorbell, GPU waited %.1f us per doorbell (at 2.46 GHz)\n",
                    static_cast<unsigned long long>(mb.layerCalls), static_cast<unsigned long long>(mb.doorbells), 100.0 * doorbells / std::max<double>(1.0, static_cast<double>(mb.layerCalls)),
                    static_cast<double>(mMissSum) / static_cast<double>(mCalls), static_cast<double>(mBusyNs) / 1000.0 / static_cast<double>(mCalls),
                    static_cast<double>(mb.waitCycles) / 2460.0 / std::max(1.0, doorbells));
            }
        }
    }

    std::mutex mInitMutex;
    bool mReady{false};
    int32_t mThreads{1};
    int32_t mBarrierThreads{1};
    kernel::CpuMissMailbox* mMailbox{nullptr};
    kernel::CpuMissMailbox* mMailboxDev{nullptr};
    kernel::CpuMissDeviceState* mStateDev{nullptr};
    std::vector<std::thread> mWorkers;
    uint64_t mCalls{0};
    uint64_t mMissSum{0};
    uint64_t mBusyNs{0};
    std::atomic<bool> mStop{false};
    alignas(64) std::atomic<int32_t> mArrived{0};
    alignas(64) std::atomic<int32_t> mGeneration{0};
    LayerWeights mLayers[kMaxLayers];
    alignas(64) float mGateUp[kernel::kCpuMissMaxSlots][2048];
    alignas(64) float mAct[kernel::kCpuMissMaxSlots][1024];
};

} // namespace cpumiss
} // namespace plugins
} // namespace trt_edgellm
