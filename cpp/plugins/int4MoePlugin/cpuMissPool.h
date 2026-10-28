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
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{
namespace cpumiss
{

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
        char const* int8 = std::getenv("EDGELLM_MOE_CPU_INT8");
        mInt8 = int8 != nullptr && std::atoi(int8) != 0;
        if (mInt8 && !__builtin_cpu_supports("avxvnni"))
        {
            std::fprintf(stderr, "EDGELLM_MOE_CPU_INT8 requested but the CPU has no AVX-VNNI; using fp32\n");
            mInt8 = false;
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
        if (char const* cores = std::getenv("EDGELLM_MOE_CPU_CORES"))
        {
            // comma-separated CPU ids, one per worker in order; workers beyond the list stay unpinned
            for (char const* c = cores; *c != '\0';)
            {
                char* end = nullptr;
                long const id = std::strtol(c, &end, 10);
                if (end == c)
                {
                    break;
                }
                mCores.push_back(static_cast<int32_t>(id));
                c = (*end == ',') ? end + 1 : end;
            }
        }
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

    // Indexed by the trailing integer of the plugin node name, which counts graph nodes (n0_9 ... n0_477 for 40 layers), not layers.
    static constexpr int32_t kMaxLayers = 1024;

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
        if (tid < static_cast<int32_t>(mCores.size()))
        {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(mCores[tid], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
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

    __attribute__((target("avx2,fma,f16c,avxvnni"))) void process(int32_t tid, uint64_t seq)
    {
        kernel::CpuMissMailbox& mb = *mMailbox;
        auto const startTime = std::chrono::steady_clock::now();
        auto mark = startTime;
        auto lap = [&](int32_t phase) {
            auto const now = std::chrono::steady_clock::now();
            mPhaseNs[tid].ns[phase] += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - mark).count());
            mark = now;
        };
        int32_t const tokens = mb.numTokens;
        int32_t const hidden = mb.hidden;
        bool const known = mb.layer >= 0 && mb.layer < kMaxLayers && mLayers[mb.layer].guW != nullptr;
        LayerWeights const& layer = mLayers[known ? mb.layer : 0];
        int32_t const numMiss = known ? mb.numMiss : 0;
        int32_t const inter = layer.inter;
        Tables const& tab = tables();
        thread_local std::vector<float> xf(static_cast<size_t>(kernel::kCpuMissMaxTokens) * kernel::kCpuMissMaxHidden);
        for (int32_t i = 0; i < tokens * hidden; i += 8)
        {
            _mm256_storeu_ps(xf.data() + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(mb.x + i))));
        }
        struct Int8Scratch
        {
            QuantisedX<kernel::kCpuMissMaxHidden> gu[kernel::kCpuMissMaxTokens];
            QuantisedX<1024> dn[kernel::kCpuMissMaxSlots];
        };
        thread_local std::unique_ptr<Int8Scratch> scratch;
        if (mInt8)
        {
            if (!scratch)
            {
                scratch = std::make_unique<Int8Scratch>();
            }
            for (int32_t tok = 0; tok < tokens; ++tok)
            {
                quantiseX(xf.data() + static_cast<size_t>(tok) * hidden, hidden, scratch->gu[tok]);
            }
        }
        lap(0);
        int32_t const guN = 2 * inter;
        for (int32_t tile = tid; tile < guN / 64; tile += mThreads)
        {
            for (int32_t m = 0; m < numMiss; ++m)
            {
                size_t const e = static_cast<size_t>(mb.expert[m]);
                auto const* wq = reinterpret_cast<int32_t const*>(layer.guW + e * layer.guWBytes);
                auto const* sq = reinterpret_cast<uint16_t const*>(layer.guS + e * layer.guSBytes);
                if (mInt8)
                {
                    gemvTileVnni(tab, wq, sq, scratch->gu[mb.token[m]], guN, tile, &mGateUp[m][tile * 64]);
                }
                else
                {
                    gemvTile(tab, wq, sq, xf.data() + static_cast<size_t>(mb.token[m]) * hidden, hidden, guN, tile, &mGateUp[m][tile * 64]);
                }
            }
        }
        lap(1);
        barrier();
        lap(2);
        int32_t const total = numMiss * inter;
        for (int32_t idx = tid; idx < total; idx += mThreads)
        {
            int32_t const m = idx / inter;
            int32_t const i = idx % inter;
            float const g = mGateUp[m][i];
            mAct[m][i] = g / (1.0F + std::exp(-g)) * mGateUp[m][inter + i];
        }
        lap(3);
        barrier();
        lap(4);
        if (mInt8)
        {
            for (int32_t m = 0; m < numMiss; ++m)
            {
                quantiseX(mAct[m], inter, scratch->dn[m]);
            }
        }
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
                auto const* wq = reinterpret_cast<int32_t const*>(layer.dnW + e * layer.dnWBytes);
                auto const* sq = reinterpret_cast<uint16_t const*>(layer.dnS + e * layer.dnSBytes);
                if (mInt8)
                {
                    gemvTileVnni(tab, wq, sq, scratch->dn[m], hidden, tile, out);
                }
                else
                {
                    gemvTile(tab, wq, sq, mAct[m], inter, hidden, tile, out);
                }
                float* y = mb.y + static_cast<size_t>(mb.token[m]) * hidden + tile * 64;
                float const w = mb.weight[m];
                for (int32_t c = 0; c < 64; ++c)
                {
                    y[c] += w * out[c];
                }
            }
        }
        lap(5);
        barrier();
        lap(6);
        if (tid == 0)
        {
            std::atomic_thread_fence(std::memory_order_release);
            mb.done = seq;
            mCalls += 1;
            mUnregistered += known ? 0 : 1;
            mMissSum += static_cast<uint64_t>(numMiss);
            mBusyNs += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTime).count());
            static bool const stats = std::getenv("EDGELLM_MOE_CPU_STATS") != nullptr;
            if (stats && mCalls % 4000 == 0)
            {
                // mean over threads, per doorbell: x->fp32, gate_up tiles, wait, SwiGLU, wait, down tiles, wait (threads other than 0 may lag a call)
                double phase[7] = {};
                for (int32_t t = 0; t < mThreads; ++t)
                {
                    for (int32_t k = 0; k < 7; ++k)
                    {
                        phase[k] += static_cast<double>(mPhaseNs[t].ns[k]);
                    }
                }
                double const per = 1000.0 * static_cast<double>(mThreads) * static_cast<double>(mCalls);
                std::fprintf(stderr, "cpu miss phases (us per doorbell, mean over threads): x %.1f, gate_up %.1f, wait1 %.1f, swiglu %.1f, wait2 %.1f, down %.1f, wait3 %.1f\n",
                    phase[0] / per, phase[1] / per, phase[2] / per, phase[3] / per, phase[4] / per, phase[5] / per, phase[6] / per);
                double const doorbells = static_cast<double>(mb.doorbells);
                std::fprintf(stderr, "cpu miss: %llu layer calls, %llu doorbells (%.1f%%), %.2f misses per doorbell, host busy %.1f us per doorbell, GPU waited %.1f us per doorbell (at 2.745 GHz), %llu wait timeouts, %llu doorbells for a layer without registered weights\n",
                    static_cast<unsigned long long>(mb.layerCalls), static_cast<unsigned long long>(mb.doorbells), 100.0 * doorbells / std::max<double>(1.0, static_cast<double>(mb.layerCalls)),
                    static_cast<double>(mMissSum) / static_cast<double>(mCalls), static_cast<double>(mBusyNs) / 1000.0 / static_cast<double>(mCalls),
                    static_cast<double>(mb.waitCycles) / 2745.0 / std::max(1.0, doorbells), static_cast<unsigned long long>(mb.timeouts), static_cast<unsigned long long>(mUnregistered));
            }
        }
    }

    std::mutex mInitMutex;
    bool mReady{false};
    bool mInt8{false};
    int32_t mThreads{1};
    int32_t mBarrierThreads{1};
    kernel::CpuMissMailbox* mMailbox{nullptr};
    kernel::CpuMissMailbox* mMailboxDev{nullptr};
    kernel::CpuMissDeviceState* mStateDev{nullptr};
    std::vector<std::thread> mWorkers;
    std::vector<int32_t> mCores;
    uint64_t mCalls{0};
    uint64_t mMissSum{0};
    uint64_t mBusyNs{0};
    uint64_t mUnregistered{0};
    std::atomic<bool> mStop{false};
    struct alignas(64) PhaseNs
    {
        uint64_t ns[7];
    };
    PhaseNs mPhaseNs[64] = {};
    alignas(64) std::atomic<int32_t> mArrived{0};
    alignas(64) std::atomic<int32_t> mGeneration{0};
    LayerWeights mLayers[kMaxLayers];
    alignas(64) float mGateUp[kernel::kCpuMissMaxSlots][2048];
    alignas(64) float mAct[kernel::kCpuMissMaxSlots][1024];
};

} // namespace cpumiss
} // namespace plugins
} // namespace trt_edgellm
