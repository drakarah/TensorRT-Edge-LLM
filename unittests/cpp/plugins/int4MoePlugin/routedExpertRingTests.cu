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

#include "common/cudaUtils.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include "plugins/int4MoePlugin/routedExpertRing.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace trt_edgellm;
using trt_edgellm::routedring::RoutedExpertRing;

namespace
{
constexpr int32_t kExperts = 64;
constexpr size_t kBpe[4] = {1024, 256, 512, 128};

__global__ void spinKernel(long long cycles)
{
    long long const start = clock64();
    while (clock64() - start < cycles)
    {
    }
}

//! Distinct bytes per (layer, tensor, expert, offset): a wrong expert, tensor, layer or offset shows up as a mismatch.
uint8_t pattern(int32_t layer, int32_t tensor, int32_t expert, size_t offset)
{
    return static_cast<uint8_t>(layer * 131 + tensor * 53 + expert * 17 + offset * 7 + 1);
}

struct HostLayer
{
    uint8_t* tensor[4]{};

    HostLayer(HostLayer const&) = delete;
    HostLayer& operator=(HostLayer const&) = delete;

    explicit HostLayer(int32_t layer)
    {
        for (int32_t t = 0; t < 4; ++t)
        {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&tensor[t]), kExperts * kBpe[t]));
            for (int32_t e = 0; e < kExperts; ++e)
            {
                for (size_t i = 0; i < kBpe[t]; ++i)
                {
                    tensor[t][e * kBpe[t] + i] = pattern(layer, t, e, i);
                }
            }
        }
    }

    ~HostLayer()
    {
        for (uint8_t* p : tensor)
        {
            cudaFreeHost(p);
        }
    }

    void const* const* src() const
    {
        return reinterpret_cast<void const* const*>(tensor);
    }
};

struct Env
{
    cudaStream_t stream{};
    int32_t* dCounts{};

    Env()
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(&dCounts, kExperts * sizeof(int32_t)));
    }

    ~Env()
    {
        cudaFree(dCounts);
        cudaStreamDestroy(stream);
    }

    //! Builds the routed list on the device for the given experts and reads it back through the ring.
    int32_t route(RoutedExpertRing& ring, std::vector<int32_t> const& experts)
    {
        std::vector<int32_t> counts(kExperts, 0);
        for (int32_t e : experts)
        {
            counts[e] = 1;
        }
        CUDA_CHECK(cudaMemcpy(dCounts, counts.data(), counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        kernel::launchBuildRoutedList(dCounts, kExperts, ring.routedListDevice(), ring.rankOfDevice(), ring.numRoutedDevice(), stream);
        return ring.readRouted(stream);
    }

    //! Walks every task of the layer the way the plugin does: acquire, enqueue the GEMM stand-in (a read of the slot after `holdCycles` of busy work),
    //! release at once, and only then wait. A refill that ignores the slot-free event overwrites a slot the stand-in has not read yet.
    void runLayer(RoutedExpertRing& ring, HostLayer const& host, int32_t layer, std::vector<int32_t> const& experts, long long holdCycles = 0)
    {
        (void) host;
        struct Task
        {
            int32_t phase, chunk, wOff, sOff;
            uint8_t* readBack;
        };
        std::vector<Task> tasks;
        for (int32_t phase = 0; phase < 2; ++phase)
        {
            for (int32_t c = 0; c < ring.chunksPerPhase(); ++c)
            {
                uint8_t* base{};
                int32_t wOff{};
                int32_t sOff{};
                ASSERT_TRUE(ring.acquire(phase, c, stream, &base, &wOff, &sOff)) << "phase " << phase << " chunk " << c;
                uint8_t* readBack{};
                CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&readBack), ring.slotBytes()));
                if (holdCycles > 0)
                {
                    spinKernel<<<1, 1, 0, stream>>>(holdCycles);
                }
                CUDA_CHECK(cudaMemcpyAsync(readBack, base, ring.slotBytes(), cudaMemcpyDeviceToHost, stream));
                ring.release(stream);
                tasks.push_back({phase, c, wOff, sOff, readBack});
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (Task const& task : tasks)
        {
            int32_t const t0 = task.phase == 0 ? 0 : 2;
            int32_t const count = ring.chunkCount(task.chunk);
            for (int32_t j = 0; j < count; ++j)
            {
                int32_t const e = experts[task.chunk * ring.chunkExperts() + j];
                for (int32_t t = 0; t < 2; ++t)
                {
                    size_t const off = (t == 0 ? static_cast<size_t>(task.wOff) : static_cast<size_t>(task.sOff)) + j * kBpe[t0 + t];
                    for (size_t i = 0; i < kBpe[t0 + t]; ++i)
                    {
                        ASSERT_EQ(task.readBack[off + i], pattern(layer, t0 + t, e, i))
                            << "layer " << layer << " phase " << task.phase << " chunk " << task.chunk << " expert " << e << " tensor " << t0 + t << " byte " << i;
                    }
                }
            }
            cudaFreeHost(task.readBack);
        }
    }
};

std::vector<int32_t> pick(std::vector<int32_t> experts)
{
    std::sort(experts.begin(), experts.end());
    return experts;
}
} // namespace

TEST(RoutedExpertRingTest, ReserveRejectsBadGeometry)
{
    RoutedExpertRing ring;
    EXPECT_FALSE(ring.reserve(kBpe, 0, 4, 4));
    EXPECT_FALSE(ring.reserve(kBpe, 1025, 4, 4));
    EXPECT_FALSE(ring.reserve(kBpe, kExperts, 0, 4));
    EXPECT_FALSE(ring.reserve(kBpe, kExperts, 4, 1));
    EXPECT_FALSE(ring.reserve(kBpe, kExperts, 4, 17));
    EXPECT_FALSE(ring.ready());
}

TEST(RoutedExpertRingTest, SecondReserveMustMatchTheFirstGeometry)
{
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 4, 3));
    EXPECT_TRUE(ring.reserve(kBpe, kExperts, 4, 3));
    size_t other[4] = {kBpe[0], kBpe[1], kBpe[2], kBpe[3] + 16};
    EXPECT_FALSE(ring.reserve(other, kExperts, 4, 3));
    EXPECT_FALSE(ring.reserve(kBpe, kExperts + 1, 4, 3));
}

TEST(RoutedExpertRingTest, ChunkLargerThanTheExpertCountIsClamped)
{
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 1000, 2));
    EXPECT_EQ(ring.chunkExperts(), kExperts);
}

TEST(RoutedExpertRingTest, SingleRoutedExpert)
{
    Env env;
    HostLayer host(0);
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 4, 3));
    std::vector<int32_t> const experts = {37};
    ASSERT_EQ(env.route(ring, experts), 1);
    ASSERT_TRUE(ring.beginLayer(host.src(), env.stream));
    env.runLayer(ring, host, 0, experts);
}

TEST(RoutedExpertRingTest, EveryExpertRouted)
{
    Env env;
    HostLayer host(0);
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 8, 4));
    std::vector<int32_t> experts(kExperts);
    for (int32_t e = 0; e < kExperts; ++e)
    {
        experts[e] = e;
    }
    ASSERT_EQ(env.route(ring, experts), kExperts);
    ASSERT_TRUE(ring.beginLayer(host.src(), env.stream));
    env.runLayer(ring, host, 0, experts);
}

TEST(RoutedExpertRingTest, ChunkBoundariesAroundTheRoutedCount)
{
    Env env;
    HostLayer host(0);
    for (int32_t n : {3, 4, 5, 7, 8, 9, 16, 17})
    {
        RoutedExpertRing ring;
        ASSERT_TRUE(ring.reserve(kBpe, kExperts, 4, 3));
        std::vector<int32_t> experts;
        for (int32_t i = 0; i < n; ++i)
        {
            experts.push_back(i * 3 + 1);
        }
        ASSERT_EQ(env.route(ring, experts), n);
        ASSERT_TRUE(ring.beginLayer(host.src(), env.stream)) << "n " << n;
        env.runLayer(ring, host, 0, experts);
    }
}

TEST(RoutedExpertRingTest, ChunkOfOneExpert)
{
    Env env;
    HostLayer host(0);
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 1, 2));
    std::vector<int32_t> const experts = pick({5, 6, 40, 63, 0});
    ASSERT_EQ(env.route(ring, experts), 5);
    ASSERT_TRUE(ring.beginLayer(host.src(), env.stream));
    env.runLayer(ring, host, 0, experts);
}

TEST(RoutedExpertRingTest, ContiguousAndScatteredExperts)
{
    Env env;
    HostLayer host(0);
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 6, 3));
    std::vector<int32_t> const experts = pick({0, 1, 2, 3, 10, 11, 20, 30, 31, 32, 33, 34, 62, 63});
    ASSERT_EQ(env.route(ring, experts), 14);
    ASSERT_TRUE(ring.beginLayer(host.src(), env.stream));
    env.runLayer(ring, host, 0, experts);
}

TEST(RoutedExpertRingTest, ManyLayersReuseTheRingAndKeepTheirOwnWeights)
{
    Env env;
    std::deque<HostLayer> layers;
    for (int32_t l = 0; l < 6; ++l)
    {
        layers.emplace_back(l);
    }
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 5, 3));
    std::mt19937 rng(5);
    for (int32_t l = 0; l < 6; ++l)
    {
        std::vector<int32_t> all(kExperts);
        for (int32_t e = 0; e < kExperts; ++e)
        {
            all[e] = e;
        }
        std::shuffle(all.begin(), all.end(), rng);
        all.resize(1 + rng() % kExperts);
        std::vector<int32_t> const experts = pick(all);
        ASSERT_EQ(env.route(ring, experts), static_cast<int32_t>(experts.size()));
        ASSERT_TRUE(ring.beginLayer(layers[l].src(), env.stream)) << "layer " << l;
        env.runLayer(ring, layers[l], l, experts);
    }
}

TEST(RoutedExpertRingTest, ASlotInUseIsNotOverwrittenByTheNextCopy)
{
    Env env;
    std::deque<HostLayer> layers;
    for (int32_t l = 0; l < 3; ++l)
    {
        layers.emplace_back(l);
    }
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 2, 2));
    std::vector<int32_t> experts;
    for (int32_t e = 0; e < kExperts; e += 2)
    {
        experts.push_back(e);
    }
    for (int32_t l = 0; l < 3; ++l)
    {
        ASSERT_EQ(env.route(ring, experts), static_cast<int32_t>(experts.size()));
        ASSERT_TRUE(ring.beginLayer(layers[l].src(), env.stream));
        env.runLayer(ring, layers[l], l, experts, 2000000);
    }
}

TEST(RoutedExpertRingTest, TasksOutNumberTheSlots)
{
    Env env;
    HostLayer host(0);
    for (int32_t slots : {2, 3, 8, 16})
    {
        RoutedExpertRing ring;
        ASSERT_TRUE(ring.reserve(kBpe, kExperts, 2, slots));
        std::vector<int32_t> experts;
        for (int32_t e = 1; e < kExperts; e += 2)
        {
            experts.push_back(e);
        }
        ASSERT_EQ(env.route(ring, experts), static_cast<int32_t>(experts.size()));
        ASSERT_TRUE(ring.beginLayer(host.src(), env.stream)) << "slots " << slots;
        env.runLayer(ring, host, 0, experts);
    }
}

TEST(RoutedExpertRingTest, TasksMustBeAcquiredInOrder)
{
    Env env;
    HostLayer host(0);
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 4, 3));
    std::vector<int32_t> const experts = pick({1, 2, 3, 4, 5, 6, 7, 8, 9});
    ASSERT_EQ(env.route(ring, experts), 9);
    ASSERT_TRUE(ring.beginLayer(host.src(), env.stream));
    uint8_t* base{};
    int32_t w{};
    int32_t s{};
    EXPECT_FALSE(ring.acquire(0, 1, env.stream, &base, &w, &s));
    EXPECT_FALSE(ring.acquire(1, 0, env.stream, &base, &w, &s));
    EXPECT_TRUE(ring.acquire(0, 0, env.stream, &base, &w, &s));
    ring.release(env.stream);
    EXPECT_TRUE(ring.acquire(0, 1, env.stream, &base, &w, &s));
    ring.release(env.stream);
    EXPECT_FALSE(ring.acquire(0, 1, env.stream, &base, &w, &s));
    CUDA_CHECK(cudaStreamSynchronize(env.stream));
}

TEST(RoutedExpertRingTest, MisuseIsRefusedNotUndefined)
{
    Env env;
    HostLayer host(0);
    RoutedExpertRing ring;
    uint8_t* base{};
    int32_t w{};
    int32_t s{};
    EXPECT_EQ(ring.readRouted(env.stream), -1);
    EXPECT_FALSE(ring.beginLayer(host.src(), env.stream));
    EXPECT_FALSE(ring.acquire(0, 0, env.stream, &base, &w, &s));
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 4, 3));
    EXPECT_FALSE(ring.beginLayer(host.src(), env.stream));
    EXPECT_FALSE(ring.acquire(0, 0, env.stream, &base, &w, &s));
    ring.release(env.stream);
    std::vector<int32_t> const experts = {2, 4, 6, 8, 10};
    ASSERT_EQ(env.route(ring, experts), 5);
    ASSERT_TRUE(ring.beginLayer(host.src(), env.stream));
    EXPECT_FALSE(ring.beginLayer(host.src(), env.stream)) << "a layer that has not finished must not be restarted";
    env.runLayer(ring, host, 0, experts);
}

TEST(RoutedExpertRingTest, EmptyOrCorruptRoutedListIsRefused)
{
    Env env;
    RoutedExpertRing ring;
    ASSERT_TRUE(ring.reserve(kBpe, kExperts, 4, 3));
    CUDA_CHECK(cudaMemset(env.dCounts, 0, kExperts * sizeof(int32_t)));
    kernel::launchBuildRoutedList(env.dCounts, kExperts, ring.routedListDevice(), ring.rankOfDevice(), ring.numRoutedDevice(), env.stream);
    EXPECT_EQ(ring.readRouted(env.stream), -1);
    int32_t const bad = kExperts + 1;
    CUDA_CHECK(cudaMemcpy(ring.numRoutedDevice(), &bad, sizeof(bad), cudaMemcpyHostToDevice));
    EXPECT_EQ(ring.readRouted(env.stream), -1);
    int32_t const negative = -3;
    CUDA_CHECK(cudaMemcpy(ring.numRoutedDevice(), &negative, sizeof(negative), cudaMemcpyHostToDevice));
    EXPECT_EQ(ring.readRouted(env.stream), -1);
}
