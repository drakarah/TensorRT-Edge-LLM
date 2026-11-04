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
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
//! Least recently used expert cache of one layer, as the lookup kernel is specified: the experts a call needs are visited in ascending order, a miss takes the
//! slot with the smallest last-use stamp among those not used by this call (lowest slot index among equals), and ids are rewritten to slot numbers.
struct HostCache
{
    int32_t numExperts;
    int32_t numSlots;
    std::vector<int32_t> slotOf, expertOf, lastUse, missList;
    int32_t counter{0};
    int32_t tally[3]{0, 0, 0};

    HostCache(int32_t experts, int32_t slots)
        : numExperts(experts)
        , numSlots(slots)
        , slotOf(experts, -1)
        , expertOf(slots, -1)
        , lastUse(slots, 0)
        , missList(1 + 2 * slots, 0)
    {
    }

    void lookup(std::vector<int32_t>& ids)
    {
        int32_t const stamp = ++counter;
        std::set<int32_t> needed;
        for (int32_t e : ids)
        {
            if (e >= 0 && e < numExperts)
            {
                needed.insert(e);
            }
        }
        int32_t misses = 0;
        int32_t distinct = 0;
        for (int32_t e : needed)
        {
            ++distinct;
            int32_t slot = slotOf[e];
            if (slot < 0)
            {
                int32_t best = -1;
                int32_t bestUse = INT_MAX;
                for (int32_t s = 0; s < numSlots; ++s)
                {
                    if (lastUse[s] != stamp && lastUse[s] < bestUse)
                    {
                        bestUse = lastUse[s];
                        best = s;
                    }
                }
                if (best < 0)
                {
                    continue;
                }
                if (expertOf[best] >= 0)
                {
                    slotOf[expertOf[best]] = -1;
                }
                slotOf[e] = best;
                expertOf[best] = e;
                missList[1 + 2 * misses] = e;
                missList[2 + 2 * misses] = best;
                ++misses;
                slot = best;
            }
            lastUse[slot] = stamp;
        }
        missList[0] = misses;
        tally[0] += 1;
        tally[1] += distinct;
        tally[2] += misses;
        for (int32_t& id : ids)
        {
            int32_t const slot = id >= 0 && id < numExperts ? slotOf[id] : -1;
            id = slot >= 0 ? slot : 0;
        }
    }
};

struct DeviceCache
{
    int32_t numExperts;
    int32_t numSlots;
    int32_t *slotOf{}, *expertOf{}, *lastUse{}, *counter{}, *missList{}, *tally{};

    DeviceCache(int32_t experts, int32_t slots)
        : numExperts(experts)
        , numSlots(slots)
    {
        std::vector<int32_t> noSlot(experts, -1), noExpert(slots, -1), zeros(slots, 0);
        CUDA_CHECK(cudaMalloc(&slotOf, experts * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&expertOf, slots * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&lastUse, slots * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&counter, sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&missList, (1 + 2 * slots) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&tally, 3 * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(slotOf, noSlot.data(), experts * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(expertOf, noExpert.data(), slots * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(lastUse, zeros.data(), slots * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(counter, 0, sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(missList, 0, (1 + 2 * slots) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(tally, 0, 3 * sizeof(int32_t)));
    }

    ~DeviceCache()
    {
        cudaFree(slotOf);
        cudaFree(expertOf);
        cudaFree(lastUse);
        cudaFree(counter);
        cudaFree(missList);
        cudaFree(tally);
    }

    ExpertCacheState view() const
    {
        return ExpertCacheState{slotOf, expertOf, lastUse, counter, missList, numSlots, tally};
    }

    template <typename T>
    static std::vector<T> read(T const* p, size_t n)
    {
        std::vector<T> v(n);
        CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

void replayAndCompare(int32_t numExperts, int32_t numSlots, int32_t hotExperts, int32_t steps, uint32_t seed, bool withTally = true)
{
    std::mt19937 rng(seed);
    HostCache host(numExperts, numSlots);
    DeviceCache device(numExperts, numSlots);
    int32_t* dIds{};
    CUDA_CHECK(cudaMalloc(&dIds, 64 * sizeof(int32_t)));
    for (int32_t step = 0; step < steps; ++step)
    {
        int32_t const count = step % 7 == 0 ? 32 : 8;
        std::vector<int32_t> ids(count);
        for (auto& e : ids)
        {
            e = static_cast<int32_t>(rng() % std::min(hotExperts, numExperts));
        }
        if (step % 97 == 0)
        {
            ids[0] = -1;
        }
        if (step % 101 == 0)
        {
            ids[1] = numExperts;
        }
        CUDA_CHECK(cudaMemcpy(dIds, ids.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice));
        ExpertCacheState view = device.view();
        view.tally = withTally ? view.tally : nullptr;
        launchExpertCacheLookup(dIds, count, numExperts, view, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        host.lookup(ids);
        auto const gotIds = DeviceCache::read(dIds, count);
        ASSERT_EQ(gotIds, ids) << "rewritten ids, step " << step << " slots " << numSlots;
        ASSERT_EQ(DeviceCache::read(device.slotOf, numExperts), host.slotOf) << "slotOf, step " << step << " slots " << numSlots;
        ASSERT_EQ(DeviceCache::read(device.expertOf, numSlots), host.expertOf) << "expertOf, step " << step;
        ASSERT_EQ(DeviceCache::read(device.lastUse, numSlots), host.lastUse) << "lastUse, step " << step;
        ASSERT_EQ(DeviceCache::read(device.counter, 1)[0], host.counter) << "counter, step " << step;
        auto const miss = DeviceCache::read(device.missList, 1 + 2 * numSlots);
        ASSERT_EQ(miss[0], host.missList[0]) << "miss count, step " << step;
        for (int32_t m = 0; m < miss[0]; ++m)
        {
            ASSERT_EQ(miss[1 + 2 * m], host.missList[1 + 2 * m]) << "missed expert " << m << ", step " << step;
            ASSERT_EQ(miss[2 + 2 * m], host.missList[2 + 2 * m]) << "assigned slot " << m << ", step " << step;
        }
        if (withTally)
        {
            auto const tally = DeviceCache::read(device.tally, 3);
            ASSERT_EQ(std::vector<int32_t>(tally.begin(), tally.end()), std::vector<int32_t>(host.tally, host.tally + 3)) << "tally, step " << step;
        }
    }
    CUDA_CHECK(cudaFree(dIds));
}
} // namespace

TEST(ExpertCacheLookupTest, MatchesTheHostLruModelOnRandomRouting)
{
    // 8 slots with up to 32 distinct experts per call exercises the case of more distinct experts than slots; 300 slots runs the single-thread fallback.
    for (int32_t numSlots : {8, 30, 40, 61, 100, 256, 300})
    {
        for (int32_t numExperts : {64, 256})
        {
            for (int32_t hot : {256, 24, 80})
            {
                replayAndCompare(numExperts, numSlots, hot, 800, 1000 + numSlots + hot);
                if (::testing::Test::HasFatalFailure())
                {
                    return;
                }
            }
        }
    }
}

TEST(ExpertCacheLookupTest, ExpertsOfTheCurrentCallAreNeverEvicted)
{
    int32_t const numExperts = 64;
    int32_t const numSlots = 8;
    DeviceCache device(numExperts, numSlots);
    int32_t* dIds{};
    CUDA_CHECK(cudaMalloc(&dIds, 8 * sizeof(int32_t)));
    for (int32_t round = 0; round < 50; ++round)
    {
        std::vector<int32_t> ids(8);
        for (int32_t i = 0; i < 8; ++i)
        {
            ids[i] = (round * 8 + i) % numExperts;
        }
        CUDA_CHECK(cudaMemcpy(dIds, ids.data(), 8 * sizeof(int32_t), cudaMemcpyHostToDevice));
        launchExpertCacheLookup(dIds, 8, numExperts, device.view(), nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto const slots = DeviceCache::read(dIds, 8);
        EXPECT_EQ(std::set<int32_t>(slots.begin(), slots.end()).size(), 8U) << "eight distinct experts need eight distinct slots, round " << round;
    }
    CUDA_CHECK(cudaFree(dIds));
}

TEST(ExpertCacheFillTest, CopiesExactlyTheMissedExpertsIntoTheirSlots)
{
    int32_t const numExperts = 32;
    int32_t const numSlots = 12;
    size_t const bytes[2] = {4096, 512};
    std::mt19937 rng(77);
    std::vector<std::vector<uint8_t>> hostSrc(2);
    uint8_t* dSrc[2];
    uint8_t* dDst[2];
    for (int32_t t = 0; t < 2; ++t)
    {
        hostSrc[t].resize(bytes[t] * numExperts);
        for (auto& b : hostSrc[t])
        {
            b = static_cast<uint8_t>(rng());
        }
        CUDA_CHECK(cudaMalloc(&dSrc[t], hostSrc[t].size()));
        CUDA_CHECK(cudaMalloc(&dDst[t], bytes[t] * numSlots));
        CUDA_CHECK(cudaMemcpy(dSrc[t], hostSrc[t].data(), hostSrc[t].size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dDst[t], 0, bytes[t] * numSlots));
    }
    ExpertStagingCopy copy{};
    for (int32_t t = 0; t < 2; ++t)
    {
        copy.src[t] = dSrc[t];
        copy.dst[t] = dDst[t];
        copy.bytesPerExpert[t] = bytes[t];
    }
    copy.numTensors = 2;

    DeviceCache device(numExperts, numSlots);
    int32_t* dIds{};
    CUDA_CHECK(cudaMalloc(&dIds, 8 * sizeof(int32_t)));
    std::vector<int32_t> ids{5, 9, 9, 17, 2, 30, 5, 11};
    CUDA_CHECK(cudaMemcpy(dIds, ids.data(), 8 * sizeof(int32_t), cudaMemcpyHostToDevice));
    launchExpertCacheLookup(dIds, 8, numExperts, device.view(), nullptr);
    launchExpertCacheFill(device.view(), copy, 8, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const miss = DeviceCache::read(device.missList, 1 + 2 * numSlots);
    ASSERT_EQ(miss[0], 6) << "six distinct experts, all missed";
    std::set<int32_t> filled;
    for (int32_t t = 0; t < 2; ++t)
    {
        auto const dst = DeviceCache::read(dDst[t], bytes[t] * numSlots);
        for (int32_t m = 0; m < miss[0]; ++m)
        {
            int32_t const expert = miss[1 + 2 * m];
            int32_t const slot = miss[2 + 2 * m];
            filled.insert(slot);
            EXPECT_EQ(0, std::memcmp(dst.data() + bytes[t] * slot, hostSrc[t].data() + bytes[t] * expert, bytes[t])) << "tensor " << t << " expert " << expert;
        }
        for (int32_t s = 0; s < numSlots; ++s)
        {
            if (filled.count(s) == 0)
            {
                EXPECT_TRUE(std::all_of(dst.begin() + bytes[t] * s, dst.begin() + bytes[t] * (s + 1), [](uint8_t b) { return b == 0; })) << "slot " << s << " was not a miss";
            }
        }
    }
    CUDA_CHECK(cudaFree(dIds));
    for (int32_t t = 0; t < 2; ++t)
    {
        CUDA_CHECK(cudaFree(dSrc[t]));
        CUDA_CHECK(cudaFree(dDst[t]));
    }
}

TEST(ExpertCacheFillTest, HitsCopyNothing)
{
    int32_t const numExperts = 16;
    int32_t const numSlots = 8;
    size_t const bytes = 256;
    uint8_t *dSrc{}, *dDst{};
    CUDA_CHECK(cudaMalloc(&dSrc, bytes * numExperts));
    CUDA_CHECK(cudaMalloc(&dDst, bytes * numSlots));
    CUDA_CHECK(cudaMemset(dSrc, 0xAB, bytes * numExperts));
    CUDA_CHECK(cudaMemset(dDst, 0x00, bytes * numSlots));
    ExpertStagingCopy copy{};
    copy.src[0] = dSrc;
    copy.dst[0] = dDst;
    copy.bytesPerExpert[0] = bytes;
    copy.numTensors = 1;
    DeviceCache device(numExperts, numSlots);
    int32_t* dIds{};
    CUDA_CHECK(cudaMalloc(&dIds, 4 * sizeof(int32_t)));
    std::vector<int32_t> ids{1, 2, 3, 4};
    for (int32_t round = 0; round < 2; ++round)
    {
        CUDA_CHECK(cudaMemcpy(dIds, ids.data(), 4 * sizeof(int32_t), cudaMemcpyHostToDevice));
        launchExpertCacheLookup(dIds, 4, numExperts, device.view(), nullptr);
        if (round == 1)
        {
            CUDA_CHECK(cudaMemset(dDst, 0x00, bytes * numSlots));
        }
        launchExpertCacheFill(device.view(), copy, 4, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    EXPECT_EQ(DeviceCache::read(device.missList, 1)[0], 0) << "second call: all hits";
    auto const dst = DeviceCache::read(dDst, bytes * numSlots);
    EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](uint8_t b) { return b == 0; })) << "no bytes copied for hits";
    CUDA_CHECK(cudaFree(dIds));
    CUDA_CHECK(cudaFree(dSrc));
    CUDA_CHECK(cudaFree(dDst));
}

TEST(ExpertCacheLookupTest, TinyCachesAndTheRegisterPathBoundary)
{
    // 1-3 slots: nearly every expert of a call overflows the cache; 255 / 256 use the warp version, 257 the single-thread fallback
    for (int32_t numSlots : {1, 2, 3, 255, 256, 257})
    {
        for (int32_t numExperts : {37, 255, 513})
        {
            replayAndCompare(numExperts, numSlots, 256, 400, 3000 + numSlots + numExperts);
            if (::testing::Test::HasFatalFailure())
            {
                return;
            }
        }
    }
}

TEST(ExpertCacheLookupTest, ExpertCountsThatAreNotAMultipleOfFourOrOfTheWarpWidth)
{
    for (int32_t numExperts : {1, 2, 5, 31, 33, 63, 65, 127, 129})
    {
        replayAndCompare(numExperts, 16, 256, 300, 4000 + numExperts);
        if (::testing::Test::HasFatalFailure())
        {
            return;
        }
    }
}

TEST(ExpertCacheLookupTest, RunsWithoutTheTallyCounters)
{
    replayAndCompare(256, 61, 256, 300, 5001, false);
    replayAndCompare(256, 300, 256, 300, 5002, false);
}

TEST(ExpertCacheLookupTest, CallsWhoseIdsAreAllInvalidOrAllTheSame)
{
    int32_t const numExperts = 64;
    int32_t const numSlots = 16;
    DeviceCache device(numExperts, numSlots);
    HostCache host(numExperts, numSlots);
    int32_t* dIds{};
    CUDA_CHECK(cudaMalloc(&dIds, 32 * sizeof(int32_t)));
    std::vector<std::vector<int32_t>> calls{
        std::vector<int32_t>(8, -1),                          // nothing valid: no expert needed, every id becomes slot 0
        std::vector<int32_t>(8, numExperts),                  // out of range
        std::vector<int32_t>(8, INT32_MAX),                   // far out of range
        std::vector<int32_t>(8, INT32_MIN),                   //
        std::vector<int32_t>(32, 5),                          // one expert, 32 times
        std::vector<int32_t>(8, 5),                           // the same expert again: a hit
        std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7},         // fills more slots
        std::vector<int32_t>{63, -1, 63, numExperts, 62, 62, 0, 1}, // valid and invalid mixed
    };
    for (auto ids : calls)
    {
        int32_t const count = static_cast<int32_t>(ids.size());
        CUDA_CHECK(cudaMemcpy(dIds, ids.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice));
        launchExpertCacheLookup(dIds, count, numExperts, device.view(), nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        host.lookup(ids);
        ASSERT_EQ(DeviceCache::read(dIds, count), ids);
        ASSERT_EQ(DeviceCache::read(device.slotOf, numExperts), host.slotOf);
        ASSERT_EQ(DeviceCache::read(device.lastUse, numSlots), host.lastUse);
        ASSERT_EQ(DeviceCache::read(device.missList, 1)[0], host.missList[0]);
    }
    CUDA_CHECK(cudaFree(dIds));
}

TEST(ExpertCacheLookupTest, EvictsTheLeastRecentlyUsedExpertAndKeepsTheRecentOnes)
{
    int32_t const numExperts = 64;
    int32_t const numSlots = 8;
    DeviceCache device(numExperts, numSlots);
    int32_t* dIds{};
    CUDA_CHECK(cudaMalloc(&dIds, 8 * sizeof(int32_t)));
    auto call = [&](std::vector<int32_t> ids)
    {
        CUDA_CHECK(cudaMemcpy(dIds, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        launchExpertCacheLookup(dIds, static_cast<int32_t>(ids.size()), numExperts, device.view(), nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    for (int32_t e = 0; e < 8; ++e)
    {
        call({e}); // experts 0..7 enter in order, expert 0 is the oldest
    }
    call({0});     // touch expert 0: expert 1 is now the oldest
    call({20});    // a miss evicts expert 1
    auto const slotOf = DeviceCache::read(device.slotOf, numExperts);
    EXPECT_EQ(slotOf[1], -1) << "the least recently used expert is evicted";
    EXPECT_GE(slotOf[0], 0) << "the touched expert stays";
    EXPECT_GE(slotOf[20], 0);
    EXPECT_EQ(slotOf[20], 1) << "the new expert takes the freed slot (expert 1 sat in slot 1)";
    for (int32_t e : {2, 3, 4, 5, 6, 7})
    {
        EXPECT_GE(slotOf[e], 0) << "expert " << e;
    }
    CUDA_CHECK(cudaFree(dIds));
}

TEST(ExpertCacheLookupTest, ACallNeverEvictsAnExpertItAlsoNeeds)
{
    // 8 slots hold experts 0..7; a call for {0..3, 100..103 mapped into range} must evict only 4..7
    int32_t const numExperts = 128;
    int32_t const numSlots = 8;
    DeviceCache device(numExperts, numSlots);
    int32_t* dIds{};
    CUDA_CHECK(cudaMalloc(&dIds, 8 * sizeof(int32_t)));
    std::vector<int32_t> first{0, 1, 2, 3, 4, 5, 6, 7};
    CUDA_CHECK(cudaMemcpy(dIds, first.data(), 8 * sizeof(int32_t), cudaMemcpyHostToDevice));
    launchExpertCacheLookup(dIds, 8, numExperts, device.view(), nullptr);
    std::vector<int32_t> second{0, 1, 2, 3, 100, 101, 102, 103};
    CUDA_CHECK(cudaMemcpy(dIds, second.data(), 8 * sizeof(int32_t), cudaMemcpyHostToDevice));
    launchExpertCacheLookup(dIds, 8, numExperts, device.view(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto const slotOf = DeviceCache::read(device.slotOf, numExperts);
    for (int32_t e : {0, 1, 2, 3, 100, 101, 102, 103})
    {
        EXPECT_GE(slotOf[e], 0) << "expert " << e << " must be resident";
    }
    for (int32_t e : {4, 5, 6, 7})
    {
        EXPECT_EQ(slotOf[e], -1) << "expert " << e << " was the oldest";
    }
    CUDA_CHECK(cudaFree(dIds));
}

namespace
{
//! Runs lookup + fill for `ids` on a cache with `numSlots` slots over `numTensors` tensors of the given per-expert sizes and checks every slot byte.
void fillAndCheck(int32_t numExperts, int32_t numSlots, std::vector<size_t> const& bytes, std::vector<int32_t> const& ids, int32_t maxMisses, uint32_t seed)
{
    int32_t const numTensors = static_cast<int32_t>(bytes.size());
    std::mt19937 rng(seed);
    std::vector<std::vector<uint8_t>> hostSrc(numTensors);
    uint8_t* dSrc[4] = {};
    uint8_t* dDst[4] = {};
    ExpertStagingCopy copy{};
    for (int32_t t = 0; t < numTensors; ++t)
    {
        hostSrc[t].resize(bytes[t] * numExperts);
        for (auto& b : hostSrc[t])
        {
            b = static_cast<uint8_t>(rng() | 1U); // never zero, so an untouched slot is recognisable
        }
        CUDA_CHECK(cudaMalloc(&dSrc[t], hostSrc[t].size()));
        CUDA_CHECK(cudaMalloc(&dDst[t], bytes[t] * numSlots));
        CUDA_CHECK(cudaMemcpy(dSrc[t], hostSrc[t].data(), hostSrc[t].size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dDst[t], 0, bytes[t] * numSlots));
        copy.src[t] = dSrc[t];
        copy.dst[t] = dDst[t];
        copy.bytesPerExpert[t] = bytes[t];
    }
    copy.numTensors = numTensors;
    DeviceCache device(numExperts, numSlots);
    int32_t* dIds{};
    CUDA_CHECK(cudaMalloc(&dIds, ids.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(dIds, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    launchExpertCacheLookup(dIds, static_cast<int32_t>(ids.size()), numExperts, device.view(), nullptr);
    launchExpertCacheFill(device.view(), copy, maxMisses, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto const miss = DeviceCache::read(device.missList, 1 + 2 * numSlots);
    std::set<int32_t> filled;
    for (int32_t m = 0; m < miss[0]; ++m)
    {
        filled.insert(miss[2 + 2 * m]);
    }
    for (int32_t t = 0; t < numTensors; ++t)
    {
        auto const dst = DeviceCache::read(dDst[t], bytes[t] * numSlots);
        for (int32_t m = 0; m < miss[0]; ++m)
        {
            EXPECT_EQ(0, std::memcmp(dst.data() + bytes[t] * miss[2 + 2 * m], hostSrc[t].data() + bytes[t] * miss[1 + 2 * m], bytes[t]))
                << "tensor " << t << " expert " << miss[1 + 2 * m] << " bytes " << bytes[t];
        }
        for (int32_t s = 0; s < numSlots; ++s)
        {
            if (filled.count(s) == 0)
            {
                EXPECT_TRUE(std::all_of(dst.begin() + bytes[t] * s, dst.begin() + bytes[t] * (s + 1), [](uint8_t b) { return b == 0; })) << "slot " << s;
            }
        }
    }
    CUDA_CHECK(cudaFree(dIds));
    for (int32_t t = 0; t < numTensors; ++t)
    {
        CUDA_CHECK(cudaFree(dSrc[t]));
        CUDA_CHECK(cudaFree(dDst[t]));
    }
}
} // namespace

TEST(ExpertCacheFillTest, ExpertSizesFromOneVectorToAMegabyte)
{
    std::vector<int32_t> ids{3, 9, 12, 17, 21, 30, 31, 40};
    for (size_t bytes : {16U, 32U, 48U, 4112U, 65536U, 524288U, 1048576U})
    {
        fillAndCheck(48, 16, {bytes}, ids, 8, 100 + static_cast<uint32_t>(bytes));
        if (::testing::Test::HasFatalFailure())
        {
            return;
        }
    }
}

TEST(ExpertCacheFillTest, FourTensorsWithDifferentSizes)
{
    // the sizes of Qwen3.6-35B-A3B: gate/up weights 1 MiB... here scaled down by 16 but with the same 16 : 1 : 8 : 1 proportions
    fillAndCheck(64, 24, {65536, 4096, 32768, 2048}, {1, 5, 9, 13, 17, 21, 25, 29}, 8, 201);
}

TEST(ExpertCacheFillTest, MoreMissesThanTheGridHasBlocksAndFewerBlocksThanMisses)
{
    std::vector<int32_t> thirtyTwo(32);
    for (int32_t i = 0; i < 32; ++i)
    {
        thirtyTwo[i] = i * 2 + 1;
    }
    fillAndCheck(128, 64, {4096, 512}, thirtyTwo, 32, 301);
    fillAndCheck(128, 64, {4096, 512}, thirtyTwo, 1, 302); // the grid is sized for one miss, the blocks loop over all 32
    fillAndCheck(128, 64, {4096, 512}, thirtyTwo, 0, 303); // a zero hint still copies every miss
}

TEST(ExpertCacheFillTest, TheLastExpertOfTheTableIsCopiedWithoutOverrunningTheSource)
{
    fillAndCheck(16, 4, {4096}, {15, 14, 13, 12}, 4, 401);
}
