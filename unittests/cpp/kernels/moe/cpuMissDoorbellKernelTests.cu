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

// The CPU miss doorbell: which missed experts are handed to the host (admit counters, per-call budget) and how the router output is rewritten for the GPU path.
#include "common/cudaUtils.h"
#include "kernels/moe/cpuMissMailbox.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <set>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
constexpr int32_t kExperts = 32;
constexpr int32_t kHidden = 64;

struct Rig
{
    CpuMissMailbox* hostMailbox{};
    CpuMissMailbox* deviceMailbox{};
    CpuMissDeviceState* state{};
    int32_t* admit{};
    int32_t* slotOf{};
    int32_t* expertOf{};
    int32_t* idx{};
    float* weights{};
    half* hidden{};

    Rig(std::vector<int32_t> const& resident)
    {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostMailbox), sizeof(CpuMissMailbox), cudaHostAllocMapped));
        std::memset(hostMailbox, 0, sizeof(CpuMissMailbox));
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&deviceMailbox), hostMailbox, 0));
        CUDA_CHECK(cudaMalloc(&state, sizeof(CpuMissDeviceState)));
        CUDA_CHECK(cudaMemset(state, 0, sizeof(CpuMissDeviceState)));
        CUDA_CHECK(cudaMalloc(&admit, (kExperts + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(admit, 0, (kExperts + 1) * sizeof(int32_t)));
        std::vector<int32_t> slot(kExperts, -1), of(resident.size() + 1, -1);
        for (size_t s = 0; s < resident.size(); ++s)
        {
            slot[resident[s]] = static_cast<int32_t>(s);
            of[s] = resident[s];
        }
        CUDA_CHECK(cudaMalloc(&slotOf, kExperts * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&expertOf, of.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(slotOf, slot.data(), slot.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(expertOf, of.data(), of.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&idx, 64 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&weights, 64 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&hidden, 4 * kHidden * sizeof(half)));
        CUDA_CHECK(cudaMemset(hidden, 0, 4 * kHidden * sizeof(half)));
    }
    ~Rig()
    {
        cudaFreeHost(hostMailbox);
        cudaFree(state);
        cudaFree(admit);
        cudaFree(slotOf);
        cudaFree(expertOf);
        cudaFree(idx);
        cudaFree(weights);
        cudaFree(hidden);
    }

    // one doorbell on a token with the given selected experts; returns (indices, weights) after the rewrite
    std::pair<std::vector<int32_t>, std::vector<float>> ring(std::vector<int32_t> const& selected, int32_t admitThreshold, int32_t budget)
    {
        std::vector<float> w(selected.size());
        for (size_t i = 0; i < w.size(); ++i)
        {
            w[i] = 0.1F + 0.01F * static_cast<float>(i);
        }
        CUDA_CHECK(cudaMemcpy(idx, selected.data(), selected.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(weights, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        ExpertCacheState cache{slotOf, expertOf, nullptr, nullptr, nullptr, 4, nullptr};
        launchCpuMissDoorbell(idx, weights, 1, static_cast<int32_t>(selected.size()), kExperts, hidden, kHidden, cache, admit, admitThreshold, budget,
            deviceMailbox, state, 3, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int32_t> outIdx(selected.size());
        std::vector<float> outW(selected.size());
        CUDA_CHECK(cudaMemcpy(outIdx.data(), idx, outIdx.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(outW.data(), weights, outW.size() * 4, cudaMemcpyDeviceToHost));
        return {outIdx, outW};
    }
};
} // namespace

TEST(CpuMissDoorbellTest, HandsEveryAdmittedMissToTheHostWithoutABudget)
{
    Rig rig({0, 1, 2, 3});
    std::vector<int32_t> const selected = {0, 10, 1, 11, 12, 2, 13, 14}; // 5 misses: 10, 11, 12, 13, 14
    auto [idx, w] = rig.ring(selected, 0, 0);
    EXPECT_EQ(rig.hostMailbox->numMiss, 5);
    EXPECT_EQ(rig.hostMailbox->layer, 3);
    std::set<int32_t> sent(rig.hostMailbox->expert, rig.hostMailbox->expert + 5);
    EXPECT_EQ(sent, (std::set<int32_t>{10, 11, 12, 13, 14}));
    for (size_t i = 0; i < selected.size(); ++i)
    {
        bool const missed = selected[i] >= 10;
        EXPECT_EQ(idx[i] == selected[i], !missed) << i; // a handed-over miss now points at a resident expert
        EXPECT_EQ(w[i] == 0.0F, missed) << i;
    }
}

TEST(CpuMissDoorbellTest, TheBudgetLimitsWhatTheHostGetsAndLeavesTheRestToTheFillPath)
{
    for (int32_t budget : {1, 2, 3, 5, 8})
    {
        Rig rig({0, 1, 2, 3});
        std::vector<int32_t> const selected = {0, 10, 1, 11, 12, 2, 13, 14};
        auto [idx, w] = rig.ring(selected, 0, budget);
        int32_t const expected = std::min(budget, 5);
        EXPECT_EQ(rig.hostMailbox->numMiss, expected) << budget;
        int32_t handed = 0;
        for (size_t i = 0; i < selected.size(); ++i)
        {
            if (selected[i] >= 10 && w[i] == 0.0F)
            {
                ++handed;
            }
            else
            {
                EXPECT_EQ(idx[i], selected[i]) << "a kept expert is untouched, budget " << budget;
                EXPECT_GT(w[i], 0.0F);
            }
        }
        EXPECT_EQ(handed, expected) << budget;
        std::set<int32_t> sent(rig.hostMailbox->expert, rig.hostMailbox->expert + expected);
        EXPECT_EQ(static_cast<int32_t>(sent.size()), expected) << "no expert is handed over twice";
    }
}

TEST(CpuMissDoorbellTest, AdmitCountersSendAnExpertToTheFillPathAfterEnoughMisses)
{
    Rig rig({0, 1, 2, 3});
    std::vector<int32_t> const selected = {10, 0, 1, 2, 3, 3, 3, 3};
    rig.ring(selected, 2, 0);
    EXPECT_EQ(rig.hostMailbox->numMiss, 1) << "first miss of expert 10: host";
    auto [idx, w] = rig.ring(selected, 2, 0);
    EXPECT_EQ(rig.hostMailbox->numMiss, 1) << "unchanged mailbox: the second call rang no doorbell";
    EXPECT_EQ(idx[0], 10) << "second miss reaches the admit threshold: the GPU fills it";
    EXPECT_GT(w[0], 0.0F);
}

TEST(CpuMissDoorbellTest, AllResidentRingsNothing)
{
    Rig rig({0, 1, 2, 3, 4, 5, 6, 7});
    auto [idx, w] = rig.ring({0, 1, 2, 3, 4, 5, 6, 7}, 0, 0);
    EXPECT_EQ(rig.hostMailbox->numMiss, 0);
    EXPECT_EQ(rig.hostMailbox->seq, 0u);
    for (size_t i = 0; i < idx.size(); ++i)
    {
        EXPECT_EQ(idx[i], static_cast<int32_t>(i));
    }
}
