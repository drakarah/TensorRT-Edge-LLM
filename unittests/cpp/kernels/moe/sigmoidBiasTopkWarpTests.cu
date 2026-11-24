/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The one-warp-per-token Inkling router (launchSigmoidBiasTopk, plain case) against the one-block reference kernel: expert ids and weight bits must be identical,
// including ties, NaN and huge logits.
#include "common/cudaUtils.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
void runCase(int32_t tokens, int32_t experts, int32_t topK, uint32_t seed, int32_t flavour)
{
    std::mt19937 g(seed);
    std::normal_distribution<float> n(0.F, 2.F);
    std::vector<float> logits(static_cast<size_t>(tokens) * experts);
    std::vector<float> bias(experts);
    for (auto& v : logits)
    {
        v = n(g);
        if (flavour == 1)
        {
            v = std::round(v); // many exact ties
        }
        if (flavour == 2 && g() % 17 == 0)
        {
            v = (g() % 3 == 0) ? std::numeric_limits<float>::quiet_NaN() : (g() % 2 ? 1e38F : -1e38F);
        }
        if (flavour == 3)
        {
            v = 0.F; // all equal: the lowest ids must win
        }
    }
    for (auto& b : bias)
    {
        b = flavour == 1 || flavour == 3 ? 0.F : n(g) * 0.01F;
    }
    float *dl, *db, *dwA, *dwB;
    int32_t *diA, *diB;
    ASSERT_EQ(cudaMalloc(&dl, logits.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&db, bias.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dwA, tokens * topK * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dwB, tokens * topK * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&diA, tokens * topK * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&diB, tokens * topK * 4), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dl, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(db, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    launchSigmoidBiasTopk(dl, db, tokens, experts, topK, 0.F, nullptr, 0, 0, 1.F, diA, dwA, nullptr);
    launchSigmoidBiasTopkBlock(dl, db, tokens, experts, topK, 0.F, nullptr, 0, 0, 1.F, diB, dwB, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<int32_t> ia(tokens * topK), ib(tokens * topK);
    std::vector<float> wa(tokens * topK), wb(tokens * topK);
    ASSERT_EQ(cudaMemcpy(ia.data(), diA, ia.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ib.data(), diB, ib.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(wa.data(), dwA, wa.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(wb.data(), dwB, wb.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < ia.size(); ++i)
    {
        ASSERT_EQ(ia[i], ib[i]) << "index " << i << " tokens " << tokens << " experts " << experts << " topK " << topK << " flavour " << flavour << " seed " << seed;
        uint32_t x, y;
        std::memcpy(&x, &wa[i], 4);
        std::memcpy(&y, &wb[i], 4);
        ASSERT_EQ(x, y) << "weight bits " << i << " (" << wa[i] << " vs " << wb[i] << ") flavour " << flavour;
    }
    for (void* d : {static_cast<void*>(dl), static_cast<void*>(db), static_cast<void*>(dwA), static_cast<void*>(dwB), static_cast<void*>(diA), static_cast<void*>(diB)})
    {
        cudaFree(d);
    }
}
} // namespace

TEST(SigmoidBiasTopkWarp, MatchesTheBlockKernelBitForBit)
{
    for (int32_t experts : {128, 64, 100, 33, 256, 200})
    {
        for (int32_t topK : {1, 2, 5, 8})
        {
            for (int32_t tokens : {1, 2, 5, 8, 40})
            {
                for (int32_t flavour : {0, 1, 2, 3})
                {
                    runCase(tokens, experts, topK, 77 + experts + topK * 3 + tokens, flavour);
                }
            }
        }
    }
}
