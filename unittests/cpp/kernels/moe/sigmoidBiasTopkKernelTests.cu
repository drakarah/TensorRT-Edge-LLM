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

// Inkling 2 routing in one kernel (scores = sigmoid(logit), selection = score + selection bias + cache bias * [resident], weights = plain scores of the chosen
// experts renormalised to one) and its opt-in tail skip, against a CPU reference.
#include "common/cudaUtils.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
struct Result
{
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

Result reference(std::vector<float> const& logits, std::vector<float> const& selectionBias, std::vector<int32_t> const& slotOf, int32_t numTokens,
    int32_t numExperts, int32_t topK, float cacheBias, int32_t tailSkip, bool renorm)
{
    Result out;
    out.ids.resize(static_cast<size_t>(numTokens) * topK);
    out.weights.resize(out.ids.size());
    for (int32_t t = 0; t < numTokens; ++t)
    {
        std::vector<float> score(numExperts), selection(numExperts);
        for (int32_t e = 0; e < numExperts; ++e)
        {
            float const z = logits[static_cast<size_t>(t) * numExperts + e];
            score[e] = std::max(1.0F / (1.0F + std::exp(-z)), 1e-30F);
            selection[e] = score[e] + selectionBias[e] + ((!slotOf.empty() && slotOf[e] >= 0) ? cacheBias : 0.0F);
        }
        std::vector<int32_t> order(numExperts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return selection[a] > selection[b]; });
        float sum = 0.0F, kept = 0.0F;
        std::vector<bool> skipped(topK);
        for (int32_t k = 0; k < topK; ++k)
        {
            sum += score[order[k]];
            skipped[k] = tailSkip > 0 && k >= tailSkip && !slotOf.empty() && slotOf[order[k]] < 0;
            kept += skipped[k] ? 0.0F : score[order[k]];
        }
        float const denom = (renorm ? kept : sum) + 1e-20F;
        for (int32_t k = 0; k < topK; ++k)
        {
            out.ids[static_cast<size_t>(t) * topK + k] = skipped[k] ? order[0] : order[k];
            out.weights[static_cast<size_t>(t) * topK + k] = skipped[k] ? 0.0F : score[order[k]] / denom;
        }
    }
    return out;
}

Result run(std::vector<float> const& logits, std::vector<float> const& selectionBias, std::vector<int32_t> const& slotOf, int32_t numTokens, int32_t numExperts,
    int32_t topK, float cacheBias, int32_t tailSkip, int32_t renorm)
{
    float *dLogits, *dBias, *dWeights;
    int32_t *dSlotOf = nullptr, *dIds;
    CUDA_CHECK(cudaMalloc(&dLogits, logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBias, selectionBias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dIds, static_cast<size_t>(numTokens) * topK * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dWeights, static_cast<size_t>(numTokens) * topK * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dLogits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, selectionBias.data(), selectionBias.size() * sizeof(float), cudaMemcpyHostToDevice));
    ExpertCacheState cache{};
    if (!slotOf.empty())
    {
        CUDA_CHECK(cudaMalloc(&dSlotOf, slotOf.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(dSlotOf, slotOf.data(), slotOf.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        cache.slotOf = dSlotOf;
    }
    launchSigmoidBiasTopk(dLogits, dBias, numTokens, numExperts, topK, cacheBias, slotOf.empty() ? nullptr : &cache, tailSkip, renorm, dIds, dWeights, nullptr);
    CUDA_CHECK(cudaGetLastError());
    Result out;
    out.ids.resize(static_cast<size_t>(numTokens) * topK);
    out.weights.resize(out.ids.size());
    CUDA_CHECK(cudaMemcpy(out.ids.data(), dIds, out.ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.weights.data(), dWeights, out.weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dLogits);
    cudaFree(dBias);
    cudaFree(dIds);
    cudaFree(dWeights);
    if (dSlotOf != nullptr)
    {
        cudaFree(dSlotOf);
    }
    return out;
}

struct Case
{
    int32_t numTokens, numExperts, topK;
    std::vector<float> logits, bias;
    std::vector<int32_t> slotOf;
};

Case makeCase(uint32_t seed, int32_t numTokens, int32_t numExperts, int32_t topK, int32_t resident)
{
    std::mt19937 g(seed);
    std::normal_distribution<float> n(0.0F, 2.0F);
    Case c{numTokens, numExperts, topK, std::vector<float>(static_cast<size_t>(numTokens) * numExperts), std::vector<float>(numExperts), std::vector<int32_t>(numExperts, -1)};
    for (auto& v : c.logits)
    {
        v = n(g);
    }
    for (auto& v : c.bias)
    {
        v = 0.05F * n(g);
    }
    std::vector<int32_t> experts(numExperts);
    std::iota(experts.begin(), experts.end(), 0);
    std::shuffle(experts.begin(), experts.end(), g);
    for (int32_t s = 0; s < resident; ++s)
    {
        c.slotOf[experts[s]] = s;
    }
    return c;
}

void expectSame(Result const& got, Result const& want)
{
    ASSERT_EQ(got.ids.size(), want.ids.size());
    for (size_t i = 0; i < got.ids.size(); ++i)
    {
        EXPECT_EQ(got.ids[i], want.ids[i]) << i;
        EXPECT_NEAR(got.weights[i], want.weights[i], 1e-5F) << i;
    }
}
} // namespace

TEST(SigmoidBiasTopkTest, MatchesTheReferenceWithoutCacheAndWithoutTailSkip)
{
    Case c = makeCase(1, 4, 128, 8, 0);
    expectSame(run(c.logits, c.bias, {}, 4, 128, 8, 0.0F, 0, 0), reference(c.logits, c.bias, {}, 4, 128, 8, 0.0F, 0, false));
}

TEST(SigmoidBiasTopkTest, WeightsOfTheChosenExpertsSumToOne)
{
    Case c = makeCase(2, 3, 128, 8, 30);
    Result r = run(c.logits, c.bias, c.slotOf, 3, 128, 8, 0.1F, 0, 0);
    for (int32_t t = 0; t < 3; ++t)
    {
        float sum = 0.0F;
        for (int32_t k = 0; k < 8; ++k)
        {
            sum += r.weights[t * 8 + k];
        }
        EXPECT_NEAR(sum, 1.0F, 1e-5F);
    }
}

TEST(SigmoidBiasTopkTest, CacheBiasPrefersResidentExperts)
{
    Case c = makeCase(3, 2, 128, 8, 40);
    expectSame(run(c.logits, c.bias, c.slotOf, 2, 128, 8, 0.15F, 0, 0), reference(c.logits, c.bias, c.slotOf, 2, 128, 8, 0.15F, 0, false));
}

TEST(SigmoidBiasTopkTest, TailSkipDropsNonResidentTailExpertsWithoutRenormalisation)
{
    for (int32_t skip : {1, 3, 4, 6})
    {
        Case c = makeCase(10 + skip, 4, 128, 8, 35);
        Result got = run(c.logits, c.bias, c.slotOf, 4, 128, 8, 0.0F, skip, 0);
        expectSame(got, reference(c.logits, c.bias, c.slotOf, 4, 128, 8, 0.0F, skip, false));
        for (int32_t t = 0; t < 4; ++t)
        {
            float sum = 0.0F;
            for (int32_t k = 0; k < 8; ++k)
            {
                int32_t const id = got.ids[t * 8 + k];
                if (k < skip)
                {
                    EXPECT_GT(got.weights[t * 8 + k], 0.0F) << "the head of the ranking is never skipped";
                }
                else if (got.weights[t * 8 + k] == 0.0F)
                {
                    EXPECT_EQ(id, got.ids[t * 8]) << "a skipped slot points at the top-ranked expert";
                }
                else
                {
                    EXPECT_GE(c.slotOf[id], 0) << "a tail expert that is kept is resident";
                }
                sum += got.weights[t * 8 + k];
            }
            EXPECT_LE(sum, 1.0F + 1e-5F);
        }
    }
}

TEST(SigmoidBiasTopkTest, TailSkipWithRenormalisationKeepsTheMassAtOne)
{
    Case c = makeCase(21, 4, 128, 8, 20);
    Result got = run(c.logits, c.bias, c.slotOf, 4, 128, 8, 0.0F, 4, 1);
    expectSame(got, reference(c.logits, c.bias, c.slotOf, 4, 128, 8, 0.0F, 4, true));
    for (int32_t t = 0; t < 4; ++t)
    {
        float sum = 0.0F;
        for (int32_t k = 0; k < 8; ++k)
        {
            sum += got.weights[t * 8 + k];
        }
        EXPECT_NEAR(sum, 1.0F, 1e-5F);
    }
}

TEST(SigmoidBiasTopkTest, TailSkipIsOffWithoutCacheState)
{
    Case c = makeCase(30, 2, 128, 8, 0);
    expectSame(run(c.logits, c.bias, {}, 2, 128, 8, 0.0F, 3, 0), reference(c.logits, c.bias, {}, 2, 128, 8, 0.0F, 0, false));
}

TEST(SigmoidBiasTopkTest, AllExpertsResidentSkipsNothing)
{
    Case c = makeCase(31, 3, 128, 8, 128);
    expectSame(run(c.logits, c.bias, c.slotOf, 3, 128, 8, 0.0F, 2, 0), reference(c.logits, c.bias, c.slotOf, 3, 128, 8, 0.0F, 0, false));
}

TEST(SigmoidBiasTopkTest, NothingResidentKeepsOnlyTheHead)
{
    Case c = makeCase(32, 3, 128, 8, 0);
    Result got = run(c.logits, c.bias, c.slotOf, 3, 128, 8, 0.0F, 2, 0);
    expectSame(got, reference(c.logits, c.bias, c.slotOf, 3, 128, 8, 0.0F, 2, false));
    for (int32_t t = 0; t < 3; ++t)
    {
        for (int32_t k = 2; k < 8; ++k)
        {
            EXPECT_EQ(got.weights[t * 8 + k], 0.0F);
        }
    }
}

TEST(SigmoidBiasTopkTest, NonFiniteLogitsNeverProduceNanWeights)
{
    Case c = makeCase(40, 2, 128, 8, 10);
    c.logits[3] = std::nanf("");
    c.logits[130] = INFINITY;
    c.logits[140] = -INFINITY;
    Result got = run(c.logits, c.bias, c.slotOf, 2, 128, 8, 0.0F, 4, 0);
    for (float w : got.weights)
    {
        EXPECT_TRUE(std::isfinite(w));
    }
}
