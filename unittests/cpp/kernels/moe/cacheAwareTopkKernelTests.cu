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
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <cstring>
#include <random>
#include <set>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
constexpr float kLowest = -3.0e38F;

float sanitize(float l)
{
    return l == l ? std::min(std::max(l, -3.0e38F), 3.0e38F) : kLowest;
}

struct RouterResult
{
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

//! Selection by logit + bias * [resident], larger score first and lower expert index among equals; weights are the softmax of the original logits over
//! the selected experts; with tailSkip > 0 a non-resident expert ranked at place tailSkip + 1 or lower is replaced by the top-ranked one with weight 0.
RouterResult referenceRouter(std::vector<float> const& logits, std::vector<int32_t> const& slotOf, int32_t numTokens, int32_t numExperts, int32_t topK,
    float bias, int32_t tailSkip)
{
    RouterResult out;
    out.ids.resize(static_cast<size_t>(numTokens) * topK);
    out.weights.resize(out.ids.size());
    for (int32_t t = 0; t < numTokens; ++t)
    {
        float const* row = logits.data() + static_cast<size_t>(t) * numExperts;
        std::vector<float> score(numExperts);
        for (int32_t e = 0; e < numExperts; ++e)
        {
            score[e] = sanitize(row[e]) + (slotOf[e] >= 0 ? bias : 0.0F);
        }
        std::vector<bool> used(numExperts, false);
        std::vector<int32_t> chosen;
        std::vector<float> chosenLogit;
        for (int32_t k = 0; k < topK; ++k)
        {
            int32_t best = -1;
            for (int32_t e = 0; e < numExperts; ++e)
            {
                if (!used[e] && (best < 0 || score[e] > score[best]))
                {
                    best = e;
                }
            }
            used[best] = true;
            chosen.push_back(best);
            chosenLogit.push_back(sanitize(row[best]));
        }
        float const mx = *std::max_element(chosenLogit.begin(), chosenLogit.end());
        float sum = 0.0F;
        std::vector<float> ex(topK);
        for (int32_t k = 0; k < topK; ++k)
        {
            ex[k] = std::exp(chosenLogit[k] - mx);
            sum += ex[k];
        }
        for (int32_t k = 0; k < topK; ++k)
        {
            int32_t id = chosen[k];
            float w = ex[k] / sum;
            if (tailSkip > 0 && k >= tailSkip && slotOf[id] < 0)
            {
                id = chosen[0];
                w = 0.0F;
            }
            out.ids[static_cast<size_t>(t) * topK + k] = id;
            out.weights[static_cast<size_t>(t) * topK + k] = w;
        }
    }
    return out;
}

RouterResult runRouter(std::vector<float> const& logits, std::vector<int32_t> const& slotOf, int32_t numTokens, int32_t numExperts, int32_t topK,
    float bias, int32_t tailSkip)
{
    float* dLogits{};
    int32_t* dSlotOf{};
    int32_t* dIds{};
    float* dWeights{};
    CUDA_CHECK(cudaMalloc(&dLogits, logits.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSlotOf, slotOf.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dIds, static_cast<size_t>(numTokens) * topK * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dWeights, static_cast<size_t>(numTokens) * topK * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dLogits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSlotOf, slotOf.data(), slotOf.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    ExpertCacheState cache{dSlotOf, nullptr, nullptr, nullptr, nullptr, 61, nullptr};
    launchCacheAwareTopk(dLogits, numTokens, numExperts, topK, bias, tailSkip, cache, dIds, dWeights, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    RouterResult out;
    out.ids.resize(static_cast<size_t>(numTokens) * topK);
    out.weights.resize(out.ids.size());
    CUDA_CHECK(cudaMemcpy(out.ids.data(), dIds, out.ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.weights.data(), dWeights, out.weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dLogits));
    CUDA_CHECK(cudaFree(dSlotOf));
    CUDA_CHECK(cudaFree(dIds));
    CUDA_CHECK(cudaFree(dWeights));
    return out;
}

std::vector<int32_t> randomSlots(std::mt19937& rng, int32_t numExperts)
{
    std::vector<int32_t> slotOf(numExperts);
    for (auto& s : slotOf)
    {
        s = rng() % 5 < 2 ? static_cast<int32_t>(rng() % 61) : -1;
    }
    return slotOf;
}

void expectSame(RouterResult const& got, RouterResult const& want, char const* what)
{
    ASSERT_EQ(got.ids, want.ids) << what;
    for (size_t i = 0; i < got.weights.size(); ++i)
    {
        EXPECT_NEAR(got.weights[i], want.weights[i], 1e-5F + 1e-5F * std::fabs(want.weights[i])) << what << " weight " << i;
    }
}
} // namespace

TEST(CacheAwareTopkTest, MatchesHostReferenceOnRandomLogits)
{
    std::mt19937 rng(2026);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    for (int32_t numExperts : {64, 128, 256, 384, 512})
    {
        for (int32_t topK : {1, 4, 8, 10})
        {
            for (int32_t numTokens : {1, 2, 4, 33})
            {
                for (float bias : {0.0F, 0.5F, 3.0F})
                {
                    for (int32_t tailSkip : {0, 3})
                    {
                        std::vector<float> logits(static_cast<size_t>(numTokens) * numExperts);
                        for (auto& l : logits)
                        {
                            l = normal(rng);
                        }
                        auto const slotOf = randomSlots(rng, numExperts);
                        expectSame(runRouter(logits, slotOf, numTokens, numExperts, topK, bias, tailSkip),
                            referenceRouter(logits, slotOf, numTokens, numExperts, topK, bias, tailSkip), "random logits");
                        if (::testing::Test::HasFatalFailure())
                        {
                            return;
                        }
                    }
                }
            }
        }
    }
}

TEST(CacheAwareTopkTest, EqualScoresPickTheLowerExpertIndexFirst)
{
    int32_t const numExperts = 256;
    std::vector<float> logits(numExperts, 1.25F);
    std::vector<int32_t> slotOf(numExperts, -1);
    slotOf[7] = 0;
    slotOf[100] = 1;
    // quantised random logits produce many ties as well
    auto const flat = runRouter(logits, slotOf, 1, numExperts, 8, 0.5F, 0);
    EXPECT_EQ(flat.ids, (std::vector<int32_t>{7, 100, 0, 1, 2, 3, 4, 5}));
    std::mt19937 rng(5);
    std::vector<float> tied(static_cast<size_t>(4) * numExperts);
    for (auto& l : tied)
    {
        l = std::round(std::normal_distribution<float>(0.0F, 2.0F)(rng) * 2.0F) / 2.0F;
    }
    auto const slots = randomSlots(rng, numExperts);
    expectSame(runRouter(tied, slots, 4, numExperts, 8, 0.5F, 0), referenceRouter(tied, slots, 4, numExperts, 8, 0.5F, 0), "tied logits");
}

TEST(CacheAwareTopkTest, NanLogitsRankLast)
{
    int32_t const numExperts = 256;
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::vector<float> logits(static_cast<size_t>(3) * numExperts);
    for (auto& l : logits)
    {
        l = rng() % 7 == 0 ? std::numeric_limits<float>::quiet_NaN() : normal(rng);
    }
    auto const slotOf = randomSlots(rng, numExperts);
    expectSame(runRouter(logits, slotOf, 3, numExperts, 8, 0.5F, 0), referenceRouter(logits, slotOf, 3, numExperts, 8, 0.5F, 0), "nan logits");
}

TEST(CacheAwareTopkTest, InfiniteLogitsGiveDistinctExpertsAndFiniteWeights)
{
    int32_t const numExperts = 256;
    int32_t const topK = 8;
    int32_t const numTokens = 3;
    std::mt19937 rng(17);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    float const inf = std::numeric_limits<float>::infinity();
    for (int32_t trial = 0; trial < 100; ++trial)
    {
        std::vector<float> logits(static_cast<size_t>(numTokens) * numExperts);
        for (auto& l : logits)
        {
            uint32_t const r = rng() % 30;
            l = r == 0 ? inf : r == 1 ? -inf : r == 2 ? std::numeric_limits<float>::quiet_NaN() : normal(rng);
        }
        if (trial % 4 == 0)
        {
            for (auto& l : logits)
            {
                l = rng() % 2 ? -inf : l;
            }
        }
        auto const slotOf = randomSlots(rng, numExperts);
        auto const got = runRouter(logits, slotOf, numTokens, numExperts, topK, 0.5F, 0);
        for (int32_t t = 0; t < numTokens; ++t)
        {
            float sum = 0.0F;
            for (int32_t k = 0; k < topK; ++k)
            {
                int32_t const id = got.ids[t * topK + k];
                float const w = got.weights[t * topK + k];
                ASSERT_GE(id, 0);
                ASSERT_LT(id, numExperts);
                ASSERT_TRUE(std::isfinite(w)) << "trial " << trial;
                sum += w;
                for (int32_t j = k + 1; j < topK; ++j)
                {
                    ASSERT_NE(id, got.ids[t * topK + j]) << "expert chosen twice, trial " << trial;
                }
            }
            EXPECT_NEAR(sum, 1.0F, 1e-4F);
        }
    }
}

TEST(CacheAwareTopkTest, WeightsAreTheSoftmaxOfTheOriginalLogitsNotTheBiasedScores)
{
    int32_t const numExperts = 64;
    std::vector<float> logits(numExperts, 0.0F);
    logits[3] = 2.0F;
    logits[9] = 1.0F;
    std::vector<int32_t> slotOf(numExperts, -1);
    slotOf[9] = 0; // resident: the bias lifts expert 9 above expert 3, but the weights still come from the logits
    auto const got = runRouter(logits, slotOf, 1, numExperts, 2, 5.0F, 0);
    ASSERT_EQ(got.ids[0], 9);
    ASSERT_EQ(got.ids[1], 3);
    float const e3 = std::exp(2.0F), e9 = std::exp(1.0F);
    EXPECT_NEAR(got.weights[0], e9 / (e3 + e9), 1e-5F);
    EXPECT_NEAR(got.weights[1], e3 / (e3 + e9), 1e-5F);
}

TEST(CacheAwareTopkTest, ExpertCountsThatAreNotAMultipleOfTheWarpWidth)
{
    // 33 and 100 leave lanes without an expert; 257 and 511 take the 16-experts-per-lane variant
    std::mt19937 rng(41);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    for (int32_t numExperts : {9, 33, 100, 250, 255, 257, 300, 511})
    {
        for (int32_t topK : {1, 8})
        {
            std::vector<float> logits(static_cast<size_t>(5) * numExperts);
            for (auto& l : logits)
            {
                l = normal(rng);
            }
            auto const slotOf = randomSlots(rng, numExperts);
            expectSame(runRouter(logits, slotOf, 5, numExperts, topK, 0.5F, 0), referenceRouter(logits, slotOf, 5, numExperts, topK, 0.5F, 0),
                "odd expert count");
            if (::testing::Test::HasFatalFailure())
            {
                return;
            }
        }
    }
}

TEST(CacheAwareTopkTest, TopKEqualToTheNumberOfExpertsAndToTheLimit)
{
    std::mt19937 rng(43);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    // every expert is chosen: the ids are a permutation and the weights sum to one
    for (auto [numExperts, topK] : {std::pair<int32_t, int32_t>{8, 8}, {16, 16}, {512, 16}, {17, 16}})
    {
        std::vector<float> logits(static_cast<size_t>(3) * numExperts);
        for (auto& l : logits)
        {
            l = normal(rng);
        }
        auto const slotOf = randomSlots(rng, numExperts);
        expectSame(runRouter(logits, slotOf, 3, numExperts, topK, 0.5F, 0), referenceRouter(logits, slotOf, 3, numExperts, topK, 0.5F, 0), "top-k at the limit");
        if (::testing::Test::HasFatalFailure())
        {
            return;
        }
    }
}

TEST(CacheAwareTopkTest, MoreThanFiveHundredTwelveExpertsUseTheBlockKernel)
{
    std::mt19937 rng(47);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    for (int32_t numExperts : {513, 600, 1024})
    {
        std::vector<float> logits(static_cast<size_t>(4) * numExperts);
        for (auto& l : logits)
        {
            l = normal(rng);
        }
        auto const slotOf = randomSlots(rng, numExperts);
        expectSame(runRouter(logits, slotOf, 4, numExperts, 8, 0.5F, 3), referenceRouter(logits, slotOf, 4, numExperts, 8, 0.5F, 3), "block kernel");
        if (::testing::Test::HasFatalFailure())
        {
            return;
        }
    }
}

TEST(CacheAwareTopkTest, TailSkipEdgeCases)
{
    int32_t const numExperts = 256;
    std::mt19937 rng(53);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::vector<float> logits(static_cast<size_t>(4) * numExperts);
    for (auto& l : logits)
    {
        l = normal(rng);
    }
    auto const slotOf = randomSlots(rng, numExperts);
    std::vector<int32_t> allResident(numExperts, 0);
    std::vector<int32_t> noneResident(numExperts, -1);
    for (int32_t tailSkip : {1, 2, 8, 9, 100})
    {
        std::vector<int32_t> const* const sets[] = {&slotOf, &allResident, &noneResident};
        for (auto const* slots : sets)
        {
            expectSame(runRouter(logits, *slots, 4, numExperts, 8, 0.5F, tailSkip), referenceRouter(logits, *slots, 4, numExperts, 8, 0.5F, tailSkip),
                "tail skip");
            if (::testing::Test::HasFatalFailure())
            {
                return;
            }
        }
    }
    // a skipped slot repeats the top-ranked expert with weight 0 and the remaining weights are not renormalised
    auto const skipped = runRouter(logits, noneResident, 1, numExperts, 8, 0.5F, 1);
    for (int32_t k = 1; k < 8; ++k)
    {
        EXPECT_EQ(skipped.ids[k], skipped.ids[0]);
        EXPECT_EQ(skipped.weights[k], 0.0F);
    }
}

TEST(CacheAwareTopkTest, RowsOfOnlyNonFiniteValues)
{
    int32_t const numExperts = 256;
    float const inf = std::numeric_limits<float>::infinity();
    float const nan = std::numeric_limits<float>::quiet_NaN();
    std::mt19937 rng(59);
    auto const slotOf = randomSlots(rng, numExperts);
    for (float fill : {nan, inf, -inf, 3.0e38F, -3.0e38F, 3.4e38F, -3.4e38F})
    {
        std::vector<float> logits(static_cast<size_t>(2) * numExperts, fill);
        auto const got = runRouter(logits, slotOf, 2, numExperts, 8, 0.5F, 0);
        for (int32_t t = 0; t < 2; ++t)
        {
            float sum = 0.0F;
            std::set<int32_t> distinct;
            for (int32_t k = 0; k < 8; ++k)
            {
                ASSERT_TRUE(std::isfinite(got.weights[t * 8 + k])) << "fill " << fill;
                sum += got.weights[t * 8 + k];
                distinct.insert(got.ids[t * 8 + k]);
            }
            EXPECT_EQ(distinct.size(), 8U) << "fill " << fill;
            EXPECT_NEAR(sum, 1.0F, 1e-4F) << "fill " << fill;
        }
    }
}

TEST(CacheAwareTopkTest, ManyTokensAndRepeatedLaunchesAreDeterministic)
{
    int32_t const numExperts = 256;
    int32_t const numTokens = 1000;
    std::mt19937 rng(61);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::vector<float> logits(static_cast<size_t>(numTokens) * numExperts);
    for (auto& l : logits)
    {
        l = normal(rng);
    }
    auto const slotOf = randomSlots(rng, numExperts);
    auto const first = runRouter(logits, slotOf, numTokens, numExperts, 8, 0.5F, 0);
    expectSame(first, referenceRouter(logits, slotOf, numTokens, numExperts, 8, 0.5F, 0), "1000 tokens");
    auto const second = runRouter(logits, slotOf, numTokens, numExperts, 8, 0.5F, 0);
    EXPECT_EQ(first.ids, second.ids);
    EXPECT_EQ(0, std::memcmp(first.weights.data(), second.weights.data(), first.weights.size() * sizeof(float)));
}

TEST(CacheAwareTopkTest, ZeroBiasIsTheExactRouterAndHugeBiasPrefersTheResidentExperts)
{
    int32_t const numExperts = 256;
    std::mt19937 rng(67);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::vector<float> logits(static_cast<size_t>(3) * numExperts);
    for (auto& l : logits)
    {
        l = normal(rng);
    }
    auto const slotOf = randomSlots(rng, numExperts);
    // bias 0: the selection ignores the cache entirely
    std::vector<int32_t> allResident(numExperts, 0);
    EXPECT_EQ(runRouter(logits, slotOf, 3, numExperts, 8, 0.0F, 0).ids, runRouter(logits, allResident, 3, numExperts, 8, 0.0F, 0).ids);
    // a bias far above the logit range selects resident experts first
    auto const got = runRouter(logits, slotOf, 3, numExperts, 8, 1.0e6F, 0);
    int32_t resident = 0;
    for (int32_t e = 0; e < numExperts; ++e)
    {
        resident += slotOf[e] >= 0 ? 1 : 0;
    }
    ASSERT_GE(resident, 8);
    for (int32_t id : got.ids)
    {
        EXPECT_GE(slotOf[id], 0) << "expert " << id << " is not resident";
    }
}
