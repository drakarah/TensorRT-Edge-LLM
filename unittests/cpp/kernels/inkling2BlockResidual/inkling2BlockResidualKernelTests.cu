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
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "kernels/inkling2BlockResidual/inkling2BlockResidualKernels.h"

using namespace trt_edgellm::kernel;

namespace
{
constexpr float kCanary = 12345.0F;
constexpr int64_t kGuardRows = 2;

struct Case
{
    int64_t tokens;
    int32_t hidden;
    int32_t states;
    int32_t queries;
    float magnitude;
    bool zeroRow;
};

void reference(Case const& c, std::vector<std::vector<float>> const& states, std::vector<float> const& queries,
    float eps, std::vector<std::vector<double>>& out)
{
    out.assign(c.queries, std::vector<double>(c.tokens * c.hidden));
    for (int64_t t = 0; t < c.tokens; ++t)
    {
        std::vector<std::vector<double>> normed(c.states, std::vector<double>(c.hidden));
        for (int32_t j = 0; j < c.states; ++j)
        {
            double ss = 0.0;
            for (int32_t h = 0; h < c.hidden; ++h)
            {
                double const v = states[j][t * c.hidden + h];
                ss += v * v;
            }
            double const inv = 1.0 / std::sqrt(ss / c.hidden + eps);
            for (int32_t h = 0; h < c.hidden; ++h)
            {
                normed[j][h] = states[j][t * c.hidden + h] * inv;
            }
        }
        for (int32_t i = 0; i < c.queries; ++i)
        {
            std::vector<double> score(c.states);
            double top = -1e300;
            for (int32_t j = 0; j < c.states; ++j)
            {
                double dot = 0.0;
                for (int32_t h = 0; h < c.hidden; ++h)
                {
                    dot += normed[j][h] * queries[static_cast<int64_t>(i) * c.hidden + h];
                }
                score[j] = dot / std::sqrt(static_cast<double>(c.hidden));
                top = std::max(top, score[j]);
            }
            double denom = 0.0;
            for (int32_t j = 0; j < c.states; ++j)
            {
                score[j] = std::exp(score[j] - top);
                denom += score[j];
            }
            for (int32_t h = 0; h < c.hidden; ++h)
            {
                double acc = 0.0;
                for (int32_t j = 0; j < c.states; ++j)
                {
                    acc += score[j] / denom * normed[j][h];
                }
                out[i][t * c.hidden + h] = acc;
            }
        }
    }
}

void runCase(Case const& c)
{
    float const eps = 1e-6F;
    std::mt19937 gen(static_cast<uint32_t>(c.tokens * 131 + c.hidden + c.states * 7 + c.queries));
    std::normal_distribution<float> dist(0.F, 1.F);
    std::vector<std::vector<float>> states(c.states, std::vector<float>(c.tokens * c.hidden));
    for (auto& s : states)
    {
        for (auto& v : s)
        {
            v = dist(gen) * c.magnitude;
        }
    }
    if (c.zeroRow)
    {
        std::fill(states[0].begin(), states[0].begin() + c.hidden, 0.F);
    }
    std::vector<float> queries(static_cast<int64_t>(c.queries) * c.hidden);
    for (auto& v : queries)
    {
        v = dist(gen);
    }

    Inkling2BlockResidualParams p{};
    p.numTokens = c.tokens;
    p.hidden = c.hidden;
    p.numStates = c.states;
    p.numQueries = c.queries;
    p.eps = eps;
    int64_t const elems = c.tokens * c.hidden;
    for (int32_t j = 0; j < c.states; ++j)
    {
        float* d = nullptr;
        ASSERT_EQ(cudaMalloc(&d, elems * 4), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d, states[j].data(), elems * 4, cudaMemcpyHostToDevice), cudaSuccess);
        p.states[j] = d;
    }
    float* dq = nullptr;
    ASSERT_EQ(cudaMalloc(&dq, queries.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dq, queries.data(), queries.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<float> canary((c.tokens + 2 * kGuardRows) * c.hidden, kCanary);
    std::vector<float*> bases(c.queries);
    for (int32_t i = 0; i < c.queries; ++i)
    {
        ASSERT_EQ(cudaMalloc(&bases[i], canary.size() * 4), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(bases[i], canary.data(), canary.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
        p.outputs[i] = bases[i] + kGuardRows * c.hidden;
    }

    launchInkling2BlockResidual(p, dq, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<std::vector<double>> ref;
    reference(c, states, queries, eps, ref);
    for (int32_t i = 0; i < c.queries; ++i)
    {
        std::vector<float> got(canary.size());
        ASSERT_EQ(cudaMemcpy(got.data(), bases[i], got.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
        for (int64_t k = 0; k < kGuardRows * c.hidden; ++k)
        {
            ASSERT_EQ(got[k], kCanary) << "front guard, query " << i;
            ASSERT_EQ(got[(kGuardRows + c.tokens) * c.hidden + k], kCanary) << "back guard, query " << i;
        }
        double worst = 0.0;
        for (int64_t k = 0; k < elems; ++k)
        {
            double const g = got[kGuardRows * c.hidden + k];
            worst = std::max(worst, std::abs(g - ref[i][k]) - 1e-4 * std::abs(ref[i][k]));
        }
        EXPECT_LT(worst, 2e-5) << "tokens " << c.tokens << " hidden " << c.hidden << " states " << c.states
                               << " queries " << c.queries << " magnitude " << c.magnitude;
    }
    for (int32_t j = 0; j < c.states; ++j)
    {
        cudaFree(const_cast<float*>(p.states[j]));
    }
    cudaFree(dq);
    for (auto* b : bases)
    {
        cudaFree(b);
    }
}
} // namespace

TEST(Inkling2BlockResidual, StateAndQueryCounts)
{
    for (int32_t states : {1, 2, 3, 5, 8})
    {
        for (int32_t queries : {1, 2, 4})
        {
            runCase({7, 2304, states, queries, 1.F, false});
        }
    }
}

TEST(Inkling2BlockResidual, TokenCountsAndOddHidden)
{
    for (int64_t tokens : {1, 2, 257, 2048})
    {
        runCase({tokens, 2304, 5, 4, 1.F, false});
    }
    runCase({9, 100, 4, 4, 1.F, false});
    runCase({9, 33, 3, 1, 1.F, false});
    runCase({3, 1, 2, 2, 1.F, false});
    runCase({3, 1023, 5, 4, 1.F, false});
    runCase({3, 1025, 5, 4, 1.F, false});
    runCase({3, 4096, 8, 4, 1.F, false});
}

TEST(Inkling2BlockResidual, LargeAndTinyMagnitudesAndZeroRow)
{
    runCase({5, 2304, 5, 4, 3e4F, false});
    runCase({5, 2304, 5, 4, 1e-4F, false});
    runCase({5, 2304, 5, 4, 1.F, true});
}
