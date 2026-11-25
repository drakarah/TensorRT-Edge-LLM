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

//! Inkling 2 norm + router kernel against a double-precision CPU reference: the norm-only launch, the one-launch small-batch path (1 to 8 tokens) and the tiled GEMM path
//! (more tokens, or too few experts for the small path to write every column), with token counts that do not divide the tile sizes.

#include "kernels/inkling2NormRoute/inkling2NormRouteKernels.h"

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace trt_edgellm::kernel;

namespace
{
struct Case
{
    int32_t tokens, hidden, experts;
};

template <typename T>
T* toDevice(std::vector<T> const& v)
{
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, v.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

void runCase(Case const& c)
{
    std::mt19937 gen(1234 + c.tokens * 7 + c.experts);
    std::normal_distribution<float> nx(0.F, 3.F), nw(1.F, 0.3F), nr(0.F, 0.05F);
    std::vector<float> x(size_t(c.tokens) * c.hidden), w(c.hidden), router(size_t(c.experts) * c.hidden), folded(router.size());
    for (auto& v : x)
    {
        v = nx(gen);
    }
    for (auto& v : w)
    {
        v = nw(gen);
    }
    for (size_t i = 0; i < router.size(); ++i)
    {
        router[i] = nr(gen);
        folded[i] = router[i] * w[i % c.hidden];
    }
    float const eps = 1e-6F;
    float* dx = toDevice(x);
    float* dw = toDevice(w);
    float* dr = c.experts > 0 ? toDevice(folded) : nullptr;
    __half* dy = nullptr;
    float* dl = nullptr;
    void* ws = nullptr;
    ASSERT_EQ(cudaMalloc(&dy, x.size() * sizeof(__half)), cudaSuccess);
    if (c.experts > 0)
    {
        ASSERT_EQ(cudaMalloc(&dl, size_t(c.tokens) * c.experts * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&ws, inkling2NormRouteWorkspaceBytes(c.tokens)), cudaSuccess);
        ASSERT_EQ(cudaMemset(dl, 0xFF, size_t(c.tokens) * c.experts * sizeof(float)), cudaSuccess);
    }
    ASSERT_EQ(cudaMemset(dy, 0xFF, x.size() * sizeof(__half)), cudaSuccess);
    launchInkling2NormRoute(dx, dw, dr, dy, dl, c.tokens, c.hidden, c.experts, eps, ws, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> y(x.size());
    ASSERT_EQ(cudaMemcpy(y.data(), dy, y.size() * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
    std::vector<float> logits(size_t(c.tokens) * c.experts);
    if (c.experts > 0)
    {
        ASSERT_EQ(cudaMemcpy(logits.data(), dl, logits.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    }
    for (int32_t t = 0; t < c.tokens; ++t)
    {
        double sq = 0;
        for (int32_t k = 0; k < c.hidden; ++k)
        {
            sq += double(x[size_t(t) * c.hidden + k]) * x[size_t(t) * c.hidden + k];
        }
        double const inv = 1.0 / std::sqrt(sq / c.hidden + eps);
        for (int32_t k = 0; k < c.hidden; ++k)
        {
            double const ref = x[size_t(t) * c.hidden + k] * inv * w[k];
            double const got = __half2float(y[size_t(t) * c.hidden + k]);
            ASSERT_NEAR(got, ref, 1.2e-3 * std::fabs(ref) + 2e-5) << "y16 token " << t << " col " << k << " case " << c.tokens << "x" << c.hidden << "x" << c.experts;
        }
        for (int32_t e = 0; e < c.experts; ++e)
        {
            double dot = 0;
            for (int32_t k = 0; k < c.hidden; ++k)
            {
                dot += double(x[size_t(t) * c.hidden + k]) * folded[size_t(e) * c.hidden + k];
            }
            ASSERT_NEAR(logits[size_t(t) * c.experts + e], dot * inv, 2e-5 * std::fabs(dot * inv) + 2e-5)
                << "logit token " << t << " expert " << e << " case " << c.tokens << "x" << c.hidden << "x" << c.experts;
        }
    }
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dr);
    cudaFree(dy);
    cudaFree(dl);
    cudaFree(ws);
}
} // namespace

TEST(Inkling2NormRoute, NormOnly)
{
    for (int32_t tokens : {1, 5, 33})
    {
        runCase({tokens, 2304, 0});
    }
    runCase({3, 100, 0});
}

TEST(Inkling2NormRoute, SmallBatchRoute)
{
    for (int32_t tokens : {1, 2, 5, 8})
    {
        runCase({tokens, 2304, 128});
    }
    runCase({3, 512, 24});
}

TEST(Inkling2NormRoute, BatchedGemmRoute)
{
    for (int32_t tokens : {9, 33, 200})
    {
        runCase({tokens, 2304, 128});
    }
    runCase({4, 512, 3});     // too few experts for the small path to write every column
    runCase({3, 3072, 128});  // hidden beyond the register-resident router row
    runCase({40, 100, 37});   // ragged tiles
}
