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

// Drafter top-16 of FP16 logits rows (kernels/inklingDrafter): sorted descending, ties to the smaller index, padded columns excluded, against a CPU reference.
#include "kernels/inklingDrafter/inklingDflash2Kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace trt_edgellm::kernel;

namespace
{
void runCase(int32_t rows, int32_t cols, int32_t validCols, uint32_t seed, bool manyTies)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.F, 3.F);
    std::vector<__half> host(static_cast<size_t>(rows) * cols);
    for (int32_t r = 0; r < rows; ++r)
    {
        for (int32_t c = 0; c < cols; ++c)
        {
            float v = dist(rng);
            if (manyTies)
            {
                v = std::round(v * 2.F) / 2.F; // coarse values: many exact ties among the top
            }
            if (c >= validCols)
            {
                v = 100.F; // the padded columns must never be selected
            }
            host[static_cast<size_t>(r) * cols + c] = __float2half(v);
        }
    }
    __half* dLogits = nullptr;
    uint64_t* dWork = nullptr;
    int32_t* dIdx = nullptr;
    float* dVal = nullptr;
    ASSERT_EQ(cudaMalloc(&dLogits, host.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dWork, dfl2Top16WorkspaceBytes(rows)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dIdx, rows * 16 * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dVal, rows * 16 * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dLogits, host.data(), host.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);
    launchDfl2Top16(dLogits, rows, cols, validCols, dWork, dIdx, dVal, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<int32_t> idx(rows * 16);
    std::vector<float> val(rows * 16);
    ASSERT_EQ(cudaMemcpy(idx.data(), dIdx, idx.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(val.data(), dVal, val.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int32_t r = 0; r < rows; ++r)
    {
        std::vector<int32_t> order(validCols);
        std::iota(order.begin(), order.end(), 0);
        auto at = [&](int32_t c) { return __half2float(host[static_cast<size_t>(r) * cols + c]); };
        std::partial_sort(order.begin(), order.begin() + 16, order.end(), [&](int32_t a, int32_t b) { return at(a) > at(b) || (at(a) == at(b) && a < b); });
        for (int32_t k = 0; k < 16; ++k)
        {
            ASSERT_EQ(idx[r * 16 + k], order[k]) << "row " << r << " rank " << k << " seed " << seed;
            ASSERT_EQ(val[r * 16 + k], at(order[k]));
        }
    }
    cudaFree(dLogits);
    cudaFree(dWork);
    cudaFree(dIdx);
    cudaFree(dVal);
}
} // namespace

TEST(DFlash2Top16, MatchesTheCpuReferenceForTheDrafterShape)
{
    for (uint32_t seed = 0; seed < 4; ++seed)
    {
        runCase(7, 200064, 200058, 100 + seed, false);
    }
}

TEST(DFlash2Top16, TiesGoToTheSmallerIndexAndPaddedColumnsAreIgnored)
{
    for (uint32_t seed = 0; seed < 4; ++seed)
    {
        runCase(7, 200064, 200058, 200 + seed, true);
    }
}

TEST(DFlash2Top16, OtherShapes)
{
    runCase(1, 4096, 4000, 301, false);
    runCase(3, 33000, 32990, 302, true);
    runCase(2, 262144, 262144, 303, false);
}
