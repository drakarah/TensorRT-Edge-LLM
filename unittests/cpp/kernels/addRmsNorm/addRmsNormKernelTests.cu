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

// Fused residual add + RMSNorm kernel against a host reference of the builder's decomposed form: FP16 add, FP32 mean of squares,
// FP32 division, then gamma in FP32 before the cast (weightBeforeCast) or in FP16 after it. The sum must be bit-exact; the norm
// may differ from the host by one FP16 step (FP32 summation order, multiply by the reciprocal instead of a division).

#include "kernels/addRmsNorm/addRmsNorm.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace trt_edgellm;

namespace
{
struct Buffers
{
    half* x{};
    half* r{};
    half* g{};
    half* sum{};
    half* norm{};
    Buffers(size_t rows, size_t hidden)
    {
        cudaMalloc(&x, rows * hidden * 2);
        cudaMalloc(&r, rows * hidden * 2);
        cudaMalloc(&g, hidden * 2);
        cudaMalloc(&sum, rows * hidden * 2);
        cudaMalloc(&norm, rows * hidden * 2);
    }
    ~Buffers()
    {
        cudaFree(x);
        cudaFree(r);
        cudaFree(g);
        cudaFree(sum);
        cudaFree(norm);
    }
};

float ulpDistance(half a, half b)
{
    return std::fabs(static_cast<float>(*reinterpret_cast<int16_t const*>(&a) - *reinterpret_cast<int16_t const*>(&b)));
}

void runCase(int32_t rows, int32_t hidden, bool withResidual, int32_t weightBeforeCast, float scale, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.F, scale);
    size_t const n = static_cast<size_t>(rows) * hidden;
    std::vector<half> hx(n), hr(n), hg(hidden), hsum(n), hnorm(n);
    for (size_t i = 0; i < n; ++i)
    {
        hx[i] = __float2half(dist(rng));
        hr[i] = __float2half(dist(rng) * 0.3F);
    }
    // One all-zero row (the eps keeps it finite) and one row with a large outlier.
    if (rows > 2)
    {
        for (int32_t i = 0; i < hidden; ++i)
        {
            hx[static_cast<size_t>(1) * hidden + i] = __float2half(0.F);
            hr[static_cast<size_t>(1) * hidden + i] = __float2half(0.F);
        }
        hx[static_cast<size_t>(2) * hidden + 5] = __float2half(900.F * scale);
    }
    for (int32_t i = 0; i < hidden; ++i)
    {
        hg[i] = __float2half(1.F + 0.5F * std::sin(0.37F * static_cast<float>(i)));
    }
    Buffers b(rows, hidden);
    cudaMemcpy(b.x, hx.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(b.r, hr.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(b.g, hg.data(), static_cast<size_t>(hidden) * 2, cudaMemcpyHostToDevice);
    cudaMemset(b.sum, 0xFF, n * 2);
    cudaMemset(b.norm, 0xFF, n * 2);
    float const eps = 1e-6F;
    kernel::launchAddRmsNorm(b.x, withResidual ? b.r : nullptr, b.g, withResidual ? b.sum : nullptr, b.norm, rows, hidden, eps,
        weightBeforeCast, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(hsum.data(), b.sum, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hnorm.data(), b.norm, n * 2, cudaMemcpyDeviceToHost);

    float worst = 0.F;
    for (int32_t m = 0; m < rows; ++m)
    {
        std::vector<half> s(hidden);
        double ss = 0.0;
        for (int32_t i = 0; i < hidden; ++i)
        {
            size_t const at = static_cast<size_t>(m) * hidden + i;
            s[i] = withResidual ? __float2half(__half2float(hx[at]) + __half2float(hr[at])) : hx[at];
            if (withResidual)
            {
                ASSERT_EQ(*reinterpret_cast<uint16_t const*>(&s[i]), *reinterpret_cast<uint16_t const*>(&hsum[at]))
                    << "sum row " << m << " col " << i;
            }
            double const f = __half2float(s[i]);
            ss += f * f;
        }
        float const stdv = std::sqrt(static_cast<float>(ss / hidden) + eps);
        for (int32_t i = 0; i < hidden; ++i)
        {
            size_t const at = static_cast<size_t>(m) * hidden + i;
            float const normed = __half2float(s[i]) / stdv;
            half const want = weightBeforeCast != 0 ? __float2half(normed * __half2float(hg[i]))
                                                    : __float2half(__half2float(__float2half(normed)) * __half2float(hg[i]));
            float const d = ulpDistance(want, hnorm[at]);
            worst = std::max(worst, d);
            ASSERT_LE(d, 2.F) << "norm row " << m << " col " << i << " want " << __half2float(want) << " got "
                              << __half2float(hnorm[at]);
        }
    }
    std::printf("rows %d hidden %d residual %d weightBeforeCast %d scale %g: worst %.0f FP16 steps\n", rows, hidden,
        withResidual ? 1 : 0, weightBeforeCast, scale, worst);
}
} // namespace

TEST(AddRmsNormKernelTest, MatchesDecomposedReference)
{
    for (int32_t hidden : {8, 64, 2048, 2056, 5120, 8192})
    {
        for (int32_t rows : {1, 3, 16, 257})
        {
            for (int32_t wbc : {0, 1})
            {
                runCase(rows, hidden, true, wbc, 1.F, 11U + static_cast<uint32_t>(rows + hidden));
                runCase(rows, hidden, false, wbc, 40.F, 12U + static_cast<uint32_t>(rows + hidden));
            }
        }
    }
    runCase(33, 5120, true, 1, 1e-3F, 5U);
}

TEST(AddRmsNormKernelTest, DISABLED_Timing)
{
    for (int32_t rows : {16, 1024})
    {
        int32_t const hidden = 5120;
        Buffers b(rows, hidden);
        cudaMemset(b.x, 0x3C, static_cast<size_t>(rows) * hidden * 2);
        cudaMemset(b.r, 0x3C, static_cast<size_t>(rows) * hidden * 2);
        cudaMemset(b.g, 0x3C, static_cast<size_t>(hidden) * 2);
        for (int32_t withResidual = 0; withResidual < 2; ++withResidual)
        {
            int32_t const iters = 200;
            for (int32_t pass = 0; pass < 2; ++pass)
            {
                cudaDeviceSynchronize();
                auto const t0 = std::chrono::steady_clock::now();
                for (int32_t i = 0; i < iters; ++i)
                {
                    kernel::launchAddRmsNorm(b.x, withResidual != 0 ? b.r : nullptr, b.g, withResidual != 0 ? b.sum : nullptr,
                        b.norm, rows, hidden, 1e-6F, 1, nullptr);
                }
                cudaDeviceSynchronize();
                double const us
                    = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
                if (pass == 1)
                {
                    std::printf("rows %d hidden %d residual %d: %.1f us per call\n", rows, hidden, withResidual, us);
                }
            }
        }
    }
}
