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

// FP8 (E4M3) weight x FP16 activation linear: the decode GEMV and the prefill tensor-core GEMM against a float64 CPU reference, plus a bandwidth benchmark.
#include "common/cudaUtils.h"
#include "kernels/fp8Linear/fp8LinearKernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
float fp8ToFloat(uint8_t b)
{
    __half_raw const h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(h));
}

uint8_t randomFp8(std::mt19937& g)
{
    // finite E4M3 only: exponent field 15 with mantissa 7 is NaN
    for (;;)
    {
        uint8_t const b = static_cast<uint8_t>(g());
        if ((b & 0x7F) != 0x7F)
        {
            return b;
        }
    }
}

struct Problem
{
    int32_t tokens, n, k;
    std::vector<uint8_t> w;
    std::vector<float> scale;
    std::vector<__half> x;
    int32_t fmt{0}; // kLinearWeightFp8 or kLinearWeightInt8
};

Problem makeProblem(uint32_t seed, int32_t tokens, int32_t n, int32_t k, float xAmp = 1.0F, int32_t fmt = kLinearWeightFp8)
{
    std::mt19937 g(seed);
    Problem p{tokens, n, k, std::vector<uint8_t>(static_cast<size_t>(n) * k), std::vector<float>(n), std::vector<__half>(static_cast<size_t>(tokens) * k)};
    p.fmt = fmt;
    for (auto& b : p.w)
    {
        if (fmt == kLinearWeightInt8)
        {
            b = static_cast<uint8_t>(g()); // every int8 value, -128 included
            continue;
        }
        // small magnitudes so the products stay moderate: exponent field 2..9
        b = static_cast<uint8_t>((g() & 0x80) | ((2 + g() % 8) << 3) | (g() & 7));
    }
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    for (auto& s : p.scale)
    {
        s = 0.002F + 0.01F * (u(g) * 0.5F + 0.5F);
    }
    for (auto& v : p.x)
    {
        v = __float2half(u(g) * xAmp);
    }
    return p;
}

std::vector<double> reference(Problem const& p)
{
    std::vector<double> y(static_cast<size_t>(p.tokens) * p.n);
    std::vector<float> wf(p.w.size());
    for (size_t i = 0; i < wf.size(); ++i)
    {
        wf[i] = p.fmt == kLinearWeightInt8 ? static_cast<float>(static_cast<int8_t>(p.w[i])) : fp8ToFloat(p.w[i]);
    }
    for (int32_t m = 0; m < p.tokens; ++m)
    {
        for (int32_t r = 0; r < p.n; ++r)
        {
            double s = 0.0;
            for (int32_t c = 0; c < p.k; ++c)
            {
                s += static_cast<double>(wf[static_cast<size_t>(r) * p.k + c]) * __half2float(p.x[static_cast<size_t>(m) * p.k + c]);
            }
            y[static_cast<size_t>(m) * p.n + r] = s * p.scale[r];
        }
    }
    return y;
}

std::vector<float> run(Problem const& p, bool scaleFp32 = true)
{
    void *dx, *dw, *dy, *ds;
    CUDA_CHECK(cudaMalloc(&dx, p.x.size() * 2));
    CUDA_CHECK(cudaMalloc(&dw, p.w.size()));
    CUDA_CHECK(cudaMalloc(&dy, static_cast<size_t>(p.tokens) * p.n * 2));
    CUDA_CHECK(cudaMemcpy(dx, p.x.data(), p.x.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, p.w.data(), p.w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dy, 0xFF, static_cast<size_t>(p.tokens) * p.n * 2)); // NaN canary
    if (scaleFp32)
    {
        CUDA_CHECK(cudaMalloc(&ds, p.n * 4));
        CUDA_CHECK(cudaMemcpy(ds, p.scale.data(), p.n * 4, cudaMemcpyHostToDevice));
    }
    else
    {
        std::vector<__half> sh(p.n);
        for (int32_t i = 0; i < p.n; ++i)
        {
            sh[i] = __float2half(p.scale[i]);
        }
        CUDA_CHECK(cudaMalloc(&ds, p.n * 2));
        CUDA_CHECK(cudaMemcpy(ds, sh.data(), p.n * 2, cudaMemcpyHostToDevice));
    }
    launchFp8Linear(dx, dw, ds, dy, p.tokens, p.n, p.k, scaleFp32, nullptr, p.fmt);
    CUDA_CHECK(cudaGetLastError());
    std::vector<__half> yh(static_cast<size_t>(p.tokens) * p.n);
    CUDA_CHECK(cudaMemcpy(yh.data(), dy, yh.size() * 2, cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    cudaFree(ds);
    std::vector<float> y(yh.size());
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = __half2float(yh[i]);
    }
    return y;
}

void expectClose(Problem const& p, bool scaleFp32 = true)
{
    auto const got = run(p, scaleFp32);
    auto const want = reference(p);
    double norm = 0.0;
    for (double v : want)
    {
        norm = std::max(norm, std::fabs(v));
    }
    double worst = 0.0;
    for (size_t i = 0; i < want.size(); ++i)
    {
        ASSERT_TRUE(std::isfinite(got[i])) << "index " << i << " tokens " << p.tokens << " n " << p.n << " k " << p.k;
        worst = std::max(worst, std::fabs(got[i] - want[i]));
    }
    // FP16 output rounding is the only error source besides the float accumulation order
    EXPECT_LT(worst / norm, 2e-3) << "tokens " << p.tokens << " n " << p.n << " k " << p.k;
}
} // namespace

TEST(Fp8LinearTest, DecodeGemvMatchesTheReferenceForInklingShapes)
{
    for (auto const& [n, k] : {std::pair{3072, 2304}, std::pair{1024, 2304}, std::pair{2304, 3072}, std::pair{384, 2304}})
    {
        for (int32_t tokens : {1, 2, 3, 4})
        {
            ASSERT_TRUE(fp8LinearUsesGemv(tokens));
            expectClose(makeProblem(100 + tokens, tokens, n, k));
        }
    }
}

TEST(Fp8LinearTest, PrefillGemmMatchesTheReferenceForInklingShapes)
{
    for (auto const& [n, k] : {std::pair{3072, 2304}, std::pair{1024, 2304}, std::pair{2304, 3072}, std::pair{384, 2304}})
    {
        for (int32_t tokens : {5, 8, 64, 100})
        {
            ASSERT_FALSE(fp8LinearUsesGemv(tokens));
            expectClose(makeProblem(200 + tokens, tokens, n, k));
        }
    }
}

TEST(Fp8LinearTest, TailsInTokensRowsAndWideTokenCounts)
{
    expectClose(makeProblem(1, 65, 200, 96)); // tokens beyond one 64-row tile, N not a multiple of 64
    expectClose(makeProblem(2, 129, 72, 32)); // the smallest K the tiles accept
    expectClose(makeProblem(3, 257, 520, 160));
    expectClose(makeProblem(4, 1024, 384, 2304));
    expectClose(makeProblem(5, 3, 8, 32));
    expectClose(makeProblem(6, 1, 8, 16 * 3)); // K = 48 is a multiple of the GEMV chunk but not of the GEMM tile: GEMV only
}

TEST(Fp8LinearTest, Fp16ScalesGiveTheSameResultAsFp32Scales)
{
    Problem p = makeProblem(7, 2, 128, 256);
    for (auto& s : p.scale)
    {
        s = __half2float(__float2half(s)); // representable in half, so both paths see identical scales
    }
    auto const a = run(p, true);
    auto const b = run(p, false);
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_EQ(a[i], b[i]);
    }
}

TEST(Fp8LinearTest, ZeroActivationsAndLargeMagnitudesStayFinite)
{
    Problem zero = makeProblem(8, 4, 64, 64, 0.0F);
    for (float v : run(zero))
    {
        EXPECT_EQ(v, 0.0F);
    }
    expectClose(makeProblem(9, 4, 64, 256, 300.0F));
    expectClose(makeProblem(10, 70, 64, 256, 300.0F));
}

TEST(Fp8LinearTest, GemvAndGemmAgreeOnTheSameRows)
{
    // the first 4 rows of a long input are what a 4-token decode step computes
    Problem big = makeProblem(11, 70, 384, 2304);
    Problem small = big;
    small.tokens = 4;
    small.x.resize(static_cast<size_t>(4) * big.k);
    auto const a = run(big);
    auto const b = run(small);
    for (size_t i = 0; i < b.size(); ++i)
    {
        EXPECT_NEAR(a[i], b[i], 2e-3 * std::fabs(b[i]) + 2e-3) << i;
    }
}

TEST(Fp8LinearBench, DecodeBandwidth)
{
    // one visit of Inkling 2: q, gate, rel, k, v, o (rows x columns); weights of 24 sets cycled so nothing stays in L2
    std::vector<std::pair<int32_t, int32_t>> const shapes = {{3072, 2304}, {3072, 2304}, {384, 2304}, {1024, 2304}, {1024, 2304}, {2304, 3072}};
    int32_t const sets = 6;
    std::vector<void*> weights;
    for (int32_t s = 0; s < sets; ++s)
    {
        for (auto const& [n, k] : shapes)
        {
            void* w;
            CUDA_CHECK(cudaMalloc(&w, static_cast<size_t>(n) * k));
            CUDA_CHECK(cudaMemset(w, 0x31, static_cast<size_t>(n) * k));
            weights.push_back(w);
        }
    }
    void *x, *y, *sc;
    CUDA_CHECK(cudaMalloc(&x, 4 * 3072 * 2));
    CUDA_CHECK(cudaMalloc(&y, 4 * 3072 * 2));
    CUDA_CHECK(cudaMalloc(&sc, 4096 * 4));
    CUDA_CHECK(cudaMemset(x, 0, 4 * 3072 * 2));
    CUDA_CHECK(cudaMemset(sc, 0, 4096 * 4));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double totalBytes = 0;
    for (auto const& [n, k] : shapes)
    {
        totalBytes += static_cast<double>(n) * k;
    }
    for (int32_t fmt : {kLinearWeightFp8, kLinearWeightInt8})
    for (int32_t tokens : {1, 2, 4})
    {
        for (int32_t warm = 0; warm < 2; ++warm)
        {
            cudaEventRecord(e0);
            for (int32_t rep = 0; rep < 20; ++rep)
            {
                for (int32_t s = 0; s < sets; ++s)
                {
                    for (size_t i = 0; i < shapes.size(); ++i)
                    {
                        launchFp8Linear(x, weights[s * shapes.size() + i], sc, y, tokens, shapes[i].first, shapes[i].second, true, nullptr, fmt);
                    }
                }
            }
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            if (warm == 1)
            {
                double const us = ms * 1000.0 / (20.0 * sets);
                printf("BENCH decode GEMV %s tokens %d: %.1f us per visit (6 linears, %.1f MB) = %.0f GB/s\n", fmt == kLinearWeightInt8 ? "int8" : "fp8", tokens, us, totalBytes / 1e6, totalBytes / (us * 1e-6) / 1e9);
            }
        }
    }
}

// INT8 weight-only path (same kernels, weight_format 1): every int8 value including -128, the decode GEMV for 1-4 tokens, the prefill GEMM, tails, FP16 scales.
TEST(Int8LinearTest, DecodeGemvMatchesTheReferenceForInklingShapes)
{
    for (auto const& [n, k] : {std::pair{3072, 2304}, std::pair{1024, 2304}, std::pair{2304, 3072}, std::pair{384, 2304}})
    {
        for (int32_t tokens : {1, 2, 3, 4})
        {
            expectClose(makeProblem(300 + tokens, tokens, n, k, 1.0F, kLinearWeightInt8));
        }
    }
}

TEST(Int8LinearTest, PrefillGemmMatchesTheReferenceForInklingShapes)
{
    for (auto const& [n, k] : {std::pair{3072, 2304}, std::pair{1024, 2304}, std::pair{2304, 3072}, std::pair{384, 2304}})
    {
        for (int32_t tokens : {5, 8, 64, 100})
        {
            expectClose(makeProblem(400 + tokens, tokens, n, k, 1.0F, kLinearWeightInt8));
        }
    }
}

TEST(Int8LinearTest, TailsAndFp16Scales)
{
    expectClose(makeProblem(11, 65, 200, 96, 1.0F, kLinearWeightInt8));
    expectClose(makeProblem(12, 129, 72, 32, 1.0F, kLinearWeightInt8));
    expectClose(makeProblem(13, 1, 8, 48, 1.0F, kLinearWeightInt8));
    expectClose(makeProblem(14, 1, 3072, 2304, 1.0F, kLinearWeightInt8), false);
    expectClose(makeProblem(15, 70, 1024, 2304, 1.0F, kLinearWeightInt8), false);
}
