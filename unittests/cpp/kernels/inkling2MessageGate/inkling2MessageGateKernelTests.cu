/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Inkling 2 message gate (BF16 weights, FP32 activations) against a float64 CPU reference: token counts from 1 to 300, hidden 2304 and other multiples of 8, rank
// 64 and 128, extreme magnitudes, guard bytes around the output.
#include "common/cudaUtils.h"
#include "kernels/inkling2MessageGate/inkling2MessageGateKernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace trt_edgellm::kernel;

namespace
{
uint16_t toBf16(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return static_cast<uint16_t>(bits >> 16); // truncation is enough: the values only need to be representable
}

float fromBf16(uint16_t b)
{
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

std::vector<uint32_t> pack(std::vector<uint16_t> const& v)
{
    std::vector<uint32_t> w(v.size() / 2);
    for (size_t i = 0; i < w.size(); ++i)
    {
        w[i] = static_cast<uint32_t>(v[2 * i]) | (static_cast<uint32_t>(v[2 * i + 1]) << 16);
    }
    return w;
}

void runCase(int64_t tokens, int32_t hidden, int32_t rank, float magnitude, uint32_t seed)
{
    std::mt19937 g(seed);
    std::normal_distribution<float> n(0.F, 1.F);
    std::vector<float> h(tokens * hidden), m(tokens * hidden);
    for (auto& v : h)
    {
        v = n(g) * magnitude;
    }
    for (auto& v : m)
    {
        v = n(g) * magnitude;
    }
    std::vector<uint16_t> down(static_cast<size_t>(rank) * 2 * hidden), forget(static_cast<size_t>(hidden) * rank), gain(static_cast<size_t>(hidden) * rank);
    for (auto& v : down)
    {
        v = toBf16(n(g) * 0.03F);
    }
    for (auto& v : forget)
    {
        v = toBf16(n(g) * 0.3F);
    }
    for (auto& v : gain)
    {
        v = toBf16(n(g) * 0.3F);
    }
    float const eps = 1e-6F;
    // reference
    std::vector<double> ref(tokens * hidden);
    for (int64_t t = 0; t < tokens; ++t)
    {
        double sh = 0, sm = 0;
        for (int32_t k = 0; k < hidden; ++k)
        {
            sh += double(h[t * hidden + k]) * h[t * hidden + k];
            sm += double(m[t * hidden + k]) * m[t * hidden + k];
        }
        double const ih = 1.0 / std::sqrt(sh / hidden + eps), im = 1.0 / std::sqrt(sm / hidden + eps);
        std::vector<double> low(rank);
        for (int32_t j = 0; j < rank; ++j)
        {
            double a = 0;
            for (int32_t k = 0; k < hidden; ++k)
            {
                a += double(fromBf16(down[size_t(j) * 2 * hidden + k])) * h[t * hidden + k] * ih;
                a += double(fromBf16(down[size_t(j) * 2 * hidden + hidden + k])) * m[t * hidden + k] * im;
            }
            low[j] = a;
        }
        for (int32_t c = 0; c < hidden; ++c)
        {
            double f = 0, gg = 0;
            for (int32_t j = 0; j < rank; ++j)
            {
                f += double(fromBf16(forget[size_t(c) * rank + j])) * low[j];
                gg += double(fromBf16(gain[size_t(c) * rank + j])) * low[j];
            }
            ref[t * hidden + c] = (1.0 - std::tanh(f)) * h[t * hidden + c] + std::tanh(gg) * m[t * hidden + c];
        }
    }
    auto const dw = pack(down);
    auto const fw = pack(forget);
    auto const gw = pack(gain);
    float *dh, *dm, *dout;
    uint32_t *ddown, *dforget, *dgain;
    void* ws;
    constexpr int64_t kGuard = 64;
    ASSERT_EQ(cudaMalloc(&dh, h.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dm, m.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dout, (h.size() + 2 * kGuard) * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&ddown, dw.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dforget, fw.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dgain, gw.size() * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&ws, inkling2MessageGateWorkspaceBytes(tokens, rank)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dm, m.data(), m.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ddown, dw.data(), dw.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dforget, fw.data(), fw.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dgain, gw.data(), gw.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<float> canary(h.size() + 2 * kGuard, 777.F);
    ASSERT_EQ(cudaMemcpy(dout, canary.data(), canary.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    launchInkling2MessageGate(dh, dm, ddown, dforget, dgain, dout + kGuard, tokens, hidden, rank, eps, ws, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> got(canary.size());
    ASSERT_EQ(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    for (int64_t i = 0; i < kGuard; ++i)
    {
        ASSERT_EQ(got[i], 777.F);
        ASSERT_EQ(got[kGuard + h.size() + i], 777.F);
    }
    double worst = 0, scale = 1e-30;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        worst = std::max(worst, std::abs(double(got[kGuard + i]) - ref[i]));
        scale = std::max(scale, std::abs(ref[i]));
    }
    EXPECT_LT(worst / scale, 2e-5) << "tokens " << tokens << " hidden " << hidden << " rank " << rank << " magnitude " << magnitude;
    for (void* p : {static_cast<void*>(dh), static_cast<void*>(dm), static_cast<void*>(dout), static_cast<void*>(ddown), static_cast<void*>(dforget),
             static_cast<void*>(dgain), ws})
    {
        cudaFree(p);
    }
}
} // namespace

TEST(Inkling2MessageGate, DecodeAndPrefillTokenCounts)
{
    for (int64_t tokens : {1, 2, 3, 5, 6, 7, 8, 9, 17, 31, 32, 33, 64, 100, 300, 1464})
    {
        runCase(tokens, 2304, 64, 1.F, 10 + tokens);
    }
}

TEST(Inkling2MessageGate, OtherShapesAndMagnitudes)
{
    runCase(4, 2304, 128, 1.F, 1);
    runCase(8, 2304, 128, 1.F, 6);
    runCase(8, 512, 64, 1.F, 7);
    runCase(3, 512, 64, 1.F, 2);
    runCase(2, 2304, 64, 1e-3F, 3);
    runCase(2, 2304, 64, 300.F, 4);
    runCase(1, 4096, 128, 1.F, 5);
}
