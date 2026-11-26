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

// Drafter block attention (kernels/inklingDrafter): 8 block rows x 32 heads (8 KV heads of 4) over a context window plus the 8 block keys, against a CPU reference
// (visibility: context key at position p is seen by row r when qpos0 + r - p < window; block keys are all visible; softmax scale 1/sqrt(128)).
#include "kernels/inklingDrafter/inklingDflash2Kernels.h"
#include <algorithm>
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
template <typename T>
T* upload(std::vector<T> const& v)
{
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, std::max<size_t>(v.size() * sizeof(T), 16)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

void runCase(int32_t nCtx, int32_t lo, int32_t window, uint32_t seed, int32_t ringCap = 0)
{
    constexpr int32_t kKv = 1024, kQ = 4096, kD = 128, kG = 4, kBlock = 8;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.F, 1.F);
    auto fill = [&](size_t n, float scale)
    {
        std::vector<__half> v(n);
        for (auto& x : v)
        {
            x = __float2half(dist(rng) * scale);
        }
        return v;
    };
    int32_t const qpos0 = lo + nCtx; // the block follows the context
    std::vector<__half> q = fill(static_cast<size_t>(kBlock) * kQ, 1.5F);
    std::vector<__half> ctxK = fill(static_cast<size_t>(lo + nCtx) * kKv, 1.5F), ctxV = fill(static_cast<size_t>(lo + nCtx) * kKv, 1.F);
    std::vector<__half> kb = fill(static_cast<size_t>(kBlock) * kKv, 1.5F), vb = fill(static_cast<size_t>(kBlock) * kKv, 1.F);
    int32_t const scoreStride = ((window + kBlock + 7) / 8) * 8;
    int32_t const nk = nCtx + kBlock;
    std::vector<__half> ringK, ringV;
    if (ringCap > 0)
    {
        // ring caches: position p in row p % ringCap (only the positions the block can see are stored)
        ringK.assign(static_cast<size_t>(ringCap) * kKv, __float2half(7.F));
        ringV.assign(static_cast<size_t>(ringCap) * kKv, __float2half(7.F));
        for (int32_t p = lo; p < lo + nCtx; ++p)
        {
            std::copy(ctxK.begin() + static_cast<size_t>(p) * kKv, ctxK.begin() + static_cast<size_t>(p + 1) * kKv, ringK.begin() + static_cast<size_t>(p % ringCap) * kKv);
            std::copy(ctxV.begin() + static_cast<size_t>(p) * kKv, ctxV.begin() + static_cast<size_t>(p + 1) * kKv, ringV.begin() + static_cast<size_t>(p % ringCap) * kKv);
        }
    }
    __half *dq = upload(q), *dK = upload(ringCap > 0 ? ringK : ctxK), *dV = upload(ringCap > 0 ? ringV : ctxV), *dkb = upload(kb), *dvb = upload(vb), *dOut = nullptr;
    float *dScores = nullptr, *dPartial = nullptr;
    ASSERT_EQ(cudaMalloc(&dOut, static_cast<size_t>(kBlock) * kQ * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dScores, static_cast<size_t>(8 * 32) * std::max(scoreStride, nk + 8) * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dPartial, static_cast<size_t>(32) * 8 * 32 * 130 * sizeof(float)), cudaSuccess);
    launchDfl2Attention(dq, dK, dV, dkb, dvb, lo, nCtx, qpos0, window, dScores, std::max(scoreStride, nk + 8), dPartial, 32, dOut, nullptr, ringCap);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(kBlock) * kQ);
    ASSERT_EQ(cudaMemcpy(out.data(), dOut, out.size() * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
    double errSq = 0, refSq = 0, maxErr = 0;
    for (int32_t g = 0; g < 8; ++g)
    {
        for (int32_t qi = 0; qi < kBlock * kG; ++qi)
        {
            int32_t const s = qi / kG, hh = qi % kG;
            std::vector<double> score(nk);
            double mx = -1e300;
            for (int32_t k = 0; k < nk; ++k)
            {
                bool const visible = k >= nCtx || (qpos0 + s - (lo + k) < window);
                if (!visible)
                {
                    score[k] = -1e300;
                    continue;
                }
                __half const* kr = k < nCtx ? &ctxK[static_cast<size_t>(lo + k) * kKv + g * kD] : &kb[static_cast<size_t>(k - nCtx) * kKv + g * kD];
                double dot = 0;
                for (int32_t d = 0; d < kD; ++d)
                {
                    dot += double(__half2float(q[static_cast<size_t>(s) * kQ + (g * kG + hh) * kD + d])) * double(__half2float(kr[d]));
                }
                score[k] = dot * 0.08838834764831845;
                mx = std::max(mx, score[k]);
            }
            double den = 0;
            std::vector<double> o(kD, 0.0);
            for (int32_t k = 0; k < nk; ++k)
            {
                if (score[k] <= -1e299)
                {
                    continue;
                }
                double const e = std::exp(score[k] - mx);
                den += e;
                __half const* vr = k < nCtx ? &ctxV[static_cast<size_t>(lo + k) * kKv + g * kD] : &vb[static_cast<size_t>(k - nCtx) * kKv + g * kD];
                for (int32_t d = 0; d < kD; ++d)
                {
                    o[d] += e * double(__half2float(vr[d]));
                }
            }
            for (int32_t d = 0; d < kD; ++d)
            {
                double const ref = o[d] / den;
                double const got = __half2float(out[static_cast<size_t>(s) * kQ + (g * kG + hh) * kD + d]);
                errSq += (got - ref) * (got - ref);
                refSq += ref * ref;
                maxErr = std::max(maxErr, std::abs(got - ref));
            }
        }
    }
    EXPECT_LT(std::sqrt(errSq / refSq), 3e-3) << "nCtx " << nCtx << " lo " << lo << " window " << window << " maxErr " << maxErr;
    for (void* p : {static_cast<void*>(dq), static_cast<void*>(dK), static_cast<void*>(dV), static_cast<void*>(dkb), static_cast<void*>(dvb),
             static_cast<void*>(dOut), static_cast<void*>(dScores), static_cast<void*>(dPartial)})
    {
        cudaFree(p);
    }
}
} // namespace

TEST(DFlash2Attention, MatchesTheCpuReferenceForShortAndLongContexts)
{
    for (int32_t nCtx : {0, 1, 7, 8, 31, 32, 33, 100, 238, 700, 2047})
    {
        runCase(nCtx, 0, 2048, 11 + nCtx);
    }
}

TEST(DFlash2Attention, WindowedContextHidesOldKeysPerRow)
{
    runCase(300, 50, 100, 21); // keys older than the window are invisible to the later block rows
    runCase(2047, 900, 2048, 22);
    runCase(500, 0, 64, 23);
}

TEST(DFlash2Attention, RingCachesGiveTheSameResult)
{
    // the window of 2,048 positions starting at lo wraps around a 2,304-row ring (position p in row p % 2,304)
    runCase(2047, 1500, 2048, 31, 2304);
    runCase(700, 5000, 2048, 32, 2304);
    runCase(300, 12345, 100, 33, 2304);
    runCase(100, 0, 2048, 34, 2304);
}
