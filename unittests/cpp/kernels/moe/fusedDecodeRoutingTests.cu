/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The fused single-block routing preparation of small decode / verify steps against a CPU reference (slot order inside an expert), including padded
// weights, lock zeroing, invalid expert ids and empty experts.
#include "common/cudaUtils.h"
#include "kernels/moe/moeAlignSumKernels.h"
#include "kernels/moe/moeMarlinIndicesKernels.h"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
template <typename T>
T* upload(std::vector<T> const& v)
{
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, std::max<size_t>(v.size() * sizeof(T), 4)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> download(T const* d, size_t n)
{
    std::vector<T> v(n);
    EXPECT_EQ(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return v;
}

void runCase(int32_t tokens, int32_t topK, int32_t experts, int32_t blockSize, uint32_t seed, bool withInvalid)
{
    int32_t const slots = tokens * topK;
    ASSERT_TRUE(canUseFusedDecodeRouting(slots, experts, blockSize));
    std::mt19937 g(seed);
    std::vector<int32_t> idx(slots);
    std::vector<float> w(slots);
    for (int32_t t = 0; t < tokens; ++t)
    {
        std::vector<int32_t> pool(experts);
        for (int32_t e = 0; e < experts; ++e)
        {
            pool[e] = e;
        }
        std::shuffle(pool.begin(), pool.end(), g);
        for (int32_t k = 0; k < topK; ++k)
        {
            idx[t * topK + k] = pool[k % experts];
            w[t * topK + k] = 0.01F + (g() % 1000) / 1000.0F;
        }
    }
    if (withInvalid && slots > 2)
    {
        idx[1] = -1;
        idx[slots - 1] = experts + 5;
    }
    int32_t const padded = slots + experts * blockSize;
    int32_t const blocks = (padded + blockSize - 1) / blockSize;
    int32_t const numLocks = 56;

    // CPU reference
    std::vector<int32_t> counts(experts, 0);
    for (int32_t s = 0; s < slots; ++s)
    {
        if (idx[s] >= 0 && idx[s] < experts)
        {
            ++counts[idx[s]];
        }
    }
    std::vector<int32_t> pc(experts), off(experts);
    int32_t total = 0;
    for (int32_t e = 0; e < experts; ++e)
    {
        pc[e] = counts[e] > 0 ? (counts[e] + blockSize - 1) / blockSize * blockSize : 0;
        off[e] = total;
        total += pc[e];
    }
    std::vector<int32_t> wantSorted(padded, -7), wantExpert(blocks, -7);
    std::vector<float> wantFlat(padded, -7.F);
    for (int32_t e = 0; e < experts; ++e)
    {
        int32_t seen = 0;
        for (int32_t s = 0; s < slots; ++s)
        {
            if (idx[s] == e)
            {
                wantSorted[off[e] + seen] = s;
                wantFlat[off[e] + seen] = w[s];
                ++seen;
            }
        }
        for (int32_t i = counts[e]; i < pc[e]; ++i)
        {
            wantSorted[off[e] + i] = slots;
            wantFlat[off[e] + i] = 0.F;
        }
        for (int32_t b = 0; b < pc[e] / blockSize; ++b)
        {
            wantExpert[off[e] / blockSize + b] = e;
        }
    }

    int32_t* dIdx = upload(idx);
    float* dW = upload(w);
    int32_t *dPc, *dOff, *dPost, *dSorted, *dExpert, *dLocks;
    float *dFlat, *dPadded;
    ASSERT_EQ(cudaMalloc(&dPc, experts * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dOff, experts * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dPost, 4), cudaSuccess);
    std::vector<int32_t> fillI(padded, -7);
    std::vector<float> fillF(padded, -7.F);
    dSorted = upload(fillI);
    dFlat = upload(fillF);
    dPadded = upload(fillF);
    std::vector<int32_t> fillE(blocks, -7);
    dExpert = upload(fillE);
    std::vector<int32_t> lockInit(numLocks + 8, 123);
    dLocks = upload(lockInit);

    launchFusedDecodeRouting(dIdx, dW, slots, experts, blockSize, dPc, dOff, dPost, dSorted, dFlat, dExpert, dPadded, padded, dLocks, numLocks, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(download(dPc, experts), pc);
    EXPECT_EQ(download(dOff, experts), off);
    EXPECT_EQ(download(dPost, 1)[0], total);
    auto const sorted = download(dSorted, padded);
    auto const flat = download(dFlat, padded);
    for (int32_t p = 0; p < total; ++p)
    {
        ASSERT_EQ(sorted[p], wantSorted[p]) << "position " << p << " tokens " << tokens << " seed " << seed;
        ASSERT_EQ(flat[p], wantFlat[p]) << "position " << p;
    }
    for (int32_t p = total; p < padded; ++p)
    {
        ASSERT_EQ(sorted[p], -7) << "wrote past the padded total at " << p; // untouched beyond numTokensPostPadded
    }
    auto const ex = download(dExpert, blocks);
    for (int32_t b = 0; b < total / blockSize; ++b)
    {
        ASSERT_EQ(ex[b], wantExpert[b]) << "block " << b;
    }
    for (int32_t b = total / blockSize; b < blocks; ++b)
    {
        ASSERT_EQ(ex[b], -7);
    }
    auto const pw = download(dPadded, padded);
    for (int32_t i = 0; i < padded; ++i)
    {
        ASSERT_EQ(pw[i], i < slots ? w[i] : 0.F) << "padded weight " << i;
    }
    auto const lk = download(dLocks, numLocks + 8);
    for (int32_t i = 0; i < numLocks; ++i)
    {
        ASSERT_EQ(lk[i], 0) << "lock " << i;
    }
    for (int32_t i = numLocks; i < numLocks + 8; ++i)
    {
        ASSERT_EQ(lk[i], 123) << "lock array overrun at " << i;
    }
    for (void* d : {static_cast<void*>(dIdx), static_cast<void*>(dW), static_cast<void*>(dPc), static_cast<void*>(dOff), static_cast<void*>(dPost),
             static_cast<void*>(dSorted), static_cast<void*>(dFlat), static_cast<void*>(dPadded), static_cast<void*>(dExpert), static_cast<void*>(dLocks)})
    {
        cudaFree(d);
    }
}
} // namespace

TEST(FusedDecodeRouting, MatchesTheReferenceForDecodeAndVerifySteps)
{
    for (int32_t tokens : {1, 2, 3, 4, 5, 8})
    {
        for (uint32_t seed = 0; seed < 6; ++seed)
        {
            runCase(tokens, 8, 128, 8, 1000 + seed + tokens * 17, false);
        }
    }
}

TEST(FusedDecodeRouting, InvalidExpertIdsAndOtherShapes)
{
    runCase(3, 8, 128, 8, 5, true);
    runCase(1, 8, 64, 8, 6, false);
    runCase(2, 4, 16, 16, 7, true);
    runCase(8, 8, 128, 8, 8, false); // 64 slots, the limit
    runCase(1, 1, 8, 8, 9, false);
}

TEST(FusedDecodeRouting, LimitsAreChecked)
{
    EXPECT_FALSE(canUseFusedDecodeRouting(65, 128, 8));
    EXPECT_FALSE(canUseFusedDecodeRouting(8, 129, 8));
    EXPECT_FALSE(canUseFusedDecodeRouting(0, 128, 8));
    EXPECT_TRUE(canUseFusedDecodeRouting(64, 128, 8));
}
