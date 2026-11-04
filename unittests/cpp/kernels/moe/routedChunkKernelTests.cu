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
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
template <typename T>
T* upload(std::vector<T> const& v)
{
    T* d{};
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
std::vector<T> download(T const* d, size_t n)
{
    std::vector<T> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

struct Routed
{
    std::vector<int32_t> list;
    std::vector<int32_t> rank;
};

Routed runList(std::vector<int32_t> const& counts, int32_t& numRouted)
{
    int32_t const e = static_cast<int32_t>(counts.size());
    int32_t* dCounts = upload(counts);
    int32_t *dList{}, *dRank{}, *dNum{};
    CUDA_CHECK(cudaMalloc(&dList, e * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dRank, e * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dNum, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dList, 0xFF, e * sizeof(int32_t)));
    launchBuildRoutedList(dCounts, e, dList, dRank, dNum, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    numRouted = download(dNum, 1)[0];
    Routed r{download(dList, e), download(dRank, e)};
    cudaFree(dCounts);
    cudaFree(dList);
    cudaFree(dRank);
    cudaFree(dNum);
    return r;
}

void checkList(std::vector<int32_t> const& counts)
{
    int32_t n = -1;
    Routed const r = runList(counts, n);
    std::vector<int32_t> expectList;
    for (size_t e = 0; e < counts.size(); ++e)
    {
        if (counts[e] > 0)
        {
            EXPECT_EQ(r.rank[e], static_cast<int32_t>(expectList.size())) << "expert " << e;
            expectList.push_back(static_cast<int32_t>(e));
        }
        else
        {
            EXPECT_EQ(r.rank[e], -1) << "expert " << e;
        }
    }
    ASSERT_EQ(n, static_cast<int32_t>(expectList.size()));
    for (int32_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(r.list[i], expectList[i]) << "rank " << i;
    }
}

//! Routing arrays as the plugin builds them: experts in id order, each owning padded blocks of moeBlockSize rows; the token id of a row is
//! encoded (expert * 1000 + position) so a slice can be checked row by row, padding rows hold the sentinel -1.
struct Routing
{
    std::vector<int32_t> counts, paddedCounts, paddedOffsets, sortedIds, blockExpert;
    int32_t numPost{0};
};

Routing makeRouting(std::vector<int32_t> const& counts, int32_t blockSize)
{
    Routing r;
    r.counts = counts;
    int32_t offset = 0;
    for (size_t e = 0; e < counts.size(); ++e)
    {
        int32_t const padded = counts[e] > 0 ? (counts[e] + blockSize - 1) / blockSize * blockSize : 0;
        r.paddedCounts.push_back(padded);
        r.paddedOffsets.push_back(offset);
        for (int32_t i = 0; i < padded; ++i)
        {
            r.sortedIds.push_back(i < counts[e] ? static_cast<int32_t>(e) * 1000 + i : -1);
        }
        for (int32_t b = 0; b < padded / blockSize; ++b)
        {
            r.blockExpert.push_back(static_cast<int32_t>(e));
        }
        offset += padded;
    }
    r.numPost = offset;
    return r;
}

//! Runs every chunk of `chunk` routed experts and checks each slice against the rows the experts own.
void checkChunks(std::vector<int32_t> const& counts, int32_t blockSize, int32_t chunk)
{
    Routing const r = makeRouting(counts, blockSize);
    int32_t n = 0;
    Routed const routed = runList(counts, n);
    int32_t const e = static_cast<int32_t>(counts.size());
    std::vector<int32_t> list(routed.list.begin(), routed.list.begin() + n);
    int32_t* dOffsets = upload(r.paddedOffsets);
    int32_t* dPadded = upload(r.paddedCounts);
    int32_t* dSorted = upload(r.sortedIds);
    int32_t* dBlocks = upload(r.blockExpert);
    int32_t* dList = upload(routed.list);
    int32_t* dRank = upload(routed.rank);
    int32_t const cap = r.numPost + blockSize;
    int32_t *dOutIds{}, *dOutBlocks{}, *dOutNum{};
    CUDA_CHECK(cudaMalloc(&dOutIds, cap * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dOutBlocks, (cap / blockSize + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dOutNum, sizeof(int32_t)));
    int32_t covered = 0;
    for (int32_t base = 0; base < n; base += chunk)
    {
        int32_t const count = std::min(chunk, n - base);
        CUDA_CHECK(cudaMemset(dOutIds, 0x7F, cap * sizeof(int32_t)));
        launchBuildRoutedChunkIndices(dOffsets, dPadded, dSorted, dBlocks, dList, dRank, base, count, blockSize, r.numPost, dOutIds, dOutBlocks, dOutNum, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        int32_t expectRows = 0;
        for (int32_t j = 0; j < count; ++j)
        {
            expectRows += r.paddedCounts[list[base + j]];
        }
        int32_t const rows = download(dOutNum, 1)[0];
        ASSERT_EQ(rows, expectRows) << "base " << base;
        std::vector<int32_t> ids = download(dOutIds, rows);
        std::vector<int32_t> blocks = download(dOutBlocks, rows / blockSize);
        int32_t at = 0;
        for (int32_t j = 0; j < count; ++j)
        {
            int32_t const expert = list[base + j];
            for (int32_t i = 0; i < r.paddedCounts[expert]; ++i, ++at)
            {
                EXPECT_EQ(ids[at], i < counts[expert] ? expert * 1000 + i : -1) << "base " << base << " expert " << expert << " row " << i;
            }
            for (int32_t b = 0; b < r.paddedCounts[expert] / blockSize; ++b)
            {
                EXPECT_EQ(blocks[(at - r.paddedCounts[expert]) / blockSize + b], j) << "expert " << expert << " block " << b;
            }
        }
        covered += count;
    }
    EXPECT_EQ(covered, n);
    (void) e;
    for (int32_t* p : {dOffsets, dPadded, dSorted, dBlocks, dList, dRank, dOutIds, dOutBlocks, dOutNum})
    {
        cudaFree(p);
    }
}
} // namespace

TEST(RoutedListTest, NoRoutedExpertGivesAnEmptyList)
{
    int32_t n = -1;
    Routed const r = runList(std::vector<int32_t>(256, 0), n);
    EXPECT_EQ(n, 0);
    for (int32_t v : r.rank)
    {
        EXPECT_EQ(v, -1);
    }
}

TEST(RoutedListTest, SingleExpertAtTheFirstAndTheLastId)
{
    for (int32_t id : {0, 255})
    {
        std::vector<int32_t> counts(256, 0);
        counts[id] = 7;
        checkList(counts);
    }
}

TEST(RoutedListTest, EveryExpertRouted)
{
    checkList(std::vector<int32_t>(256, 3));
}

TEST(RoutedListTest, ExpertCountsThatAreNotAPowerOfTwo)
{
    std::mt19937 rng(1);
    for (int32_t e : {1, 2, 7, 33, 128, 255, 257, 511, 512, 1000, 1024})
    {
        std::vector<int32_t> counts(e);
        for (auto& c : counts)
        {
            c = rng() % 3 == 0 ? static_cast<int32_t>(rng() % 50) : 0;
        }
        checkList(counts);
    }
}

TEST(RoutedListTest, RandomPatterns)
{
    std::mt19937 rng(7);
    for (int32_t round = 0; round < 40; ++round)
    {
        std::vector<int32_t> counts(256);
        int32_t const density = 1 + rng() % 100;
        for (auto& c : counts)
        {
            c = static_cast<int32_t>(rng() % 100) < density ? 1 + static_cast<int32_t>(rng() % 900) : 0;
        }
        checkList(counts);
    }
}

TEST(RoutedListTest, MoreThanTheSupportedNumberOfExpertsThrows)
{
    int32_t* d{};
    CUDA_CHECK(cudaMalloc(&d, 4096 * sizeof(int32_t)));
    EXPECT_THROW(launchBuildRoutedList(d, 1025, d, d, d, nullptr), std::invalid_argument);
    EXPECT_THROW(launchBuildRoutedList(d, 0, d, d, d, nullptr), std::invalid_argument);
    cudaFree(d);
}

TEST(RoutedChunkIndicesTest, ChunkOfOne)
{
    std::vector<int32_t> counts(64, 0);
    counts[3] = 5;
    counts[10] = 16;
    counts[63] = 1;
    checkChunks(counts, 8, 1);
}

TEST(RoutedChunkIndicesTest, ChunkEqualToTheRoutedCountAndLargerThanIt)
{
    std::vector<int32_t> counts(64, 0);
    for (int32_t e : {2, 9, 30, 31, 50})
    {
        counts[e] = 3 + e;
    }
    checkChunks(counts, 16, 5);
    checkChunks(counts, 16, 8);
    checkChunks(counts, 16, 64);
}

TEST(RoutedChunkIndicesTest, LastChunkIsPartial)
{
    std::vector<int32_t> counts(256, 0);
    for (int32_t e = 0; e < 256; e += 3)
    {
        counts[e] = 1 + e % 20;
    }
    checkChunks(counts, 8, 16);
    checkChunks(counts, 8, 7);
}

TEST(RoutedChunkIndicesTest, AllTokensOnOneExpert)
{
    std::vector<int32_t> counts(256, 0);
    counts[100] = 5000;
    checkChunks(counts, 64, 16);
}

TEST(RoutedChunkIndicesTest, EveryExpertRoutedAcrossBlockSizes)
{
    std::mt19937 rng(3);
    std::vector<int32_t> counts(256);
    for (auto& c : counts)
    {
        c = 1 + static_cast<int32_t>(rng() % 200);
    }
    for (int32_t blockSize : {8, 16, 32, 64})
    {
        for (int32_t chunk : {1, 4, 16, 32, 255, 256})
        {
            checkChunks(counts, blockSize, chunk);
        }
    }
}

TEST(RoutedChunkIndicesTest, CountsThatFillBlocksExactlyAndOffByOne)
{
    std::vector<int32_t> counts(32, 0);
    counts[1] = 16;
    counts[2] = 17;
    counts[3] = 15;
    counts[4] = 1;
    counts[5] = 32;
    checkChunks(counts, 16, 2);
}

TEST(RoutedChunkIndicesTest, RandomRoutings)
{
    std::mt19937 rng(11);
    for (int32_t round = 0; round < 30; ++round)
    {
        std::vector<int32_t> counts(256);
        int32_t const density = 1 + rng() % 100;
        for (auto& c : counts)
        {
            c = static_cast<int32_t>(rng() % 100) < density ? 1 + static_cast<int32_t>(rng() % 400) : 0;
        }
        if (std::all_of(counts.begin(), counts.end(), [](int32_t c) { return c == 0; }))
        {
            counts[17] = 1;
        }
        int32_t const blockSizes[] = {8, 16, 32, 64};
        checkChunks(counts, blockSizes[rng() % 4], 1 + static_cast<int32_t>(rng() % 40));
    }
}

TEST(RoutedChunkIndicesTest, EmptyChunkWritesZeroRows)
{
    std::vector<int32_t> counts(16, 0);
    counts[4] = 3;
    Routing const r = makeRouting(counts, 8);
    int32_t n = 0;
    Routed const routed = runList(counts, n);
    int32_t* dOffsets = upload(r.paddedOffsets);
    int32_t* dPadded = upload(r.paddedCounts);
    int32_t* dSorted = upload(r.sortedIds);
    int32_t* dBlocks = upload(r.blockExpert);
    int32_t* dList = upload(routed.list);
    int32_t* dRank = upload(routed.rank);
    int32_t *dOutIds{}, *dOutBlocks{}, *dOutNum{};
    CUDA_CHECK(cudaMalloc(&dOutIds, 64 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dOutBlocks, 64 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dOutNum, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(dOutNum, 0x55, sizeof(int32_t)));
    launchBuildRoutedChunkIndices(dOffsets, dPadded, dSorted, dBlocks, dList, dRank, 0, 0, 8, r.numPost, dOutIds, dOutBlocks, dOutNum, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(download(dOutNum, 1)[0], 0);
    for (int32_t* p : {dOffsets, dPadded, dSorted, dBlocks, dList, dRank, dOutIds, dOutBlocks, dOutNum})
    {
        cudaFree(p);
    }
}
