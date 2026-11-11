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

// Native INT8-pool paged prefill (CuteDslFMHAV2Runner::runPagedInt8): the pool stays INT8, the kernel expands tiles in
// shared memory and applies the K scale to the softmax scale and the V scale to the normalised output.
// The kernel itself is checked bit for bit against the FP16 kernel in kernelSrcs/fmha_v2_cutedsl/
// test_fmha_v2_paged_i8.py; this file covers what the runner adds: the strided pool view, page-table plumbing, scale
// handling and its contract.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h"
#include "testUtils.h"

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{

int32_t constexpr kPAGE = 128;
int32_t constexpr kHEAD_DIM = 256;

int32_t smVersionForTest()
{
    int32_t sm = getSMVersion();
    applyThorSMRenumberWAR(sm);
    return sm;
}

struct Int8Case
{
    int32_t batch{1};
    std::vector<int32_t> seqLensQ;  // per batch
    std::vector<int32_t> seqLensK;  // per batch, logical KV length including the cached prefix
    int32_t seqLenQCapacity{1};     // padded Q extent
    int32_t numQHeads{6};
    int32_t numKVHeads{2};
    float kScale{0.0123F};
    float vScale{0.0071F};
    float attentionScale{1.F / 16.F};
    bool sharePages{false}; // every request maps the same physical pages (prefix-cache hit)
};

size_t idx4(int32_t a, int32_t b, int32_t c, int32_t d, int32_t nb, int32_t nc, int32_t nd)
{
    return static_cast<size_t>(((static_cast<int64_t>(a) * nb + b) * nc + c) * nd + d);
}

void runInt8Case(Int8Case const& c)
{
    int32_t const sm = smVersionForTest();
    if (!CuteDslFMHAV2Runner::canImplementPagedInt8(kHEAD_DIM, sm, false))
    {
        GTEST_SKIP() << "INT8 paged FMHA-v2 is not built for SM" << sm;
    }
    int32_t const maxK = *std::max_element(c.seqLensK.begin(), c.seqLensK.end());
    int32_t const livePages = (maxK + kPAGE - 1) / kPAGE;
    int32_t const maxPagesPerSeq = livePages + 1; // one unmapped (poisoned) logical slot at the end
    int32_t const pagesPerPlane = c.batch * livePages + 2;
    int32_t const numFlatPages = 2 * pagesPerPlane;

    std::mt19937 rng(1234U + static_cast<uint32_t>(maxK));
    std::uniform_int_distribution<int32_t> codeDist(-127, 127);
    std::uniform_real_distribution<float> qDist(-1.5F, 1.5F);

    // Codes and queries.
    std::vector<int8_t> kCodes(static_cast<size_t>(c.batch) * maxK * c.numKVHeads * kHEAD_DIM);
    std::vector<int8_t> vCodes(kCodes.size());
    for (auto& v : kCodes)
    {
        v = static_cast<int8_t>(codeDist(rng));
    }
    for (auto& v : vCodes)
    {
        v = static_cast<int8_t>(codeDist(rng));
    }
    if (c.sharePages)
    {
        size_t const perBatch = static_cast<size_t>(maxK) * c.numKVHeads * kHEAD_DIM;
        for (int32_t b = 1; b < c.batch; ++b)
        {
            std::copy(kCodes.begin(), kCodes.begin() + perBatch, kCodes.begin() + b * perBatch);
            std::copy(vCodes.begin(), vCodes.begin() + perBatch, vCodes.begin() + b * perBatch);
        }
    }
    std::vector<half> q(static_cast<size_t>(c.batch) * c.seqLenQCapacity * c.numQHeads * kHEAD_DIM);
    for (auto& v : q)
    {
        v = __float2half(qDist(rng));
    }

    // Pool: extreme poison everywhere (K plane -128, V plane 127), then the live tokens scattered over scrambled pages.
    size_t const poolElems = static_cast<size_t>(numFlatPages) * kPAGE * c.numKVHeads * kHEAD_DIM;
    std::vector<int8_t> pool(poolElems);
    for (int32_t page = 0; page < numFlatPages; ++page)
    {
        int8_t const poison = page < pagesPerPlane ? int8_t{-128} : int8_t{127};
        std::fill_n(pool.begin() + static_cast<size_t>(page) * kPAGE * c.numKVHeads * kHEAD_DIM,
            static_cast<size_t>(kPAGE) * c.numKVHeads * kHEAD_DIM, poison);
    }
    std::vector<int32_t> pageList(static_cast<size_t>(c.batch) * 2 * maxPagesPerSeq, 0);
    for (int32_t b = 0; b < c.batch; ++b)
    {
        int32_t const batchPages = (c.seqLensK[b] + kPAGE - 1) / kPAGE;
        int32_t const owner = c.sharePages ? 0 : b;
        for (int32_t lp = 0; lp < batchPages; ++lp)
        {
            // Reverse the K pages and rotate the V pages so neither plane is identity-mapped.
            int32_t const kPage = 1 + owner * livePages + (livePages - lp - 1);
            int32_t const vPage = pagesPerPlane + 1 + owner * livePages + ((lp + 1) % livePages);
            pageList[static_cast<size_t>((b * 2) * maxPagesPerSeq + lp)] = kPage;
            pageList[static_cast<size_t>((b * 2 + 1) * maxPagesPerSeq + lp)] = vPage;
        }
        for (int32_t lp = batchPages; lp < maxPagesPerSeq; ++lp)
        {
            pageList[static_cast<size_t>((b * 2) * maxPagesPerSeq + lp)] = 0; // poisoned K page
            pageList[static_cast<size_t>((b * 2 + 1) * maxPagesPerSeq + lp)] = pagesPerPlane; // poisoned V page
        }
        for (int32_t t = 0; t < c.seqLensK[b]; ++t)
        {
            int32_t const lp = t / kPAGE;
            int32_t const tp = t % kPAGE;
            int32_t const kPage = pageList[static_cast<size_t>((b * 2) * maxPagesPerSeq + lp)];
            int32_t const vPage = pageList[static_cast<size_t>((b * 2 + 1) * maxPagesPerSeq + lp)];
            for (int32_t h = 0; h < c.numKVHeads; ++h)
            {
                for (int32_t d = 0; d < kHEAD_DIM; ++d)
                {
                    size_t const src = idx4(b, t, h, d, maxK, c.numKVHeads, kHEAD_DIM);
                    pool[idx4(kPage, tp, h, d, kPAGE, c.numKVHeads, kHEAD_DIM)] = kCodes[src];
                    pool[idx4(vPage, tp, h, d, kPAGE, c.numKVHeads, kHEAD_DIM)] = vCodes[src];
                }
            }
        }
    }
    std::vector<int32_t> cuQ(static_cast<size_t>(c.batch + 1), 0);
    std::vector<int32_t> cuK(static_cast<size_t>(c.batch + 1), 0);
    for (int32_t b = 0; b < c.batch; ++b)
    {
        cuQ[b + 1] = cuQ[b] + c.seqLensQ[b];
        cuK[b + 1] = cuK[b] + c.seqLensK[b];
    }

    rt::Tensor qT({c.batch, c.seqLenQCapacity, c.numQHeads, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor poolT({numFlatPages, kPAGE, c.numKVHeads, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor pageT({c.batch, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor outT({c.batch, c.seqLenQCapacity, c.numQHeads, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuQT({c.batch + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKT({c.batch + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(qT, q);
    copyHostToDevice(poolT, pool);
    copyHostToDevice(pageT, pageList);
    copyHostToDevice(cuQT, cuQ);
    copyHostToDevice(cuKT, cuK);
    CUDA_CHECK(cudaMemset(outT.rawPointer(), 0, outT.getMemoryCapacity()));

    cudaStream_t stream = nullptr;
    CuteDslFMHAV2Runner runner(
        c.numQHeads, c.numKVHeads, kHEAD_DIM, c.batch, c.seqLenQCapacity, maxPagesPerSeq * kPAGE);
    ASSERT_TRUE(runner.runPagedInt8(qT.rawPointer(), poolT.rawPointer(), pageT.dataPointer<int32_t>(),
        outT.rawPointer(), cuQT.dataPointer<int32_t>(), cuKT.dataPointer<int32_t>(), numFlatPages, maxPagesPerSeq,
        kPAGE, stream, c.attentionScale, c.kScale, c.vScale));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    auto const out = copyDeviceToHost<half>(outT);

    // Reference over the exact dequantised values (float32).
    int32_t const group = c.numQHeads / c.numKVHeads;
    int64_t checked = 0;
    for (int32_t b = 0; b < c.batch; ++b)
    {
        int32_t const sq = c.seqLensQ[b];
        int32_t const sk = c.seqLensK[b];
        for (int32_t i = 0; i < sq; ++i)
        {
            int32_t const keyEnd = sk - sq + i;
            for (int32_t h = 0; h < c.numQHeads; ++h)
            {
                int32_t const kh = h / group;
                std::vector<float> w(static_cast<size_t>(keyEnd + 1));
                float m = -std::numeric_limits<float>::infinity();
                for (int32_t t = 0; t <= keyEnd; ++t)
                {
                    float s = 0.F;
                    for (int32_t d = 0; d < kHEAD_DIM; ++d)
                    {
                        s += __half2float(q[idx4(b, i, h, d, c.seqLenQCapacity, c.numQHeads, kHEAD_DIM)])
                            * (static_cast<float>(kCodes[idx4(b, t, kh, d, maxK, c.numKVHeads, kHEAD_DIM)]) * c.kScale);
                    }
                    w[t] = s * c.attentionScale;
                    m = std::max(m, w[t]);
                }
                float sum = 0.F;
                for (auto& x : w)
                {
                    x = std::exp(x - m);
                    sum += x;
                }
                for (int32_t d = 0; d < kHEAD_DIM; ++d)
                {
                    float o = 0.F;
                    for (int32_t t = 0; t <= keyEnd; ++t)
                    {
                        o += w[t] / sum
                            * (static_cast<float>(vCodes[idx4(b, t, kh, d, maxK, c.numKVHeads, kHEAD_DIM)]) * c.vScale);
                    }
                    float const got = __half2float(out[idx4(b, i, h, d, c.seqLenQCapacity, c.numQHeads, kHEAD_DIM)]);
                    ASSERT_FALSE(std::isnan(got)) << "NaN at b=" << b << " i=" << i << " h=" << h << " d=" << d;
                    // P is rounded to FP16 before P.V, so the absolute error scales with the largest value magnitude
                    // (127 * vScale) as well as with |o|.
                    float const tol = 1e-3F + 4e-3F * std::abs(o) + 6e-4F * 127.F * c.vScale;
                    ASSERT_NEAR(got, o, tol)
                        << "b=" << b << " i=" << i << " h=" << h << " d=" << d << " Sq=" << sq << " Sk=" << sk;
                    ++checked;
                }
            }
        }
    }
    ASSERT_GT(checked, 0);
}

} // namespace

TEST(PagedInt8Test, capabilityTruthTable)
{
    EXPECT_TRUE(CuteDslFMHAV2Runner::canImplementPagedInt8(256, 87, false));
    EXPECT_TRUE(CuteDslFMHAV2Runner::canImplementPagedInt8(256, 80, false));
    EXPECT_FALSE(CuteDslFMHAV2Runner::canImplementPagedInt8(256, 87, true)) << "sliding window not implemented";
    EXPECT_FALSE(CuteDslFMHAV2Runner::canImplementPagedInt8(128, 87, false));
    EXPECT_FALSE(CuteDslFMHAV2Runner::canImplementPagedInt8(512, 87, false));
    EXPECT_FALSE(CuteDslFMHAV2Runner::canImplementPagedInt8(256, 86, false)) << "needs 147 KB of shared memory";
    EXPECT_FALSE(CuteDslFMHAV2Runner::canImplementPagedInt8(256, 100, false));
}

TEST(PagedInt8Test, singleQueryRow)
{
    runInt8Case({1, {1}, {1}, 1});
}

TEST(PagedInt8Test, singleQueryRowOverPrefix)
{
    runInt8Case({1, {1}, {300}, 1});
}

TEST(PagedInt8Test, tileAndPageBoundaries)
{
    // Br=128 / Bc=64 / page=128 edges: one less, equal, one more.
    for (int32_t s : {63, 64, 65, 127, 128, 129})
    {
        SCOPED_TRACE(s);
        runInt8Case({1, {s}, {s}, s});
    }
}

TEST(PagedInt8Test, chunkedPrefillOverCachedPrefix)
{
    runInt8Case({1, {100}, {100 + 257}, 100});
}

TEST(PagedInt8Test, raggedBatchWithPaddedQueryExtent)
{
    runInt8Case({3, {100, 37, 128}, {400, 237, 128}, 128});
}

TEST(PagedInt8Test, pagesSharedBetweenRequests)
{
    Int8Case c{2, {40, 40}, {300, 300}, 40};
    c.sharePages = true;
    runInt8Case(c);
}

TEST(PagedInt8Test, scaleExtremes)
{
    for (auto const& [ks, vs] : {std::pair<float, float>{1.F, 1.F}, {1e-4F, 3e-5F}, {7.3F, 0.0039F}, {0.5F, 4.F}})
    {
        SCOPED_TRACE(ks);
        Int8Case c{1, {70}, {200}, 70};
        c.kScale = ks;
        c.vScale = vs;
        runInt8Case(c);
    }
}

TEST(PagedInt8Test, groupSizeOne)
{
    Int8Case c{1, {65}, {65}, 65};
    c.numQHeads = 2;
    c.numKVHeads = 2;
    runInt8Case(c);
}

TEST(PagedInt8Test, contractViolationsThrow)
{
    int32_t const sm = smVersionForTest();
    if (!CuteDslFMHAV2Runner::canImplementPagedInt8(kHEAD_DIM, sm, false))
    {
        GTEST_SKIP() << "INT8 paged FMHA-v2 is not built for SM" << sm;
    }
    rt::Tensor q({1, 1, 2, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor pool({4, kPAGE, 2, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor pages({1, 2, 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor out({1, 1, 2, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cu({2}, rt::DeviceType::kGPU, DataType::kINT32);
    cudaStream_t stream = nullptr;
    auto run = [&](CuteDslFMHAV2Runner& r, float scale, float ks, float vs, int32_t tokensPerPage, void const* qp) {
        return r.runPagedInt8(qp, pool.rawPointer(), pages.dataPointer<int32_t>(), out.rawPointer(),
            cu.dataPointer<int32_t>(), cu.dataPointer<int32_t>(), 4, 1, tokensPerPage, stream, scale, ks, vs);
    };
    CuteDslFMHAV2Runner ok(2, 2, kHEAD_DIM, 1, 1, kPAGE);
    EXPECT_THROW(run(ok, 0.0625F, 0.F, 1.F, kPAGE, q.rawPointer()), std::runtime_error) << "zero K scale";
    EXPECT_THROW(run(ok, 0.0625F, 1.F, -1.F, kPAGE, q.rawPointer()), std::runtime_error) << "negative V scale";
    EXPECT_THROW(run(ok, 0.0625F, std::numeric_limits<float>::quiet_NaN(), 1.F, kPAGE, q.rawPointer()),
        std::runtime_error)
        << "NaN scale";
    EXPECT_THROW(run(ok, 0.0625F, 1.F, std::numeric_limits<float>::infinity(), kPAGE, q.rawPointer()),
        std::runtime_error)
        << "infinite scale";
    EXPECT_THROW(run(ok, 0.0625F, 1.F, 1.F, 64, q.rawPointer()), std::runtime_error) << "page size must be 128";
    EXPECT_THROW(run(ok, 0.0625F, 1.F, 1.F, kPAGE, nullptr), std::runtime_error) << "null Q";
    CuteDslFMHAV2Runner wrongCapacity(2, 2, kHEAD_DIM, 1, 1, 2 * kPAGE);
    EXPECT_THROW(run(wrongCapacity, 0.0625F, 1.F, 1.F, kPAGE, q.rawPointer()), std::runtime_error)
        << "runner capacity must equal maxPagesPerSeq * page";
    CuteDslFMHAV2Runner head128(2, 2, 128, 1, 1, kPAGE);
    EXPECT_THROW(run(head128, 0.0625F, 1.F, 1.F, kPAGE, q.rawPointer()), std::runtime_error)
        << "only head dim 256 is implemented";
}
