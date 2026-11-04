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

// Decode attention on the packed K8V4 cache (INT8 K, V rows packed to 4 bits with an fp16 scale and zero per 32 channels,
// see cpp/kernels/posEncoding/kvInt4Quant.h). The reference runs in double precision on the dequantised K and on V decoded
// by the host reference of the packed format, so a wrong nibble, group, token pair, tile or page shows up as a large error.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/posEncoding/kvInt4Quant.h"
#include "testUtils.h"
#include "xqaJitTestUtils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
constexpr int32_t kHEAD = 256;
constexpr int32_t kTOKENS_PER_PAGE = 128;
constexpr int32_t kNUM_PATTERNS = 8;

//! One V row of 256 channels; the pattern is picked by the token so every test mixes all of them.
void fillVRow(std::mt19937& gen, int32_t pattern, std::vector<float>& row, float largeAmplitude)
{
    std::uniform_real_distribution<float> uni(-1.0F, 1.0F);
    std::normal_distribution<float> normal(0.0F, 0.1F);
    row.assign(kHEAD, 0.0F);
    switch (pattern)
    {
    case 0:
        for (auto& x : row) x = uni(gen);
        break;
    case 1: // heavy-tailed: every group has its own amplitude over four decades
        for (int32_t g = 0; g < 8; ++g)
        {
            float const amp = std::pow(10.0F, 3.0F * (uni(gen) * 0.5F + 0.5F) - 2.0F);
            for (int32_t i = 0; i < 32; ++i) row[g * 32 + i] = amp * uni(gen);
        }
        break;
    case 2: // constant groups: scale 0, the zero point carries the value
        for (int32_t g = 0; g < 8; ++g)
        {
            float const c = uni(gen) * 2.0F;
            for (int32_t i = 0; i < 32; ++i) row[g * 32 + i] = c;
        }
        break;
    case 3: // zeros except for one group
    {
        int32_t const g = static_cast<int32_t>(gen() % 8);
        for (int32_t i = 0; i < 32; ++i) row[g * 32 + i] = uni(gen);
        break;
    }
    case 4: // tiny values: fp16 scales in the subnormal range
        for (auto& x : row) x = uni(gen) * 2.0e-5F;
        break;
    case 5: // large values, scales up to hundreds in the extreme test
        for (auto& x : row) x = uni(gen) * largeAmplitude;
        break;
    case 6: // one outlier channel per row on top of small noise
        for (auto& x : row) x = normal(gen);
        row[gen() % kHEAD] = 8.0F * (uni(gen) < 0 ? -1.0F : 1.0F);
        break;
    default: // offset groups: all values positive, far from zero
        for (int32_t g = 0; g < 8; ++g)
        {
            float const base = 1.0F + 2.0F * g;
            for (int32_t i = 0; i < 32; ++i) row[g * 32 + i] = base + uni(gen) * 0.5F;
        }
        break;
    }
    for (auto& x : row) x = __half2float(__float2half(x)); // the writer sees fp16 values
}

struct PackedCase
{
    int32_t numQHeads{8};
    int32_t numKVHeads{2};
    std::vector<int32_t> lengths;
    bool paged{false};
    uint32_t multiBlockSplits{1};
    uint32_t seed{1};
    int32_t fixedPattern{-1};    //!< >= 0: every row uses this pattern
    float largeAmplitude{6.0F}; //!< amplitude of pattern 5
    bool peaked{false};         //!< sequence b attends to token b only (every q head of a sequence is the same vector)
};

//! Fills garbage that must never influence the result: fp16 NaN pairs and random bytes.
void fillGarbage(std::vector<int8_t>& bytes, std::mt19937& gen)
{
    for (size_t i = 0; i < bytes.size(); ++i)
    {
        bytes[i] = static_cast<int8_t>(gen());
    }
    // Make a quarter of the metadata-looking words fp16 NaN so a read of stale metadata is loud.
    for (size_t row = 0; row * 256 + 256 <= bytes.size(); ++row)
    {
        if (row % 4 == 0)
        {
            for (int32_t g = 0; g < 8; ++g)
            {
                uint32_t const nan2 = 0x7E007E00U;
                std::memcpy(&bytes[row * 256 + vInt4PackedScaleByte(g) * 1], &nan2, 4);
            }
        }
    }
}

void runPackedDecode(PackedCase const& tc)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    int32_t const batch = static_cast<int32_t>(tc.lengths.size());
    int32_t const H = tc.numKVHeads;
    int32_t const maxLen = *std::max_element(tc.lengths.begin(), tc.lengths.end());
    int32_t const maxPages = (maxLen + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;
    int32_t const capacity = tc.paged ? maxPages * kTOKENS_PER_PAGE : ((maxLen + 63) / 64) * 64;
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(kHEAD));
    std::mt19937 gen(tc.seed);
    std::uniform_real_distribution<float> uni(-1.0F, 1.0F);

    // Q, K (int8 with a per-tensor scale) and V rows.
    std::vector<half> q(static_cast<size_t>(batch) * tc.numQHeads * kHEAD);
    for (auto& x : q) x = __float2half(uni(gen));
    float const kScale = (tc.peaked ? 8.0F : 1.0F) / 127.0F;
    std::vector<int8_t> kQuant(static_cast<size_t>(batch) * H * maxLen * kHEAD);
    for (auto& x : kQuant) x = static_cast<int8_t>(std::lround(uni(gen) * 127.0F));
    if (tc.peaked)
    {
        // Every q head of sequence b is head 0's vector and K is zero except at token b, aligned with it: the logit gap is
        // about 32, so the output is the V row of token b and any token mix-up is a large error.
        for (int32_t b = 0; b < batch; ++b)
        {
            for (int32_t qh = 1; qh < tc.numQHeads; ++qh)
                std::copy_n(&q[(static_cast<size_t>(b) * tc.numQHeads) * kHEAD], kHEAD,
                    &q[(static_cast<size_t>(b) * tc.numQHeads + qh) * kHEAD]);
            for (int32_t h = 0; h < H; ++h)
                for (int32_t s = 0; s < tc.lengths[b]; ++s)
                    for (int32_t d = 0; d < kHEAD; ++d)
                    {
                        float const qd = __half2float(q[(static_cast<size_t>(b) * tc.numQHeads) * kHEAD + d]);
                        kQuant[((static_cast<size_t>(b) * H + h) * maxLen + s) * kHEAD + d]
                            = static_cast<int8_t>(s == b ? (qd >= 0.0F ? 127 : -127) : 0);
                    }
        }
    }
    std::vector<uint8_t> vRows(static_cast<size_t>(batch) * H * maxLen * kHEAD, 0); // packed rows
    std::vector<float> vDecoded(vRows.size(), 0.0F);
    {
        std::vector<float> row;
        for (int32_t b = 0; b < batch; ++b)
            for (int32_t h = 0; h < H; ++h)
                for (int32_t s = 0; s < tc.lengths[b]; ++s)
                {
                    size_t const at = ((static_cast<size_t>(b) * H + h) * maxLen + s) * kHEAD;
                    fillVRow(gen, tc.fixedPattern >= 0 ? tc.fixedPattern : (s + 3 * h + b) % kNUM_PATTERNS, row, tc.largeAmplitude);
                    std::vector<uint8_t> packed(kHEAD, 0);
                    vInt4PackRowHost(row.data(), packed.data());
                    std::copy(packed.begin(), packed.end(), vRows.begin() + at);
                    vInt4UnpackRowHost(packed.data(), &vDecoded[at]);
                }
    }

    // Device cache image, filled with garbage first. Rows past a sequence's length and the unused tail of every V row keep it.
    std::vector<int8_t> cache;
    std::vector<int32_t> pageList;
    auto putRow = [&](size_t headSlot, bool isV, size_t src) {
        // headSlot indexes 256-byte rows of the cache image.
        if (isV)
        {
            // Only the 160 packed bytes are written; the tail stays garbage.
            std::memcpy(&cache[headSlot * kHEAD], &vRows[src], kVInt4PackedUsedBytes);
        }
        else
        {
            std::memcpy(&cache[headSlot * kHEAD], &kQuant[src], kHEAD);
        }
    };
    if (!tc.paged)
    {
        cache.assign(static_cast<size_t>(batch) * 2 * H * capacity * kHEAD, 0);
        fillGarbage(cache, gen);
        for (int32_t b = 0; b < batch; ++b)
            for (int32_t h = 0; h < H; ++h)
                for (int32_t s = 0; s < tc.lengths[b]; ++s)
                {
                    size_t const src = ((static_cast<size_t>(b) * H + h) * maxLen + s) * kHEAD;
                    putRow((((static_cast<size_t>(b) * 2 + 0) * H + h) * capacity + s), false, src);
                    putRow((((static_cast<size_t>(b) * 2 + 1) * H + h) * capacity + s), true, src);
                }
    }
    else
    {
        // K pages 0..P-1 and V pages P..2P-1 of one pool, scrambled; a page is [tokens][heads] rows.
        int32_t const totalPages = batch * maxPages;
        std::vector<int32_t> perm(totalPages);
        for (int32_t i = 0; i < totalPages; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), gen);
        cache.assign(static_cast<size_t>(2) * totalPages * kTOKENS_PER_PAGE * H * kHEAD, 0);
        fillGarbage(cache, gen);
        pageList.assign(static_cast<size_t>(batch) * 2 * maxPages, -1);
        for (int32_t b = 0; b < batch; ++b)
        {
            int32_t const pagesNeeded = (tc.lengths[b] + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;
            for (int32_t p = 0; p < pagesNeeded; ++p)
            {
                int32_t const kPage = perm[b * maxPages + p];
                pageList[(static_cast<size_t>(b) * 2 + 0) * maxPages + p] = kPage;
                pageList[(static_cast<size_t>(b) * 2 + 1) * maxPages + p] = totalPages + perm[b * maxPages + p];
            }
            for (int32_t h = 0; h < H; ++h)
                for (int32_t s = 0; s < tc.lengths[b]; ++s)
                {
                    int32_t const p = s / kTOKENS_PER_PAGE;
                    int32_t const t = s % kTOKENS_PER_PAGE;
                    size_t const src = ((static_cast<size_t>(b) * H + h) * maxLen + s) * kHEAD;
                    int32_t const kPage = pageList[(static_cast<size_t>(b) * 2 + 0) * maxPages + p];
                    int32_t const vPage = pageList[(static_cast<size_t>(b) * 2 + 1) * maxPages + p];
                    putRow((static_cast<size_t>(kPage) * kTOKENS_PER_PAGE + t) * H + h, false, src);
                    putRow((static_cast<size_t>(vPage) * kTOKENS_PER_PAGE + t) * H + h, true, src);
                }
        }
    }

    // Reference in double precision.
    int32_t const ratio = tc.numQHeads / H;
    std::vector<double> ref(static_cast<size_t>(batch) * tc.numQHeads * kHEAD, 0.0);
    for (int32_t b = 0; b < batch; ++b)
        for (int32_t qh = 0; qh < tc.numQHeads; ++qh)
        {
            int32_t const h = qh / ratio;
            std::vector<double> score(tc.lengths[b]);
            double best = -1e300;
            for (int32_t s = 0; s < tc.lengths[b]; ++s)
            {
                size_t const at = ((static_cast<size_t>(b) * H + h) * maxLen + s) * kHEAD;
                double acc = 0.0;
                for (int32_t d = 0; d < kHEAD; ++d)
                {
                    acc += static_cast<double>(__half2float(q[(static_cast<size_t>(b) * tc.numQHeads + qh) * kHEAD + d]))
                        * static_cast<double>(kQuant[at + d]) * kScale;
                }
                score[s] = acc * attentionScale;
                best = std::max(best, score[s]);
            }
            double sum = 0.0;
            for (auto& x : score) { x = std::exp(x - best); sum += x; }
            for (int32_t s = 0; s < tc.lengths[b]; ++s)
            {
                size_t const at = ((static_cast<size_t>(b) * H + h) * maxLen + s) * kHEAD;
                for (int32_t d = 0; d < kHEAD; ++d)
                {
                    ref[(static_cast<size_t>(b) * tc.numQHeads + qh) * kHEAD + d] += score[s] / sum * vDecoded[at + d];
                }
            }
        }

    // Launch.
    ASSERT_TRUE(canCompileXQAKernel(tc.numQHeads, H, kHEAD, smVersion, DataType::kHALF, DataType::kINT4));
    ASSERT_TRUE(loadXQAJitKernelForTest(smVersion, DataType::kHALF, DataType::kINT4, kHEAD, tc.numQHeads, H, false, false,
        tc.paged ? kTOKENS_PER_PAGE : 0));
    DecoderXQARunner runner(DataType::kHALF, DataType::kINT4, batch, tc.numQHeads, H, kHEAD, smVersion);
    auto params = runner.initXQAParams();
    thrust::device_vector<half> qDevice(q);
    thrust::device_vector<int8_t> cacheDevice(cache);
    thrust::device_vector<int32_t> lengthsDevice(tc.lengths);
    thrust::device_vector<int32_t> pageListDevice(pageList);
    thrust::device_vector<half> outDevice(ref.size(), __float2half(0.0F));
    params.qInputPtr = thrust::raw_pointer_cast(qDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(cacheDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(lengthsDevice.data());
    params.kvCache.capacity = capacity;
    if (tc.paged)
    {
        params.kvCache.pageList = thrust::raw_pointer_cast(pageListDevice.data());
        params.kvCache.tokensPerPage = kTOKENS_PER_PAGE;
    }
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.attentionScale = attentionScale;
    params.kScale = kScale;
    params.vScale = 1.0F; // the scales are inside the packed rows
    thrust::device_vector<int32_t> semaphores(tc.multiBlockSplits > 1 ? batch * H : 0, 0);
    size_t const scratchBytes = tc.multiBlockSplits > 1
        ? xqaMultiBlockScratchUpperBound(kHEAD, static_cast<size_t>(batch) * H * tc.multiBlockSplits)
        : 0;
    thrust::device_vector<int8_t> scratch(scratchBytes, 0);
    if (tc.multiBlockSplits > 1)
    {
        params.nbSubSeqPerSeq = tc.multiBlockSplits;
        params.semaphores = thrust::raw_pointer_cast(semaphores.data());
        params.scratch = thrust::raw_pointer_cast(scratch.data());
        params.scratchBytes = scratchBytes;
    }
    cudaStream_t stream{nullptr};
    runner.dispatchXQAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    thrust::host_vector<half> out(outDevice.size());
    thrust::copy(outDevice.begin(), outDevice.end(), out.begin());
    // The tolerance follows the largest V value of the head: fp16 softmax weights cost about 1e-3 of it.
    std::vector<double> vMax(static_cast<size_t>(batch) * H, 0.0);
    for (int32_t b = 0; b < batch; ++b)
        for (int32_t h = 0; h < H; ++h)
            for (int32_t s = 0; s < tc.lengths[b]; ++s)
                for (int32_t d = 0; d < kHEAD; ++d)
                    vMax[static_cast<size_t>(b) * H + h] = std::max(vMax[static_cast<size_t>(b) * H + h],
                        static_cast<double>(std::fabs(vDecoded[((static_cast<size_t>(b) * H + h) * maxLen + s) * kHEAD + d])));
    double worst = 0.0;
    double meanRatio = 0.0;
    for (size_t i = 0; i < out.size(); ++i)
    {
        float const got = __half2float(out[i]);
        ASSERT_TRUE(std::isfinite(got)) << "non-finite output at " << i;
        size_t const b = i / (static_cast<size_t>(tc.numQHeads) * kHEAD);
        size_t const qh = (i / kHEAD) % tc.numQHeads;
        double const err = std::fabs(got - ref[i]);
        double const tol = 1e-7 + 2e-3 * vMax[b * H + qh / ratio] + 1e-2 * std::fabs(ref[i]);
        worst = std::max(worst, err / tol);
        meanRatio += err / tol;
        EXPECT_LE(err, tol) << "output " << i << " (batch " << b << ", head " << qh << ", channel " << i % kHEAD
                            << "): got " << got << " expected " << ref[i];
    }
    meanRatio /= static_cast<double>(out.size());
    EXPECT_LT(meanRatio, 0.2) << "worst error / tolerance " << worst;
    std::cout << "XQA packed-V decode: batch " << batch << " q_heads " << tc.numQHeads << " kv_heads " << H << " paged "
              << tc.paged << " splits " << tc.multiBlockSplits << " max_len " << maxLen << " worst err/tol " << worst
              << std::endl;
}

std::vector<int32_t> tileEdgeLengths()
{
    // Around the 16-token mma step, the 32-token V tile, the 64-token K tile and the 128-token page.
    return {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513};
}
} // namespace

TEST(XQAPackedVDecodingTest, ContiguousTileAndPageEdgeLengths)
{
    runPackedDecode({8, 2, tileEdgeLengths(), false, 1, 11});
}

TEST(XQAPackedVDecodingTest, PagedTileAndPageEdgeLengthsScrambledPages)
{
    runPackedDecode({8, 2, tileEdgeLengths(), true, 1, 12});
}

TEST(XQAPackedVDecodingTest, HeadRatios)
{
    for (auto const& [qHeads, kvHeads] : {std::pair<int32_t, int32_t>{4, 2}, {16, 2}, {8, 1}, {2, 1}, {12, 2}})
    {
        SCOPED_TRACE("q_heads=" + std::to_string(qHeads) + " kv_heads=" + std::to_string(kvHeads));
        runPackedDecode({qHeads, kvHeads, {1, 33, 130, 300}, true, 1, 20});
        runPackedDecode({qHeads, kvHeads, {1, 33, 130, 300}, false, 1, 21});
    }
}

TEST(XQAPackedVDecodingTest, LongContextSingleAndMultiBlock)
{
    for (uint32_t splits : {1U, 2U, 4U, 7U})
    {
        SCOPED_TRACE("splits=" + std::to_string(splits));
        runPackedDecode({8, 2, {4099, 2500}, true, splits, 30});
    }
}

TEST(XQAPackedVDecodingTest, SingleTokenAndSingleSequence)
{
    runPackedDecode({8, 2, {1}, true, 1, 40});
    runPackedDecode({8, 2, {1}, false, 1, 41});
    runPackedDecode({8, 2, {128}, true, 1, 42});
    runPackedDecode({8, 2, {129}, true, 1, 43});
}

TEST(XQAPackedVDecodingTest, SeveralSeeds)
{
    for (uint32_t seed = 100; seed < 106; ++seed)
    {
        SCOPED_TRACE("seed=" + std::to_string(seed));
        runPackedDecode({8, 2, {37, 400, 129, 64}, seed % 2 == 0, 1, seed});
    }
}

TEST(XQAPackedVDecodingTest, ExtremeValueRangesOnePatternAtATime)
{
    for (int32_t pattern = 0; pattern < kNUM_PATTERNS; ++pattern)
    {
        SCOPED_TRACE("pattern=" + std::to_string(pattern));
        PackedCase tc{8, 2, {1, 33, 130, 257}, true, 1, 300U + static_cast<uint32_t>(pattern)};
        tc.fixedPattern = pattern;
        tc.largeAmplitude = 3000.0F;
        runPackedDecode(tc);
    }
}

TEST(XQAPackedVDecodingTest, PeakedAttentionSelectsEachTokenOfEveryTile)
{
    for (int32_t length : {40, 70, 130, 260})
    {
        for (bool paged : {false, true})
        {
            SCOPED_TRACE("length=" + std::to_string(length) + " paged=" + std::to_string(paged));
            PackedCase tc{8, 2, std::vector<int32_t>(length, length), paged, 1, 700U + static_cast<uint32_t>(length)};
            tc.peaked = true;
            runPackedDecode(tc);
        }
    }
}

//! The writer's output, read back from the pool, is what the decoder must see: the format is checked across the two kernels
//! without a hand-built row.
TEST(XQAPackedVDecodingTest, WriterOutputIsReadByTheDecoder)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    constexpr int32_t kBATCH = 2;
    constexpr int32_t kQ_HEADS = 8;
    constexpr int32_t kKV_HEADS = 2;
    constexpr int32_t kPAGES = 2;
    constexpr int32_t kNUM_PAGES = kBATCH * kPAGES;
    constexpr int32_t kCAPACITY = kPAGES * kTOKENS_PER_PAGE;
    std::vector<int32_t> const lengths{kCAPACITY, 150};
    std::mt19937 gen(8080U);
    std::uniform_real_distribution<float> uni(-1.0F, 1.0F);

    // Prefill: write both sequences' K and V through the real writer.
    std::vector<half> qIn(static_cast<size_t>(kBATCH) * kCAPACITY * kQ_HEADS * kHEAD);
    std::vector<half> kIn(static_cast<size_t>(kBATCH) * kCAPACITY * kKV_HEADS * kHEAD);
    std::vector<half> vIn(kIn.size());
    std::vector<float> row;
    for (auto& x : qIn) x = __float2half(uni(gen));
    for (auto& x : kIn) x = __float2half(uni(gen));
    for (size_t r = 0; r * kHEAD < vIn.size(); ++r)
    {
        fillVRow(gen, static_cast<int32_t>(r % kNUM_PATTERNS), row, 6.0F);
        for (int32_t d = 0; d < kHEAD; ++d) vIn[r * kHEAD + d] = __float2half(row[d]);
    }
    cudaStream_t stream{nullptr};
    rt::Tensor cosSin(rt::Coords{1, kCAPACITY, kHEAD}, rt::DeviceType::kGPU, DataType::kFLOAT);
    initializeNormalRopeCosSin(cosSin.dataPointer<float>(), 10000.0F, 1.0F, 1.0F, kHEAD, kCAPACITY, stream);
    rt::Tensor qT(rt::Coords{kBATCH, kCAPACITY, kQ_HEADS, kHEAD}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kT(rt::Coords{kBATCH, kCAPACITY, kKV_HEADS, kHEAD}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vT(rt::Coords{kBATCH, kCAPACITY, kKV_HEADS, kHEAD}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(qT, qIn);
    copyHostToDevice(kT, kIn);
    copyHostToDevice(vT, vIn);
    rt::Tensor endLens(rt::Coords{kBATCH}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(endLens, std::vector<int32_t>(kBATCH, kCAPACITY)); // both sequences write the whole tensor
    std::vector<int32_t> pageTableHost(static_cast<size_t>(kBATCH) * 2 * kPAGES);
    for (int32_t b = 0; b < kBATCH; ++b)
        for (int32_t p = 0; p < kPAGES; ++p)
        {
            int32_t const page = kNUM_PAGES - 1 - (b * kPAGES + p); // scrambled
            pageTableHost[(b * 2 + 0) * kPAGES + p] = page;
            pageTableHost[(b * 2 + 1) * kPAGES + p] = page + kNUM_PAGES;
        }
    rt::Tensor pageTable(rt::Coords{kBATCH, 2, kPAGES}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(pageTable, pageTableHost);
    rt::Tensor pool(
        rt::Coords{2, kNUM_PAGES, kTOKENS_PER_PAGE, kKV_HEADS, kHEAD}, rt::DeviceType::kGPU, DataType::kINT8);
    copyHostToDevice(pool, std::vector<int8_t>(pool.getShape().volume(), static_cast<int8_t>(-128)));
    float const kScale = 1.0F / 20.0F;
    {
        ScopedVInt4WriteMode const packedRows(true);
        launchApplyRopeWriteKV(cosSin, endLens, qT, kT, vT, pool, kScale, 0.05F, stream, false,
            pageTable.dataPointer<int32_t>(), kPAGES);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int8_t> const poolHost = copyDeviceToHost<int8_t>(pool);

    // Reference from the bytes the writer produced.
    auto rowAt = [&](int32_t plane, int32_t b, int32_t s, int32_t h) {
        int32_t const page = pageTableHost[(b * 2 + plane) * kPAGES + s / kTOKENS_PER_PAGE];
        return &poolHost[((static_cast<size_t>(page) * kTOKENS_PER_PAGE + s % kTOKENS_PER_PAGE) * kKV_HEADS + h) * kHEAD];
    };
    std::vector<half> q(static_cast<size_t>(kBATCH) * kQ_HEADS * kHEAD);
    for (auto& x : q) x = __float2half(uni(gen));
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(kHEAD));
    std::vector<double> ref(q.size(), 0.0);
    std::vector<double> vMax(static_cast<size_t>(kBATCH) * kKV_HEADS, 0.0);
    for (int32_t b = 0; b < kBATCH; ++b)
        for (int32_t qh = 0; qh < kQ_HEADS; ++qh)
        {
            int32_t const h = qh / (kQ_HEADS / kKV_HEADS);
            std::vector<double> score(lengths[b]);
            std::vector<std::vector<float>> vDecoded(lengths[b], std::vector<float>(kHEAD));
            double best = -1e300;
            for (int32_t s = 0; s < lengths[b]; ++s)
            {
                int8_t const* kRow = rowAt(0, b, s, h);
                double acc = 0.0;
                for (int32_t d = 0; d < kHEAD; ++d)
                    acc += static_cast<double>(__half2float(q[(static_cast<size_t>(b) * kQ_HEADS + qh) * kHEAD + d]))
                        * kRow[d] * kScale;
                score[s] = acc * attentionScale;
                best = std::max(best, score[s]);
                vInt4UnpackRowHost(reinterpret_cast<uint8_t const*>(rowAt(1, b, s, h)), vDecoded[s].data());
            }
            double sum = 0.0;
            for (auto& x : score) { x = std::exp(x - best); sum += x; }
            for (int32_t s = 0; s < lengths[b]; ++s)
                for (int32_t d = 0; d < kHEAD; ++d)
                {
                    ref[(static_cast<size_t>(b) * kQ_HEADS + qh) * kHEAD + d] += score[s] / sum * vDecoded[s][d];
                    vMax[static_cast<size_t>(b) * kKV_HEADS + h]
                        = std::max(vMax[static_cast<size_t>(b) * kKV_HEADS + h], static_cast<double>(std::fabs(vDecoded[s][d])));
                }
        }

    ASSERT_TRUE(loadXQAJitKernelForTest(
        smVersion, DataType::kHALF, DataType::kINT4, kHEAD, kQ_HEADS, kKV_HEADS, false, false, kTOKENS_PER_PAGE));
    DecoderXQARunner runner(DataType::kHALF, DataType::kINT4, kBATCH, kQ_HEADS, kKV_HEADS, kHEAD, smVersion);
    auto params = runner.initXQAParams();
    thrust::device_vector<half> qDevice(q);
    thrust::device_vector<int32_t> lengthsDevice(lengths);
    thrust::device_vector<half> outDevice(ref.size(), __float2half(0.0F));
    params.qInputPtr = thrust::raw_pointer_cast(qDevice.data());
    params.kvCache.data = pool.rawPointer();
    params.kvCache.pageList = pageTable.dataPointer<int32_t>();
    params.kvCache.tokensPerPage = kTOKENS_PER_PAGE;
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(lengthsDevice.data());
    params.kvCache.capacity = kCAPACITY;
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.attentionScale = attentionScale;
    params.kScale = kScale;
    params.vScale = 1.0F;
    runner.dispatchXQAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    thrust::host_vector<half> out(outDevice.size());
    thrust::copy(outDevice.begin(), outDevice.end(), out.begin());
    double worst = 0.0;
    for (size_t i = 0; i < out.size(); ++i)
    {
        size_t const b = i / (static_cast<size_t>(kQ_HEADS) * kHEAD);
        size_t const h = ((i / kHEAD) % kQ_HEADS) / (kQ_HEADS / kKV_HEADS);
        double const tol = 1e-7 + 2e-3 * vMax[b * kKV_HEADS + h] + 1e-2 * std::fabs(ref[i]);
        double const err = std::fabs(__half2float(out[i]) - ref[i]);
        worst = std::max(worst, err / tol);
        EXPECT_LE(err, tol) << "output " << i;
    }
    std::cout << "XQA packed-V decode of writer output: worst err/tol " << worst << std::endl;
}
