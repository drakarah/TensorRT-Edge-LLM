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

//! Inkling 2 attention kernel against a double-precision CPU reference: paging, padding slots, windows, relative
//! extent, clipping, extreme sink/gate/score values, NaN isolation and guard canaries around the KV pool.

#include "common/cudaUtils.h"
#include "kernels/inkling2Attention/inkling2AttentionKernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
constexpr int32_t kHd = kInkling2HeadDim;
constexpr int32_t kRd = kInkling2RelDim;
constexpr int32_t kPage = kInkling2TokensPerPage;
constexpr float kSentinel = 77.0F;

struct Case
{
    int32_t nh = 6, nkv = 2, extent = 64, window = 0, batch = 1, maxPages = 4, numPages = 10;
    float alpha = 0.1F, nFloor = 16.0F, qClip = 22.627417F, relClip = 8.0F;
    float scaling = 1.0F / std::sqrt(128.0F);
};

float h2f(half v)
{
    return __half2float(v);
}

std::vector<half> toHalf(std::vector<float> const& v)
{
    std::vector<half> out(v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        out[i] = __float2half(v[i]);
    }
    return out;
}

template <typename T>
T* upload(std::vector<T> const& host)
{
    T* d{};
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(host.size(), 1) * sizeof(T)));
    if (!host.empty())
    {
        CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return d;
}

struct Data
{
    // Per packed token slot.
    std::vector<float> q, rel, sink, gate, k, v;
    std::vector<float> qScale, relScale, proj; // proj [extent, 16]
    std::vector<int32_t> offsets, lengths, past;
    std::vector<int32_t> table; // [batch, 2, maxPages]
    int32_t slots = 0;
};

std::vector<float> normal(size_t n, float scale, std::mt19937& gen)
{
    std::normal_distribution<float> dist(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v)
    {
        x = h2f(__float2half(dist(gen))); // stay on the FP16 grid so the reference sees the same inputs
    }
    return v;
}

//! counts[b] real tokens for sequence b with past[b] cached tokens; width > 0 selects entry-padded slots.
Data makeData(Case const& c, std::vector<int32_t> const& past, std::vector<int32_t> const& counts, int32_t width,
    uint32_t seed, float qScaleMag = 0.8F, float kMag = 1.5F)
{
    std::mt19937 gen(seed);
    Data d;
    d.past = past;
    d.lengths = counts;
    d.offsets = {0};
    for (int32_t b = 0; b < c.batch; ++b)
    {
        d.offsets.push_back(d.offsets.back() + (width > 0 ? width : counts[b]));
    }
    d.slots = d.offsets.back();
    d.q = normal(size_t(d.slots) * c.nh * kHd, qScaleMag, gen);
    d.rel = normal(size_t(d.slots) * c.nh * kRd, 0.8F, gen);
    d.sink = normal(size_t(d.slots) * c.nh, 0.8F, gen);
    d.gate = normal(size_t(d.slots) * c.nh * kHd, 0.8F, gen);
    d.k = normal(size_t(d.slots) * c.nkv * kHd, kMag, gen);
    d.v = normal(size_t(d.slots) * c.nkv * kHd, 1.0F, gen);
    d.qScale = normal(size_t(c.nh) * kHd, 0.1F, gen);
    d.relScale = normal(size_t(c.nh) * kRd, 0.1F, gen);
    for (auto& x : d.qScale)
    {
        x += 1.0F;
    }
    for (auto& x : d.relScale)
    {
        x += 1.0F;
    }
    d.proj = normal(size_t(c.extent) * kRd, 0.5F, gen);
    // Scrambled pages; V entries sit on the flattened plane axis (K page + numPages).
    std::vector<int32_t> perm(c.numPages);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), gen);
    d.table.assign(size_t(c.batch) * 2 * c.maxPages, 0);
    size_t cursor = 0;
    for (int32_t b = 0; b < c.batch; ++b)
    {
        int32_t const total = past[b] + counts[b];
        for (int32_t p = 0; p < (total + kPage - 1) / kPage; ++p)
        {
            int32_t const page = perm.at(cursor++);
            d.table[(size_t(b) * 2 + 0) * c.maxPages + p] = page;
            d.table[(size_t(b) * 2 + 1) * c.maxPages + p] = page + c.numPages;
        }
    }
    return d;
}

struct Pool
{
    std::vector<float> host; // [2 * numPages, 128, nkv, 128]
    size_t rowElems;
    float& at(int32_t flatPage, int32_t slot, int32_t head, int32_t d)
    {
        return host[((size_t(flatPage) * kPage + slot) * rowElems / kHd + head) * kHd + d];
    }
};

Pool newPool(Case const& c)
{
    Pool p;
    p.rowElems = size_t(c.nkv) * kHd;
    p.host.assign(size_t(2) * c.numPages * kPage * p.rowElems, kSentinel);
    return p;
}

//! Reference for slot t of the packed batch; pool must already hold the keys/values of positions < past + length.
void reference(Case const& c, Data const& d, Pool& pool, int32_t t, int32_t seq, std::vector<double>& out)
{
    int32_t const local = t - d.offsets[seq];
    int32_t const qpos = d.past[seq] + local;
    int32_t const rep = c.nh / c.nkv;
    double const logScale = c.window == 0 ? 1.0 + c.alpha * std::log(std::max((qpos + 1.0) / c.nFloor, 1.0)) : 1.0;
    out.assign(size_t(c.nh) * kHd, 0.0);
    for (int32_t h = 0; h < c.nh; ++h)
    {
        std::vector<double> q(kHd);
        double ss = 0;
        for (int32_t i = 0; i < kHd; ++i)
        {
            q[i] = double(d.q[(size_t(t) * c.nh + h) * kHd + i]) * d.qScale[size_t(h) * kHd + i];
            ss += q[i] * q[i];
        }
        double const qFactor = std::min(double(c.qClip) / (std::sqrt(ss) + 1e-6), 1.0) * logScale;
        std::vector<double> rs(kRd);
        double rss = 0;
        for (int32_t i = 0; i < kRd; ++i)
        {
            rs[i] = double(d.rel[(size_t(t) * c.nh + h) * kRd + i]) * d.relScale[size_t(h) * kRd + i];
            rss += rs[i] * rs[i];
        }
        double const rFactor = std::min(double(c.relClip) / (std::sqrt(rss) + 1e-6), 1.0) * logScale;
        int32_t const lo = c.window > 0 ? std::max(0, qpos - c.window + 1) : 0;
        std::vector<double> score;
        for (int32_t s = lo; s <= qpos; ++s)
        {
            int32_t const page = d.table[(size_t(seq) * 2) * c.maxPages + s / kPage];
            double dot = 0;
            for (int32_t i = 0; i < kHd; ++i)
            {
                dot += q[i] * qFactor * pool.at(page, s % kPage, h / rep, i);
            }
            double bias = 0;
            int32_t const dist = qpos - s;
            if (dist < c.extent)
            {
                for (int32_t i = 0; i < kRd; ++i)
                {
                    bias += rs[i] * rFactor * d.proj[size_t(dist) * kRd + i];
                }
            }
            score.push_back(dot * c.scaling + bias);
        }
        double const sink = d.sink[size_t(t) * c.nh + h];
        double m = sink;
        for (double s : score)
        {
            m = std::max(m, s);
        }
        double den = std::exp(sink - m);
        std::vector<double> acc(kHd, 0.0);
        for (size_t j = 0; j < score.size(); ++j)
        {
            double const w = std::exp(score[j] - m);
            den += w;
            int32_t const s = lo + int32_t(j);
            int32_t const vPage = d.table[(size_t(seq) * 2 + 1) * c.maxPages + s / kPage];
            for (int32_t i = 0; i < kHd; ++i)
            {
                acc[i] += w * pool.at(vPage, s % kPage, h / rep, i);
            }
        }
        for (int32_t i = 0; i < kHd; ++i)
        {
            double const g = 1.0 / (1.0 + std::exp(-double(d.gate[(size_t(t) * c.nh + h) * kHd + i])));
            out[size_t(h) * kHd + i] = acc[i] / den * g;
        }
    }
}

struct Result
{
    std::vector<float> out; // [slots, nh * 128]
    Pool pool;
};

//! Runs the kernel for one step, updating the host pool image with the appended rows first (for the reference).
Result run(Case const& c, Data const& d, Pool pool)
{
    int32_t const nTok = d.slots;
    half* dq = upload(toHalf(d.q));
    half* drel = upload(toHalf(d.rel));
    half* dsink = upload(toHalf(d.sink));
    half* dgate = upload(toHalf(d.gate));
    half* dk = upload(toHalf(d.k));
    half* dv = upload(toHalf(d.v));
    float* dqs = upload(d.qScale);
    float* drs = upload(d.relScale);
    float* dproj = upload(d.proj);
    half* dpool = upload(toHalf(pool.host));
    int32_t* doff = upload(d.offsets);
    int32_t* dlen = upload(d.lengths);
    int32_t* dpast = upload(d.past);
    int32_t* dtab = upload(d.table);
    half* dout{};
    CUDA_CHECK(cudaMalloc(&dout, std::max<size_t>(size_t(nTok) * c.nh * kHd, 1) * sizeof(half)));
    std::vector<half> junk(size_t(nTok) * c.nh * kHd, __float2half(9.0F));
    if (!junk.empty())
    {
        CUDA_CHECK(cudaMemcpy(dout, junk.data(), junk.size() * sizeof(half), cudaMemcpyHostToDevice));
    }
    Inkling2AttentionParams p{};
    p.numTokens = nTok;
    p.batchSize = c.batch;
    p.numQHeads = c.nh;
    p.numKVHeads = c.nkv;
    p.relExtent = c.extent;
    p.window = c.window;
    p.maxPagesPerSeq = c.maxPages;
    p.scaling = c.scaling;
    p.logAlpha = c.alpha;
    p.logNFloor = c.nFloor;
    p.qClip = c.qClip;
    p.relClip = c.relClip;
    launchInkling2Attention(
        p, dq, drel, dsink, dgate, dqs, drs, dproj, dk, dv, dpool, doff, dlen, dpast, dtab, dout, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    Result r;
    std::vector<half> hout(size_t(nTok) * c.nh * kHd);
    if (!hout.empty())
    {
        CUDA_CHECK(cudaMemcpy(hout.data(), dout, hout.size() * sizeof(half), cudaMemcpyDeviceToHost));
    }
    r.out.resize(hout.size());
    for (size_t i = 0; i < hout.size(); ++i)
    {
        r.out[i] = h2f(hout[i]);
    }
    std::vector<half> hpool(pool.host.size());
    CUDA_CHECK(cudaMemcpy(hpool.data(), dpool, hpool.size() * sizeof(half), cudaMemcpyDeviceToHost));
    r.pool = pool;
    for (size_t i = 0; i < hpool.size(); ++i)
    {
        r.pool.host[i] = h2f(hpool[i]);
    }
    for (void* ptr :
        {(void*) dq, (void*) drel, (void*) dsink, (void*) dgate, (void*) dk, (void*) dv, (void*) dqs, (void*) drs,
            (void*) dproj, (void*) dpool, (void*) doff, (void*) dlen, (void*) dpast, (void*) dtab, (void*) dout})
    {
        CUDA_CHECK(cudaFree(ptr));
    }
    return r;
}

//! Writes the expected appended K/V rows of the real tokens into a copy of the pool.
Pool expectedPool(Case const& c, Data const& d, Pool pool)
{
    for (int32_t b = 0; b < c.batch; ++b)
    {
        for (int32_t i = 0; i < d.lengths[b]; ++i)
        {
            int32_t const t = d.offsets[b] + i, pos = d.past[b] + i;
            int32_t const kPageIdx = d.table[(size_t(b) * 2) * c.maxPages + pos / kPage];
            int32_t const vPageIdx = d.table[(size_t(b) * 2 + 1) * c.maxPages + pos / kPage];
            if (kPageIdx < 0 || vPageIdx < 0)
            {
                continue; // unmapped logical page: the kernel writes nothing
            }
            for (int32_t hd = 0; hd < c.nkv; ++hd)
            {
                for (int32_t x = 0; x < kHd; ++x)
                {
                    pool.at(kPageIdx, pos % kPage, hd, x) = d.k[(size_t(t) * c.nkv + hd) * kHd + x];
                    pool.at(vPageIdx, pos % kPage, hd, x) = d.v[(size_t(t) * c.nkv + hd) * kHd + x];
                }
            }
        }
    }
    return pool;
}

//! Checks the kernel output for all real tokens and the pool contents (appended rows and untouched sentinels).
void expectMatches(Case const& c, Data const& d, Pool const& before, double tol = 3e-3)
{
    Pool expected = expectedPool(c, d, before);
    Result r = run(c, d, before);
    ASSERT_EQ(r.pool.host.size(), expected.host.size());
    size_t pooledMismatch = 0;
    for (size_t i = 0; i < expected.host.size(); ++i)
    {
        pooledMismatch += r.pool.host[i] != expected.host[i];
    }
    EXPECT_EQ(pooledMismatch, 0U) << "KV pool differs: appended rows wrong or unrelated pages written";
    Pool pool = expected;
    for (int32_t b = 0; b < c.batch; ++b)
    {
        for (int32_t slot = d.offsets[b]; slot < d.offsets[b + 1]; ++slot)
        {
            bool const real = slot - d.offsets[b] < d.lengths[b];
            if (!real)
            {
                for (int32_t i = 0; i < c.nh * kHd; ++i)
                {
                    ASSERT_EQ(r.out[size_t(slot) * c.nh * kHd + i], 0.0F) << "padding slot " << slot;
                }
                continue;
            }
            std::vector<double> ref;
            reference(c, d, pool, slot, b, ref);
            double maxErr = 0, refNorm = 0, errNorm = 0;
            for (int32_t i = 0; i < c.nh * kHd; ++i)
            {
                double const got = r.out[size_t(slot) * c.nh * kHd + i];
                ASSERT_TRUE(std::isfinite(got)) << "slot " << slot << " elem " << i;
                maxErr = std::max(maxErr, std::fabs(got - ref[i]));
                errNorm += (got - ref[i]) * (got - ref[i]);
                refNorm += ref[i] * ref[i];
            }
            EXPECT_LT(std::sqrt(errNorm / std::max(refNorm, 1e-12)), tol) << "slot " << slot << " relative error";
            EXPECT_LT(maxErr, 8e-3 + 0.01 * std::sqrt(refNorm / double(c.nh * kHd))) << "slot " << slot;
        }
    }
}

//! Fills the pool with the cached (past) rows of every sequence, so decode-style steps see real history.
Pool poolWithHistory(Case const& c, Data& d, std::vector<int32_t> const& past, uint32_t seed)
{
    Pool pool = newPool(c);
    std::mt19937 gen(seed);
    for (int32_t b = 0; b < c.batch; ++b)
    {
        for (int32_t pos = 0; pos < past[b]; ++pos)
        {
            int32_t const kPageIdx = d.table[(size_t(b) * 2) * c.maxPages + pos / kPage];
            int32_t const vPageIdx = d.table[(size_t(b) * 2 + 1) * c.maxPages + pos / kPage];
            auto kk = normal(size_t(c.nkv) * kHd, 1.5F, gen), vv = normal(size_t(c.nkv) * kHd, 1.0F, gen);
            if (kPageIdx < 0 || vPageIdx < 0)
            {
                continue;
            }
            for (int32_t hd = 0; hd < c.nkv; ++hd)
            {
                for (int32_t x = 0; x < kHd; ++x)
                {
                    pool.at(kPageIdx, pos % kPage, hd, x) = kk[size_t(hd) * kHd + x];
                    pool.at(vPageIdx, pos % kPage, hd, x) = vv[size_t(hd) * kHd + x];
                }
            }
        }
    }
    return pool;
}
} // namespace

TEST(Inkling2Attention, PrefillFullAttentionOnInklingHeadCounts)
{
    Case c;
    c.nh = 24;
    c.nkv = 8;
    c.maxPages = 3;
    c.numPages = 6;
    c.extent = 128;
    Data d = makeData(c, {0}, {200}, 0, 1);
    expectMatches(c, d, newPool(c));
}

TEST(Inkling2Attention, SlidingWindowAndExtentBoundaries)
{
    // window and extent chosen so the relative bias stops exactly at, one before and one after the window edge
    for (auto [window, extent] :
        {std::pair<int32_t, int32_t>{20, 20}, {20, 19}, {20, 21}, {1, 8}, {130, 64}, {128, 128}})
    {
        Case c;
        c.window = window;
        c.extent = extent;
        Data d = makeData(c, {0}, {260}, 0, 2);
        expectMatches(c, d, newPool(c));
    }
}

TEST(Inkling2Attention, DecodeAtPageBoundariesWithHistory)
{
    for (int32_t pos : {0, 1, 127, 128, 129, 255, 256, 383})
    {
        Case c;
        c.numPages = 8;
        Data d = makeData(c, {pos}, {1}, 0, 3);
        Pool pool = poolWithHistory(c, d, {pos}, 4);
        expectMatches(c, d, pool);
    }
}

TEST(Inkling2Attention, MultiRowStepsAndWindowSpanningPages)
{
    Case c;
    c.window = 150;
    Data d = makeData(c, {100}, {40}, 0, 5);
    Pool pool = poolWithHistory(c, d, {100}, 6);
    expectMatches(c, d, pool);
}

TEST(Inkling2Attention, BatchWithDifferentPastLengthsAndEntryPadding)
{
    Case c;
    c.batch = 3;
    c.maxPages = 3;
    c.numPages = 12;
    std::vector<int32_t> past{130, 0, 50}, counts{6, 2, 0};
    Data d = makeData(c, past, counts, 8, 7);
    Pool pool = poolWithHistory(c, d, past, 8);
    expectMatches(c, d, pool);
}

TEST(Inkling2Attention, ZeroTokensIsANoOp)
{
    Case c;
    Data d = makeData(c, {5}, {0}, 0, 9);
    Pool pool = poolWithHistory(c, d, {5}, 10);
    Result r = run(c, d, pool);
    EXPECT_TRUE(r.out.empty());
    for (size_t i = 0; i < pool.host.size(); ++i)
    {
        ASSERT_EQ(r.pool.host[i], pool.host[i]);
    }
}

TEST(Inkling2Attention, LastSlotOfTheLastPage)
{
    Case c;
    c.maxPages = 2;
    c.numPages = 4;
    Data d = makeData(c, {255}, {1}, 0, 11);
    Pool pool = poolWithHistory(c, d, {255}, 12);
    expectMatches(c, d, pool);
}

TEST(Inkling2Attention, QueryAndRelativeClipsAreActive)
{
    Case c;
    c.qClip = 2.0F;
    c.relClip = 0.3F;
    Data d = makeData(c, {0}, {120}, 0, 13, 3.0F);
    expectMatches(c, d, newPool(c));
}

TEST(Inkling2Attention, LogLengthScalingGrowsWithPosition)
{
    Case c;
    c.alpha = 0.7F;
    c.nFloor = 2.0F;
    Data d = makeData(c, {0}, {200}, 0, 14);
    expectMatches(c, d, newPool(c));
    c.window = 100; // sliding layers never scale
    expectMatches(c, d, newPool(c));
}

TEST(Inkling2Attention, LargeKeysKeepTheSoftmaxStable)
{
    Case c;
    Data d = makeData(c, {0}, {64}, 0, 15, 0.8F, 40.0F); // scores of order 100
    expectMatches(c, d, newPool(c), 5e-3);
}

TEST(Inkling2Attention, ExtremeSinkAndGateValues)
{
    Case c;
    Data d = makeData(c, {0}, {32}, 0, 16);
    for (size_t i = 0; i < d.sink.size(); ++i)
    {
        d.sink[i] = (i % 3 == 0) ? 60.0F : (i % 3 == 1) ? -60.0F : 0.0F; // sink dominates / vanishes / neutral
    }
    for (size_t i = 0; i < d.gate.size(); ++i)
    {
        d.gate[i] = (i % 4 == 0) ? 30.0F : (i % 4 == 1) ? -30.0F : 0.5F;
    }
    expectMatches(c, d, newPool(c));
}

TEST(Inkling2Attention, ZeroQueryUsesTheClipEpsilon)
{
    Case c;
    Data d = makeData(c, {0}, {16}, 0, 17);
    std::fill(d.q.begin(), d.q.end(), 0.0F);
    std::fill(d.rel.begin(), d.rel.end(), 0.0F);
    expectMatches(c, d, newPool(c));
}

TEST(Inkling2Attention, NaNInOneTokenStaysInThatToken)
{
    Case c;
    Data d = makeData(c, {0}, {8}, 0, 18);
    int32_t const bad = 3;
    d.q[size_t(bad) * c.nh * kHd] = std::numeric_limits<float>::quiet_NaN();
    Result r = run(c, d, newPool(c));
    for (int32_t t = 0; t < 8; ++t)
    {
        bool anyNaN = false, anyInf = false;
        for (int32_t i = 0; i < c.nh * kHd; ++i)
        {
            float const v = r.out[size_t(t) * c.nh * kHd + i];
            anyNaN |= std::isnan(v);
            anyInf |= std::isinf(v);
        }
        EXPECT_FALSE(anyInf);
        // later tokens only read the earlier keys and values, which are finite: they must not be affected
        if (t != bad)
        {
            EXPECT_FALSE(anyNaN) << "token " << t;
        }
        else
        {
            EXPECT_TRUE(anyNaN);
        }
    }
}

//! Marks the logical pages [0, firstMapped) of every sequence as unmapped (-1), the way bounded window storage does.
void unmapLeadingPages(Case const& c, Data& d, int32_t firstMapped)
{
    for (int32_t b = 0; b < c.batch; ++b)
    {
        for (int32_t p = 0; p < firstMapped; ++p)
        {
            d.table[(size_t(b) * 2 + 0) * c.maxPages + p] = -1;
            d.table[(size_t(b) * 2 + 1) * c.maxPages + p] = -1;
        }
    }
}

TEST(Inkling2Attention, UnmappedPagesOutsideTheWindowAreNeverRead)
{
    // window 100 at position 600: the live keys sit in pages 3 and 4, pages 0-2 are unmapped (-1)
    Case c;
    c.window = 100;
    c.maxPages = 8;
    c.numPages = 20;
    Data d = makeData(c, {600}, {1}, 0, 31);
    unmapLeadingPages(c, d, 3);
    Pool pool = poolWithHistory(c, d, {600}, 32);
    expectMatches(c, d, pool);
}

TEST(Inkling2Attention, WindowPrefillChunkOverUnmappedHistory)
{
    // a 70-token chunk after 400 cached tokens; window 64 reaches back into page 2 only
    Case c;
    c.window = 64;
    c.maxPages = 8;
    c.numPages = 20;
    Data d = makeData(c, {400}, {70}, 0, 33);
    unmapLeadingPages(c, d, 2);
    Pool pool = poolWithHistory(c, d, {400}, 34);
    expectMatches(c, d, pool);
}

TEST(Inkling2Attention, FullyUnmappedTableWritesNothingAndStaysFinite)
{
    // graph-capture warmup: the sparse table holds -1 everywhere
    Case c;
    c.window = 64;
    c.maxPages = 4;
    c.numPages = 6;
    Data d = makeData(c, {0}, {3}, 0, 35);
    unmapLeadingPages(c, d, c.maxPages);
    Pool before = newPool(c);
    Result r = run(c, d, before);
    for (size_t i = 0; i < before.host.size(); ++i)
    {
        ASSERT_EQ(r.pool.host[i], kSentinel) << "pool element " << i << " was written through an unmapped page";
    }
    for (float v : r.out)
    {
        ASSERT_TRUE(std::isfinite(v));
    }
}

TEST(Inkling2Attention, PartiallyUnmappedAppendSkipsOnlyTheUnmappedRows)
{
    // a decode token whose own page is unmapped is not written; its neighbours in mapped pages are
    Case c;
    c.window = 0;
    c.maxPages = 4;
    c.numPages = 8;
    Data d = makeData(c, {0}, {2}, 0, 37);
    unmapLeadingPages(c, d, 1); // page 0 unmapped, both tokens live in page 0
    Pool before = newPool(c);
    Result r = run(c, d, before);
    for (size_t i = 0; i < before.host.size(); ++i)
    {
        ASSERT_EQ(r.pool.host[i], kSentinel);
    }
}
