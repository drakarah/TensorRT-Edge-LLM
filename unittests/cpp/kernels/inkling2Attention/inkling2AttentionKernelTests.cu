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
#include <cuda_fp8.h>
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
    std::vector<int8_t> int8Bytes; // INT8 runs: the raw pool rows of 132 bytes
};

//! Runs the kernel for one step, updating the host pool image with the appended rows first (for the reference).
//! INT8 KV row image: 128 values, the FP16 scale (absmax / 127, 1 for a zero row) and 2 pad bytes.
constexpr int32_t kInt8Row = kHd + 4;

std::vector<int8_t> int8PoolImage(std::vector<float> const& host)
{
    size_t const rows = host.size() / kHd;
    std::vector<int8_t> bytes(rows * kInt8Row, 0);
    for (size_t r = 0; r < rows; ++r)
    {
        float amax = 0.F;
        for (int32_t x = 0; x < kHd; ++x)
        {
            amax = std::max(amax, std::fabs(host[r * kHd + x]));
        }
        half const sh = __float2half(amax > 0.F ? amax / 127.F : 1.F);
        float const inv = 1.F / h2f(sh);
        for (int32_t x = 0; x < kHd; ++x)
        {
            int32_t const q = std::max(-127, std::min(127, static_cast<int32_t>(std::lrintf(host[r * kHd + x] * inv))));
            bytes[r * kInt8Row + x] = static_cast<int8_t>(q);
        }
        uint16_t const bits = __half_as_ushort(sh);
        bytes[r * kInt8Row + kHd] = static_cast<int8_t>(bits & 0xFF);
        bytes[r * kInt8Row + kHd + 1] = static_cast<int8_t>(bits >> 8);
    }
    return bytes;
}

std::vector<float> int8PoolDequant(std::vector<int8_t> const& bytes, size_t floats)
{
    size_t const rows = floats / kHd;
    std::vector<float> out(floats);
    for (size_t r = 0; r < rows; ++r)
    {
        uint16_t const bits = static_cast<uint16_t>(static_cast<uint8_t>(bytes[r * kInt8Row + kHd]) | (static_cast<uint8_t>(bytes[r * kInt8Row + kHd + 1]) << 8));
        float const scale = h2f(__ushort_as_half(bits));
        for (int32_t x = 0; x < kHd; ++x)
        {
            out[r * kHd + x] = static_cast<float>(bytes[r * kInt8Row + x]) * scale;
        }
    }
    return out;
}

Result run(Case const& c, Data const& d, Pool pool, bool fp8 = false, bool int8 = false)
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
    // FP8 pools hold one byte per element (E4M3); the host image is rounded through it.
    std::vector<uint8_t> poolBytes;
    half* dpool = nullptr;
    if (int8)
    {
        dpool = reinterpret_cast<half*>(upload(int8PoolImage(pool.host)));
    }
    else if (fp8)
    {
        poolBytes.resize(pool.host.size());
        for (size_t i = 0; i < pool.host.size(); ++i)
        {
            poolBytes[i] = __nv_fp8_e4m3(pool.host[i]).__x;
        }
        dpool = reinterpret_cast<half*>(upload(poolBytes));
    }
    else
    {
        dpool = upload(toHalf(pool.host));
    }
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
    // INKLING2_LEGACY=1 runs the simple kernel (the reference implementation), otherwise the tiled kernels
    bool const legacy = std::getenv("INKLING2_LEGACY") != nullptr;
    size_t const wsBytes = legacy ? 0 : inkling2AttentionWorkspaceBytes(nTok, c.nh);
    void* dws = nullptr;
    if (wsBytes > 0)
    {
        CUDA_CHECK(cudaMalloc(&dws, wsBytes));
        CUDA_CHECK(cudaMemset(dws, 0xFF, wsBytes)); // garbage the kernels must fully overwrite where they read
    }
    Inkling2AttentionParams p{};
    p.numTokens = nTok;
    p.batchSize = c.batch;
    p.numQHeads = c.nh;
    p.numKVHeads = c.nkv;
    p.relExtent = c.extent;
    p.window = c.window;
    p.maxPagesPerSeq = c.maxPages;
    p.kvFp8 = fp8 ? 1 : 0;
    p.kvInt8 = int8 ? 1 : 0;
    p.scaling = c.scaling;
    p.logAlpha = c.alpha;
    p.logNFloor = c.nFloor;
    p.qClip = c.qClip;
    p.relClip = c.relClip;
    launchInkling2Attention(
        p, dq, drel, dsink, dgate, dqs, drs, dproj, dk, dv, dpool, doff, dlen, dpast, dtab, dout, dws, wsBytes, nullptr);
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
    r.pool = pool;
    if (int8)
    {
        std::vector<int8_t> bytes(pool.host.size() / kHd * kInt8Row);
        CUDA_CHECK(cudaMemcpy(bytes.data(), dpool, bytes.size(), cudaMemcpyDeviceToHost));
        r.pool.host = int8PoolDequant(bytes, pool.host.size());
        // the pad bytes of every row must stay zero where the kernel wrote the row
        r.int8Bytes = bytes;
    }
    else if (fp8)
    {
        std::vector<uint8_t> bytes(pool.host.size());
        CUDA_CHECK(cudaMemcpy(bytes.data(), dpool, bytes.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < bytes.size(); ++i)
        {
            __nv_fp8_e4m3 v;
            v.__x = bytes[i];
            r.pool.host[i] = static_cast<float>(v);
        }
    }
    else
    {
        std::vector<half> hpool(pool.host.size());
        CUDA_CHECK(cudaMemcpy(hpool.data(), dpool, hpool.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < hpool.size(); ++i)
        {
            r.pool.host[i] = h2f(hpool[i]);
        }
    }
    for (void* ptr :
        {(void*) dq, (void*) drel, (void*) dsink, (void*) dgate, (void*) dk, (void*) dv, (void*) dqs, (void*) drs,
            (void*) dproj, (void*) dpool, (void*) doff, (void*) dlen, (void*) dpast, (void*) dtab, (void*) dout, dws})
    {
        if (ptr != nullptr)
        {
            CUDA_CHECK(cudaFree(ptr));
        }
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
void expectMatches(Case const& c, Data const& d, Pool const& before, double tol = 3e-3, double maxErrScale = 0.01)
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
            EXPECT_LT(maxErr, 8e-3 + maxErrScale * std::sqrt(refNorm / double(c.nh * kHd))) << "slot " << slot;
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
    Data d = makeData(c, {0}, {64}, 0, 15, 0.8F, 40.0F); // scores of order 100: the tensor-core path rounds q to half (relative 5e-4 of a score of 100)
    expectMatches(c, d, newPool(c), 5e-3, 0.02);
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

//! Rounds every cached and new K/V value to the E4M3 grid, so an FP16 pool and an FP8 pool hold identical numbers.
void roundKvToFp8(Data& d, Pool& pool)
{
    auto rnd = [](float& x) { x = static_cast<float>(__nv_fp8_e4m3(x)); };
    for (auto& x : d.k)
    {
        rnd(x);
    }
    for (auto& x : d.v)
    {
        rnd(x);
    }
    for (auto& x : pool.host)
    {
        if (x != kSentinel)
        {
            rnd(x);
        }
    }
}

void expectFp8MatchesFp16(Case const& c, Data& d, Pool pool)
{
    roundKvToFp8(d, pool);
    Result ref = run(c, d, pool, false);
    Result got = run(c, d, pool, true);
    ASSERT_EQ(ref.out.size(), got.out.size());
    for (size_t i = 0; i < ref.out.size(); ++i)
    {
        ASSERT_TRUE(std::isfinite(got.out[i]));
        ASSERT_NEAR(got.out[i], ref.out[i], 2e-3F + 2e-3F * std::fabs(ref.out[i])) << "output element " << i;
    }
    // the appended rows are the FP8 image of the (already grid-aligned) inputs; untouched rows keep the sentinel
    // (the sentinel 77 is itself not exactly on the grid; compare the FP8 round-trip)
    for (size_t i = 0; i < ref.pool.host.size(); ++i)
    {
        float const expected = ref.pool.host[i] == kSentinel ? static_cast<float>(__nv_fp8_e4m3(kSentinel)) : ref.pool.host[i];
        ASSERT_EQ(got.pool.host[i], expected) << "pool element " << i;
    }
}

TEST(Inkling2Attention, Fp8KvPoolMatchesFp16OnTheSameGridValuesPrefill)
{
    Case c;
    c.nh = 24;
    c.nkv = 8;
    c.maxPages = 3;
    c.numPages = 8;
    Data d = makeData(c, {0}, {150}, 0, 41);
    expectFp8MatchesFp16(c, d, newPool(c));
}

TEST(Inkling2Attention, Fp8KvPoolMatchesFp16DecodeWithHistoryAndWindow)
{
    Case c;
    c.window = 100;
    c.maxPages = 8;
    c.numPages = 20;
    Data d = makeData(c, {600}, {1}, 0, 43);
    unmapLeadingPages(c, d, 3);
    Pool pool = poolWithHistory(c, d, {600}, 44);
    expectFp8MatchesFp16(c, d, pool);
}

TEST(Inkling2Attention, Fp8KvSaturatesInsteadOfOverflowing)
{
    // |value| far above the E4M3 range (448) is clamped, never NaN/Inf
    Case c;
    c.maxPages = 2;
    c.numPages = 4;
    Data d = makeData(c, {0}, {2}, 0, 45);
    for (auto& x : d.k)
    {
        x = 5000.0F;
    }
    for (auto& x : d.v)
    {
        x = -5000.0F;
    }
    Result got = run(c, d, newPool(c), true);
    for (float v : got.out)
    {
        ASSERT_TRUE(std::isfinite(v));
    }
    bool sawMax = false;
    for (float v : got.pool.host)
    {
        sawMax = sawMax || std::fabs(v) == 448.0F;
        ASSERT_TRUE(std::isfinite(v));
    }
    EXPECT_TRUE(sawMax);
}


//! Rounds every cached and new K/V row to the INT8 row grid (per-row FP16 scale), so an FP16 pool and an INT8 pool hold the same numbers (up to the FP16
//! rounding of the scale, which re-quantising a grid value reproduces exactly).
void roundKvToInt8(Data& d, Pool& pool, Case const& c)
{
    auto roundRows = [](std::vector<float>& v)
    {
        for (size_t r = 0; r < v.size() / kHd; ++r)
        {
            float amax = 0.F;
            for (int32_t x = 0; x < kHd; ++x)
            {
                amax = std::max(amax, std::fabs(v[r * kHd + x]));
            }
            float const scale = h2f(__float2half(amax > 0.F ? amax / 127.F : 1.F));
            for (int32_t x = 0; x < kHd; ++x)
            {
                int32_t const q = std::max(-127, std::min(127, static_cast<int32_t>(std::lrintf(v[r * kHd + x] / scale))));
                v[r * kHd + x] = h2f(__float2half(static_cast<float>(q) * scale));
            }
        }
    };
    (void) c;
    roundRows(d.k);
    roundRows(d.v);
    roundRows(pool.host);
}

void expectInt8MatchesFp16(Case const& c, Data& d, Pool pool)
{
    roundKvToInt8(d, pool, c);
    Result ref = run(c, d, pool, false, false);
    Result got = run(c, d, pool, false, true);
    ASSERT_EQ(ref.out.size(), got.out.size());
    for (size_t i = 0; i < ref.out.size(); ++i)
    {
        ASSERT_TRUE(std::isfinite(got.out[i]));
        ASSERT_NEAR(got.out[i], ref.out[i], 3e-3F + 3e-3F * std::fabs(ref.out[i])) << "output element " << i;
    }
    // the appended rows are the INT8 image of the grid-aligned inputs (within the FP16 rounding of the scale); untouched rows keep their image
    for (size_t i = 0; i < ref.pool.host.size(); ++i)
    {
        ASSERT_NEAR(got.pool.host[i], ref.pool.host[i], 2e-3F * std::fabs(ref.pool.host[i]) + 1e-3F) << "pool element " << i;
    }
    // every row written by the kernel has zero pad bytes behind its scale
    for (size_t r = 0; r < got.int8Bytes.size() / kInt8Row; ++r)
    {
        ASSERT_EQ(got.int8Bytes[r * kInt8Row + kHd + 2], 0) << "pad byte of row " << r;
        ASSERT_EQ(got.int8Bytes[r * kInt8Row + kHd + 3], 0) << "pad byte of row " << r;
    }
}

TEST(Inkling2Attention, Int8KvPoolMatchesFp16OnTheSameGridValuesPrefill)
{
    Case c;
    c.nh = 24;
    c.nkv = 8;
    c.maxPages = 3;
    c.numPages = 8;
    Data d = makeData(c, {0}, {150}, 0, 51);
    expectInt8MatchesFp16(c, d, newPool(c));
}

TEST(Inkling2Attention, Int8KvPoolMatchesFp16DecodeWithHistoryAndWindow)
{
    Case c;
    c.window = 100;
    c.maxPages = 8;
    c.numPages = 20;
    Data d = makeData(c, {600}, {1}, 0, 53);
    unmapLeadingPages(c, d, 3);
    Pool pool = poolWithHistory(c, d, {600}, 54);
    expectInt8MatchesFp16(c, d, pool);
}

TEST(Inkling2Attention, Int8KvPoolMultiRowStepsFullAttentionAndGroupedHeads)
{
    Case c;
    c.nh = 24;
    c.nkv = 8;
    c.maxPages = 6;
    c.numPages = 16;
    Data d = makeData(c, {300}, {4}, 0, 55);
    Pool pool = poolWithHistory(c, d, {300}, 56);
    expectInt8MatchesFp16(c, d, pool);
}

TEST(Inkling2Attention, Int8KvZeroRowsAndHugeValuesStayFinite)
{
    Case c;
    c.maxPages = 2;
    c.numPages = 4;
    Data d = makeData(c, {0}, {3}, 0, 57);
    for (size_t i = 0; i < d.k.size(); ++i)
    {
        d.k[i] = (i / kHd) % 3 == 0 ? 0.F : 3000.F * (static_cast<int32_t>(i % 7) - 3);
        d.v[i] = (i / kHd) % 2 == 0 ? 0.F : -2000.F;
    }
    Result got = run(c, d, newPool(c), false, true);
    for (float v : got.out)
    {
        ASSERT_TRUE(std::isfinite(v));
    }
    for (float v : got.pool.host)
    {
        ASSERT_TRUE(std::isfinite(v));
    }
}

TEST(Inkling2Attention, Int8KvErrorAgainstUnquantisedFp16IsSmall)
{
    // quantisation noise of INT8 rows on realistic (normal) K/V: the attention output stays within about 1.5% relative L2 of the FP16 pool
    Case c;
    c.nh = 24;
    c.nkv = 8;
    c.maxPages = 6;
    c.numPages = 16;
    Data d = makeData(c, {400}, {1}, 0, 59);
    Pool pool = poolWithHistory(c, d, {400}, 60);
    Result ref = run(c, d, pool, false, false);
    Result got = run(c, d, pool, false, true);
    double err = 0, norm = 0;
    for (size_t i = 0; i < ref.out.size(); ++i)
    {
        err += double(got.out[i] - ref.out[i]) * (got.out[i] - ref.out[i]);
        norm += double(ref.out[i]) * ref.out[i];
    }
    EXPECT_LT(std::sqrt(err / std::max(norm, 1e-12)), 0.015);
}

TEST(Inkling2Attention, GroupSizesAndKeySplits)
{
    // GQA group 1, 2, 3, 4, 8 (query tile 24, 12, 8, 6, 3) with a decode step deep enough for several key splits
    for (auto [nh, nkv] : {std::pair<int32_t, int32_t>{4, 4}, {4, 2}, {6, 2}, {8, 2}, {8, 1}})
    {
        Case c;
        c.nh = nh;
        c.nkv = nkv;
        c.maxPages = 30;
        c.numPages = 60;
        Data d = makeData(c, {3000}, {1}, 0, 51 + nh);
        Pool pool = poolWithHistory(c, d, {3000}, 52);
        expectMatches(c, d, pool);
    }
}

TEST(Inkling2Attention, ChunkSpanningSeveralQueryTilesAndKeyTiles)
{
    // 70 new tokens (several 8-token tiles, the last one partial) over 200 cached tokens, full attention and window
    for (int32_t window : {0, 90})
    {
        Case c;
        c.window = window;
        c.maxPages = 4;
        c.numPages = 10;
        Data d = makeData(c, {200}, {70}, 0, 53);
        Pool pool = poolWithHistory(c, d, {200}, 54);
        expectMatches(c, d, pool);
    }
}

TEST(Inkling2Attention, TiledKernelAgreesWithTheLegacyKernel)
{
    // same data through both kernels (the harness picks by INKLING2_LEGACY; compare via two runs of the helper)
    Case c;
    c.nh = 24;
    c.nkv = 8;
    c.window = 100;
    c.maxPages = 8;
    c.numPages = 20;
    Data d = makeData(c, {600}, {5}, 0, 55);
    unmapLeadingPages(c, d, 3);
    Pool pool = poolWithHistory(c, d, {600}, 56);
    setenv("INKLING2_LEGACY", "1", 1);
    Result legacy = run(c, d, pool);
    unsetenv("INKLING2_LEGACY");
    Result tiled = run(c, d, pool);
    ASSERT_EQ(legacy.out.size(), tiled.out.size());
    for (size_t i = 0; i < legacy.out.size(); ++i)
    {
        ASSERT_NEAR(legacy.out[i], tiled.out[i], 2e-3F + 2e-3F * std::fabs(legacy.out[i])) << "element " << i;
    }
}

//! Timing of the legacy and the tiled kernels at long context (numbers only, nothing is asserted).
TEST(Inkling2AttentionBench, LongContextTiming)
{
    struct Shape
    {
        char const* name;
        int32_t tokens, past, window;
    };
    for (Shape const& shape : {Shape{"decode  96K full", 1, 96000, 0}, Shape{"decode  96K window1024", 1, 96000, 1024},
             Shape{"prefill 1024 @50K full", 1024, 50000, 0}, Shape{"prefill 1024 @50K window1024", 1024, 50000, 1024}})
    {
        int32_t const nh = 24, nkv = 8, maxPages = (shape.past + shape.tokens + 127) / 128 + 1;
        int32_t const numPages = maxPages;
        size_t const poolElems = size_t(2) * numPages * kPage * nkv * kHd;
        half* pool{};
        CUDA_CHECK(cudaMalloc(&pool, poolElems * sizeof(half)));
        CUDA_CHECK(cudaMemset(pool, 0x3c, poolElems * sizeof(half)));
        std::vector<int32_t> table(2 * maxPages);
        for (int32_t i = 0; i < maxPages; ++i)
        {
            table[i] = i;
            table[maxPages + i] = i + numPages;
        }
        std::vector<int32_t> offsets{0, shape.tokens}, lengths{shape.tokens}, past{shape.past};
        auto* dtab = upload(table);
        auto* doff = upload(offsets);
        auto* dlen = upload(lengths);
        auto* dpast = upload(past);
        size_t const nTok = shape.tokens;
        half *dq, *drel, *dsink, *dgate, *dk, *dv, *dout;
        CUDA_CHECK(cudaMalloc(&dq, nTok * nh * kHd * 2));
        CUDA_CHECK(cudaMalloc(&dgate, nTok * nh * kHd * 2));
        CUDA_CHECK(cudaMalloc(&dout, nTok * nh * kHd * 2));
        CUDA_CHECK(cudaMalloc(&drel, nTok * nh * kRd * 2));
        CUDA_CHECK(cudaMalloc(&dsink, nTok * nh * 2));
        CUDA_CHECK(cudaMalloc(&dk, nTok * nkv * kHd * 2));
        CUDA_CHECK(cudaMalloc(&dv, nTok * nkv * kHd * 2));
        for (half* x : {dq, dgate, drel, dsink, dk, dv})
        {
            CUDA_CHECK(cudaMemset(x, 0x2e, nTok * 2));
        }
        std::vector<float> ones(size_t(nh) * kHd, 1.0F), proj(size_t(1024) * kRd, 0.1F);
        float* dqs = upload(ones);
        float* drs = upload(ones);
        float* dproj = upload(proj);
        Inkling2AttentionParams p{};
        p.numTokens = shape.tokens;
        p.batchSize = 1;
        p.numQHeads = nh;
        p.numKVHeads = nkv;
        p.relExtent = 1024;
        p.window = shape.window;
        p.maxPagesPerSeq = maxPages;
        p.scaling = 1.0F / std::sqrt(128.0F);
        p.logAlpha = 0.1F;
        p.logNFloor = 16.0F;
        p.qClip = 22.6F;
        p.relClip = 8.0F;
        size_t const wsBytes = inkling2AttentionWorkspaceBytes(shape.tokens, nh);
        void* dws{};
        CUDA_CHECK(cudaMalloc(&dws, wsBytes));
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
        for (bool tiled : {false, true})
        {
            auto launch = [&]()
            {
                launchInkling2Attention(p, dq, drel, dsink, dgate, dqs, drs, dproj, dk, dv, pool, doff, dlen, dpast,
                    dtab, dout, tiled ? dws : nullptr, tiled ? wsBytes : 0, nullptr);
            };
            launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            int32_t const iters = shape.tokens > 1 ? 3 : 20;
            CUDA_CHECK(cudaEventRecord(a));
            for (int32_t i = 0; i < iters; ++i)
            {
                launch();
            }
            CUDA_CHECK(cudaEventRecord(b));
            CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            printf("BENCH %-30s %-6s %9.3f ms\n", shape.name, tiled ? "tiled" : "legacy", ms / iters);
        }
        for (void* ptr : {(void*) pool, (void*) dtab, (void*) doff, (void*) dlen, (void*) dpast, (void*) dq,
                 (void*) dgate, (void*) dout, (void*) drel, (void*) dsink, (void*) dk, (void*) dv, (void*) dqs,
                 (void*) drs, (void*) dproj, dws})
        {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
}
