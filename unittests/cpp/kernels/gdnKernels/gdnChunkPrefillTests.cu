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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "common/cudaUtils.h"
#include "kernels/gdnKernels/gdnChunkPrefill.h"

#ifdef CUTE_DSL_GDN_ENABLED
#include "kernels/gdnKernels/cuteDslGDNRunner.h"
#endif

using namespace trt_edgellm;

namespace
{

struct Problem
{
    int32_t n, seqLen, h, hv;
    std::vector<int32_t> lengths;
    std::vector<__half> q, k, v, a, b, dtBias;
    std::vector<float> aLog, pool;
    std::vector<int32_t> rows; // pool row per sequence
    int32_t poolRows;
};

Problem makeProblem(int32_t n, int32_t seqLen, int32_t h, int32_t hv, std::vector<int32_t> lengths, uint32_t seed)
{
    Problem p{n, seqLen, h, hv, std::move(lengths)};
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(-3.f, 1.f);
    auto fill = [&](std::vector<__half>& x, size_t count, float s)
    {
        x.resize(count);
        for (auto& e : x)
            e = __float2half(nd(rng) * s);
    };
    size_t const tok = static_cast<size_t>(n) * seqLen;
    fill(p.q, tok * h * 128, 1.f);
    fill(p.k, tok * h * 128, 1.f);
    fill(p.v, tok * hv * 128, 1.f);
    fill(p.a, tok * hv, 1.f);
    fill(p.b, tok * hv, 1.f);
    fill(p.dtBias, hv, 0.5f);
    p.aLog.resize(hv);
    for (auto& e : p.aLog)
        e = ud(rng);
    // Sequence i lives in pool row n - i; row 0 belongs to nobody (sentinel 7).
    p.poolRows = n + 1;
    p.rows.resize(n);
    p.pool.assign(static_cast<size_t>(p.poolRows) * hv * 128 * 128, 7.f);
    for (int32_t i = 0; i < n; ++i)
    {
        p.rows[i] = n - i;
        float* s = p.pool.data() + static_cast<size_t>(p.rows[i]) * hv * 128 * 128;
        for (size_t e = 0; e < static_cast<size_t>(hv) * 128 * 128; ++e)
            s[e] = nd(rng) * 0.05f;
    }
    return p;
}

// Token-by-token fp64 recurrence (the sequential kernel's math). Returns o [n, seqLen, hv, 128] and updates pool.
std::vector<double> reference(Problem const& p, std::vector<double>& pool)
{
    int32_t const D = 128;
    std::vector<double> o(static_cast<size_t>(p.n) * p.seqLen * p.hv * D, 0.0);
    pool.assign(p.pool.begin(), p.pool.end());
    for (int32_t i = 0; i < p.n; ++i)
    {
        for (int32_t j = 0; j < p.hv; ++j)
        {
            int32_t const iH = j / (p.hv / p.h);
            double* S = pool.data() + (static_cast<size_t>(p.rows[i]) * p.hv + j) * D * D;
            for (int32_t t = 0; t < p.lengths[i]; ++t)
            {
                size_t const qk = ((static_cast<size_t>(i) * p.seqLen + t) * p.h + iH) * D;
                size_t const vo = ((static_cast<size_t>(i) * p.seqLen + t) * p.hv + j) * D;
                size_t const ab = (static_cast<size_t>(i) * p.seqLen + t) * p.hv + j;
                double nq = 1e-6, nk = 1e-6;
                for (int32_t d = 0; d < D; ++d)
                {
                    nq += std::pow(__half2float(p.q[qk + d]), 2);
                    nk += std::pow(__half2float(p.k[qk + d]), 2);
                }
                std::vector<double> qe(D), ke(D);
                for (int32_t d = 0; d < D; ++d)
                {
                    qe[d] = __half2float(p.q[qk + d]) / std::sqrt(nq) / std::sqrt(128.0);
                    ke[d] = __half2float(p.k[qk + d]) / std::sqrt(nk);
                }
                double const x = __half2float(p.a[ab]) + __half2float(p.dtBias[j]);
                double const sp = x > 20 ? x : std::log1p(std::exp(x));
                double const g = std::exp(-std::exp(static_cast<double>(p.aLog[j])) * sp);
                double const beta = 1.0 / (1.0 + std::exp(-static_cast<double>(__half2float(p.b[ab]))));
                for (int32_t e = 0; e < D * D; ++e)
                    S[e] *= g;
                for (int32_t c = 0; c < D; ++c)
                {
                    double sk = 0;
                    for (int32_t r = 0; r < D; ++r)
                        sk += S[r * D + c] * ke[r];
                    double const nv = beta * (__half2float(p.v[vo + c]) - sk);
                    for (int32_t r = 0; r < D; ++r)
                        S[r * D + c] += ke[r] * nv;
                }
                for (int32_t c = 0; c < D; ++c)
                {
                    double acc = 0;
                    for (int32_t r = 0; r < D; ++r)
                        acc += S[r * D + c] * qe[r];
                    o[vo + c] = acc;
                }
            }
        }
    }
    return o;
}

template <typename T>
T* toDevice(std::vector<T> const& x)
{
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, x.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

struct DeviceProblem
{
    __half *q, *k, *v, *a, *b, *dt, *o;
    float *aLog, *pool;
    int32_t *rows, *lengths;
    void* ws;
    explicit DeviceProblem(Problem const& p)
    {
        q = toDevice(p.q), k = toDevice(p.k), v = toDevice(p.v), a = toDevice(p.a), b = toDevice(p.b);
        dt = toDevice(p.dtBias), aLog = toDevice(p.aLog), pool = toDevice(p.pool), rows = toDevice(p.rows);
        lengths = toDevice(p.lengths);
        CUDA_CHECK(cudaMalloc(&o, p.v.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemset(o, 0xff, p.v.size() * sizeof(__half))); // NaN: every position must be written
        CUDA_CHECK(cudaMalloc(&ws, kernel::gdnChunkPrefillWorkspaceBytes(p.n, p.h, p.hv)));
    }
    ~DeviceProblem()
    {
        for (void* x : {static_cast<void*>(q), static_cast<void*>(k), static_cast<void*>(v), static_cast<void*>(a),
                 static_cast<void*>(b), static_cast<void*>(dt), static_cast<void*>(o), static_cast<void*>(aLog),
                 static_cast<void*>(pool), static_cast<void*>(rows), static_cast<void*>(lengths), ws})
            cudaFree(x);
    }
    cudaError_t run(Problem const& p) const
    {
        return kernel::gdnChunkPrefill(q, k, v, a, b, aLog, dt, pool, rows, p.poolRows, lengths, o, p.n, p.seqLen, p.h,
            p.hv, ws, nullptr);
    }
};

void runChunkPrefillTest(int32_t n, int32_t seqLen, int32_t h, int32_t hv, std::vector<int32_t> lengths)
{
    Problem const p = makeProblem(n, seqLen, h, hv, lengths, 1234);
    DeviceProblem dp(p);
    ASSERT_EQ(dp.run(p), cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> poolRef;
    std::vector<double> const oRef = reference(p, poolRef);
    std::vector<__half> oGot(p.v.size());
    std::vector<float> poolGot(p.pool.size());
    CUDA_CHECK(cudaMemcpy(oGot.data(), dp.o, oGot.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(poolGot.data(), dp.pool, poolGot.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Relative to the largest reference magnitude: fp16 tensor-core operands give ~1e-3.
    double oMax = 0, oErr = 0, sMax = 0, sErr = 0;
    size_t nan = 0;
    for (size_t i = 0; i < oRef.size(); ++i)
    {
        double const got = __half2float(oGot[i]);
        nan += std::isnan(got) ? 1 : 0;
        oMax = std::max(oMax, std::abs(oRef[i]));
        oErr = std::max(oErr, std::abs(got - oRef[i]));
    }
    for (size_t i = 0; i < poolRef.size(); ++i)
    {
        sMax = std::max(sMax, std::abs(poolRef[i]));
        sErr = std::max(sErr, std::abs(poolGot[i] - poolRef[i]));
    }
    std::printf("chunk prefill n=%d T=%d h=%d hv=%d: rel err o %.2e state %.2e\n", n, seqLen, h, hv, oErr / oMax,
        sErr / sMax);
    EXPECT_EQ(nan, 0u) << "unwritten output positions";
    EXPECT_LT(oErr / oMax, 5e-3);
    EXPECT_LT(sErr / sMax, 5e-3);
    for (size_t e = 0; e < static_cast<size_t>(hv) * 128 * 128; ++e)
    {
        ASSERT_EQ(poolGot[e], 7.f) << "pool row 0 (no sequence) was written";
    }
}

} // namespace

TEST(GDNChunkPrefill, SingleShortSequence)
{
    runChunkPrefillTest(1, 50, 4, 8, {50});
}

TEST(GDNChunkPrefill, VariableLengthsQwenHeads)
{
    runChunkPrefillTest(2, 300, 16, 32, {300, 177});
}

TEST(GDNChunkPrefill, CrossesSegmentBoundary)
{
    runChunkPrefillTest(1, kernel::kGDN_CHUNK_SEGMENT + 100, 2, 4, {kernel::kGDN_CHUNK_SEGMENT + 100});
}

#ifdef CUTE_DSL_GDN_ENABLED
// Chunked vs the sequential CuTe DSL prefill at the Qwen3.5-4B shape. Run with
// --gtest_also_run_disabled_tests --gtest_filter='*ChunkPrefillBenchmark*'.
TEST(GDNChunkPrefill, DISABLED_ChunkPrefillBenchmark)
{
    for (int32_t const seqLen : {512, 2048, 8192})
    {
        Problem const p = makeProblem(1, seqLen, 16, 32, {seqLen}, 7);
        DeviceProblem dp(p);
        int32_t* ctx = dp.lengths;
        GDNParams params{};
        params.q = dp.q, params.k = dp.k, params.v = dp.v, params.a = dp.a, params.b = dp.b;
        params.A_log = dp.aLog, params.dt_bias = dp.dt, params.h0_source = dp.pool, params.o = dp.o;
        params.context_lengths = ctx, params.state_indices = dp.rows, params.state_pool_rows = p.poolRows;
        params.n = 1, params.seq_len = seqLen, params.h = 16, params.hv = 32, params.k_dim = 128, params.v_dim = 128;
        params.use_prefill = true;
        params.smVersion = getSMVersion();
        CuteDslGDNRunner runner;
        auto time = [&](auto&& fn)
        {
            for (int32_t i = 0; i < 3; ++i)
                fn();
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0);
            for (int32_t i = 0; i < 10; ++i)
                fn();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return ms / 10;
        };
        float const seq = time([&] { runner.run(params, nullptr); });
        float const chunk = time([&] { dp.run(p); });
        std::printf("T=%5d: sequential %.2f ms, chunked %.2f ms per layer call (%.2fx)\n", seqLen, seq, chunk,
            seq / chunk);
    }
}
#endif
