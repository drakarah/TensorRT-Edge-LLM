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

// Gated (SwiGLU) input kernels of the INT4 GEMM plugin:
//  * launchSiluMulGather (both kernel forms) against the FP64 value of SiLU(gate) * up with a column gather;
//  * launchQuantizeRowsInt8PermutedSiluMul must be bit-identical to launchSiluMulGather followed by
//    launchQuantizeRowsInt8Permuted (the two-kernel path it replaces), codes and row scales, with and without a permutation.

#include "kernels/int4W4A8/int4W4A8Quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace trt_edgellm;

namespace
{
template <typename T>
struct DeviceBuffer
{
    T* p{};
    size_t n{};
    explicit DeviceBuffer(size_t count)
        : n(count)
    {
        cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T));
    }
    ~DeviceBuffer()
    {
        cudaFree(p);
    }
    void upload(std::vector<T> const& host)
    {
        cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    std::vector<T> download() const
    {
        std::vector<T> host(n);
        cudaMemcpy(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return host;
    }
};

void runCase(int32_t M, int32_t K, bool withPerm, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.F, 2.5F);
    size_t const n = static_cast<size_t>(M) * K;
    std::vector<half> hg(n), hu(n);
    for (size_t i = 0; i < n; ++i)
    {
        hg[i] = __float2half(dist(rng));
        hu[i] = __float2half(dist(rng));
    }
    if (M > 1)
    {
        // An all-zero row (scale falls back to 1) and extreme gates (sigmoid saturates both ways).
        std::fill(hu.begin() + K, hu.begin() + 2 * static_cast<size_t>(K), __float2half(0.F));
        hg[0] = __float2half(60.F);
        hg[1] = __float2half(-60.F);
    }
    std::vector<int32_t> hperm(K);
    std::iota(hperm.begin(), hperm.end(), 0);
    std::shuffle(hperm.begin(), hperm.end(), rng);

    DeviceBuffer<half> g(n), u(n), prod(n);
    DeviceBuffer<int32_t> perm(K);
    DeviceBuffer<int8_t> q1(n), q2(n);
    DeviceBuffer<float> s1(M), s2(M);
    g.upload(hg);
    u.upload(hu);
    perm.upload(hperm);
    int32_t const* permPtr = withPerm ? perm.p : nullptr;

    kernel::launchSiluMulGather(g.p, u.p, permPtr, prod.p, M, K, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<half> const hprod = prod.download();
    double worst = 0.0;
    for (int32_t m = 0; m < M; ++m)
    {
        for (int32_t j = 0; j < K; ++j)
        {
            size_t const src = static_cast<size_t>(m) * K + (withPerm ? hperm[j] : j);
            double const gv = __half2float(hg[src]);
            double const want = gv / (1.0 + std::exp(-gv)) * static_cast<double>(__half2float(hu[src]));
            double const got = __half2float(hprod[static_cast<size_t>(m) * K + j]);
            double const err = std::fabs(got - want);
            worst = std::max(worst, err / (std::fabs(want) + 1e-4));
            // Three FP16 roundings (sigmoid, gate * sigmoid, product), each half a step of 2^-11 relative; the floor covers intermediates rounded in the FP16 subnormal range.
            ASSERT_LE(err, 1.5e-3 * std::fabs(want) + 2e-6) << "row " << m << " col " << j << " want " << want << " got " << got;
        }
    }

    if (K % 128 == 0 && K <= kernel::kInt4W4A8MaxK)
    {
        kernel::launchQuantizeRowsInt8Permuted(prod.p, q1.p, s1.p, M, K, nullptr);
        kernel::launchQuantizeRowsInt8PermutedSiluMul(g.p, u.p, permPtr, q2.p, s2.p, M, K, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<int8_t> const a = q1.download();
        std::vector<int8_t> const b = q2.download();
        std::vector<float> const sa = s1.download();
        std::vector<float> const sb = s2.download();
        ASSERT_EQ(std::memcmp(a.data(), b.data(), n), 0) << "INT8 codes differ, M " << M << " K " << K << " perm " << withPerm;
        ASSERT_EQ(std::memcmp(sa.data(), sb.data(), static_cast<size_t>(M) * sizeof(float)), 0) << "row scales differ";
    }
    std::printf("M %d K %d perm %d: product relative error %.2e against the FP64 value, fused quantiser identical\n", M, K,
        withPerm ? 1 : 0, worst);
}
} // namespace

TEST(Int4W4A8GatedInputTest, MatchesTwoKernelPath)
{
    for (int32_t K : {128, 5120, 17408, 20480})
    {
        for (int32_t M : {1, 2, 16, 65, 300})
        {
            runCase(M, K, false, 3U + static_cast<uint32_t>(M + K));
            runCase(M, K, true, 4U + static_cast<uint32_t>(M + K));
        }
    }
    runCase(7, 8, true, 9U);   // FP16 path only (K not a multiple of 128)
    runCase(5, 4104, true, 9U);
    runCase(3, 24576, true, 9U); // wider than the staged row: element-gather kernel with a permutation
}

TEST(Int4W4A8GatedInputTest, DISABLED_Timing)
{
    int32_t const K = 17408;
    for (int32_t M : {16, 1024})
    {
        size_t const n = static_cast<size_t>(M) * K;
        DeviceBuffer<half> g(n), u(n), prod(n);
        DeviceBuffer<int32_t> perm(K);
        DeviceBuffer<int8_t> q(n);
        DeviceBuffer<float> s(M);
        std::vector<int32_t> hperm(K);
        std::iota(hperm.begin(), hperm.end(), 0);
        std::mt19937 rng(1);
        std::shuffle(hperm.begin(), hperm.end(), rng);
        perm.upload(hperm);
        cudaMemset(g.p, 0x3C, n * 2);
        cudaMemset(u.p, 0x3C, n * 2);
        auto const time = [&](char const* what, auto&& fn) {
            for (int32_t pass = 0; pass < 2; ++pass)
            {
                cudaDeviceSynchronize();
                auto const t0 = std::chrono::steady_clock::now();
                for (int32_t i = 0; i < 100; ++i)
                {
                    fn();
                }
                cudaDeviceSynchronize();
                double const us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 100;
                if (pass == 1)
                {
                    std::printf("M %d K %d %s: %.1f us per call\n", M, K, what, us);
                }
            }
        };
        time("silu-mul-gather (FP16 path)", [&] { kernel::launchSiluMulGather(g.p, u.p, perm.p, prod.p, M, K, nullptr); });
        time("row quantiser of the product", [&] { kernel::launchQuantizeRowsInt8Permuted(prod.p, q.p, s.p, M, K, nullptr); });
        time("fused silu-mul quantiser", [&] {
            kernel::launchQuantizeRowsInt8PermutedSiluMul(g.p, u.p, perm.p, q.p, s.p, M, K, nullptr);
        });
    }
}
