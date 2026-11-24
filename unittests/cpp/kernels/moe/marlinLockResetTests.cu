/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Marlin resets every lock it takes back to zero (the last slice of a column), so a workspace that starts zeroed is still zeroed after a GEMM: the down GEMM of
// the MoE plugin can skip its own lock memset after the gate-up GEMM of the same call. Checked for the decode shapes of Inkling 2 with random routing,
// and the results of the memset-free path must equal those of the path with the memset.
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/moe/moe_marlin/moeMarlin.h"
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{
constexpr int32_t kPool = 128;
constexpr int32_t kBlock = 8;

void fillRandomBytes(Tensor& t, int64_t bytes, uint32_t seed, bool scaleLike)
{
    std::mt19937 g(seed);
    std::vector<uint8_t> h(bytes);
    for (auto& b : h)
    {
        b = scaleLike ? static_cast<uint8_t>((6 + g() % 6) << 3 | (g() & 7)) : static_cast<uint8_t>(g());
    }
    CUDA_CHECK(cudaMemcpy(t.rawPointer(), h.data(), h.size(), cudaMemcpyHostToDevice));
}

std::vector<half> run(Tensor const& input, Tensor& output, Tensor const& weights, Tensor const& scales, Tensor const& globalScales, Tensor const& sorted,
    Tensor const& experts, Tensor const& post, Tensor const& topkW, Tensor& workspace, int64_t topK, bool mulW, bool locksZeroed, int64_t outElems)
{
    kernel::moeNvfp4A16MarlinGemm(input, output, weights, scales, globalScales, sorted, experts, post, topkW, workspace, kBlock, topK, mulW, nullptr, locksZeroed);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> out(outElems);
    CUDA_CHECK(cudaMemcpy(out.data(), output.rawPointer(), out.size() * 2, cudaMemcpyDeviceToHost));
    return out;
}
} // namespace

TEST(MarlinLockReset, WorkspaceStaysZeroAndResultsMatchTheMemsetPath)
{
    int32_t sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int32_t const numLocks = 4 * sms;
    for (auto const& [k, n, rows, topK] : {std::tuple<int32_t, int32_t, int32_t, int32_t>{2304, 2304, 1, 8}, {1152, 2304, 8, 1}})
    {
        int32_t const slots = rows * topK;
        int64_t const capacity = static_cast<int64_t>(slots) * kBlock;
        int64_t const ws = kernel::getMoeMarlinWorkspaceSize(capacity, n, kBlock, sms);
        Tensor input({rows, k}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
        Tensor output({slots, n}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
        Tensor weights({kPool, k / 16, 2 * n}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
        fillRandomBytes(weights, static_cast<int64_t>(kPool) * k / 16 * 2 * n * 4, 1, false);
        Tensor scales({kPool, k / 16, n}, DeviceType::kGPU, nvinfer1::DataType::kINT8);
        fillRandomBytes(scales, static_cast<int64_t>(kPool) * k / 16 * n, 2, true);
        Tensor globalScales({kPool}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
        CUDA_CHECK(cudaMemset(globalScales.rawPointer(), 0x3c, kPool * 2));
        {
            std::mt19937 g(3);
            std::vector<half> x(static_cast<size_t>(rows) * k);
            for (auto& v : x)
            {
                v = __float2half((static_cast<int32_t>(g() % 2001) - 1000) / 1000.0F);
            }
            CUDA_CHECK(cudaMemcpy(input.rawPointer(), x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        }
        Tensor sorted({capacity}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
        Tensor experts({slots}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
        Tensor post({1}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
        Tensor topkW({capacity}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        std::vector<float> w(capacity, 0.5F);
        CUDA_CHECK(cudaMemcpy(topkW.rawPointer(), w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        Tensor wsFree({ws}, DeviceType::kGPU, nvinfer1::DataType::kINT32);  // starts zeroed once, never memset again
        Tensor wsMemset({ws}, DeviceType::kGPU, nvinfer1::DataType::kINT32); // the reference path
        CUDA_CHECK(cudaMemset(wsFree.rawPointer(), 0, ws * 4));
        CUDA_CHECK(cudaMemset(wsMemset.rawPointer(), 0, ws * 4));
        std::mt19937 g(11);
        for (int32_t iter = 0; iter < 60; ++iter)
        {
            std::vector<int32_t> pool(kPool);
            for (int32_t i = 0; i < kPool; ++i)
            {
                pool[i] = i;
            }
            std::shuffle(pool.begin(), pool.end(), g);
            std::vector<int32_t> sortedHost(capacity, slots), expertHost(slots);
            for (int32_t s = 0; s < slots; ++s)
            {
                sortedHost[s * kBlock] = s;
                expertHost[s] = pool[s];
            }
            int32_t const postHost = slots * kBlock;
            CUDA_CHECK(cudaMemcpy(sorted.rawPointer(), sortedHost.data(), capacity * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(experts.rawPointer(), expertHost.data(), slots * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(post.rawPointer(), &postHost, 4, cudaMemcpyHostToDevice));
            auto const ref = run(input, output, weights, scales, globalScales, sorted, experts, post, topkW, wsMemset, topK, rows == 8, false, int64_t{slots} * n);
            auto const got = run(input, output, weights, scales, globalScales, sorted, experts, post, topkW, wsFree, topK, rows == 8, true, int64_t{slots} * n);
            std::vector<int32_t> locks(numLocks);
            CUDA_CHECK(cudaMemcpy(locks.data(), wsFree.rawPointer(), numLocks * 4, cudaMemcpyDeviceToHost));
            for (int32_t i = 0; i < numLocks; ++i)
            {
                ASSERT_EQ(locks[i], 0) << "lock " << i << " not reset, K " << k << " N " << n << " iteration " << iter;
            }
            for (size_t i = 0; i < ref.size(); ++i)
            {
                ASSERT_EQ(__half_as_ushort(ref[i]), __half_as_ushort(got[i])) << "output " << i << " K " << k << " iteration " << iter;
            }
        }
    }
}
