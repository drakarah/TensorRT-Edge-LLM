/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/mambaCacheManager.h"

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// GDN dimensions the replay commit supports: dk = dv = 128; one k-head, two v-heads, convDim = 2 * h * dk + hv * dv.
constexpr int32_t kHv{2};
constexpr int32_t kH{1};
constexpr int32_t kConvDim{2 * kH * 128 + kHv * 128};
constexpr int32_t kConvKernel{4};
constexpr int32_t kLayers{3};
constexpr int32_t kMaxBatch{2};
constexpr int32_t kVerifyRows{8};

size_t rowBytes()
{
    return static_cast<size_t>(kHv) * 128 * 128 * sizeof(float);
}

int32_t minimumScratchRows()
{
    size_t const needed = kernel::gdnTreeScratchBytes(kH, kHv);
    return static_cast<int32_t>((needed + rowBytes() - 1) / rowBytes());
}

rt::MambaCacheManager::Config makeConfig(int32_t scratchRows, bool replay = false, int32_t maxIntermediate = kVerifyRows)
{
    rt::MambaCacheManager::Config config{kLayers, kMaxBatch, kHv, 128, 128, kConvDim, kConvKernel, maxIntermediate,
        DataType::kFLOAT, DataType::kHALF};
    config.recurrentStateNumGroups = replay ? 1 : 0;
    config.specVerifyUsesReplay = replay;
    config.recurrentScratchRows = scratchRows;
    return config;
}

template <typename T>
rt::Tensor uploadTensor(std::vector<T> const& host, rt::Coords shape, DataType type, char const* name)
{
    rt::Tensor tensor(shape, rt::DeviceType::kGPU, type, name);
    EXPECT_EQ(cudaMemcpy(tensor.rawPointer(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return tensor;
}

template <typename T>
std::vector<T> download(rt::Tensor const& tensor, size_t count)
{
    std::vector<T> host(count);
    EXPECT_EQ(cudaMemcpy(host.data(), tensor.rawPointer(), count * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return host;
}

} // namespace

TEST(MambaScratchRowsTest, RecurrentIntermediateHasTheScratchRowsAndConvKeepsOneRowPerToken)
{
    rt::MambaCacheManager scratch(makeConfig(2), nullptr);
    ASSERT_TRUE(scratch.hasIntermediateRecurrentStates());
    rt::Coords const rec = scratch.getIntermediateRecurrentState(0).getShape();
    EXPECT_EQ(rec[0], kMaxBatch);
    EXPECT_EQ(rec[1], 2) << "the scratch rows, not the verify length";
    EXPECT_EQ(rec[2], kHv);
    rt::Coords const conv = scratch.getIntermediateConvState(0).getShape();
    EXPECT_EQ(conv[1], kVerifyRows);

    rt::MambaCacheManager legacy(makeConfig(0), nullptr);
    EXPECT_EQ(legacy.getIntermediateRecurrentState(0).getShape()[1], kVerifyRows);
}

TEST(MambaScratchRowsTest, TheScratchLayoutIsSmallerByTheDroppedRows)
{
    rt::MambaCacheManager scratch(makeConfig(2), nullptr);
    rt::MambaCacheManager legacy(makeConfig(0), nullptr);
    int64_t const scratchBytes = scratch.getIntermediateRecurrentState(0).getShape().volume();
    int64_t const legacyBytes = legacy.getIntermediateRecurrentState(0).getShape().volume();
    EXPECT_EQ(legacyBytes * 2, scratchBytes * kVerifyRows);
}

TEST(MambaScratchRowsTest, ReshapeKeepsTheScratchRowsWhileConvFollowsTheVerifyLength)
{
    rt::MambaCacheManager scratch(makeConfig(2), nullptr);
    for (int32_t activeBatch : {1, 2})
    {
        for (int32_t seqLen : {8, 5, 1})
        {
            scratch.reshapeIntermediateStates(activeBatch, seqLen);
            rt::Coords const rec = scratch.getIntermediateRecurrentState(1).getShape();
            EXPECT_EQ(rec[0], activeBatch);
            EXPECT_EQ(rec[1], 2);
            rt::Coords const conv = scratch.getIntermediateConvState(1).getShape();
            EXPECT_EQ(conv[0], activeBatch);
            EXPECT_EQ(conv[1], seqLen);
        }
    }
    // The legacy layout still follows the verify length.
    rt::MambaCacheManager legacy(makeConfig(0), nullptr);
    legacy.reshapeIntermediateStates(2, 5);
    EXPECT_EQ(legacy.getIntermediateRecurrentState(0).getShape()[1], 5);
}

TEST(MambaScratchRowsTest, InvalidConfigurationsAreRefused)
{
    EXPECT_THROW(rt::MambaCacheManager(makeConfig(-1), nullptr), std::runtime_error);
    EXPECT_THROW(rt::MambaCacheManager(makeConfig(2, /*replay=*/true), nullptr), std::runtime_error)
        << "scratch rows are a GDN snapshot-layout feature";
    // Without spec-verify buffers there is nothing to size.
    EXPECT_NO_THROW(rt::MambaCacheManager(makeConfig(2, false, /*maxIntermediate=*/0), nullptr));
    rt::MambaCacheManager none(makeConfig(2, false, 0), nullptr);
    EXPECT_FALSE(none.hasIntermediateRecurrentStates());
}

TEST(MambaScratchRowsTest, TheFullStateScatterRefusesAScratchLayout)
{
    rt::MambaCacheManager scratch(makeConfig(2), nullptr);
    scratch.reshapeIntermediateStates(1, kVerifyRows);
    rt::Tensor ids({1, 2}, rt::DeviceType::kGPU, DataType::kINT32, "ids");
    rt::Tensor lengths({1}, rt::DeviceType::kGPU, DataType::kINT32, "lengths");
    rt::Tensor indices({1}, rt::DeviceType::kGPU, DataType::kINT32, "indices");
    EXPECT_THROW(scratch.scatterAcceptedTreeStates(ids, lengths, indices, nullptr), std::runtime_error);
}

TEST(MambaScratchRowsTest, ReplayCommitRefusesAnIntermediateBufferBelowTheTreeVerifyScratch)
{
    ASSERT_GE(minimumScratchRows(), 2) << "the test shape must need more than one row to have an undersized case";
    rt::MambaCacheManager tooSmall(makeConfig(minimumScratchRows() - 1), nullptr);
    tooSmall.reshapeIntermediateStates(1, kVerifyRows);
    std::vector<int32_t> const hostIds{0, 1, 2};
    rt::Tensor ids = uploadTensor(hostIds, {1, 3}, DataType::kINT32, "ids");
    rt::Tensor lengths = uploadTensor(std::vector<int32_t>{3}, {1}, DataType::kINT32, "lengths");
    rt::Tensor indices = uploadTensor(std::vector<int32_t>{0}, {1}, DataType::kINT32, "indices");
    EXPECT_THROW(tooSmall.replayCommitAcceptedTreeStates(ids, lengths, indices, nullptr), std::runtime_error);
}

TEST(MambaScratchRowsTest, ReplayCommitRefusesMoreAcceptedNodesThanTheVerifyTree)
{
    rt::MambaCacheManager scratch(makeConfig(minimumScratchRows()), nullptr);
    scratch.reshapeIntermediateStates(1, 3);
    std::vector<int32_t> const hostIds{0, 1, 2, 3};
    rt::Tensor ids = uploadTensor(hostIds, {1, 4}, DataType::kINT32, "ids");
    rt::Tensor lengths = uploadTensor(std::vector<int32_t>{4}, {1}, DataType::kINT32, "lengths");
    rt::Tensor indices = uploadTensor(std::vector<int32_t>{0}, {1}, DataType::kINT32, "indices");
    EXPECT_THROW(scratch.replayCommitAcceptedTreeStates(ids, lengths, indices, nullptr), std::runtime_error);
}

//! The commit reads the stash cells at the head of each batch's intermediate buffer; the buffer's row count only sets
//! where the next batch's cells start. Same cells at their layout's offsets must give the same committed state.
TEST(MambaScratchRowsTest, ReplayCommitGivesTheSameStatesAsTheTokenMajorLayout)
{
    int32_t const scratchRows = minimumScratchRows();
    rt::MambaCacheManager legacy(makeConfig(0), nullptr);
    rt::MambaCacheManager scratch(makeConfig(scratchRows), nullptr);
    legacy.reshapeIntermediateStates(kMaxBatch, kVerifyRows);
    scratch.reshapeIntermediateStates(kMaxBatch, kVerifyRows);

    std::mt19937 rng(20261111);
    std::uniform_real_distribution<float> dist(-0.5F, 0.5F);
    size_t const nodeBytes = kernel::gdnTreeStashNodeBytes(kH, kHv);
    size_t const stashCellBytes = static_cast<size_t>(kVerifyRows) * nodeBytes;
    size_t const stateElements = static_cast<size_t>(kMaxBatch) * kHv * 128 * 128;
    size_t const convElements = static_cast<size_t>(kMaxBatch) * kConvDim * kConvKernel;

    for (int32_t layer = 0; layer < kLayers; ++layer)
    {
        std::vector<float> state(stateElements);
        for (float& value : state)
        {
            value = dist(rng);
        }
        std::vector<__half> conv(convElements);
        for (__half& value : conv)
        {
            value = __float2half(dist(rng));
        }
        std::vector<__half> interConv(static_cast<size_t>(kMaxBatch) * kVerifyRows * kConvDim * kConvKernel);
        for (__half& value : interConv)
        {
            value = __float2half(dist(rng));
        }
        std::vector<float> cells(stashCellBytes / sizeof(float) * kMaxBatch);
        for (float& value : cells)
        {
            value = dist(rng);
        }
        for (rt::MambaCacheManager* manager : {&legacy, &scratch})
        {
            ASSERT_EQ(cudaMemcpy(manager->getRecurrentState(layer).rawPointer(), state.data(),
                          state.size() * sizeof(float), cudaMemcpyHostToDevice),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpy(manager->getConvState(layer).rawPointer(), conv.data(), conv.size() * sizeof(__half),
                          cudaMemcpyHostToDevice),
                cudaSuccess);
            ASSERT_EQ(cudaMemcpy(manager->getIntermediateConvState(layer).rawPointer(), interConv.data(),
                          interConv.size() * sizeof(__half), cudaMemcpyHostToDevice),
                cudaSuccess);
            int32_t const rows = manager == &legacy ? kVerifyRows : scratchRows;
            size_t const stride = static_cast<size_t>(rows) * rowBytes();
            std::vector<char> buffer(stride * kMaxBatch, 0);
            for (int32_t b = 0; b < kMaxBatch; ++b)
            {
                std::memcpy(buffer.data() + b * stride, reinterpret_cast<char const*>(cells.data()) + b * stashCellBytes,
                    stashCellBytes);
            }
            ASSERT_EQ(cudaMemcpy(manager->getIntermediateRecurrentState(layer).rawPointer(), buffer.data(),
                          buffer.size(), cudaMemcpyHostToDevice),
                cudaSuccess);
        }
    }

    std::vector<int32_t> const hostIds{0, 2, 5, 0, 1, 3};
    std::vector<int32_t> const hostLengths{3, 2};
    std::vector<int32_t> const hostIndices{1, 0};
    for (rt::MambaCacheManager* manager : {&legacy, &scratch})
    {
        rt::Tensor ids = uploadTensor(hostIds, {2, 3}, DataType::kINT32, "ids");
        rt::Tensor lengths = uploadTensor(hostLengths, {2}, DataType::kINT32, "lengths");
        rt::Tensor indices = uploadTensor(hostIndices, {2}, DataType::kINT32, "indices");
        manager->replayCommitAcceptedTreeStates(ids, lengths, indices, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    for (int32_t layer = 0; layer < kLayers; ++layer)
    {
        auto const legacyState = download<float>(legacy.getRecurrentState(layer), stateElements);
        auto const scratchState = download<float>(scratch.getRecurrentState(layer), stateElements);
        EXPECT_EQ(std::memcmp(legacyState.data(), scratchState.data(), stateElements * sizeof(float)), 0)
            << "committed recurrent state differs in layer " << layer;
        auto const legacyConv = download<__half>(legacy.getConvState(layer), convElements);
        auto const scratchConv = download<__half>(scratch.getConvState(layer), convElements);
        EXPECT_EQ(std::memcmp(legacyConv.data(), scratchConv.data(), convElements * sizeof(__half)), 0)
            << "committed conv state differs in layer " << layer;
    }
}
