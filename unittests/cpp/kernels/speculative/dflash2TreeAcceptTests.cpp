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
#include "common/tensor.h"
#include "kernels/speculative/dflash2TreeAccept.h"
#include "testUtils.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm;

namespace
{

// Tree (per batch): node 0 root; 1 <- 0 (token 10); 2 <- 0 (token 20); 3 <- 1 (token 30); 4 <- 2 (token 40).
constexpr int32_t kNodes = 5;
constexpr int32_t kK = 3;
constexpr int32_t kProposalLen = 4;

struct AcceptResult
{
    std::vector<int32_t> tokens;
    std::vector<int32_t> indices;
    std::vector<int32_t> lengths;
};

AcceptResult run(std::vector<float> const& probs, std::vector<int32_t> const& ids, std::vector<float> const& uniforms,
    bool greedy, std::vector<int32_t> const& remaining, int32_t batch)
{
    std::vector<int32_t> treeTokens;
    std::vector<int32_t> treeParents;
    std::vector<int32_t> valid;
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t t : {99, 10, 20, 30, 40})
        {
            treeTokens.push_back(t);
        }
        for (int32_t p : {-1, 0, 0, 1, 2})
        {
            treeParents.push_back(p);
        }
        valid.push_back(kNodes);
    }
    rt::Tensor probsT({batch, kNodes, kK}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor idsT({batch, kNodes, kK}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor tokT({batch, kNodes}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor parT({batch, kNodes}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor validT({batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor uniT({batch, 2 * kProposalLen + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    rt::Tensor remT({batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor accIds({batch, kProposalLen + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor accIdx({batch, kProposalLen + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor accLen({batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(probsT, probs);
    copyHostToDevice(idsT, ids);
    copyHostToDevice(tokT, treeTokens);
    copyHostToDevice(parT, treeParents);
    copyHostToDevice(validT, valid);
    copyHostToDevice(uniT, uniforms);
    copyHostToDevice(remT, remaining);
    kernel::dflash2TreeSampleAccept(probsT, idsT, tokT, parT, validT, uniT, kProposalLen, greedy, &remT, accIds, accIdx,
        accLen, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    return {copyDeviceToHost<int32_t>(accIds), copyDeviceToHost<int32_t>(accIdx), copyDeviceToHost<int32_t>(accLen)};
}

std::vector<float> baseProbs()
{
    // node 0: ids {10,20,99} p {.2,.7,.1}; node 1: {30,31,32} {.5,.4,.1}; node 2: {40,41,42} {.6,.3,.1};
    // node 3, 4: {7,8,9} {.5,.3,.2}.
    return {.2F, .7F, .1F, .5F, .4F, .1F, .6F, .3F, .1F, .5F, .3F, .2F, .5F, .3F, .2F};
}

std::vector<int32_t> baseIds()
{
    return {10, 20, 99, 30, 31, 32, 40, 41, 42, 7, 8, 9, 7, 8, 9};
}

} // namespace

TEST(DFlash2TreeAcceptTest, greedyFollowsTheBestChildAndEmitsTheBonus)
{
    auto const r = run(baseProbs(), baseIds(), std::vector<float>(2 * kProposalLen + 1, 0.5F), true, {8}, 1);
    ASSERT_EQ(r.lengths[0], 3);
    EXPECT_EQ((std::vector<int32_t>{r.tokens[0], r.tokens[1], r.tokens[2]}), (std::vector<int32_t>{20, 40, 7}));
    EXPECT_EQ((std::vector<int32_t>{r.indices[0], r.indices[1], r.indices[2]}), (std::vector<int32_t>{0, 2, 4}));
}

TEST(DFlash2TreeAcceptTest, sampledDrawUsesTheDepthUniformAndStopsWithoutAMatchingChild)
{
    std::vector<float> u(2 * kProposalLen + 1, 0.5F);
    u[0] = 0.15F; // node 0: cumulative {.2, .9, 1.0} -> id 10 -> child 1
    u[1] = 0.95F; // node 1: cumulative {.5, .9, 1.0} -> id 32, no child -> bonus
    auto const r = run(baseProbs(), baseIds(), u, false, {8}, 1);
    ASSERT_EQ(r.lengths[0], 2);
    EXPECT_EQ((std::vector<int32_t>{r.tokens[0], r.tokens[1]}), (std::vector<int32_t>{10, 32}));
    EXPECT_EQ((std::vector<int32_t>{r.indices[0], r.indices[1]}), (std::vector<int32_t>{0, 1}));
}

TEST(DFlash2TreeAcceptTest, remainingGenerationCapsTheAcceptedLength)
{
    auto const r = run(baseProbs(), baseIds(), std::vector<float>(2 * kProposalLen + 1, 0.5F), true, {2}, 1);
    ASSERT_EQ(r.lengths[0], 2);
    EXPECT_EQ((std::vector<int32_t>{r.tokens[0], r.tokens[1]}), (std::vector<int32_t>{20, 40}));
}

TEST(DFlash2TreeAcceptTest, batchesAreIndependent)
{
    // Batch 1 repeats batch 0's tree but its caller-supplied probabilities are the same; remaining differs.
    auto p = baseProbs();
    auto i = baseIds();
    auto p2 = p;
    p.insert(p.end(), p2.begin(), p2.end());
    auto i2 = i;
    i.insert(i.end(), i2.begin(), i2.end());
    std::vector<float> u(2 * (2 * kProposalLen + 1), 0.5F);
    auto const r = run(p, i, u, true, {8, 1}, 2);
    EXPECT_EQ(r.lengths[0], 3);
    EXPECT_EQ(r.lengths[1], 1);
    EXPECT_EQ(r.tokens[kProposalLen + 1], 20);
}
