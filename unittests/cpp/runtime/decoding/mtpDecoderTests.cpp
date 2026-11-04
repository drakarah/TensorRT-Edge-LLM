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

// MTP speculative decoding assembled around two substitute engines.

#include "substituteEngine.h"

using namespace trt_edgellm;
using namespace substitute_engine;

namespace
{

Json makeMtpDraftConfig()
{
    Json config = makeSpecDraftConfig("mtp", kVerifySize);
    // MTP's draft consumes the base hidden state unchanged, so this equals the base hidden size.
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

class MtpAssemblyTest : public SpecAssemblyTest
{
protected:
    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("mtpAssemblyTests", makeSpecBaseConfig("mtp", kVerifySize), makeMtpDraftConfig());
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }
};

TEST_F(MtpAssemblyTest, AssemblesTwoEnginesWithoutAnyEngineFileOnDisk)
{
    expectAssembledFromArtifactsAlone("mtp");
}

TEST_F(MtpAssemblyTest, RunsTheDraftChainThenOneBaseVerificationPerRound)
{
    expectChainProposalThenOneVerification();
}

TEST_F(MtpAssemblyTest, CompactsTheBatchWhenOneSlotFinishesAheadOfTheOther)
{
    // MTP's per-slot state is the walked chain in its own draft cache. The other three decoders are held to this
    // too; leaving the mode that has been in production longest out of it would be the odd exception.
    expectSurvivorKeepsDecodingAfterCompaction(kVerifySize);
}

TEST_F(MtpAssemblyTest, RunsTheDraftPrefillAfterEveryBasePassOfAPromptLongerThanOneEnginePass)
{
    // The draft layer's KV cache needs the base hidden state of every prompt position, and the base leaves only its
    // current pass in the shared hidden-state buffer. A prompt longer than the engine's max input length therefore has
    // to run the draft prefill after each base pass, on that pass's rows, instead of once at the end.
    //
    // Given a prompt of more than two engine passes (the tiny deployment's engines take 32 tokens per pass)
    // When the request runs
    // Then the base prefill runs in passes of at most that length, each followed by a draft prefill of the same
    // length, and the passes together cover the whole prompt on both engines
    using ::testing::_;
    constexpr int32_t kEnginePass{32};

    std::vector<char> order;
    std::vector<int32_t> basePasses;
    std::vector<int32_t> draftPasses;
    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto const record = [this, &order](std::vector<int32_t>& lengths, char tag) {
        return [this, &order, &lengths, tag](
                   int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
            mLogits = map.get(binding_names::kLogits);
            order.push_back(tag);
            lengths.push_back(dims.seqLen);
            return true;
        };
    };
    EXPECT_CALL(*baseEngine, prepare(kPrefillProfile, _, _, _)).WillRepeatedly(record(basePasses, 'B'));
    EXPECT_CALL(*draftEngine, prepare(kPrefillProfile, _, _, _)).WillRepeatedly(record(draftPasses, 'D'));
    // Declaring an expectation for one profile makes a call with the other one unexpected; the decode steps only need
    // the logits binding captured.
    auto const bindLogits = [this](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t) {
        mLogits = map.get(binding_names::kLogits);
        return true;
    };
    EXPECT_CALL(*baseEngine, prepare(kDecodeProfile, _, _, _)).WillRepeatedly(bindLogits);
    EXPECT_CALL(*draftEngine, prepare(kDecodeProfile, _, _, _)).WillRepeatedly(bindLogits);

    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    auto const request = makeGreedyRequest(std::string(5 * kEnginePass, 'a'), kVerifySize);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    expectResponseCoversEverySlot(response, 1);

    ASSERT_GE(basePasses.size(), 3U);
    EXPECT_THAT(basePasses, ::testing::Each(::testing::Le(kEnginePass)));
    EXPECT_EQ(draftPasses, basePasses) << "each draft prefill must cover exactly the rows of the base pass before it";
    std::string expectedOrder;
    for (size_t pass = 0; pass < basePasses.size(); ++pass)
    {
        expectedOrder += "BD";
    }
    EXPECT_EQ(std::string(order.begin(), order.end()), expectedOrder);
}

TEST_F(MtpAssemblyTest, RunsOneBasePrefillAndOneDraftPrefillForAPromptThatFitsOneEnginePass)
{
    // The per-pass draft prefill is the long-prompt path; a prompt that fits one pass keeps the single prefill pair.
    using ::testing::_;
    std::vector<char> order;
    auto baseEngine = makeEngine();
    auto draftEngine = makeEngine();
    auto const record = [this, &order](char tag) {
        return [this, &order, tag](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t) {
            mLogits = map.get(binding_names::kLogits);
            order.push_back(tag);
            return true;
        };
    };
    EXPECT_CALL(*baseEngine, prepare(kPrefillProfile, _, _, _)).WillRepeatedly(record('B'));
    EXPECT_CALL(*draftEngine, prepare(kPrefillProfile, _, _, _)).WillRepeatedly(record('D'));
    auto const bindLogits = [this](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t) {
        mLogits = map.get(binding_names::kLogits);
        return true;
    };
    EXPECT_CALL(*baseEngine, prepare(kDecodeProfile, _, _, _)).WillRepeatedly(bindLogits);
    EXPECT_CALL(*draftEngine, prepare(kDecodeProfile, _, _, _)).WillRepeatedly(bindLogits);

    auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
    auto const request = makeGreedyRequest("a", kVerifySize);
    rt::LLMGenerationResponse response;
    ASSERT_TRUE(runtime.handleRequest(request, response, mStream));
    EXPECT_EQ(std::string(order.begin(), order.end()), "BD");
}

} // namespace
