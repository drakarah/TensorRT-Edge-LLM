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

#include <algorithm>
#include <cstdlib>

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

// --------------------------------------------------------------------------
// n-gram lookup windows (EDGELLM_MTP_LOOKUP): the engines are built for up to 8 verify rows, the drafter's own window
// is kVerifySize. The substitute engines produce one token forever, so once a few tokens exist every later context
// has a long earlier match and the lookup always has a proposal.
// --------------------------------------------------------------------------

constexpr int32_t kLookupEngineWindow{8};

Json makeLookupBaseConfig()
{
    return makeSpecBaseConfig("mtp", kLookupEngineWindow);
}

Json makeLookupDraftConfig()
{
    Json config = makeSpecDraftConfig("mtp", kLookupEngineWindow);
    config["base_model_hidden_size"] = config["hidden_size"];
    return config;
}

class MtpLookupTest : public SpecAssemblyTest
{
protected:
    void TearDown() override
    {
        unsetenv("EDGELLM_MTP_LOOKUP");
        unsetenv("EDGELLM_MTP_LOOKUP_MIN");
        unsetenv("EDGELLM_MTP_LOOKUP_PROMPT");
        SpecAssemblyTest::TearDown();
    }

    std::filesystem::path stageModelDir() override
    {
        return writeSpecModelDir("mtpLookupTests", makeLookupBaseConfig(), makeLookupDraftConfig());
    }

    rt::SpecDecodeDraftingConfig drafting() const override
    {
        return makeMtpDrafting();
    }

    struct Run
    {
        std::vector<int32_t> verifyRows;  //!< seqLen of every base decode-profile forward (one verification each)
        int32_t draftProposalForwards{0}; //!< draft decode-profile forwards (chain steps beyond the first)
        rt::LLMGenerationResponse response;
    };

    //! Generate `maxGenerateLength` tokens for `slots` identical prompts and record the verification windows.
    Run generate(int64_t maxGenerateLength, int32_t slots = 1)
    {
        using ::testing::_;
        Run run;
        auto baseEngine = makeEngine();
        auto draftEngine = makeEngine();
        auto const bindLogits = [this](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t) {
            mLogits = map.get(binding_names::kLogits);
            return true;
        };
        EXPECT_CALL(*baseEngine, prepare(kPrefillProfile, _, _, _)).WillRepeatedly(bindLogits);
        EXPECT_CALL(*draftEngine, prepare(kPrefillProfile, _, _, _)).WillRepeatedly(bindLogits);
        EXPECT_CALL(*baseEngine, prepare(kDecodeProfile, _, _, _))
            .WillRepeatedly([this, &run](int32_t, rt::InferenceDims const& dims, rt::TensorMap const& map, cudaStream_t) {
                mLogits = map.get(binding_names::kLogits);
                run.verifyRows.push_back(static_cast<int32_t>(dims.seqLen));
                return true;
            });
        // The draft's decode-profile forwards are the proposal chain steps (after the first) and the accept pass.
        EXPECT_CALL(*draftEngine, prepare(kDecodeProfile, _, _, _))
            .WillRepeatedly([this, &run](int32_t, rt::InferenceDims const&, rt::TensorMap const& map, cudaStream_t) {
                mLogits = map.get(binding_names::kLogits);
                run.draftProposalForwards += 1;
                return true;
            });
        auto runtime = makeRuntime(makeArtifacts(std::move(baseEngine), std::move(draftEngine)));
        auto request = makeGreedyRequest("a", maxGenerateLength);
        for (int32_t slot = 1; slot < slots; ++slot)
        {
            request.requests.push_back(makeGreedyRequest("a", maxGenerateLength).requests.front());
        }
        EXPECT_TRUE(runtime.handleRequest(request, run.response, mStream));
        return run;
    }
};

TEST_F(MtpLookupTest, WithoutTheSwitchEveryRoundVerifiesOnlyTheDraftersWindow)
{
    auto const run = generate(60);
    ASSERT_FALSE(run.verifyRows.empty());
    EXPECT_THAT(run.verifyRows, ::testing::Each(kVerifySize));
    ASSERT_EQ(run.response.outputIds.size(), 1U);
    EXPECT_EQ(run.response.outputIds[0].size(), 60U);
}

TEST_F(MtpLookupTest, RepeatedOutputGetsAWiderVerifyWindowAndSkipsTheDraftProposal)
{
    auto const off = generate(60);
    setenv("EDGELLM_MTP_LOOKUP", "7", 1);
    auto const on = generate(60);

    // Some rounds verified a window wider than the drafter's, never wider than the engine takes.
    EXPECT_GT(*std::max_element(on.verifyRows.begin(), on.verifyRows.end()), kVerifySize);
    EXPECT_THAT(on.verifyRows, ::testing::Each(::testing::AllOf(::testing::Ge(2), ::testing::Le(kLookupEngineWindow))));
    // A lookup round commits more per verification, so the same 60 tokens take fewer rounds ...
    EXPECT_LT(on.verifyRows.size(), off.verifyRows.size());
    // ... and it skips the draft chain, so the draft layer ran fewer decode forwards than one round-trip each.
    EXPECT_LT(on.draftProposalForwards, off.draftProposalForwards);
    // The output is exactly as long as requested and made of the one token the engines emit.
    ASSERT_EQ(on.response.outputIds.size(), 1U);
    EXPECT_THAT(on.response.outputIds[0], ::testing::AllOf(::testing::SizeIs(60), ::testing::Each(kZeroLogitsToken)));
    EXPECT_EQ(on.response.finishReasons[0], rt::FinishReason::kLength);
}

TEST_F(MtpLookupTest, ALengthCapShorterThanTheWideWindowIsNeverExceeded)
{
    setenv("EDGELLM_MTP_LOOKUP", "7", 1);
    for (int64_t cap : {1, 2, 3, 5, 7, 9, 13})
    {
        auto const run = generate(cap);
        ASSERT_EQ(run.response.outputIds.size(), 1U);
        EXPECT_EQ(run.response.outputIds[0].size(), static_cast<size_t>(cap)) << "cap " << cap;
        EXPECT_EQ(run.response.finishReasons[0], rt::FinishReason::kLength) << "cap " << cap;
    }
}

TEST_F(MtpLookupTest, TheWindowNeverExceedsWhatTheEngineWasBuiltFor)
{
    setenv("EDGELLM_MTP_LOOKUP", "50", 1);
    auto const run = generate(80);
    ASSERT_FALSE(run.verifyRows.empty());
    EXPECT_LE(*std::max_element(run.verifyRows.begin(), run.verifyRows.end()), kLookupEngineWindow);
    EXPECT_GT(*std::max_element(run.verifyRows.begin(), run.verifyRows.end()), kVerifySize);
}

TEST_F(MtpLookupTest, ALookupNoWiderThanTheDraftersWindowIsIgnored)
{
    // A lookup of 2 tokens makes a window of 3 rows, which the drafter already covers (4): nothing to switch on.
    setenv("EDGELLM_MTP_LOOKUP", "2", 1);
    auto const run = generate(40);
    ASSERT_FALSE(run.verifyRows.empty());
    EXPECT_THAT(run.verifyRows, ::testing::Each(kVerifySize));
}

TEST_F(MtpLookupTest, ZeroOrJunkInTheSwitchLeavesTheLookupOff)
{
    for (char const* value : {"0", "-3", "abc", ""})
    {
        setenv("EDGELLM_MTP_LOOKUP", value, 1);
        auto const run = generate(30);
        ASSERT_FALSE(run.verifyRows.empty()) << "value '" << value << "'";
        EXPECT_THAT(run.verifyRows, ::testing::Each(kVerifySize)) << "value '" << value << "'";
    }
}

TEST_F(MtpLookupTest, AMinimumProposalLengthLongerThanAnyProposalKeepsTheDrafterWindow)
{
    // Proposals are at most 7 tokens; asking for 9 before a lookup is offered means it is never offered.
    setenv("EDGELLM_MTP_LOOKUP", "7", 1);
    setenv("EDGELLM_MTP_LOOKUP_MIN", "9", 1);
    auto const run = generate(60);
    ASSERT_FALSE(run.verifyRows.empty());
    EXPECT_THAT(run.verifyRows, ::testing::Each(kVerifySize));
}

TEST_F(MtpLookupTest, ABatchOfTwoNeverUsesALookupWindow)
{
    // The lookup is a single-sequence feature: two slots share one verify size and one set of tensors.
    setenv("EDGELLM_MTP_LOOKUP", "7", 1);
    auto const run = generate(40, 2);
    ASSERT_FALSE(run.verifyRows.empty());
    EXPECT_THAT(run.verifyRows, ::testing::Each(2 * kVerifySize));
    ASSERT_EQ(run.response.outputIds.size(), 2U);
    for (auto const& ids : run.response.outputIds)
    {
        EXPECT_EQ(ids.size(), 40U);
    }
}

TEST_F(MtpLookupTest, ASecondRequestStartsAFreshSequenceInTheCache)
{
    // The cache indexes outputs across requests; a new request must not continue the previous one's n-grams
    // (the separator), and a second request on the same runtime still produces exactly its length.
    setenv("EDGELLM_MTP_LOOKUP", "7", 1);
    auto first = generate(24);
    auto second = generate(24);
    ASSERT_EQ(first.response.outputIds[0].size(), 24U);
    ASSERT_EQ(second.response.outputIds[0].size(), 24U);
    EXPECT_THAT(second.response.outputIds[0], ::testing::Each(kZeroLogitsToken));
}


} // namespace
