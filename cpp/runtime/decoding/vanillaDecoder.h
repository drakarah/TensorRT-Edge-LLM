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

#pragma once

#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/draftSource.h"
#include "runtime/decoding/lookupDraftPolicy.h"
#include "runtime/decoding/suffixDraftCache.h"

#include <array>
#include <chrono>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

class VanillaDecoder final : public DecodingStrategy
{
public:
    explicit VanillaDecoder(DecodingRuntimeContext& runtime);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kVanilla;
    }

    char const* name() const noexcept override
    {
        return "vanilla";
    }

    //! Samples from the last prompt position only.
    bool supportsChunkedPrefill() const noexcept override
    {
        return true;
    }
    bool isSpeculative() const noexcept override
    {
        return false;
    }

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;
    //! Inkling drafter (EDGELLM_INKLING_SPEC=drafter | hybrid): it needs the taps of every prompt position, pass by pass.
    bool needsPerPassDraftUpdate() const noexcept override
    {
        return mDrafter != nullptr;
    }
    static int32_t sequenceToken(DecodingInferenceContext const& context, int32_t position);
    bool drafterContextMatches(DecodingInferenceContext const& context, int32_t length) const;
    void noteDrafterTokens(DecodingInferenceContext const& context, int32_t pos0, int32_t rows);
    bool absorbPrefillPass(DecodingInferenceContext& context) override;
    bool initializeForGeneration(DecodingInferenceContext& context) override;

    DecodingKvHeadroom requiredKvHeadroom() const override
    {
        return {/*.baseExtraTokens=*/std::max(1, mChainWindow), /*.draftExtraTokens=*/0};
    }

    int64_t getRequiredContextMemorySize() const noexcept override
    {
        return 0;
    }
    void setContextMemory(Tensor&) override {}

    bool hasSystemPromptKVCache(SystemPromptCacheKey const&) const override
    {
        return false;
    }
    void restoreSystemPromptKVCache(SystemPromptCacheKey const&, int32_t, cudaStream_t) override {}
    bool runSystemPromptPrefill(DecodingInferenceContext&) override
    {
        return true;
    }
    void saveSystemPromptKVCache(SystemPromptCacheKey const&, std::string const&, std::vector<tokenizer::Rank> const&,
        int32_t, cudaStream_t) override
    {
    }

    void resetForNewSequences(Tensor&, cudaStream_t) override {}
    void onBatchEvict(std::vector<int32_t> const&, int32_t, int32_t, Tensor&, cudaStream_t) override {}

private:
    //! Inkling 2 chain verification (EDGELLM_INKLING_SPEC=lookup | cost:<T>): a step verifies [committed token, drafts...] in one engine pass and
    //! commits the accepted prefix. Needs an engine built with EDGELLM_INKLING_VERIFY and a greedy single-sequence request; every other request, and every
    //! step without a proposal, runs the plain single-token step. Greedy output equals plain decoding by construction.
    bool chainApplies(DecodingInferenceContext const& context) const;
    bool decodeStepChain(DecodingInferenceContext& context);
    bool decodeStepPlain(DecodingInferenceContext& context);
    //! Runs one chain pass of ids.size() tokens at past length pastLength; the logits are left in outputLogits [T, vocab].
    bool runChainPass(std::vector<int32_t> const& ids, ResidentRef ref, RequestId requestId, int32_t pastLength,
        RaggedBatchBuilder& builder, RaggedExecutionBatch& batch, StepId stepId, cudaStream_t stream, bool captureOnly);
    void prepareChainTensors();
    void indexLookupTokens(DecodingInferenceContext& context);
    bool feedDrafterPrefill(DecodingInferenceContext& context, bool lastPass);

    DecodingRuntimeContext& mRuntime;
    int32_t mChainWindow{0};      //!< Largest verified window (0 = chain verification off).
    int32_t mChainMinProposal{2}; //!< Shortest lookup proposal worth a verify.
    int32_t mChainFixed{0};       //!< cost mode: always verify this many tokens (fillers when the lookup has no proposal).
    int32_t mLookupIndexed{0};
    SuffixDraftCache mLookupCache;
    std::unique_ptr<LookupDraftPolicy> mLookupPolicy;
    Tensor mChainArgmax;          //!< GPU [maxWindow, 1] INT32
    Tensor mChainHostArgmax;      //!< CPU [maxWindow]
    Tensor mChainUniforms;        //!< GPU [maxWindow] FP32, request-stable uniforms of the sampled verify rows
    Tensor mChainHostUniforms;    //!< CPU [maxWindow]
    uint64_t mChainSamplingOffset{0}; //!< device RNG offset of the sampled verify rows when sampling is not request-stable
    Tensor mChainAcceptLength;    //!< GPU [1]
    Tensor mChainAcceptedIds;     //!< GPU [1, maxWindow]
    Tensor mChainHostAcceptLength;
    Tensor mChainHostAcceptedIds;
    std::unique_ptr<IDraftSource> mDrafter;
    bool mDrafterPrefillOpen{false}; //!< a chunked prefill has fed passes and awaits its last one
    std::vector<int32_t> mDrafterTokens; //!< token ids whose taps the drafter holds (positions 0..size-1): a request with a reused prefix resumes from them
    bool mDrafterValid{false};   //!< the drafter has the taps of every position of this request
    bool mAutoK{false};          //!< EDGELLM_INKLING_DRAFT_K=auto: the verified draft count follows the running acceptance
    std::array<double, 7> mPrefixAcceptance{0.70, 0.47, 0.33, 0.24, 0.18, 0.14, 0.10}; //!< P(at least j + 1 drafts accepted), offline prior
    int32_t mDraftK{3};          //!< drafted tokens verified per round
    bool mHybrid{false};         //!< prefer a long lookup proposal over the drafter's
    Tensor mDrafterIds;          //!< GPU [1, 1] INT32
    Tensor mDrafterEmb;          //!< GPU [1, 1, hidden] FP16
    Tensor mDrafterHostTokens;   //!< pinned [8] INT32
    int64_t mStatDraftRounds{0};
    int64_t mStatDraftAccepted{0};
    int64_t mStatRounds{0};
    int64_t mStatChainRounds{0};
    int64_t mStatTokens{0};
    int64_t mStatAccepted{0};
    std::array<int64_t, 9> mStatWindowRounds{};
    std::array<double, 9> mStatWindowMs{};
    std::array<int64_t, 9> mStatWindowAccepted{};
};

} // namespace rt
} // namespace trt_edgellm
