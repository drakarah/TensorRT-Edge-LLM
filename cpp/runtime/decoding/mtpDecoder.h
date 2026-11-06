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

#include "common/hashUtils.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/decoding/lookupDraftPolicy.h"
#include "runtime/decoding/suffixDraftCache.h"
#include "runtime/state/externalWeightManager.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

// TODO(follow-up): Simplify MTPDecoder to deliver the simplest chain-based
// speculative drafting + verification. AR + tree-based proposal logic should
// live exclusively in EagleDecoder.
class MTPDecoder final : public DecodingStrategy
{
public:
    MTPDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        SpecDecodeDraftingConfig const& draftingConfig, std::unique_ptr<EngineExecutor> draftExecutor,
        ExternalWeightManager draftWeights, cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kMTP;
    }

    char const* name() const noexcept override
    {
        return "mtp";
    }

    bool isSpeculative() const noexcept override
    {
        return true;
    }

    DecodingStrategyCapabilities capabilities() const noexcept override
    {
        return {/*.ownsBaseVerificationCudaGraphs=*/false, /*.supportsLosslessSampling=*/!mUseTree,
            /*.maxSamplingSupport=*/0, /*.fallbackToVanillaForNonGreedySampling=*/mUseTree,
            /*.requiresDefaultDecoderCudaGraphs=*/false, /*.supportsPresencePenalty=*/!mUseTree};
    }

    DecodingKvHeadroom requiredKvHeadroom() const override;

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;

    //! Hybrid+MTP endpoint reuse runs the draft prefill pre-publication (mirrors EagleDecoder). Default MTP keeps its
    //! decode-round-0 draft prefill: this override is a no-op unless context.hybridMtpEndpointReuse is set.
    bool initializeForGeneration(DecodingInferenceContext& context) override;

    //! Chain drafting only. The draft layer's KV cache needs the base hidden state of every prompt position, so a
    //! prompt longer than one engine pass runs the draft prefill after each base pass (draftPrefillPerPass, or the
    //! per-pass loop of LLMRankRuntime::runHybridMtpPrefill) instead of once at the end. The tree variant keeps the
    //! single-pass contract.
    bool supportsChunkedPrefill() const noexcept override
    {
        return !mUseTree;
    }

    int64_t getRequiredContextMemorySize() const noexcept override;
    void setContextMemory(Tensor& memory) override;

    bool hasSystemPromptKVCache(SystemPromptCacheKey const& key) const override;
    void restoreSystemPromptKVCache(
        SystemPromptCacheKey const& key, int32_t residentSlot, cudaStream_t stream) override;
    bool runSystemPromptPrefill(DecodingInferenceContext& context) override;
    void saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
        std::vector<tokenizer::Rank> const& tokenizedPrompt, int32_t promptIdsLength, cudaStream_t stream) override;

    void resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream) override;
    void onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch,
        Tensor& deviceBatchMapping, cudaStream_t stream) override;

private:
    bool runDraftModelPrefill(DecodingInferenceContext& context);
    bool constructDraftProposal(DecodingInferenceContext& context);
    bool buildTreeVerifyInputs(int32_t activeBatchSize, cudaStream_t stream);
    bool runBaseModelVerification(DecodingInferenceContext& context);
    bool runDraftModelAcceptToken(DecodingInferenceContext& context);

    //! n-gram lookup window (EDGELLM_MTP_LOOKUP=<longest proposal>). Chain mode, batch 1, no guided decoding.
    //! When the lookup finds a continuation of the context in earlier output (and the prompt, with
    //! EDGELLM_MTP_LOOKUP_PROMPT) and the policy expects more tokens per millisecond than the drafter's own window,
    //! the round verifies [last token, lookup tokens...] instead of the draft layer's chain and skips the draft
    //! proposal. Verification is unchanged, so the output is. Returns true when this round's window was replaced.
    bool prepareLookupWindow(DecodingInferenceContext& context);
    void indexLookupTokens(DecodingInferenceContext& context);
    void observeLookupRound(DecodingInferenceContext& context);

    DecodingRuntimeContext& mRuntime;
    HybridCacheManager& mDraftCacheManager;

    std::unique_ptr<EngineExecutor> mDraftExecutor;
    TensorMap mDraftTensorMap;
    //! Externalized draft-engine weights, published into mDraftTensorMap.
    ExternalWeightManager mDraftExternalWeightManager;

    Tensor mDraftProposalSize;
    Tensor mDraftAttentionMask;
    Tensor mDraftTokenIdsFullTable;
    Tensor mDraftTokenScoreFullTable;
    Tensor mDraftTokenPredecessorFullTable;
    Tensor mDraftVocabMappingTable;
    Tensor mDraftRootTokenId;
    Tensor mDraftTokenIdsTable;
    Tensor mDraftTokenScoresTable;
    Tensor mDraftTokenIntermediateScores;
    Tensor mDraftTokenIntermediateParents;
    Tensor mAcceptedTokenIds;
    Tensor mAcceptedTokenIndices;
    Tensor mAcceptLength;
    Tensor mSampledIds;        //!< sampled base token per verify row (sampling requests)
    uint64_t mSamplingOffset{0};
    Tensor mHostAcceptLengths;
    Tensor mHostAcceptedTokenIds;

    //! Tree drafting (draftingTopK > 1): the chain drafter keeps one full logits row
    //! per proposal depth and the greedy tree builder grows a prefix-closed,
    //! score-prioritized verify tree from them.
    bool mUseTree{false};
    Tensor mStackedDraftLogits; //!< [maxBatch, draftingStep+1, draftVocab] per-depth chain logits
    Tensor mTreeTokenIds;       //!< [maxBatch, verifySize] flattened tree token ids
    Tensor mTreeNodeScores;     //!< [maxBatch, verifySize] prefix log-prob scores
    Tensor mValidCounts;        //!< [maxBatch] valid node counts
    Tensor mVerifyTreeMask;     //!< [maxBatch, verifySize, verifySize] unpacked accept mask
    Tensor mTreeBuildWorkspace; //!< ddtreeBuild temporary workspace

    hash_utils::HashMap<SystemPromptCacheKey, SystemPromptKVCache> mSystemPromptKVCacheDraft;

    // Lookup window state (see prepareLookupWindow). mLookupMaxLength == 0: off.
    int32_t mLookupMaxLength{0};     //!< Longest lookup proposal; the widest window is mLookupMaxLength + 1 rows.
    int32_t mLookupMinLength{2};     //!< Shortest proposal worth offering to the policy.
    int32_t mLookupPromptTokens{0};  //!< Index up to this many prompt tokens too (0 = outputs only).
    int32_t mServedVerifySize{0};    //!< The drafter's own window (the engine's served verify size).
    SuffixDraftCache mLookupCache;
    std::unique_ptr<LookupDraftPolicy> mLookupPolicy;
    int32_t mLookupIndexed{0};       //!< Generated tokens of the running request already in the cache.
    bool mLookupRound{false};        //!< This round's window came from the lookup.
    int32_t mRoundWindow{0};         //!< Rows verified this round.
    int32_t mRoundMatch{0};          //!< Context tokens matched by this round's lookup proposal.
    std::chrono::steady_clock::time_point mRoundStart;
    int64_t mStatRounds{0};
    int64_t mStatLookupRounds{0};
    int64_t mStatLookupAccepted{0};
    int64_t mStatTokens{0};
};

} // namespace rt
} // namespace trt_edgellm
