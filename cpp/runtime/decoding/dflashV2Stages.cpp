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

#include "runtime/decoding/dflashDecoder.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/speculative/dflash2CandidateSelector.h"
#include "kernels/speculative/dflash2TreeAccept.h"
#include "kernels/speculative/ddtreeKernels.h"
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "kernels/speculative/dsparkKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "kernels/speculative/requestStableRngKernels.h"
#include "kernels/speculative/speculativeSampling.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/dflashDecodeUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/requestStableRng.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
Tensor takeRequiredSelectorTensor(std::vector<Tensor>& tensors, std::string const& name)
{
    auto const it
        = std::find_if(tensors.begin(), tensors.end(), [&](Tensor const& tensor) { return tensor.getName() == name; });
    check::check(it != tensors.end(), "DFlash2 selector sidecar is missing tensor '" + name + "'");
    Tensor output = std::move(*it);
    tensors.erase(it);
    return output;
}

void validateSelectorCodebook(Tensor const& tensor, std::string const& name, int32_t vocabSize, int32_t rank)
{
    check::check(tensor.getDataType() == nvinfer1::DataType::kHALF, name + " must be FP16");
    check::check(tensor.getShape().getNumDims() == 2, name + " must be rank-2");
    check::check(tensor.getShape()[0] == vocabSize && tensor.getShape()[1] == rank,
        name + " must have shape [" + std::to_string(vocabSize) + ", " + std::to_string(rank) + "]");
}
} // namespace

void DFlashDecoder::initializeDFlash2(
    std::filesystem::path const& engineDir, ExternalWeightManager draftWeights, cudaStream_t stream)
{
    auto const& deployment = mRuntime.deployment;
    auto const& baseCfg = deployment.base;
    ELLM_CHECK(deployment.specConfig.has_value(), "DFlashDecoder: specConfig is required.");
    ELLM_CHECK(deployment.draft.has_value(), "DFlashDecoder: draft config is required.");
    ELLM_CHECK(isCachedBlockDraftMode(baseCfg.specDecodeType),
        "DFlashDecoder requires a cached-block speculative base engine.");
    ELLM_CHECK(baseCfg.specDecodeType == mBlockDraft.userMode,
        "DFlashDecoder normalized user mode does not match the base engine config.");
    ELLM_CHECK(baseCfg.specDecodeType == SpecDecodeMode::kDFlash && baseCfg.dflashVersion == DFlashVersion::kV2,
        "DFlashDecoder requires spec_decode_type=dflash with dflash_config.version=2.");
    bool const treeMode = mBlockDraft.treePolicy == dflash_utils::BlockDraftTreePolicy::kDDTree;
    ELLM_CHECK(mBlockDraft.blockSize >= 2 && mBlockDraft.blockSize <= 16
            && mBlockDraft.proposalLen == mBlockDraft.blockSize - 1
            && (treeMode ? (mBlockDraft.verifySize > mBlockDraft.blockSize && mBlockDraft.verifySize <= kernel::kGDN_TREE_CHUNK_MAX_NODES
                               && mBlockDraft.candidateTopK >= 2)
                         : mBlockDraft.verifySize == mBlockDraft.blockSize),
        "DFlashDecoder requires block_size=verify_size (linear) or block_size < verify_size <= 16 with "
        "draft_top_k >= 2 (DDTree), block_size in [2, 16].");

    int32_t const maxBatch = deployment.maxRuntimeBatchSize();

    ELLM_CHECK(mDraftExecutor != nullptr, std::string(userModeName()) + " decoding requires a validated draft engine.");

    mDraftInputsEmbeds = Tensor({maxBatch, mBlockDraft.blockSize, mBlockDraft.draftHiddenSize}, DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "DFlash2Draft::inputsEmbeds");
    mDraftTargetHidden = Tensor({maxBatch, mBlockDraft.blockSize, mBlockDraft.baseOutputHiddenDim}, DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "DFlash2Draft::targetHiddenScratch");
    int32_t const proposalLen = mBlockDraft.proposalLen;
    int32_t const selectorTopK = deployment.draft->specSelectorTopK;
    int32_t const selectorRank = deployment.draft->specSelectorRank;
    mDraftTokenIds
        = Tensor({maxBatch, proposalLen}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2Draft::proposalIds");
    mProposalSupportIds = Tensor({maxBatch, proposalLen, selectorTopK}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "DFlash2Draft::proposalSupportIds");
    mProposalSupportProbs = Tensor({maxBatch, proposalLen, selectorTopK}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "DFlash2Draft::proposalSupportProbs");
    mProposalUnaryValues = Tensor({maxBatch, proposalLen, selectorTopK}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "DFlash2Draft::proposalUnaryValues");
    mProposalProjectedHidden = Tensor({maxBatch, proposalLen, selectorRank}, DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "DFlash2Draft::proposalProjectedHidden");
    mProposalUniforms = Tensor(
        {maxBatch, proposalLen}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DFlash2Draft::proposalUniforms");
    mSamplingTemperatures
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DFlash2Draft::temperatures");
    mProposalGreedyMask = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2Draft::greedyMask");
    mLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2Draft::lastAcceptedTokens");

    int32_t const packedMaskLen = divUp(mBlockDraft.blockSize, 32);
    mDraftPackedAttentionMask = Tensor({maxBatch, mBlockDraft.blockSize, packedMaskLen}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "DFlash2Draft::packedMask");
    mDraftAttentionPosId = Tensor(
        {maxBatch, mBlockDraft.blockSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2Draft::positionIds");
    mDraftContextLengths
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2Draft::contextLengths");
    mDraftDeltaLenCommit
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2Draft::deltaLenCommit");
    mDraftDeltaLens = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2Draft::deltaLens");

    mDraftTensorMap.set(binding_names::kInputsEmbeds, mDraftInputsEmbeds);
    mDraftTensorMap.set(binding_names::kDFlashTargetHiddenConcat, mDraftTargetHidden);
    mDraftTensorMap.set(binding_names::kSpecProposalSupportIds, mProposalSupportIds);
    mDraftTensorMap.set(binding_names::kSpecProposalUnaryValues, mProposalUnaryValues);
    mDraftTensorMap.set(binding_names::kSpecProposalProjectedHidden, mProposalProjectedHidden);
    mDraftTensorMap.set(binding_names::kAttentionMask, mDraftPackedAttentionMask);
    mDraftTensorMap.set(binding_names::kAttentionPosId, mDraftAttentionPosId);
    LLMEngineConfig const& draftCfg = *deployment.draft;
    initializeDraftRaggedBindings(draftCfg);

    // KV cache bindings: bind to draft cache manager's combined KV cache (index 1). Unified on
    // the paged-pool view — the engine's past/present_key_values_i binding for DFlash's own draft
    // cache is the same [2, numPages, kTOKENS_PER_PAGE, numKVHeads, headDim] contract as the main
    // model and EAGLE/MTP drafts.
    auto& kvMgr = mDraftCacheManager.getKVCacheManager();
    int32_t localAttnIdx = 0;
    for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(draftCfg.layerTypes.size()); ++absIdx)
    {
        if (draftCfg.layerTypes[absIdx] != HybridCacheManager::LayerType::kAttention)
        {
            continue;
        }
        auto& combinedKV = kvMgr.getCombinedKVCache(localAttnIdx);
        mDraftTensorMap.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/true), combinedKV);
        mDraftTensorMap.set(binding_names::formatKVCacheName(localAttnIdx, /*isPast=*/false), combinedKV);
        ++localAttnIdx;
    }
    mDraftExternalWeightManager = std::move(draftWeights);
    mDraftExternalWeightManager.registerTensorMapEntries(mDraftTensorMap);

    std::string const selectorFile = draftCfg.dflash2SelectorFile.empty()
        ? std::string(binding_names::kDFlash2SelectorFileName)
        : draftCfg.dflash2SelectorFile;
    auto const selectorPath = engineDir / selectorFile;
    ELLM_CHECK(std::filesystem::exists(selectorPath), "DFlash2 selector sidecar missing: " + selectorPath.string());
    std::vector<Tensor> selectorTensors;
    ELLM_CHECK(safetensors::loadSafetensors(selectorPath, selectorTensors, stream),
        "Failed to load DFlash2 selector sidecar from " + selectorPath.string());
    mSelectorPredecessorCodebook = takeRequiredSelectorTensor(selectorTensors, "predecessor_codebook");
    mSelectorSuccessorCodebook = takeRequiredSelectorTensor(selectorTensors, "successor_codebook");
    check::check(selectorTensors.empty(), "DFlash2 selector sidecar contains unexpected tensors");
    validateSelectorCodebook(
        mSelectorPredecessorCodebook, "predecessor_codebook", draftCfg.outputVocabSize, selectorRank);
    validateSelectorCodebook(mSelectorSuccessorCodebook, "successor_codebook", draftCfg.outputVocabSize, selectorRank);

    mHostDraftInputIds = Tensor(
        {maxBatch, mBlockDraft.blockSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash2Draft::hostInputIds");
    mHostLastAcceptedTokens
        = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash2Draft::hostLastAcceptedTokens");
    mHostDeltaLens = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash2Draft::hostDeltaLens");

    mVerifyTokenIds = Tensor(
        {maxBatch, mBlockDraft.verifySize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2::verifyTokenIds");
    int32_t const maxAcceptBufferSize = mBlockDraft.verifySize;
    mAcceptedTokenIds = Tensor(
        {maxBatch, maxAcceptBufferSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2::acceptedTokenIds");
    mAcceptedTokenIndices = Tensor(
        {maxBatch, maxAcceptBufferSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2::acceptedTokenIndices");
    mAcceptLength = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2::acceptLength");
    mHostAcceptLengths = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash2::hostAcceptLengths");
    mHostAcceptedTokenIds = Tensor(
        {maxBatch, maxAcceptBufferSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash2::hostAcceptedIds");
    if (treeMode)
    {
        // DDTree over the drafter's top-k candidates. The builder reads [batch, rows, columns] sparse logits with one
        // leading placeholder row (rows 1..proposalLen carry the candidates, firstCandidateLogitsRow = 1).
        int32_t const candRows = mBlockDraft.blockSize;
        mTreeCandidateLogits = Tensor({maxBatch, candRows, selectorTopK}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
            "DFlash2::treeCandidateLogits");
        mTreeCandidateIds = Tensor({maxBatch, candRows, selectorTopK}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
            "DFlash2::treeCandidateIds");
        mTreeTokenIds = Tensor(
            {maxBatch, mBlockDraft.verifySize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2::treeTokenIds");
        mTreeNodeScores = Tensor({maxBatch, mBlockDraft.verifySize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
            "DFlash2::treeNodeScores");
        mValidCounts = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2::validCounts");
        mVerifyTreeMask = Tensor({maxBatch, mBlockDraft.verifySize, mBlockDraft.verifySize}, DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "DFlash2::verifyTreeMask");
        size_t const buildWorkspaceSize = kernel::getDDTreeBuildWorkspaceSize(
            maxBatch, candRows, mBlockDraft.verifySize, selectorTopK, mBlockDraft.candidateTopK);
        ELLM_CHECK(buildWorkspaceSize > 0, "DFlash2: DDTree build workspace size must be > 0.");
        mBuildWorkspace = Tensor({static_cast<int64_t>(buildWorkspaceSize)}, DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "DFlash2::buildWorkspace");
    }
    mRemainingGenerationLengths
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2::remainingGenerationLengths");
    mHostRemainingGenerationLengths
        = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash2::hostRemainingGenerationLengths");

    constexpr int32_t kMaxTargetTopK{128};
    mTargetTopKValues = Tensor({maxBatch * mBlockDraft.verifySize, kMaxTargetTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "DFlash2::targetTopKValues");
    mTargetTopKIds = Tensor({maxBatch * mBlockDraft.verifySize, kMaxTargetTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kINT32, "DFlash2::targetTopKIds");
    mTargetTopKProbs = Tensor({maxBatch * mBlockDraft.verifySize, kMaxTargetTopK}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "DFlash2::targetTopKProbs");
    mTargetProbabilities = Tensor({maxBatch * mBlockDraft.verifySize, mRuntime.deployment.base.outputVocabSize},
        DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DFlash2::targetProbabilities");
    mProposalLengths = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "DFlash2::proposalLengths");
    mAcceptUniforms = Tensor(
        {maxBatch, 2 * proposalLen + 1}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "DFlash2::acceptUniforms");
    mRequestSeeds = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT64, "DFlash2::requestSeeds");
    mNextAbsolutePositions
        = Tensor({maxBatch}, DeviceType::kGPU, nvinfer1::DataType::kINT64, "DFlash2::nextAbsolutePositions");
    mHostRequestSeeds = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT64, "DFlash2::hostRequestSeeds");
    mHostNextPositions = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT64, "DFlash2::hostNextPositions");
    mHostTemperatures = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "DFlash2::hostTemperatures");
    mHostGreedyMask = Tensor({maxBatch}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "DFlash2::hostGreedyMask");

    LOG_INFO(
        "DFlashDecoder initialized: user_mode=%s, proposal_attention=%s, tree_policy=%s, blockSize=%d, "
        "proposalLen=%d, verifySize=%d, runtimeDraftTopK=%d, selectorTopK=%d, maskTokenId=%d, maxBatch=%d, "
        "draftHiddenSize=%d, baseOutputHiddenDim=%d, draftVocabSize=%d",
        userModeName(), dflash_utils::proposalAttentionPolicyName(mBlockDraft.proposalAttention),
        dflash_utils::blockDraftTreePolicyName(mBlockDraft.treePolicy), mBlockDraft.blockSize, mBlockDraft.proposalLen,
        mBlockDraft.verifySize, mBlockDraft.candidateTopK, selectorTopK, mBlockDraft.maskTokenId, maxBatch,
        mBlockDraft.draftHiddenSize, mBlockDraft.baseOutputHiddenDim, mBlockDraft.draftVocabSize);
}

bool DFlashDecoder::prepareV2Proposal(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(
        nvtx_dflash2_prepare_verify, "DFlashDecoder::prepareBlockDraftVerifyInputs", nvtx_colors::LIGHT_ORANGE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifySize = mBlockDraft.verifySize;
    if (mBlockDraft.treePolicy == dflash_utils::BlockDraftTreePolicy::kDDTree)
    {
        if (!buildV2TreeVerifyInputs(context))
        {
            return false;
        }
        if (context.hasGuidedDecoding)
        {
            // One grammar state per tree node (the node's root path): the base logits of every verify row are masked
            // with it in runBaseVerification, so the lossless accept kernel only sees grammar-legal probabilities.
            mRuntime.guidedDecoder.captureDraftTree(mTreeTokenIds, mRuntime.base.pipelineIO.specTreeParentIds,
                std::ref(mValidCounts), activeBatchSize, verifySize, context.stream);
        }
        return checkCudaLastError("prepare DFlash2 guided DDTree inputs");
    }
    check::check(mVerifyTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mDraftTokenIds.reshape({activeBatchSize, mBlockDraft.proposalLen}), "Tensor reshape failed");
    kernel::dsparkBuildVerifyTokens(mLastAcceptedTokens, mDraftTokenIds, mVerifyTokenIds, activeBatchSize,
        mBlockDraft.proposalLen, mBlockDraft.proposalLen, context.stream);
    check::check(mProposalLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    kernel::dsparkFillProposalLengths(mProposalLengths, activeBatchSize, mBlockDraft.proposalLen, context.stream);
    prepareLinearTreeBaseVerificationMetadata(activeBatchSize, verifySize, context.stream);

    copyVerifyTokenIdsToBaseInput(activeBatchSize, verifySize, context.stream);
    if (context.hasGuidedDecoding)
    {
        mRuntime.guidedDecoder.captureDraftChains(mVerifyTokenIds, activeBatchSize, verifySize, context.stream);
    }
    if (!checkCudaLastError("prepare DFlash verify inputs"))
    {
        return false;
    }
    return true;
}

bool DFlashDecoder::runV2Acceptance(DecodingInferenceContext& context, int32_t verifySize, int32_t maxAcceptLength)
{
    int32_t const activeBatchSize = context.activeBatchSize;
    bool const targetNonGreedy
        = ::trt_edgellm::shouldUseNonGreedySampling(context.temperature, context.topK, context.topP);
    int32_t targetTopK = targetNonGreedy ? static_cast<int32_t>(context.topK) : 1;
    int32_t const targetRows = activeBatchSize * verifySize;
    bool useDenseTarget = targetNonGreedy && targetTopK <= 0;
    if (useDenseTarget && mBlockDraft.treePolicy == dflash_utils::BlockDraftTreePolicy::kDDTree)
    {
        // The tree accepts through the target's top-k probabilities, so a request without top-k (top-p only) is bounded
        // to the largest supported support: the nucleus of a language model's next-token distribution is far smaller
        // than that, and only a very flat row loses tail mass beyond the 128 most likely tokens.
        static bool const sWarned = [] {
            LOG_WARNING("DFlash2 DDTree: a request without top_k samples from the %d most likely tokens.",
                dflash_utils::kDFlash2MaxSamplingSupport);
            return true;
        }();
        (void) sWarned;
        targetTopK = dflash_utils::kDFlash2MaxSamplingSupport;
        useDenseTarget = false;
    }
    if (useDenseTarget)
    {
        int32_t const vocabSize = mRuntime.deployment.base.outputVocabSize;
        check::check(mRuntime.base.pipelineIO.outputLogits.reshape({targetRows, vocabSize}), "Tensor reshape failed");
        check::check(mTargetProbabilities.reshape({targetRows, vocabSize}), "Tensor reshape failed");
        topPProbabilitiesFromLogits(mRuntime.base.pipelineIO.outputLogits, mTargetProbabilities, context.temperature,
            context.topP, mRuntime.sampling.workspace, context.stream);
        check::check(mTargetProbabilities.reshape({activeBatchSize, verifySize, vocabSize}), "Tensor reshape failed");
    }
    else
    {
        ELLM_CHECK(targetTopK >= 1 && targetTopK <= dflash_utils::kDFlash2MaxSamplingSupport,
            "DFlash2 optimized sparse verification requires bounded sampling support.");
        check::check(mTargetTopKValues.reshape({targetRows, targetTopK}), "Tensor reshape failed");
        check::check(mTargetTopKIds.reshape({targetRows, targetTopK}), "Tensor reshape failed");
        check::check(mTargetTopKProbs.reshape({targetRows, targetTopK}), "Tensor reshape failed");
        selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::ref(mTargetTopKValues), mTargetTopKIds, targetTopK,
            mRuntime.sampling.workspace, context.stream);
        kernel::speculativeNormalizeTopKTopP(mTargetTopKValues, mTargetTopKProbs,
            targetNonGreedy ? context.temperature : 1.0F, targetNonGreedy ? context.topP : 1.0F, context.stream);
        check::check(mTargetTopKProbs.reshape({activeBatchSize, verifySize, targetTopK}), "Tensor reshape failed");
        check::check(mTargetTopKIds.reshape({activeBatchSize, verifySize, targetTopK}), "Tensor reshape failed");
    }
    check::check(mDraftTokenIds.reshape({activeBatchSize, mBlockDraft.proposalLen}), "Tensor reshape failed");
    check::check(mProposalSupportIds.reshape(
                     {activeBatchSize, mBlockDraft.proposalLen, mRuntime.deployment.draft->specSelectorTopK}),
        "Tensor reshape failed");
    check::check(mProposalSupportProbs.reshape(
                     {activeBatchSize, mBlockDraft.proposalLen, mRuntime.deployment.draft->specSelectorTopK}),
        "Tensor reshape failed");
    check::check(mRemainingGenerationLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mHostRemainingGenerationLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostRemainingGenerationLengths = mHostRemainingGenerationLengths.dataPointer<int32_t>();
    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        hostRemainingGenerationLengths[batchIdx]
            = std::max(0, context.maxGenerateLength - context.currentGenerateLengths[batchIdx]);
    }
    CUDA_CHECK(cudaMemcpyAsync(mRemainingGenerationLengths.rawPointer(), hostRemainingGenerationLengths,
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    if (mBlockDraft.treePolicy == dflash_utils::BlockDraftTreePolicy::kDDTree)
    {
        ELLM_CHECK(!useDenseTarget, "DFlash2 DDTree needs a bounded sampling support: set top_k (1..128) or sample greedily.");
        kernel::dflash2TreeSampleAccept(mTargetTopKProbs, mTargetTopKIds, mTreeTokenIds,
            mRuntime.base.pipelineIO.specTreeParentIds, mValidCounts, mAcceptUniforms, mBlockDraft.proposalLen,
            !targetNonGreedy, &mRemainingGenerationLengths, mAcceptedTokenIds, mAcceptedTokenIndices, mAcceptLength,
            context.stream);
        if (!checkCudaLastError("DFlash2 DDTree accept"))
        {
            return false;
        }
        commitAcceptedTreePath(context, verifySize, maxAcceptLength);
        dumpV2Accepted(context, maxAcceptLength);
        return true;
    }
    if (useDenseTarget)
    {
        kernel::speculativeDenseTargetAccept(mTargetProbabilities, mProposalSupportProbs, mProposalSupportIds,
            mDraftTokenIds, mProposalLengths, mAcceptUniforms, mAcceptedTokenIds, mAcceptLength, &mAcceptedTokenIndices,
            context.stream, &mRemainingGenerationLengths);
    }
    else
    {
        kernel::speculativeSparseAccept(mTargetTopKProbs, mTargetTopKIds, mProposalSupportProbs, mProposalSupportIds,
            mDraftTokenIds, mProposalLengths, mAcceptUniforms, mAcceptedTokenIds, mAcceptLength, &mAcceptedTokenIndices,
            context.stream, &mRemainingGenerationLengths);
    }
    if (!checkCudaLastError("DFlash2 accept"))
    {
        return false;
    }
    check::check(mRuntime.base.pipelineIO.baseHiddenStates.reshape(
                     {activeBatchSize, maxAcceptLength, mBlockDraft.baseOutputHiddenDim}),
        "Tensor reshape failed");
    auto& cacheManager = mRuntime.base.cacheManager;
    cacheManager.commitSequenceLength(mAcceptLength, context.stream);
    auto& mambaManager = cacheManager.getMambaCacheManager();
    if (mambaManager.hasIntermediateRecurrentStates() || mambaManager.hasIntermediateConvStates())
    {
        check::check(kernel::gdnTreeChunkVerifyEnabled(verifySize),
            "DFlash2 GDN tree verification exceeds the supported node count");
        mambaManager.replayCommitAcceptedTreeStates(
            mAcceptedTokenIndices, mAcceptLength, mRuntime.base.pipelineIO.stateIndices, context.stream);
    }

    dumpV2Accepted(context, maxAcceptLength);
    return true;
}

void DFlashDecoder::applyV2RequestControls(DecodingInferenceContext& context, int32_t verifySize)
{
    if (context.activeBatchSize != 1)
    {
        return;
    }
    bool const tree = mBlockDraft.treePolicy == dflash_utils::BlockDraftTreePolicy::kDDTree;
    auto& logits = mRuntime.base.pipelineIO.outputLogits;
    auto& budget = context.reasoningBudget;
    if (budget.enabled())
    {
        // Row r of a chain predicts the (r+1)-th next token; in a tree a row at depth d does. Counting ends after
        // `remaining` more tokens, so the rows at depth `remaining` emit the forced token (a forcing that is already
        // running is continued one token per round, in the root row).
        decoder_utils::syncReasoningBudget(context);
        if (auto const forced = budget.forcedToken())
        {
            forceTokenInRow(logits, 0, *forced, context.stream);
        }
        else if (budget.state() == ReasoningBudget::State::kCounting && budget.remaining() < verifySize)
        {
            int32_t const depthToForce = budget.remaining();
            std::vector<int32_t> ids(static_cast<size_t>(verifySize));
            std::vector<int32_t> parents(static_cast<size_t>(verifySize));
            CUDA_CHECK(cudaMemcpyAsync(ids.data(), mVerifyTokenIds.rawPointer(), ids.size() * sizeof(int32_t),
                cudaMemcpyDeviceToHost, context.stream));
            if (tree)
            {
                CUDA_CHECK(cudaMemcpyAsync(parents.data(), mRuntime.base.pipelineIO.specTreeParentIds.rawPointer(),
                    parents.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(context.stream));
            if (!tree)
            {
                for (int32_t i = 0; i < verifySize; ++i)
                {
                    parents[static_cast<size_t>(i)] = i - 1;
                }
            }
            for (int32_t row = 0; row < verifySize; ++row)
            {
                // Depth of the row and whether the thinking already ends inside its root path.
                int32_t depth = 0;
                bool naturalEnd = false;
                bool valid = row == 0 || parents[static_cast<size_t>(row)] >= 0;
                for (int32_t node = row; valid && node > 0 && depth <= verifySize;)
                {
                    naturalEnd = naturalEnd || budget.isEndToken(ids[static_cast<size_t>(node)]);
                    node = parents[static_cast<size_t>(node)];
                    ++depth;
                    valid = node >= 0;
                }
                if (!valid || depth != depthToForce || naturalEnd)
                {
                    continue;
                }
                // The character before the forced token has to be complete; otherwise the budget machine decides in
                // a later round.
                if (budget.tokenEndsUtf8(ids[static_cast<size_t>(row)]))
                {
                    forceTokenInRow(logits, row, budget.firstForced(), context.stream);
                }
            }
        }
    }
    if (context.presencePenalty != 0.0f)
    {
        if (context.presenceFlagsPending)
        {
            CUDA_CHECK(cudaMemsetAsync(mRuntime.sampling.presenceSeen.rawPointer(), 0,
                mRuntime.sampling.presenceSeen.getMemoryCapacity(), context.stream));
            context.presenceFlagsPending = false;
        }
        applyTreePresencePenalty(logits, mVerifyTokenIds, tree ? &mRuntime.base.pipelineIO.specTreeParentIds : nullptr,
            mRuntime.sampling.presenceSeen, verifySize, context.presencePenalty, context.stream);
    }
}

void DFlashDecoder::markV2PresenceSeen(DecodingInferenceContext& context, int32_t verifySize, int32_t maxAcceptLength)
{
    if (context.activeBatchSize != 1 || context.presencePenalty == 0.0f)
    {
        return;
    }
    markTreeSeen(mRuntime.sampling.presenceSeen, mVerifyTokenIds, mAcceptedTokenIds, mAcceptLength, verifySize,
        maxAcceptLength, context.activeBatchSize, context.stream);
}

bool DFlashDecoder::buildV2TreeVerifyInputs(DecodingInferenceContext& context)
{
    NVTX_SCOPED_RANGE(nvtx_dflash2_ddtree_build, "DFlashDecoder::buildV2TreeVerifyInputs", nvtx_colors::LIGHT_ORANGE);
    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const verifySize = mBlockDraft.verifySize;
    int32_t const rows = mBlockDraft.blockSize;
    int32_t const proposalLen = mBlockDraft.proposalLen;
    int32_t const columns = mRuntime.deployment.draft->specSelectorTopK;
    auto& io = mRuntime.base.pipelineIO;

    check::check(mTreeTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mTreeNodeScores.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mValidCounts.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(io.specTreeParentIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(io.specTreeDepths.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(mVerifyTokenIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(io.specDecodePositionIds.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");
    check::check(io.packedAttentionMask.reshape({activeBatchSize, verifySize, static_cast<int64_t>(divUp(verifySize, 32))}),
        "Tensor reshape failed");
    check::check(mVerifyTreeMask.reshape({activeBatchSize, verifySize, verifySize}), "Tensor reshape failed");
    check::check(io.contextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(io.selectTokenIndices.reshape({activeBatchSize, verifySize}), "Tensor reshape failed");

    // Copy the proposal rows into rows 1..proposalLen of the builder's [batch, rows, columns] inputs.
    check::check(mProposalUnaryValues.reshape({activeBatchSize, proposalLen, columns}), "Tensor reshape failed");
    check::check(mProposalSupportIds.reshape({activeBatchSize, proposalLen, columns}), "Tensor reshape failed");
    check::check(mTreeCandidateLogits.reshape({activeBatchSize * rows, columns}), "Tensor reshape failed");
    check::check(mTreeCandidateIds.reshape({activeBatchSize * rows, columns}), "Tensor reshape failed");
    size_t const srcPitch = static_cast<size_t>(proposalLen) * columns * sizeof(float);
    size_t const dstPitch = static_cast<size_t>(rows) * columns * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(mTreeCandidateLogits.rawPointer()) + columns * sizeof(float),
        dstPitch, mProposalUnaryValues.rawPointer(), srcPitch, srcPitch, activeBatchSize, cudaMemcpyDeviceToDevice,
        context.stream));
    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(mTreeCandidateIds.rawPointer()) + columns * sizeof(int32_t),
        dstPitch, mProposalSupportIds.rawPointer(), srcPitch, srcPitch, activeBatchSize, cudaMemcpyDeviceToDevice,
        context.stream));

    Tensor const& baseKVCacheLengths = mRuntime.base.cacheManager.getKVCacheLengths();
    kernel::DDTreeBuildParams const buildParams{
        {mTreeCandidateLogits, mLastAcceptedTokens, baseKVCacheLengths, nullptr, nullptr, 0.0F, &mTreeCandidateIds},
        {mTreeTokenIds, io.specTreeDepths, io.specTreeParentIds, mTreeNodeScores, mValidCounts, mVerifyTokenIds,
            io.specDecodePositionIds, io.packedAttentionMask, mVerifyTreeMask, io.contextLengths,
            io.selectTokenIndices},
        mBlockDraft.candidateTopK, mBuildWorkspace.rawPointer(), static_cast<size_t>(mBuildWorkspace.getMemoryCapacity()),
        context.stream, /*firstCandidateLogitsRow=*/1};
    kernel::ddtreeBuild(buildParams);

    copyVerifyTokenIdsToBaseInput(activeBatchSize, verifySize, context.stream);
    return checkCudaLastError("prepare DFlash2 DDTree verify inputs");
}

namespace
{
FILE* v2DumpFile()
{
    static FILE* file = []() -> FILE* {
        char const* path = std::getenv("EDGELLM_DFLASH2_DUMP");
        return (path != nullptr && path[0] != '\0') ? std::fopen(path, "a") : nullptr;
    }();
    return file;
}

template <typename T>
std::vector<T> copyToHost(Tensor const& tensor, size_t count, cudaStream_t stream)
{
    std::vector<T> host(count);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tensor.rawPointer(), count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}
} // namespace

void DFlashDecoder::dumpV2Proposal(DecodingInferenceContext& context)
{
    FILE* file = v2DumpFile();
    if (file == nullptr || context.activeBatchSize != 1)
    {
        return;
    }
    size_t const len = static_cast<size_t>(mBlockDraft.proposalLen);
    size_t const topK = static_cast<size_t>(mRuntime.deployment.draft->specSelectorTopK);
    auto const anchor = copyToHost<int32_t>(mLastAcceptedTokens, 1, context.stream);
    auto const chain = copyToHost<int32_t>(mDraftTokenIds, len, context.stream);
    auto const ids = copyToHost<int32_t>(mProposalSupportIds, len * topK, context.stream);
    auto const unary = copyToHost<float>(mProposalUnaryValues, len * topK, context.stream);
    std::fprintf(file, "{\"t\":\"P\",\"anchor\":%d,\"chain\":[", anchor[0]);
    for (size_t i = 0; i < len; ++i)
    {
        std::fprintf(file, "%s%d", i ? "," : "", chain[i]);
    }
    std::fprintf(file, "],\"ids\":[");
    for (size_t i = 0; i < len * topK; ++i)
    {
        std::fprintf(file, "%s%d", i ? "," : "", ids[i]);
    }
    std::fprintf(file, "],\"unary\":[");
    for (size_t i = 0; i < len * topK; ++i)
    {
        std::fprintf(file, "%s%.4f", i ? "," : "", unary[i]);
    }
    std::fprintf(file, "]}\n");
    std::fflush(file);
}

void DFlashDecoder::dumpV2Accepted(DecodingInferenceContext& context, int32_t maxAcceptLength)
{
    FILE* file = v2DumpFile();
    if (file == nullptr || context.activeBatchSize != 1)
    {
        return;
    }
    auto const length = copyToHost<int32_t>(mAcceptLength, 1, context.stream);
    int32_t const n = std::min(length[0], maxAcceptLength);
    auto const tokens = copyToHost<int32_t>(mAcceptedTokenIds, static_cast<size_t>(std::max(n, 0)), context.stream);
    std::fprintf(file, "{\"t\":\"A\",\"n\":%d,\"tokens\":[", n);
    for (int32_t i = 0; i < n; ++i)
    {
        std::fprintf(file, "%s%d", i ? "," : "", tokens[static_cast<size_t>(i)]);
    }
    std::fprintf(file, "]}\n");
    std::fflush(file);
}

} // namespace rt
} // namespace trt_edgellm
