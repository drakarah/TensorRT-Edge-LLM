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

#include "runtime/decoding/vanillaDecoder.h"
#include "kernels/speculative/mtpStateScatterKernels.h"
#include "runtime/decoding/draftSource.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/debug/layerDebugger.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/guidedDecoder.h"
#include "runtime/decoding/logitBias.h"
#include "runtime/decoding/requestStableRng.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr int32_t kDecodeProfile{1};

} // namespace

VanillaDecoder::VanillaDecoder(DecodingRuntimeContext& runtime)
    : mRuntime(runtime)
{
    int32_t const engineWindow = runtime.deployment.base.inklingVerifyTokens;
    char const* const mode = std::getenv("EDGELLM_INKLING_SPEC");
    if (engineWindow > 1 && mode != nullptr && mode[0] != '\0')
    {
        std::string const m(mode);
        mChainWindow = engineWindow;
        if (char const* const maxEnv = std::getenv("EDGELLM_INKLING_SPEC_MAX"))
        {
            mChainWindow = std::clamp(std::atoi(maxEnv), 2, engineWindow);
        }
        if (m.rfind("cost:", 0) == 0)
        {
            mChainFixed = std::clamp(std::atoi(m.c_str() + 5), 2, mChainWindow);
        }
        if (char const* const minEnv = std::getenv("EDGELLM_INKLING_LOOKUP_MIN"))
        {
            mChainMinProposal = std::max(1, std::atoi(minEnv));
        }
        mLookupPolicy = std::make_unique<LookupDraftPolicy>(mChainWindow);
        prepareChainTensors();
        if (m == "drafter" || m == "hybrid")
        {
            char const* const file = std::getenv("EDGELLM_INKLING_DRAFTER");
            ELLM_CHECK(file != nullptr, "EDGELLM_INKLING_SPEC=drafter needs EDGELLM_INKLING_DRAFTER=<drafter weights file>");
            mHybrid = m == "hybrid";
            if (char const* const k = std::getenv("EDGELLM_INKLING_DRAFT_K"))
            {
                mAutoK = std::string(k) == "auto";
                mDraftK = mAutoK ? 3 : std::clamp(std::atoi(k), 1, std::min(mChainWindow - 1, 7));
            }
            mDraftK = std::min(mDraftK, mChainWindow - 1);
            mDrafter = createDraftSource(file, runtime.deployment.base.maxKVCacheCapacity, runtime.deployment.base.maxPhysicalTokens, static_cast<cudaStream_t>(nullptr));
            mDrafterIds = Tensor({1, 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "VanillaDecoder::drafterIds");
            mDrafterEmb = Tensor({1, 1, runtime.deployment.base.hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "VanillaDecoder::drafterEmb");
            mDrafterHostTokens = Tensor({8}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "VanillaDecoder::drafterHostTokens");
        }
        LOG_INFO("Inkling chain verification on: mode=%s, window=%d, fixed=%d, min proposal=%d", mode, mChainWindow,
            mChainFixed, mChainMinProposal);
    }
}

void VanillaDecoder::prepareChainTensors()
{
    int32_t const w = mChainWindow;
    mChainArgmax = Tensor({w, 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "VanillaDecoder::chainArgmax");
    mChainHostArgmax = Tensor({w}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "VanillaDecoder::chainHostArgmax");
    mChainAcceptLength = Tensor({1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "VanillaDecoder::chainAcceptLength");
    mChainAcceptedIds = Tensor({1, w}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "VanillaDecoder::chainAcceptedIds");
    mChainHostAcceptLength
        = Tensor({1}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "VanillaDecoder::chainHostAcceptLength");
    mChainHostAcceptedIds
        = Tensor({1, w}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "VanillaDecoder::chainHostAcceptedIds");
}

bool VanillaDecoder::chainApplies(DecodingInferenceContext const& context) const
{
    return mChainWindow > 1 && context.activeBatchSize == 1 && context.generationRound > 0 && !context.hasGuidedDecoding
        && context.numLogprobs == 0 && !context.hasLogitBias && context.presencePenalty == 0.0f
        && !context.reasoningBudget.enabled()
        && !shouldUseNonGreedySampling(context.temperature, context.topK, context.topP);
}

void VanillaDecoder::indexLookupTokens(DecodingInferenceContext& context)
{
    // Outputs of the running request feed the n-gram cache (EDGELLM_INKLING_LOOKUP_PROMPT=<n> also indexes the last n prompt tokens)
    std::vector<int32_t> const& tokens = context.tokenIds[0];
    int32_t const generated = std::min<int32_t>(context.currentGenerateLengths[0], static_cast<int32_t>(tokens.size()));
    mLookupIndexed = std::min(mLookupIndexed, generated);
    if (mLookupIndexed == 0)
    {
        mLookupCache.beginSequence();
        static int32_t const promptTokens = [] {
            char const* const v = std::getenv("EDGELLM_INKLING_LOOKUP_PROMPT");
            return v != nullptr ? std::max(0, std::atoi(v)) : 0;
        }();
        if (promptTokens > 0)
        {
            size_t const promptEnd = tokens.size() - generated;
            size_t const promptBegin = promptEnd > static_cast<size_t>(promptTokens) ? promptEnd - promptTokens : 0;
            for (size_t pos = promptBegin; pos < promptEnd; ++pos)
            {
                mLookupCache.append(tokens[pos]);
            }
        }
    }
    for (size_t pos = tokens.size() - generated + mLookupIndexed; pos < tokens.size(); ++pos)
    {
        mLookupCache.append(tokens[pos]);
    }
    mLookupIndexed = generated;
}

namespace
{
//! EDGELLM_INKLING_DRAFT_DUMP=<prefix>: every context update appends its taps (FP32 rows) to <prefix>.taps and every draft one line "anchorPos anchorToken d0..d6"
//! to <prefix>.drafts, for the offline comparison with the PyTorch drafter (one request per run).
void dumpTaps(float const* devTaps, int32_t rows, cudaStream_t stream)
{
    static char const* const prefix = std::getenv("EDGELLM_INKLING_DRAFT_DUMP");
    if (prefix == nullptr)
    {
        return;
    }
    std::vector<float> host(static_cast<size_t>(rows) * kInklingTapsWidth);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), devTaps, host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    FILE* f = std::fopen((std::string(prefix) + ".taps").c_str(), "ab");
    std::fwrite(host.data(), sizeof(float), host.size(), f);
    std::fclose(f);
}

void dumpDraft(int32_t pos, int32_t anchor, int32_t const* drafts)
{
    static char const* const prefix = std::getenv("EDGELLM_INKLING_DRAFT_DUMP");
    if (prefix == nullptr)
    {
        return;
    }
    FILE* f = std::fopen((std::string(prefix) + ".drafts").c_str(), "a");
    std::fprintf(f, "%d %d", pos, anchor);
    for (int32_t i = 0; i < 7; ++i)
    {
        std::fprintf(f, " %d", drafts[i]);
    }
    std::fprintf(f, "\n");
    std::fclose(f);
}
} // namespace

bool VanillaDecoder::feedDrafterPrefill(DecodingInferenceContext& context, bool lastPass)
{
    if (mDrafter == nullptr || context.activeBatchSize != 1)
    {
        return true;
    }
    int32_t rows = context.effectivePrefillLengths[0];
    int32_t pos0 = context.committedLengths[0] - rows; // the pass has been committed already
    if (!lastPass)
    {
        mDrafterPrefillOpen = true;
    }
    else if (mDrafterPrefillOpen)
    {
        // after a chunked prefill effectivePrefillLengths covers the whole suffix again: the last pass is what the drafter has not seen yet
        pos0 = mDrafter->contextLength();
        rows = context.committedLengths[0] - pos0;
        mDrafterPrefillOpen = false;
    }
    if (pos0 == 0)
    {
        mDrafter->beginRequest();
        mDrafterValid = true;
    }
    if (!mDrafterValid || mDrafter->contextLength() != pos0 || rows <= 0 || rows > mRuntime.deployment.base.maxPhysicalTokens)
    {
        // a reused prefix (or an unexpected pass layout): the drafter lacks those positions, so this request runs without it
        LOG_WARNING("Inkling drafter prefill feed skipped: rows %d, pos0 %d, drafter context %d", rows, pos0, mDrafter->contextLength());
        mDrafterValid = false;
        return true;
    }
    dumpTaps(mRuntime.base.pipelineIO.inklingTaps.dataPointer<float>(), rows, context.stream);
    mDrafter->observeTaps(mRuntime.base.pipelineIO.inklingTaps.dataPointer<float>(), rows, pos0, context.stream);
    return true;
}

bool VanillaDecoder::absorbPrefillPass(DecodingInferenceContext& context)
{
    return feedDrafterPrefill(context, /*lastPass=*/false);
}

bool VanillaDecoder::initializeForGeneration(DecodingInferenceContext& context)
{
    return feedDrafterPrefill(context, /*lastPass=*/true);
}

bool VanillaDecoder::decodeStep(DecodingInferenceContext& context)
{
    int32_t const first = context.committedLengths.empty() ? 0 : context.committedLengths[0];
    bool const ok = chainApplies(context) ? decodeStepChain(context) : decodeStepPlain(context);
    if (ok && mDrafter != nullptr && mDrafterValid && context.activeBatchSize == 1)
    {
        // the rows this step committed (the committed token and the accepted drafts) are the first rows of the taps it produced
        int32_t const rows = context.committedLengths[0] - first;
        if (rows > 0 && mDrafter->contextLength() == first)
        {
            dumpTaps(mRuntime.base.pipelineIO.inklingTaps.dataPointer<float>(), rows, context.stream);
            mDrafter->observeTaps(mRuntime.base.pipelineIO.inklingTaps.dataPointer<float>(), rows, first, context.stream);
        }
        else if (rows > 0)
        {
            LOG_WARNING("Inkling drafter lost sync (rows %d, first %d, drafter context %d): drafting off for this request", rows, first, mDrafter->contextLength());
            mDrafterValid = false;
        }
    }
    return ok;
}

bool VanillaDecoder::runChainPass(std::vector<int32_t> const& ids, ResidentRef ref, RequestId requestId,
    int32_t pastLength, RaggedBatchBuilder& builder, RaggedExecutionBatch& batch, StepId stepId, cudaStream_t stream,
    bool captureOnly)
{
    int32_t const window = static_cast<int32_t>(ids.size());
    auto& io = mRuntime.base.pipelineIO;
    ScheduledStep step;
    step.id = stepId;
    step.sequences.push_back(ScheduledSequence{SequenceIdentity{requestId, ref}, SequenceWork::kContext, window,
        pastLength, HostTokenRange{ids.data(), window, 0, window}, /*selectLastTokenLogits=*/false});
    io.waitForStepHostStaging();
    builder.buildInto(step, batch);
    batch.logitsIndices.resize(static_cast<size_t>(window));
    for (int32_t i = 0; i < window; ++i)
    {
        batch.logitsIndices[static_cast<size_t>(i)] = i;
    }
    batch.logitsToSequence.assign(static_cast<size_t>(window), 0);
    batch.shape.numLogits = window;

    check::check(mRuntime.preprocess.idsInput.reshape({1, window}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), batch.hostTokenIds.rawPointer(),
        window * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    io.recordStepHostUploads(stream);
    check::check(io.inputsEmbeds.reshape({1, window, mRuntime.deployment.base.hiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), io.inputsEmbeds, stream);
    prepareRaggedExecutionBindings(
        io, mRuntime.base.sharedResources, mRuntime.deployment.base, batch, /*kvCacheIndex=*/0, stream);
    check::check(io.contextSequenceCountCarrier.reshape({0}), "Tensor reshape failed");
    check::check(io.inputsEmbeds.reshape({window, mRuntime.deployment.base.hiddenSize}), "Tensor reshape failed");
    check::check(io.outputLogits.reshape({window, mRuntime.deployment.base.outputVocabSize}), "Tensor reshape failed");
    if (!io.inklingTaps.isEmpty())
    {
        check::check(io.inklingTaps.reshape({window, kInklingTapsWidth}), "Tensor reshape failed");
    }
    mRuntime.base.cacheManager.getMambaCacheManager().reshapeIntermediateStates(1, window);
    auto const dims = mRuntime.deployment.base.specVerifyDims(1, window);
    if (captureOnly)
    {
        return mRuntime.base.captureGraph(dims, stream);
    }
    bool ok = mRuntime.base.executor.prepare(kDecodeProfile, dims, mRuntime.base.tensorMap, stream);
    if (ok)
    {
        ok = mRuntime.base.executor.execute(stream);
    }
    return ok;
}

bool VanillaDecoder::decodeStepChain(DecodingInferenceContext& context)
{
    auto const roundStart = std::chrono::steady_clock::now();
    std::vector<int32_t> const& tokens = context.tokenIds[0];
    indexLookupTokens(context);
    int32_t match = 0;
    std::vector<int32_t> proposal
        = mLookupCache.proposeWithMatch(tokens.data(), static_cast<int32_t>(tokens.size()), mChainWindow - 1, match);
    int32_t window = 1;
    bool lookupPick = false;
    bool drafterPick = false;
    if (mDrafter != nullptr && mDrafterValid && mDrafter->contextLength() == context.committedLengths[0]
        && !(mHybrid && static_cast<int32_t>(proposal.size()) >= mChainMinProposal && match >= 6))
    {
        // anchor embedding -> drafter
        cudaStream_t const s0 = context.stream;
        int32_t const anchor = tokens.back();
        CUDA_CHECK(cudaMemcpyAsync(mDrafterIds.rawPointer(), &anchor, sizeof(int32_t), cudaMemcpyHostToDevice, s0));
        CUDA_CHECK(cudaStreamSynchronize(s0)); // `anchor` is a local
        kernel::embeddingLookup(mDrafterIds, mRuntime.preprocess.embedding.table, mRuntime.preprocess.embedding.scalesAsOptional(), mDrafterEmb, s0);
        mDrafter->draft(mDrafterEmb.rawPointer(), anchor, mDrafterHostTokens.dataPointer<int32_t>(), s0);
        CUDA_CHECK(cudaStreamSynchronize(s0));
        dumpDraft(mDrafter->contextLength(), anchor, mDrafterHostTokens.dataPointer<int32_t>());
        int32_t k = mDraftK;
        if (mAutoK)
        {
            // verify cost per window (measured on this engine, ms) and the running prefix acceptance per draft position: the K with the most tokens per ms
            static constexpr std::array<double, 9> kVerifyMs{0.0, 38.5, 49.0, 60.7, 66.8, 87.0, 92.0, 96.7, 101.0};
            double best = 1.0 / kVerifyMs[1];
            k = 0;
            double expected = 1.0;
            for (int32_t kk = 1; kk <= std::min(mChainWindow - 1, 7); ++kk)
            {
                expected += mPrefixAcceptance[static_cast<size_t>(kk - 1)];
                double const rate = expected / kVerifyMs[static_cast<size_t>(kk + 1)];
                if (rate > best)
                {
                    best = rate;
                    k = kk;
                }
            }
        }
        if (k == 0)
        {
            drafterPick = false; // not worth a verify window: plain step
            lookupPick = false;
        }
        else
        {
            proposal.assign(mDrafterHostTokens.dataPointer<int32_t>(), mDrafterHostTokens.dataPointer<int32_t>() + k);
            window = k + 1;
            lookupPick = true;
            drafterPick = true;
        }
    }
    else if (mChainFixed > 0)
    {
        window = mChainFixed;
        // fillers when the lookup has less: ordinary tokens of the context at scattered positions (distinct tokens route to distinct experts, as real drafts do);
        // only the pass cost is of interest, so these are almost never accepted
        while (static_cast<int32_t>(proposal.size()) < window - 1)
        {
            uint64_t const h = (static_cast<uint64_t>(mStatRounds) * 2654435761ULL + proposal.size() * 40503ULL + 12345ULL) * 11400714819323198485ULL;
            proposal.push_back(tokens[static_cast<size_t>((h >> 20) % tokens.size())]);
        }
        proposal.resize(static_cast<size_t>(window - 1));
        lookupPick = true;
    }
    else if (static_cast<int32_t>(proposal.size()) >= mChainMinProposal)
    {
        auto const pick = mLookupPolicy->choose(1, static_cast<int32_t>(proposal.size()), match);
        if (pick.lookup)
        {
            window = pick.window;
            proposal.resize(static_cast<size_t>(window - 1));
            lookupPick = true;
        }
    }
    if (!lookupPick)
    {
        bool const ok = decodeStepPlain(context);
        double const ms
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - roundStart).count();
        mLookupPolicy->observe(false, 1, 0, 0, ms);
        mStatRounds += 1;
        mStatTokens += 1;
        mStatWindowRounds[1] += 1;
        mStatWindowMs[1] += ms;
        return ok;
    }

    cudaStream_t const stream = context.stream;
    std::vector<int32_t> ids(static_cast<size_t>(window));
    ids[0] = tokens.back();
    std::copy(proposal.begin(), proposal.end(), ids.begin() + 1);
    check::check(context.raggedBatchBuilder.has_value(), "Ragged batch builder is not initialized");
    if (!runChainPass(ids, context.residentRefs[0], context.requestIds[0], context.committedLengths[0],
            *context.raggedBatchBuilder, context.raggedExecutionBatch, context.nextStepId++, stream, false))
    {
        LOG_ERROR("Failed to execute the Inkling chain verification step.");
        return false;
    }
    auto& io = mRuntime.base.pipelineIO;
    check::check(mChainArgmax.reshape({window, 1}), "Tensor reshape failed");
    selectAllTopK(io.outputLogits, std::nullopt, mChainArgmax, 1, mRuntime.sampling.workspace, stream);
    check::check(mChainHostArgmax.reshape({window}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mChainHostArgmax.rawPointer(), mChainArgmax.rawPointer(), window * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int32_t const* const argmax = mChainHostArgmax.dataPointer<int32_t>();
    int32_t accepted = 0;
    while (accepted < window - 1 && ids[static_cast<size_t>(accepted + 1)] == argmax[accepted])
    {
        ++accepted;
    }
    int32_t const count = accepted + 1; // verified rows that stay: the committed token and the accepted drafts; the bonus token is the next anchor
    check::check(mChainAcceptLength.reshape({1}), "Tensor reshape failed");
    check::check(mChainAcceptedIds.reshape({1, window}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mChainAcceptLength.rawPointer(), &count, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mChainAcceptedIds.rawPointer(), argmax, count * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // the host values live in tensors of this decoder: nothing may read them after this call returns
    decoder_utils::clampAcceptLengthsToRemainingGeneration(context, mChainAcceptLength, stream);
    mRuntime.base.cacheManager.commitSequenceLength(mChainAcceptLength, stream);
    mRuntime.base.cacheManager.getMambaCacheManager().scatterAcceptedConvStatesOnly(
        mChainAcceptLength, io.stateIndices, window, stream);
    decoder_utils::appendAcceptedTokens(context, mChainHostAcceptLength, mChainHostAcceptedIds, mChainAcceptLength,
        mChainAcceptedIds, window, mRuntime.tokenizer, stream, window - 1);

    double const ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - roundStart).count();
    mLookupPolicy->observe(true, window, accepted, match, ms);
    mStatRounds += 1;
    mStatChainRounds += 1;
    mStatTokens += count;
    mStatAccepted += accepted;
    mStatWindowRounds[static_cast<size_t>(window)] += 1;
    mStatWindowMs[static_cast<size_t>(window)] += ms;
    mStatWindowAccepted[static_cast<size_t>(window)] += accepted;
    if (drafterPick)
    {
        for (int32_t j = 0; j < window - 1; ++j)
        {
            mPrefixAcceptance[static_cast<size_t>(j)] += 0.02 * ((accepted > j ? 1.0 : 0.0) - mPrefixAcceptance[static_cast<size_t>(j)]);
        }
        mStatDraftRounds += 1;
        mStatDraftAccepted += accepted;
    }
    if (mStatRounds % 16 == 0 || context.finishedStates[0])
    {
        std::string perWindow;
        for (int32_t w = 1; w <= mChainWindow; ++w)
        {
            if (mStatWindowRounds[static_cast<size_t>(w)] > 0)
            {
                perWindow += format::fmtstr(" w%d:%lldx %.1fms acc %.2f", w,
                    static_cast<long long>(mStatWindowRounds[static_cast<size_t>(w)]),
                    mStatWindowMs[static_cast<size_t>(w)] / mStatWindowRounds[static_cast<size_t>(w)],
                    static_cast<double>(mStatWindowAccepted[static_cast<size_t>(w)])
                        / mStatWindowRounds[static_cast<size_t>(w)]);
            }
        }
        LOG_INFO("Inkling chain: %lld rounds (%lld verified windows, %lld drafted with %.2f accepted each), %.2f tokens per round;%s",
            static_cast<long long>(mStatRounds), static_cast<long long>(mStatChainRounds), static_cast<long long>(mStatDraftRounds),
            mStatDraftRounds > 0 ? static_cast<double>(mStatDraftAccepted) / mStatDraftRounds : 0.0,
            static_cast<double>(mStatTokens) / mStatRounds, perWindow.c_str());
    }
    return true;
}

bool VanillaDecoder::decodeStepPlain(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kLLM_GENERATION, context.stream);
    NVTX_SCOPED_RANGE(nvtx_vanilla_decoding,
        ("VANILLA_DECODING[R" + std::to_string(context.generationRound) + "," + std::to_string(context.activeBatchSize)
            + "]")
            .c_str(),
        nvtx_colors::BLUE);

    int32_t const activeBatchSize = context.activeBatchSize;
    StepId raggedStepId{0};
    context.scheduledStep.id = context.nextStepId++;
    context.scheduledStep.sequences.resize(static_cast<size_t>(activeBatchSize));
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::vector<int32_t> const& tokens = context.tokenIds[static_cast<size_t>(i)];
        context.scheduledStep.sequences[static_cast<size_t>(i)] = ScheduledSequence{
            SequenceIdentity{context.requestIds[static_cast<size_t>(i)], context.residentRefs[static_cast<size_t>(i)]},
            SequenceWork::kDecode, 1, context.committedLengths[static_cast<size_t>(i)],
            HostTokenRange{
                tokens.data(), static_cast<int32_t>(tokens.size()), static_cast<int32_t>(tokens.size()) - 1, 1},
            true};
    }
    check::check(context.raggedBatchBuilder.has_value(), "Ragged batch builder is not initialized");
    mRuntime.base.pipelineIO.waitForStepHostStaging();
    RaggedBatchBuilder::validateRuntimeAdapterStep(
        context.scheduledStep, context.activeBatchSize, context.requestIds, context.residentRefs);
    context.raggedBatchBuilder->buildInto(context.scheduledStep, context.raggedExecutionBatch);
    RaggedExecutionBatch const& batch = context.raggedExecutionBatch;
    raggedStepId = context.scheduledStep.id;

    check::check(mRuntime.preprocess.idsInput.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mRuntime.preprocess.idsInput.rawPointer(), batch.hostTokenIds.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));
    mRuntime.base.pipelineIO.recordStepHostUploads(context.stream);

    check::check(
        mRuntime.base.pipelineIO.inputsEmbeds.reshape({activeBatchSize, 1, mRuntime.deployment.base.hiddenSize}),
        "Tensor reshape failed");
    kernel::embeddingLookup(mRuntime.preprocess.idsInput, mRuntime.preprocess.embedding.table,
        mRuntime.preprocess.embedding.scalesAsOptional(), mRuntime.base.pipelineIO.inputsEmbeds, context.stream);
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->embed(mRuntime.preprocess.idsInput, context.stream);
    }
    if (mRuntime.preprocess.gemma4Ple)
    {
        mRuntime.preprocess.gemma4Ple->reshapeOutputsTokenMajor(batch.shape.physicalTokens);
    }
    prepareRaggedExecutionBindings(mRuntime.base.pipelineIO, mRuntime.base.sharedResources, mRuntime.deployment.base,
        batch, /*kvCacheIndex=*/0, context.stream);
    check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape(
                     {batch.shape.physicalTokens, mRuntime.deployment.base.hiddenSize}),
        "Ragged decode embedding reshape failed");

    check::check(
        mRuntime.base.pipelineIO.outputLogits.reshape({activeBatchSize, mRuntime.deployment.base.outputVocabSize}),
        "Tensor reshape failed");

    mRuntime.preprocess.stepPreparer.prepare(
        InferencePhase::kDecode, activeBatchSize, mRuntime.base.cacheManager, mRuntime.base.pipelineIO, context.stream);
    mRuntime.base.pipelineIO.recordStepHostUploads(context.stream);
    if (mRuntime.preprocess.deepstack)
    {
        mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
    }

    auto const decodeDims = mRuntime.deployment.base.decodeDims(activeBatchSize);
    bool decodingStatus
        = mRuntime.base.executor.prepare(kDecodeProfile, decodeDims, mRuntime.base.tensorMap, context.stream);
    if (decodingStatus)
    {
        decodingStatus = mRuntime.base.executor.execute(context.stream);
    }
    if (decodingStatus)
    {
        context.completionSequences.resize(static_cast<size_t>(activeBatchSize));
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            context.completionSequences[static_cast<size_t>(i)]
                = CompletionSequence{context.requestIds[static_cast<size_t>(i)],
                    context.residentRefs[static_cast<size_t>(i)], context.committedLengths[static_cast<size_t>(i)]};
        }
        context.raggedExecutionBatch.validateCommitSnapshot(raggedStepId, context.completionSequences);
        mRuntime.base.cacheManager.commitSequenceLength(/*increment=*/1, context.stream);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            ++context.committedLengths[static_cast<size_t>(i)];
        }
    }
    if (!decodingStatus)
    {
        LOG_ERROR("Failed to execute vanilla decoding step for base model.");
        return false;
    }

    // GCOVR_EXCL_START
    if (context.hasLogitBias)
    {
        applyLogitBias(mRuntime.logitBias, mRuntime.base.pipelineIO.outputLogits, context, context.stream);
    }
    // GCOVR_EXCL_STOP

    if (context.presencePenalty != 0.0f)
    {
        if (context.presenceFlagsPending)
        {
            CUDA_CHECK(cudaMemsetAsync(mRuntime.sampling.presenceSeen.rawPointer(), 0,
                mRuntime.sampling.presenceSeen.getMemoryCapacity(), context.stream));
            context.presenceFlagsPending = false;
        }
        applyPresencePenalty(mRuntime.base.pipelineIO.outputLogits, mRuntime.preprocess.idsInput,
            mRuntime.sampling.presenceSeen, context.presencePenalty, context.stream);
    }

    if (context.hasGuidedDecoding)
    {
        // One logits row per active slot. Sits after the forward was enqueued, so the host-side
        // mask fill overlaps it; the stream keeps the ordering.
        applyGuidedDecodingMask(mRuntime.guidedDecoder, context, mRuntime.base.pipelineIO.outputLogits, activeBatchSize,
            /*rowsPerSlot=*/1, context.stream);
    }

    if (context.reasoningBudget.enabled())
    {
        decoder_utils::syncReasoningBudget(context);
        if (auto const forced = context.reasoningBudget.forcedToken())
        {
            forceToken(mRuntime.base.pipelineIO.outputLogits, *forced, context.stream);
        }
    }

    check::check(mRuntime.sampling.indices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        constexpr uint64_t kSAMPLING_SEED = 42U;
        constexpr uint64_t kSAMPLING_OFFSET = 0U;
        constexpr uint64_t kRANDOM_LANE = 0U;
        SamplingParams params(activeBatchSize, mRuntime.deployment.base.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        rt::Tensor const* rowUniforms{nullptr};
        if (context.useRequestStableSampling)
        {
            check::check(mRuntime.sampling.hostUniforms.reshape({activeBatchSize}), "Tensor reshape failed");
            check::check(mRuntime.sampling.uniforms.reshape({activeBatchSize}), "Tensor reshape failed");
            for (int32_t batch = 0; batch < activeBatchSize; ++batch)
            {
                uint64_t const position = requestStableNextAbsolutePosition(
                    context.rawBatchedInputIds[batch].size(), context.currentGenerateLengths[batch]);
                mRuntime.sampling.hostUniforms.dataPointer<float>()[batch] = requestStableUniform(
                    context.samplingSeeds[batch], position, SpecRandomPurpose::kTarget, kRANDOM_LANE);
            }
            CUDA_CHECK(
                cudaMemcpyAsync(mRuntime.sampling.uniforms.rawPointer(), mRuntime.sampling.hostUniforms.rawPointer(),
                    activeBatchSize * sizeof(float), cudaMemcpyHostToDevice, context.stream));
            rowUniforms = &mRuntime.sampling.uniforms;
        }
        topKtopPSamplingFromLogits(mRuntime.base.pipelineIO.outputLogits, mRuntime.sampling.indices, params,
            mRuntime.sampling.workspace, context.stream, kSAMPLING_SEED, kSAMPLING_OFFSET, rowUniforms);
    }
    else
    {
        constexpr int32_t kSAMPLING_TOP_K = 1;
        selectAllTopK(mRuntime.base.pipelineIO.outputLogits, std::nullopt, mRuntime.sampling.indices, kSAMPLING_TOP_K,
            mRuntime.sampling.workspace, context.stream);
    }

    // Capture the indices while still in the output vocabulary, the space the matchers work
    // in. Rides the round's single sync below.
    if (context.hasGuidedDecoding)
    {
        check::check(mRuntime.sampling.hostOutputSpaceIds.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mRuntime.sampling.hostOutputSpaceIds.dataPointer<int32_t>(),
            mRuntime.sampling.indices.rawPointer(), activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost,
            context.stream));
    }

    if (mRuntime.deployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(mRuntime.sampling.indices, mRuntime.sampling.baseVocabMappingTable, context.stream);
    }

    if (mRuntime.tokenBroadcast
        && !mRuntime.tokenBroadcast(mRuntime.sampling.indices.rawPointer(), activeBatchSize, context.stream))
    {
        LOG_ERROR("Failed to broadcast vanilla decode sampled tokens for parallel rank %d.", mRuntime.parallelRank);
        return false;
    }

    // Enqueue logprobs extraction + D2H before the round's single synchronization so the
    // copies ride the same sync as the sampled-token D2H below.
    if (context.numLogprobs > 0)
    {
        decoder_utils::enqueueLogprobsD2H(
            mRuntime.base.pipelineIO.outputLogits, activeBatchSize, mRuntime, context.numLogprobs, context.stream);
    }

    check::check(mRuntime.sampling.hostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mRuntime.sampling.hostSelectedTokenIds.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mRuntime.sampling.indices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Few-layer-validation debug: dump this decode round. The KV cache is committed (line above) so
    // tokenIds[i].size() == the committed cache length for this round.
    if (context.layerDebugger != nullptr)
    {
        std::vector<int32_t> validLengths(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            validLengths[i] = static_cast<int32_t>(context.tokenIds[i].size());
        }
        context.layerDebugger->dumpRound(mRuntime.base.cacheManager, *mRuntime.base.sharedResources.kvPageTables[0],
            mRuntime.base.sharedResources.getSwaKVPageTable(0), mRuntime.base.pipelineIO.outputLogits, validLengths,
            context.batchIndexMapping, context.residentRefs, hostSelectedTokenIdsData, activeBatchSize, context.stream);

        // Teacher-forcing — feed the golden's tokens instead of our own (no-op unless
        // EDGELLM_FORCE_TOKENS_FILE is set). After the dump, so the dump keeps our own sampled token.
        context.layerDebugger->applyForcedTokens(
            context.currentGenerateLengths, context.batchIndexMapping, hostSelectedTokenIdsData, activeBatchSize);
    }

    decoder_utils::appendSampledTokens(context, hostSelectedTokenIdsData, activeBatchSize);

    if (context.hasGuidedDecoding)
    {
        advanceGuidedDecoding(mRuntime.guidedDecoder, context,
            mRuntime.sampling.hostOutputSpaceIds.dataPointer<int32_t>(), activeBatchSize);
    }

    if (context.numLogprobs > 0)
    {
        decoder_utils::collectLogprobsFromHost(mRuntime, context, activeBatchSize, context.numLogprobs);
    }

    return true;
}

bool VanillaDecoder::captureCudaGraphs(cudaStream_t stream)
{
    bool baseVanillaDecodingCaptureStatus{true};
    RaggedEngineContract const contract{TokenLayoutBackend::kEntryPaddedCompatibility,
        mRuntime.deployment.base.maxNumSequences, mRuntime.deployment.base.maxQueryLength,
        mRuntime.deployment.base.maxPhysicalTokens, mRuntime.deployment.base.recurrentPoolRows,
        /* mixedStepSupported = */ false};
    RaggedBatchBuilder builder(contract);
    RaggedExecutionBatch batch;
    builder.reserve(batch);
    for (int32_t batchSize = 1; batchSize <= mRuntime.maxRuntimeBatchSize; ++batchSize)
    {
        check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, 1, mRuntime.deployment.base.hiddenSize}),
            "Tensor reshape failed");
        check::check(
            mRuntime.base.pipelineIO.outputLogits.reshape({batchSize, mRuntime.deployment.base.outputVocabSize}),
            "Tensor reshape failed");
        check::check(mRuntime.base.pipelineIO.selectTokenIndices.reshape({batchSize, 1}), "Tensor reshape failed");
        check::check(mRuntime.preprocess.idsInput.reshape({batchSize, 1}), "Tensor reshape failed");
        std::vector<int32_t> dummyTokens(static_cast<size_t>(batchSize), 0);
        std::vector<ScheduledSequence> sequences(static_cast<size_t>(batchSize));
        for (int32_t i = 0; i < batchSize; ++i)
        {
            sequences[static_cast<size_t>(i)]
                = ScheduledSequence{SequenceIdentity{static_cast<RequestId>(i) + 1, ResidentRef{i, 1}},
                    SequenceWork::kDecode, 1, 0, HostTokenRange{dummyTokens.data() + i, 1, 0, 1}, true};
        }
        ScheduledStep const step{1, std::move(sequences)};
        builder.buildInto(step, batch);
        if (mRuntime.preprocess.gemma4Ple)
        {
            mRuntime.preprocess.gemma4Ple->reshapeOutputsTokenMajor(batch.shape.physicalTokens);
        }
        prepareRaggedExecutionBindings(mRuntime.base.pipelineIO, mRuntime.base.sharedResources,
            mRuntime.deployment.base, batch, /*kvCacheIndex=*/0, stream);
        check::check(mRuntime.base.pipelineIO.inputsEmbeds.reshape({batchSize, mRuntime.deployment.base.hiddenSize}),
            "Ragged graph-capture embedding reshape failed");

        mRuntime.preprocess.stepPreparer.prepare(
            InferencePhase::kDecode, batchSize, mRuntime.base.cacheManager, mRuntime.base.pipelineIO, stream);
        if (mRuntime.preprocess.deepstack)
        {
            mRuntime.preprocess.deepstack->useZeroTarget(mRuntime.base.tensorMap);
        }

        auto const decodeDims = mRuntime.deployment.base.decodeDims(batchSize);
        baseVanillaDecodingCaptureStatus &= mRuntime.base.captureGraph(decodeDims, stream);
    }
    if (mChainWindow > 1 && mRuntime.maxRuntimeBatchSize >= 1)
    {
        // chain verification: one graph per window size
        RaggedBatchBuilder chainBuilder(contract);
        RaggedExecutionBatch chainBatch;
        chainBuilder.reserve(chainBatch);
        for (int32_t window = 2; window <= mChainWindow; ++window)
        {
            std::vector<int32_t> const dummy(static_cast<size_t>(window), 0);
            baseVanillaDecodingCaptureStatus &= runChainPass(dummy, ResidentRef{0, 1}, 1, 0, chainBuilder, chainBatch, 1,
                stream, /*captureOnly=*/true);
        }
    }
    return baseVanillaDecodingCaptureStatus;
}

} // namespace rt
} // namespace trt_edgellm
