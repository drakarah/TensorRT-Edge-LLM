/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/decoding/inklingDflash2Drafter.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/inklingDrafter/inklingDflash2Kernels.h"
#include "kernels/inklingDrafter/inklingDrafterKernels.h"
#include "kernels/speculative/dflash2CandidateSelector.h"
#include "kernels/speculative/dflash2GroupedDynamicConv.h"
#include "sampler/sampling.h"

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr float kTheta = 10000.F;
constexpr int32_t kVocab = 200058;
constexpr int32_t kVocabPadded = 200064;
constexpr int32_t kWindow = 2048;
constexpr int32_t kMaxSplits = 32;
constexpr int32_t kH = 2304;
constexpr int32_t kGroups = 144; //!< 2304 / group size 16
} // namespace

Tensor const& InklingDflash2Drafter::w(std::string const& name) const
{
    auto const it = mW.find(name);
    ELLM_CHECK(it != mW.end(), "Inkling DFlash2 drafter weight missing: " + name);
    return it->second;
}

Tensor InklingDflash2Drafter::make(std::vector<int64_t> const& dims, nvinfer1::DataType type, std::string const& name) const
{
    nvinfer1::Dims shape{};
    shape.nbDims = static_cast<int32_t>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i)
    {
        shape.d[i] = dims[i];
    }
    return Tensor(Coords(shape), DeviceType::kGPU, type, "InklingDflash2Drafter::" + name);
}

InklingDflash2Drafter::InklingDflash2Drafter(std::filesystem::path const& weightsFile, int32_t capacity, int32_t maxAppendRows, cudaStream_t stream)
    : mCapacity(capacity)
    , mMaxAppendRows(maxAppendRows)
{
    std::vector<Tensor> tensors;
    ELLM_CHECK(safetensors::loadSafetensors(weightsFile, tensors, stream), "Failed to load the Inkling DFlash2 drafter weights: " + weightsFile.string());
    for (Tensor& t : tensors)
    {
        std::string const name = t.getName();
        mW.emplace(name, std::move(t));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    using DT = nvinfer1::DataType;
    mScoreStride = ((kWindow + kBlock + 7) / 8) * 8;
    for (int32_t i = 0; i < kLayers; ++i)
    {
        mCtxK[i] = make({mCapacity, 1024}, DT::kHALF, "ctxK" + std::to_string(i));
        mCtxV[i] = make({mCapacity, 1024}, DT::kHALF, "ctxV" + std::to_string(i));
    }
    mCtxIn = make({mMaxAppendRows, 4 * kH}, DT::kHALF, "ctxIn");
    mFcOut = make({mMaxAppendRows, kH}, DT::kHALF, "fcOut");
    mDelta = make({mMaxAppendRows, kH}, DT::kHALF, "delta");
    mKRaw = make({mMaxAppendRows, 1024}, DT::kHALF, "kRaw");
    mVRaw = make({mMaxAppendRows, 1024}, DT::kHALF, "vRaw");
    mXf = make({1, kBlock, kH}, DT::kFLOAT, "x");
    mH = make({1, kBlock, kH}, DT::kHALF, "h");
    mCoPre = make({1, kBlock, 2, kGroups}, DT::kHALF, "coPre");
    mCoPost = make({1, kBlock, 2, kGroups}, DT::kHALF, "coPost");
    mHc = make({1, kBlock, kH}, DT::kHALF, "hc");
    mQRaw = make({kBlock, 4096}, DT::kHALF, "qRaw");
    mKBRaw = make({kBlock, 1024}, DT::kHALF, "kbRaw");
    mVBRaw = make({kBlock, 1024}, DT::kHALF, "vbRaw");
    mQ = make({kBlock, 4096}, DT::kHALF, "q");
    mKb = make({kBlock, 1024}, DT::kHALF, "kb");
    mVb = make({kBlock, 1024}, DT::kHALF, "vb");
    mAttn = make({1, kBlock, 4096}, DT::kHALF, "attn");
    mO = make({1, kBlock, kH}, DT::kHALF, "o");
    mG = make({kBlock, 6912}, DT::kHALF, "g");
    mU = make({kBlock, 6912}, DT::kHALF, "u");
    mAct = make({kBlock, 6912}, DT::kHALF, "act");
    mDown = make({1, kBlock, kH}, DT::kHALF, "down");
    mHidN = make({kBlock, kH}, DT::kHALF, "hidN");
    mScores = make({8 * 32, mScoreStride}, DT::kFLOAT, "scores");
    mPartial = make({kMaxSplits, 8 * 32, 128}, DT::kFLOAT, "partial");
    mProj = make({kDrafts, 256}, DT::kHALF, "proj");
    mProjected = make({1, kDrafts, 256}, DT::kHALF, "projected");
    mLogits = make({kDrafts, kVocabPadded}, DT::kHALF, "logits");
    mLogitsF = make({kDrafts, kVocabPadded}, DT::kFLOAT, "logitsF");
    mTopVals = make({kDrafts, kTopK}, DT::kFLOAT, "topVals");
    mTopIdx = make({kDrafts, kTopK}, DT::kINT32, "topIdx");
    mCand = make({1, kDrafts, kTopK}, DT::kINT32, "cand");
    mUnary = make({1, kDrafts, kTopK}, DT::kFLOAT, "unary");
    size_t const workspace = getSelectAllTopKWorkspaceSize(kDrafts, kVocabPadded, kTopK);
    mTopWorkspace = make({static_cast<int64_t>(std::max<size_t>(workspace, 1))}, DT::kINT8, "topWorkspace");
    mAnchorTok = make({1}, DT::kINT32, "anchorTok");
    mUniforms = make({1, kDrafts}, DT::kFLOAT, "uniforms");
    mTemperatures = make({1}, DT::kFLOAT, "temperatures");
    mGreedy = make({1}, DT::kINT32, "greedy");
    mDraftTok = make({1, kDrafts}, DT::kINT32, "draftTok");
    mProbs = make({1, kDrafts, kTopK}, DT::kFLOAT, "probs");
    float const one = 1.F;
    int32_t const greedy = 1;
    CUDA_CHECK(cudaMemsetAsync(mUniforms.rawPointer(), 0, kDrafts * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(mTemperatures.rawPointer(), &one, sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mGreedy.rawPointer(), &greedy, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_INFO("Inkling DFlash2 drafter loaded from %s (%zu tensors, context capacity %d)", weightsFile.c_str(), mW.size(), mCapacity);
}

//! EDGELLM_INKLING_DRAFT_DEBUG=<prefix>: the first draft call writes <prefix>.<tag> with the raw bytes of selected tensors (offline comparison).
void InklingDflash2Drafter::debugDump(Tensor const& t, std::string const& tag, cudaStream_t stream)
{
    static char const* const prefix = std::getenv("EDGELLM_INKLING_DRAFT_DEBUG");
    if (prefix == nullptr || mDebugDone)
    {
        return;
    }
    size_t const bytes = static_cast<size_t>(t.getMemoryCapacity());
    std::vector<char> host(bytes);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host.data(), t.rawPointer(), bytes, cudaMemcpyDeviceToHost));
    FILE* f = std::fopen((std::string(prefix) + "." + tag).c_str(), "wb");
    std::fwrite(host.data(), 1, bytes, f);
    std::fclose(f);
}

void InklingDflash2Drafter::linear(std::string const& name, void const* x, void* y, int32_t rows, int32_t n, int32_t k, cudaStream_t stream) const
{
    for (int32_t r = 0; r < rows; r += 8)
    {
        int32_t const m = std::min(8, rows - r);
        kernel::launchDrafterGemv(static_cast<__half const*>(x) + static_cast<size_t>(r) * k, w(name).rawPointer(), w(name + ".s").rawPointer(),
            static_cast<__half*>(y) + static_cast<size_t>(r) * n, m, n, k, stream);
    }
}

void InklingDflash2Drafter::convPrepare(std::string const& prefix, int32_t layer, Tensor const& in, Tensor& out, cudaStream_t stream)
{
    std::string const p = "l" + std::to_string(layer) + "." + prefix;
    // the coefficient projection is stored as two halves ([288, 2304] each): the pre-side and the post-side kernels of the convolution, both from the same input
    linear(p + "_pre", in.rawPointer(), mCoPre.rawPointer(), kBlock, 2 * kGroups, kH, stream);
    linear(p + "_post", in.rawPointer(), mCoPost.rawPointer(), kBlock, 2 * kGroups, kH, stream);
    CUDA_CHECK(kernel::launchDFlash2GroupedDynamicConv(in, mCoPre, w(p + "_base0"), std::nullopt, out, stream));
}

void InklingDflash2Drafter::convFinish(std::string const& prefix, int32_t layer, Tensor const& hidden, cudaStream_t stream)
{
    std::string const p = "l" + std::to_string(layer) + "." + prefix;
    CUDA_CHECK(kernel::launchDFlash2GroupedDynamicConv(hidden, mCoPost, w(p + "_base1"), std::cref(mXf), mXf, stream));
}

void InklingDflash2Drafter::observeTaps(float const* taps, int32_t rows, int32_t pos0, cudaStream_t stream)
{
    ELLM_CHECK(pos0 == mCtxLen, "Inkling DFlash2 drafter context is not contiguous");
    ELLM_CHECK(rows > 0 && rows <= mMaxAppendRows && pos0 + rows <= mCapacity, "Inkling DFlash2 drafter context update out of range");
    kernel::launchDfl2CtxPrep(taps, rows, w("target_final_norm").dataPointer<float>(), mCtxIn.dataPointer<__half>(), stream);
    linear("fc", mCtxIn.rawPointer(), mFcOut.rawPointer(), rows, kH, 4 * kH, stream);
    kernel::launchDfl2RmsHalf(mFcOut.dataPointer<__half>(), w("hidden_norm").dataPointer<float>(), mDelta.dataPointer<__half>(), rows, stream);
    for (int32_t i = 0; i < kLayers; ++i)
    {
        std::string const p = "l" + std::to_string(i) + ".";
        linear(p + "k", mDelta.rawPointer(), mKRaw.rawPointer(), rows, 1024, kH, stream);
        linear(p + "v", mDelta.rawPointer(), mVRaw.rawPointer(), rows, 1024, kH, stream);
        kernel::launchDrafterCtxKv(mKRaw.dataPointer<__half>(), mVRaw.dataPointer<__half>(), rows, pos0, w(p + "k_norm").dataPointer<float>(), kTheta,
            mCtxK[i].dataPointer<__half>(), mCtxV[i].dataPointer<__half>(), stream);
    }
    mCtxLen = pos0 + rows;
}

void InklingDflash2Drafter::draft(void const* anchorEmbedding, int32_t anchorToken, int32_t* hostTokens, cudaStream_t stream)
{
    int32_t const a = mCtxLen;
    ELLM_CHECK(a > 0, "Inkling DFlash2 drafter has no context");
    kernel::launchDrafterInitBlock(static_cast<__half const*>(anchorEmbedding), w("embed_norm").dataPointer<float>(), w("mask_embed").dataPointer<float>(),
        mXf.dataPointer<float>(), stream);
    int32_t const lo = std::max(0, a - (kWindow - 1));
    for (int32_t i = 0; i < kLayers; ++i)
    {
        std::string const p = "l" + std::to_string(i) + ".";
        // attention
        kernel::launchDrafterRms(mXf.dataPointer<float>(), w(p + "ln_in").dataPointer<float>(), mH.dataPointer<__half>(), kBlock, 0, stream);
        convPrepare("ac", i, mH, mHc, stream);
        linear(p + "q", mHc.rawPointer(), mQRaw.rawPointer(), kBlock, 4096, kH, stream);
        linear(p + "k", mHc.rawPointer(), mKBRaw.rawPointer(), kBlock, 1024, kH, stream);
        linear(p + "v", mHc.rawPointer(), mVBRaw.rawPointer(), kBlock, 1024, kH, stream);
        kernel::launchDfl2QkvPrep(mQRaw.dataPointer<__half>(), mKBRaw.dataPointer<__half>(), mVBRaw.dataPointer<__half>(), w(p + "q_norm").dataPointer<float>(),
            w(p + "k_norm").dataPointer<float>(), a, kTheta, mQ.dataPointer<__half>(), mKb.dataPointer<__half>(), mVb.dataPointer<__half>(), stream);
        kernel::launchDfl2Attention(mQ.dataPointer<__half>(), mCtxK[i].dataPointer<__half>(), mCtxV[i].dataPointer<__half>(), mKb.dataPointer<__half>(),
            mVb.dataPointer<__half>(), lo, a - lo, a, kWindow, mScores.dataPointer<float>(), mScoreStride, mPartial.dataPointer<float>(), kMaxSplits,
            mAttn.dataPointer<__half>(), stream);
        linear(p + "o", mAttn.rawPointer(), mO.rawPointer(), kBlock, kH, 4096, stream);
        convFinish("ac", i, mO, stream);
        // MLP
        kernel::launchDrafterRms(mXf.dataPointer<float>(), w(p + "ln_post").dataPointer<float>(), mH.dataPointer<__half>(), kBlock, 0, stream);
        convPrepare("mc", i, mH, mHc, stream);
        linear(p + "gate", mHc.rawPointer(), mG.rawPointer(), kBlock, 6912, kH, stream);
        linear(p + "up", mHc.rawPointer(), mU.rawPointer(), kBlock, 6912, kH, stream);
        kernel::launchDrafterSwiglu(mG.dataPointer<__half>(), mU.dataPointer<__half>(), mAct.dataPointer<__half>(), kBlock * 6912, stream);
        linear(p + "down", mAct.rawPointer(), mDown.rawPointer(), kBlock, kH, 6912, stream);
        convFinish("mc", i, mDown, stream);
        debugDump(mXf, "x" + std::to_string(i), stream);
    }
    kernel::launchDrafterRms(mXf.dataPointer<float>(), w("norm").dataPointer<float>(), mHidN.dataPointer<__half>(), kBlock, 0, stream);
    debugDump(mHidN, "hidN", stream);
    __half const* rowsOut = mHidN.dataPointer<__half>() + kH; // block rows 1..7 are the proposal rows
    linear("sel_proj", rowsOut, mProjected.rawPointer(), kDrafts, 256, kH, stream);
    linear("U", rowsOut, mLogits.rawPointer(), kDrafts, kVocabPadded, kH, stream);
    kernel::launchDfl2LogitsToFloat(mLogits.dataPointer<__half>(), mLogitsF.dataPointer<float>(), kDrafts, kVocabPadded, kVocab, stream);
    check::check(mCand.reshape({kDrafts, kTopK}), "Tensor reshape failed");
    selectAllTopK(mLogitsF, std::ref(mTopVals), mCand, kTopK, mTopWorkspace, stream);
    check::check(mCand.reshape({1, kDrafts, kTopK}), "Tensor reshape failed");
    kernel::launchDfl2GatherValues(mLogitsF.dataPointer<float>(), mCand.dataPointer<int32_t>(), kDrafts, kVocabPadded, kTopK, mUnary.dataPointer<float>(), stream);
    CUDA_CHECK(cudaMemcpyAsync(mAnchorTok.rawPointer(), &anchorToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    kernel::launchDFlash2CandidateSelector(mCand, mUnary, mProjected, mAnchorTok, w("cb_pred"), w("cb_succ"), mUniforms, mTemperatures, mGreedy, mDraftTok, mProbs, stream);
    CUDA_CHECK(cudaMemcpyAsync(hostTokens, mDraftTok.rawPointer(), kDrafts * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    mDebugDone = true;
}

} // namespace rt
} // namespace trt_edgellm
