/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "runtime/decoding/inklingDrafter.h"
#include "runtime/decoding/inklingDflash2Drafter.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <cstdlib>

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/inklingDrafter/inklingDrafterKernels.h"

namespace trt_edgellm
{
namespace rt
{
namespace
{
constexpr float kTheta = 1000000.F;
constexpr int32_t kVocabPadded = 200064;
constexpr int32_t kMaxSplits = 32;
} // namespace

Tensor const& InklingDrafter::w(std::string const& name) const
{
    auto const it = mW.find(name);
    ELLM_CHECK(it != mW.end(), "Inkling drafter weight missing: " + name);
    return it->second;
}

Tensor InklingDrafter::make(std::vector<int64_t> const& dims, nvinfer1::DataType type, std::string const& name) const
{
    nvinfer1::Dims shape{};
    shape.nbDims = static_cast<int32_t>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i)
    {
        shape.d[i] = dims[i];
    }
    return Tensor(Coords(shape), DeviceType::kGPU, type, "InklingDrafter::" + name);
}

InklingDrafter::InklingDrafter(std::filesystem::path const& weightsFile, int32_t capacity, int32_t maxAppendRows, cudaStream_t stream)
    : mCapacity(capacity)
    , mMaxAppendRows(maxAppendRows)
{
    std::vector<Tensor> tensors;
    ELLM_CHECK(safetensors::loadSafetensors(weightsFile, tensors, stream), "Failed to load the Inkling drafter weights: " + weightsFile.string());
    for (Tensor& t : tensors)
    {
        std::string const name = t.getName();
        mW.emplace(name, std::move(t));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    using DT = nvinfer1::DataType;
    int64_t const H = kernel::kDrafterHidden;
    mScoreStride = ((mCapacity + kBlock + 7) / 8) * 8;
    for (int32_t i = 0; i < 3; ++i)
    {
        mCtxK[i] = make({mCapacity, 1024}, DT::kHALF, "ctxK" + std::to_string(i));
        mCtxV[i] = make({mCapacity, 1024}, DT::kHALF, "ctxV" + std::to_string(i));
    }
    mDflareAll = make({3, H}, DT::kFLOAT, "dflareAll");
    for (int32_t i = 0; i < 3; ++i)
    {
        Tensor const& src = w("l" + std::to_string(i) + ".dflare");
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(mDflareAll.rawPointer()) + i * H * sizeof(float), src.rawPointer(), H * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
    mLastFeat = make({H}, DT::kFLOAT, "lastFeat");
    mCtxn = make({3, mMaxAppendRows, H}, DT::kHALF, "ctxn");
    mKRaw = make({mMaxAppendRows, 1024}, DT::kHALF, "kRaw");
    mVRaw = make({mMaxAppendRows, 1024}, DT::kHALF, "vRaw");
    mX = make({kBlock, H}, DT::kFLOAT, "x");
    mH = make({kBlock, H}, DT::kHALF, "h");
    mQRaw = make({kBlock, 3072}, DT::kHALF, "qRaw");
    mKBRaw = make({kBlock, 1024}, DT::kHALF, "kbRaw");
    mVBRaw = make({kBlock, 1024}, DT::kHALF, "vbRaw");
    mQ = make({kBlock, 3072}, DT::kHALF, "q");
    mKb = make({kBlock, 1024}, DT::kHALF, "kb");
    mVb = make({kBlock, 1024}, DT::kHALF, "vb");
    mAttn = make({kBlock, 3072}, DT::kHALF, "attn");
    mO = make({kBlock, H}, DT::kHALF, "o");
    mG = make({kBlock, 9216}, DT::kHALF, "g");
    mU = make({kBlock, 9216}, DT::kHALF, "u");
    mAct = make({kBlock, 9216}, DT::kHALF, "act");
    mDown = make({kBlock, H}, DT::kHALF, "down");
    mHidN = make({kBlock, H}, DT::kHALF, "hidN");
    mXs = make({kDrafts, 2 * H}, DT::kHALF, "xs");
    mInv = make({kDrafts, 768}, DT::kHALF, "inv");
    mBase = make({kDrafts, kVocabPadded}, DT::kHALF, "base");
    mCorr = make({kVocabPadded}, DT::kHALF, "corr");
    mO256 = make({256}, DT::kHALF, "o256");
    mScores = make({192, mScoreStride}, DT::kFLOAT, "scores");
    mPartial = make({kMaxSplits, 192, 128}, DT::kFLOAT, "partial");
    mState[0] = make({256}, DT::kFLOAT, "state0");
    mState[1] = make({256}, DT::kFLOAT, "state1");
    mArgVal = make({128}, DT::kFLOAT, "argVal");
    mArgIdx = make({128}, DT::kINT32, "argIdx");
    mDraftTok = make({kDrafts}, DT::kINT32, "draftTok");
    mPrevTok = make({1}, DT::kINT32, "prevTok");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_INFO("Inkling drafter loaded from %s (%zu tensors, context capacity %d)", weightsFile.c_str(), mW.size(), mCapacity);
}

//! EDGELLM_INKLING_DRAFT_DEBUG=<prefix>: the first draft call writes <prefix>.<tag> with the raw bytes of selected tensors (offline comparison).
void InklingDrafter::debugDump(Tensor const& t, std::string const& tag, cudaStream_t stream)
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

void InklingDrafter::linear(std::string const& name, void const* x, void* y, int32_t rows, int32_t n, int32_t k, cudaStream_t stream) const
{
    // 8 tokens per launch (a long context update re-reads the weights per group of 8 rows)
    for (int32_t r = 0; r < rows; r += 8)
    {
        int32_t const m = std::min(8, rows - r);
        kernel::launchDrafterGemv(static_cast<__half const*>(x) + static_cast<size_t>(r) * k, w(name).rawPointer(), w(name + ".s").rawPointer(),
            static_cast<__half*>(y) + static_cast<size_t>(r) * n, m, n, k, stream);
    }
}

void InklingDrafter::append(float const* taps, int32_t rows, int32_t pos0, cudaStream_t stream)
{
    ELLM_CHECK(pos0 == mCtxLen, "Inkling drafter context is not contiguous");
    ELLM_CHECK(rows > 0 && rows <= mMaxAppendRows && pos0 + rows <= mCapacity, "Inkling drafter context update out of range");
    int64_t const H = kernel::kDrafterHidden;
    kernel::launchDrafterCtxMix(taps, rows, mMaxAppendRows, w("alpha").dataPointer<float>(), mDflareAll.dataPointer<float>(), w("markov_dflare").dataPointer<float>(),
        w("target_final_norm").dataPointer<float>(), mCtxn.dataPointer<__half>(), mLastFeat.dataPointer<float>(), stream);
    for (int32_t i = 0; i < 3; ++i)
    {
        std::string const p = "l" + std::to_string(i) + ".";
        void const* x = mCtxn.dataPointer<__half>() + static_cast<size_t>(i) * mMaxAppendRows * H;
        linear(p + "kctx", x, mKRaw.rawPointer(), rows, 1024, 2304, stream);
        linear(p + "vctx", x, mVRaw.rawPointer(), rows, 1024, 2304, stream);
        kernel::launchDrafterCtxKv(mKRaw.dataPointer<__half>(), mVRaw.dataPointer<__half>(), rows, pos0, w(p + "k_norm_ctx").dataPointer<float>(), kTheta,
            mCtxK[i].dataPointer<__half>(), mCtxV[i].dataPointer<__half>(), stream);
    }
    mCtxLen = pos0 + rows;
}

void InklingDrafter::draft(void const* anchorEmbedding, int32_t anchorToken, int32_t* hostTokens, cudaStream_t stream)
{
    int32_t const a = mCtxLen;
    ELLM_CHECK(a > 0, "Inkling drafter has no context");
    int64_t const H = kernel::kDrafterHidden;
    kernel::launchDrafterInitBlock(static_cast<__half const*>(anchorEmbedding), w("embed_norm").dataPointer<float>(), w("mask_embed").dataPointer<float>(),
        mX.dataPointer<float>(), stream);
    for (int32_t i = 0; i < 3; ++i)
    {
        std::string const p = "l" + std::to_string(i) + ".";
        kernel::launchDrafterRms(mX.dataPointer<float>(), w(p + "ln_in").dataPointer<float>(), mH.dataPointer<__half>(), kBlock, 0, stream);
        linear(p + "q", mH.rawPointer(), mQRaw.rawPointer(), kBlock, 3072, 2304, stream);
        linear(p + "k", mH.rawPointer(), mKBRaw.rawPointer(), kBlock, 1024, 2304, stream);
        linear(p + "v", mH.rawPointer(), mVBRaw.rawPointer(), kBlock, 1024, 2304, stream);
        kernel::launchDrafterQkvPrep(mQRaw.dataPointer<__half>(), mKBRaw.dataPointer<__half>(), mVBRaw.dataPointer<__half>(), w(p + "q_norm").dataPointer<float>(),
            w(p + "k_norm").dataPointer<float>(), a, kTheta, mQ.dataPointer<__half>(), mKb.dataPointer<__half>(), mVb.dataPointer<__half>(), stream);
        bool const local = (i == 0 || i == 2);
        int32_t const lo = local ? std::max(0, a - 1024) : 0;
        kernel::launchDrafterAttention(mQ.dataPointer<__half>(), mCtxK[i].dataPointer<__half>(), mCtxV[i].dataPointer<__half>(), mKb.dataPointer<__half>(),
            mVb.dataPointer<__half>(), lo, a - lo, mScores.dataPointer<float>(), mScoreStride, mPartial.dataPointer<float>(), kMaxSplits, mAttn.dataPointer<__half>(),
            stream);
        linear(p + "o", mAttn.rawPointer(), mO.rawPointer(), kBlock, 2304, 3072, stream);
        kernel::launchDrafterAddResidual(mX.dataPointer<float>(), mO.dataPointer<__half>(), kBlock * 2304, stream);
        kernel::launchDrafterRms(mX.dataPointer<float>(), w(p + "ln_post").dataPointer<float>(), mH.dataPointer<__half>(), kBlock, 0, stream);
        linear(p + "gate", mH.rawPointer(), mG.rawPointer(), kBlock, 9216, 2304, stream);
        linear(p + "up", mH.rawPointer(), mU.rawPointer(), kBlock, 9216, 2304, stream);
        kernel::launchDrafterSwiglu(mG.dataPointer<__half>(), mU.dataPointer<__half>(), mAct.dataPointer<__half>(), kBlock * 9216, stream);
        linear(p + "down", mAct.rawPointer(), mDown.rawPointer(), kBlock, 2304, 9216, stream);
        kernel::launchDrafterAddResidual(mX.dataPointer<float>(), mDown.dataPointer<__half>(), kBlock * 2304, stream);
        debugDump(mX, "x" + std::to_string(i), stream);
    }
    kernel::launchDrafterRms(mX.dataPointer<float>(), w("final_norm_drafter").dataPointer<float>(), mHidN.dataPointer<__half>(), kBlock, 0, stream);
    debugDump(mHidN, "hidN", stream);
    kernel::launchDrafterBuildInv(mHidN.dataPointer<__half>(), mLastFeat.dataPointer<float>(), mXs.dataPointer<__half>(), stream);
    linear("rnn_inv", mXs.rawPointer(), mInv.rawPointer(), kDrafts, 768, 2 * 2304, stream);
    linear("U", mHidN.dataPointer<__half>() + H, mBase.rawPointer(), kDrafts, kVocabPadded, 2304, stream);
    CUDA_CHECK(cudaMemsetAsync(mState[0].rawPointer(), 0, 256 * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(mPrevTok.rawPointer(), &anchorToken, sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    for (int32_t j = 0; j < kDrafts; ++j)
    {
        kernel::launchDrafterMarkovGate(mInv.dataPointer<__half>(), w("rnn_b").dataPointer<float>(), w("rnn_state_w").dataPointer<float>(),
            w("markov_w1").dataPointer<__half>(), mPrevTok.dataPointer<int32_t>(), mState[j & 1].dataPointer<float>(), mState[(j + 1) & 1].dataPointer<float>(),
            mO256.dataPointer<__half>(), j, stream);
        linear("Mt", mO256.rawPointer(), mCorr.rawPointer(), 1, kVocabPadded, 256, stream);
        kernel::launchDrafterArgmax(mBase.dataPointer<__half>(), mCorr.dataPointer<__half>(), w("bU").dataPointer<float>(), kVocabPadded, j,
            mArgVal.dataPointer<float>(), mArgIdx.dataPointer<int32_t>(), mDraftTok.dataPointer<int32_t>(), mPrevTok.dataPointer<int32_t>(), stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(hostTokens, mDraftTok.rawPointer(), kDrafts * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    mDebugDone = true;
}

std::unique_ptr<IDraftSource> createDraftSource(
    std::filesystem::path const& weightsFile, int32_t capacity, int32_t maxAppendRows, cudaStream_t stream)
{
    std::string type = "inkling2_dspark_int8";
    std::filesystem::path const sidecar = std::filesystem::path(weightsFile).replace_extension(".json");
    if (std::filesystem::exists(sidecar))
    {
        std::ifstream in(sidecar);
        std::string const text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto const key = text.find("\"type\"");
        if (key != std::string::npos)
        {
            auto const q1 = text.find('"', text.find(':', key) + 1);
            auto const q2 = text.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
            {
                type = text.substr(q1 + 1, q2 - q1 - 1);
            }
        }
    }
    if (type == "inkling2_dspark_int8")
    {
        return std::make_unique<InklingDrafter>(weightsFile, capacity, maxAppendRows, stream);
    }
    if (type == "inkling2_dflash2_int8")
    {
        return std::make_unique<InklingDflash2Drafter>(weightsFile, capacity, maxAppendRows, stream);
    }
    throw std::runtime_error("Unknown draft source type '" + type + "' in " + sidecar.string());
}

} // namespace rt
} // namespace trt_edgellm
