/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common/tensor.h"
#include "runtime/decoding/draftSource.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

namespace trt_edgellm
{
namespace rt
{

//! The DFlash2 block drafter of Inkling 2 Mini (incoai/Inkling-2-Mini-DFlash2: 5 layers with dynamic grouped convolutions around attention and MLP, fc over the target's four taps,
//! a rank-256 candidate selector over the top-16 candidates of the target's LM head) with INT8 weights. Keeps the context K/V of every committed position and drafts 7 tokens
//! after an anchor in one pass. Weights come from the file written by tools/inkling2_dflash2_int8.py.
class InklingDflash2Drafter final : public IDraftSource
{
public:
    static constexpr int32_t kBlock = 8;
    static constexpr int32_t kDrafts = 7;
    static constexpr int32_t kLayers = 5;
    static constexpr int32_t kTopK = 16;

    InklingDflash2Drafter(std::filesystem::path const& weightsFile, int32_t capacity, int32_t maxAppendRows, cudaStream_t stream);

    char const* name() const noexcept override
    {
        return "inkling2_dflash2_int8";
    }
    int32_t maxDrafts() const noexcept override
    {
        return kDrafts;
    }
    int32_t tapWidth() const noexcept override
    {
        return 4 * 2304;
    }
    int32_t contextLength() const noexcept override
    {
        return mCtxLen;
    }
    void beginRequest() noexcept override
    {
        mCtxLen = 0;
    }
    bool truncateContext(int32_t length) noexcept override
    {
        if (length < 0 || length > mCtxLen)
        {
            return false;
        }
        mCtxLen = length;
        return true;
    }
    void observeTaps(float const* taps, int32_t rows, int32_t pos0, cudaStream_t stream) override;
    void draft(void const* anchorEmbedding, int32_t anchorToken, int32_t* hostTokens, cudaStream_t stream) override;

private:
    void debugDump(Tensor const& t, std::string const& tag, cudaStream_t stream);
    Tensor const& w(std::string const& name) const;
    Tensor make(std::vector<int64_t> const& dims, nvinfer1::DataType type, std::string const& name) const;
    //! y[rows, n] = x[rows, k] * W^T (INT8 per-row scales).
    void linear(std::string const& name, void const* x, void* y, int32_t rows, int32_t n, int32_t k, cudaStream_t stream) const;
    //! One convolution side pair on the block: coefficient GEMVs of x, the pre side applied to x (out), the post coefficients left in mCoPost.
    void convPrepare(std::string const& prefix, int32_t layer, Tensor const& in, Tensor& out, cudaStream_t stream);
    void convFinish(std::string const& prefix, int32_t layer, Tensor const& hidden, cudaStream_t stream);

    int32_t mCapacity;
    int32_t mMaxAppendRows;
    int32_t mCtxLen{0};
    bool mDebugDone{false};
    int32_t mScoreStride{0};
    std::map<std::string, Tensor> mW;
    Tensor mCtxK[kLayers], mCtxV[kLayers];
    Tensor mCtxIn, mFcOut, mDelta, mKRaw, mVRaw;
    Tensor mXf, mH, mCoPre, mCoPost, mHc, mQRaw, mKBRaw, mVBRaw, mQ, mKb, mVb, mAttn, mO, mG, mU, mAct, mDown, mHidN;
    Tensor mScores, mPartial;
    Tensor mProj, mLogits, mLogitsF, mTopVals, mTopIdx, mTopWorkspace, mTopKeys, mCand, mUnary, mProjected;
    Tensor mAnchorTok, mUniforms, mTemperatures, mGreedy, mDraftTok, mProbs;
};

} // namespace rt
} // namespace trt_edgellm
