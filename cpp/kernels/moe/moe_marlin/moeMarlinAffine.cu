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

/*
 * Entry point for the Flint affine W4A16 MoE Marlin GEMM (see moeMarlinAffine.h).
 * Same translation-unit style as moeMarlin.cu: ops and the instantiation file are included here (they have a
 * .cuh extension so that the kernels/*.cu glob does not compile them separately).
 */

#include "moeMarlinAffine.h"
#include "moeMarlin.h"
#include <cuda_runtime.h>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/stringUtils.h"
#include "marlin/scalar_type.hpp"

#include <algorithm>
#include <cstdint>

#include "marlin_moe_wna16_affine/ops.cuh"
#include "marlin_moe_wna16_affine/sm80_kernel_float16_u4_float16_affine.cuh"

namespace trt_edgellm
{
namespace kernel
{

using namespace trt_edgellm::format;

void moeAffineW4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& scales, rt::Tensor const& biasPrime, rt::Tensor const& sortedTokenIds,
    rt::Tensor const& expertIds, rt::Tensor const& numTokensPostPadded, rt::Tensor const& topkWeights,
    rt::Tensor& workspace, int64_t moeBlockSize, int64_t topK, bool mulTopkWeights, cudaStream_t stream)
{
    auto const inputShape = input.getShape();
    auto const outputShape = output.getShape();
    auto const weightsShape = weights.getShape();
    auto const scalesShape = scales.getShape();
    auto const biasShape = biasPrime.getShape();

    check::check(inputShape.getNumDims() == 2, "Input must be 2D tensor [numTokens, hiddenDim]");
    check::check(outputShape.getNumDims() == 2, "Output must be 2D tensor [numTokens*topK, outDim]");
    check::check(weightsShape.getNumDims() == 3, "Weights must be 3D tensor [numExperts, K/16, N*16/8]");
    check::check(scalesShape.getNumDims() == 3, "Scales must be 3D tensor [numExperts, numGroups, outDim]");
    check::check(biasShape == scalesShape, "biasPrime must have the same shape as scales");

    int64_t const numTokens = inputShape[0];
    int64_t const hiddenDim = inputShape[1];
    int64_t const numExperts = weightsShape[0];
    int64_t const numGroups = scalesShape[1];
    int64_t const outDim = scalesShape[2];

    check::check(hiddenDim % 64 == 0 && numGroups == hiddenDim / 64,
        fmtstr("Affine kernel requires group size 64: numGroups %ld, hiddenDim %ld", numGroups, hiddenDim));
    bool const validOutputShape = (outputShape[0] == numTokens * topK) || (outputShape[0] == numTokens);
    check::check(validOutputShape,
        fmtstr("Output shape[0] %ld must be either numTokens*topK=%ld or numTokens=%ld", outputShape[0],
            numTokens * topK, numTokens));
    check::check(outputShape[1] == outDim, fmtstr("Output shape[1] %ld != outDim %ld", outputShape[1], outDim));

    check::check(input.getDataType() == nvinfer1::DataType::kHALF, "Input must be FP16");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "Output must be FP16");
    check::check(weights.getDataType() == nvinfer1::DataType::kINT32, "Weights must be INT32 (Marlin packed)");
    check::check(scales.getDataType() == nvinfer1::DataType::kHALF, "Scales must be FP16");
    check::check(biasPrime.getDataType() == nvinfer1::DataType::kHALF, "biasPrime must be FP16");
    check::check(sortedTokenIds.getDataType() == nvinfer1::DataType::kINT32, "sortedTokenIds must be INT32");
    check::check(expertIds.getDataType() == nvinfer1::DataType::kINT32, "expertIds must be INT32");
    check::check(numTokensPostPadded.getDataType() == nvinfer1::DataType::kINT32, "numTokensPostPadded must be INT32");
    check::check(topkWeights.getDataType() == nvinfer1::DataType::kFLOAT, "topkWeights must be FP32");
    check::check(workspace.getDataType() == nvinfer1::DataType::kINT32, "workspace must be INT32");

    if (moeBlockSize != 8)
    {
        check::check(moeBlockSize % 16 == 0, fmtstr("moeBlockSize %ld must be divisible by 16", moeBlockSize));
        check::check(
            moeBlockSize >= 16 && moeBlockSize <= 64, fmtstr("moeBlockSize %ld must be in [16, 64]", moeBlockSize));
    }

    int dev;
    int sms;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

    int64_t const numTokensPadded = sortedTokenIds.getShape()[0];
    int64_t locksSize = std::min(
        (outDim / marlin_moe_wna16_affine::min_thread_n) * static_cast<int64_t>(divUp(numTokensPadded, moeBlockSize)),
        static_cast<int64_t>(sms * 4));
    locksSize = static_cast<int64_t>(divUp(locksSize, static_cast<int64_t>(4))) * 4;
    // Workspace layout is identical to the AWQ path: [locks][c_tmp]; sized by getMoeMarlinWorkspaceSize().
    int64_t const requiredWorkspace = getMoeMarlinWorkspaceSize(numTokensPadded, outDim, moeBlockSize, sms);
    check::check(workspace.getShape().volume() >= requiredWorkspace,
        fmtstr("Marlin workspace has %ld elements but requires at least %ld", workspace.getShape().volume(),
            requiredWorkspace));

    int32_t* workspaceBasePtr = static_cast<int32_t*>(workspace.rawPointer());
    void* locksPtr = workspaceBasePtr;
    void* cTmpPtr = workspaceBasePtr + locksSize;
    CUDA_CHECK(cudaMemsetAsync(locksPtr, 0, locksSize * sizeof(int32_t), stream));

    marlin_moe_wna16_affine::marlin_mm(const_cast<void*>(input.rawPointer()),
        const_cast<void*>(weights.rawPointer()), output.rawPointer(), cTmpPtr,
        nullptr, // b_bias (not used)
        nullptr, // a_scales
        const_cast<void*>(scales.rawPointer()),
        nullptr, // global_scale
        const_cast<void*>(biasPrime.rawPointer()), // b' (affine offset)
        nullptr, nullptr, nullptr,                  // g_idx, perm, a_tmp
        const_cast<void*>(sortedTokenIds.rawPointer()), const_cast<void*>(expertIds.rawPointer()),
        const_cast<void*>(numTokensPostPadded.rawPointer()), const_cast<void*>(topkWeights.rawPointer()),
        static_cast<int>(moeBlockSize), static_cast<int>(numExperts), static_cast<int>(topK), mulTopkWeights,
        static_cast<int>(numTokens), static_cast<int>(outDim), static_cast<int>(hiddenDim), locksPtr,
        trt_edgellm::marlin_dtypes::kFloat16, trt_edgellm::marlin_dtypes::kU4, trt_edgellm::marlin_dtypes::kFloat16,
        trt_edgellm::marlin_dtypes::kFloat16,
        false, // has_bias
        false, // has_act_order
        true,  // is_k_full
        false, // has_zp (affine offset is passed through the b' pointer)
        static_cast<int>(numGroups), 64, dev, stream,
        -1, // thread_k (auto)
        -1, // thread_n (auto)
        sms,
        -1,    // blocks_per_sm (auto)
        false, // use_atomic_add
        true,  // use_fp32_reduce
        false  // is_zp_float
    );
}

} // namespace kernel
} // namespace trt_edgellm
