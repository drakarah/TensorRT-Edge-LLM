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

#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

Gemma4EmbeddingPreprocessor::Gemma4EmbeddingPreprocessor(std::filesystem::path const& engineDir,
    LLMEngineConfig const& config, int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream,
    std::optional<Tensor> checkpointTable)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.pleEnabled, "Gemma4EmbeddingPreprocessor constructed while PLE is disabled");
    ELLM_CHECK(maxBatchSize > 0, "Gemma4EmbeddingPreprocessor requires positive max batch size");
    ELLM_CHECK(maxSeqLen > 0, "Gemma4EmbeddingPreprocessor requires positive max sequence length");

    if (checkpointTable.has_value())
    {
        mPleTable = std::move(*checkpointTable);
    }
    else
    {
        std::filesystem::path const plePath = engineDir / binding_names::kPleEmbeddingFileName;
        std::vector<Tensor> pleTensors;
        ELLM_CHECK(safetensors::loadSafetensors(plePath, pleTensors, stream),
            "Failed to load " + std::string(binding_names::kPleEmbeddingFileName)
                + " from model directory: " + engineDir.string());
        // Either one fp16/bf16 tensor "weight", or an INT8 "weight" plus fp16 "scale"
        // [vocab, num_ple_inputs] (one scale per token-layer slice), written by the
        // edge-llm-optimization tools/quantize_ple.py tool.
        ELLM_CHECK(pleTensors.size() == 1 || pleTensors.size() == 2,
            "ple_embedding.safetensors must contain weight, or weight and scale");
        for (auto& tensor : pleTensors)
        {
            if (tensor.getName() == "weight")
            {
                mPleTable = std::move(tensor);
            }
            else if (tensor.getName() == "scale")
            {
                mPleScales = std::move(tensor);
            }
        }
        ELLM_CHECK(mPleTable.getShape().getNumDims() == 2, "ple_embedding.safetensors must contain a tensor named weight");
    }

    auto const pleShape = mPleTable.getShape();
    ELLM_CHECK(pleShape.getNumDims() == 2, "PLE table must be 2D [vocab, num_layers * hidden]");
    ELLM_CHECK(pleShape[1] == static_cast<int64_t>(mConfig.numPleInputs) * mConfig.pleHiddenSize,
        "PLE table second dimension must equal num_ple_inputs * ple_hidden_size");
    mPleInt8 = mPleTable.getDataType() == nvinfer1::DataType::kINT8;
    if (mPleInt8)
    {
        ELLM_CHECK(mPleScales.getShape().getNumDims() == 2 && mPleScales.getDataType() == nvinfer1::DataType::kHALF,
            "INT8 PLE table requires an FP16 scale tensor [vocab, num_ple_inputs]");
    }
    else
    {
        ELLM_CHECK(mPleTable.getDataType() == nvinfer1::DataType::kHALF
                || mPleTable.getDataType() == nvinfer1::DataType::kBF16,
            "PLE table must be FP16, BF16 or INT8 with scales");
    }
    // The engine input dtype is the dequantised dtype: fp16 for an INT8 table.
    mOutputDataType = mPleInt8 ? nvinfer1::DataType::kHALF : mPleTable.getDataType();

    mPleOutputBuffer = Tensor({mConfig.numPleInputs, maxBatchSize, maxSeqLen, mConfig.pleHiddenSize}, DeviceType::kGPU,
        mOutputDataType, "Gemma4EmbeddingPreprocessor::mPleOutputBuffer");

    mPleOutputViews.reserve(mConfig.numPleInputs);
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews.emplace_back(makeTokenMajorOutputViewForLayer(idx, maxBatchSize * maxSeqLen));
        tensorMap.set(mPleOutputViews.back().getName(), mPleOutputViews.back());
    }

    LOG_INFO("Initialized Gemma4 PLE preprocessor: table=%s (%s) outputBuffer=%s numPleInputs=%d pleHiddenSize=%d",
        mPleTable.getShape().formatString().c_str(), mPleInt8 ? "INT8 + fp16 scales" : "fp16/bf16",
        mPleOutputBuffer.getShape().formatString().c_str(), mConfig.numPleInputs, mConfig.pleHiddenSize);
}

Tensor Gemma4EmbeddingPreprocessor::makeTokenMajorOutputViewForLayer(int32_t layerIdx, int64_t physicalTokens)
{
    ELLM_CHECK(layerIdx >= 0 && layerIdx < mConfig.numPleInputs, "Gemma4 PLE layer index out of range");
    auto const outputShape = mPleOutputBuffer.getShape();
    ELLM_CHECK(physicalTokens > 0, "Gemma4 PLE physical token count must be positive");
    ELLM_CHECK(
        physicalTokens <= outputShape[1] * outputShape[2], "Gemma4 PLE physical token count exceeds buffer capacity");

    int64_t const layerOutputCapacityBytes = outputShape[1] * outputShape[2] * mConfig.pleHiddenSize
        * static_cast<int64_t>(utils::getTypeSize(mOutputDataType));
    void* const layerOutputPtr
        = static_cast<void*>(static_cast<char*>(mPleOutputBuffer.rawPointer()) + layerIdx * layerOutputCapacityBytes);
    return Tensor(layerOutputPtr, Coords{physicalTokens, mConfig.pleHiddenSize}, DeviceType::kGPU, mOutputDataType,
        binding_names::formatPleTokenEmbedsName(layerIdx));
}

void Gemma4EmbeddingPreprocessor::reshapeOutputsTokenMajor(int64_t physicalTokens)
{
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews[idx] = makeTokenMajorOutputViewForLayer(idx, physicalTokens);
    }
}

void Gemma4EmbeddingPreprocessor::embed(Tensor const& tokenIds, cudaStream_t stream)
{
    auto const tokenShape = tokenIds.getShape();
    ELLM_CHECK(tokenShape.getNumDims() == 2, "Gemma4 PLE token IDs must be [batch, seq_len]");
    if (mPleInt8)
    {
        kernel::gemma4PleGatherInt8(tokenIds, mPleTable, mPleScales, mPleOutputBuffer, mConfig.numPleInputs,
            mConfig.pleHiddenSize, mConfig.imageTokenId, mConfig.audioTokenId, stream);
    }
    else
    {
        kernel::gemma4PleGather(tokenIds, mPleTable, mPleOutputBuffer, mConfig.numPleInputs, mConfig.pleHiddenSize,
            mConfig.imageTokenId, mConfig.audioTokenId, stream);
    }
    reshapeOutputsTokenMajor(tokenShape[0] * tokenShape[1]);
}

} // namespace rt
} // namespace trt_edgellm
