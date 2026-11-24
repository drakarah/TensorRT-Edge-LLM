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

#pragma once

#include <NvInferRuntime.h>

#include <string>
#include <vector>

namespace trt_edgellm::plugins
{

//! Inkling 2 short convolution with a residual, static taps plus optional per-token dynamic taps, over a rolling
//! per-sequence state that is updated in place.
//!
//! Inputs: x [T, C], taps [C, k] (FP32, lag 0 = current token), state [rows, stateC >= state_offset + C, k],
//! query_start_offsets [B+1], query_lengths [B], state_indices [B], and dyn [T, groups * k] when groups > 0 (FP16
//! except taps and the metadata). Outputs: y [T, C], present_state (aliases state). y = scale * (sum_j (taps + dyn) *
//! x[t - j] + x[t]). With the optional attribute intermediate=1 a third output holds the state after every token
//! ([T, C, k]), the rollback source of speculative verification.
class Inkling2ShortConvPlugin : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime
{
public:
    Inkling2ShortConvPlugin(std::string const& name, int32_t channels, int32_t kernelSize, int32_t groups, float scale,
        int32_t stateOffset, int32_t intermediate = 0);

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int32_t numInputs() const noexcept;
    int32_t numOutputs() const noexcept;

    std::string mLayerName;
    std::string mNamespace;
    int32_t mChannels;
    int32_t mKernelSize;
    int32_t mGroups;
    float mScale;
    int32_t mStateOffset;
    int32_t mIntermediate; //!< > 0: third output [min(T, mIntermediate), C, k] with the state after each token of a step of at most mIntermediate tokens (speculative verification).
    std::vector<nvinfer1::PluginField> mData;
    nvinfer1::PluginFieldCollection mFields{};
};

class Inkling2ShortConvPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Inkling2ShortConvPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static std::vector<nvinfer1::PluginField> mAttributes;
    nvinfer1::PluginFieldCollection mFields{};
    std::string mNamespace;
};

} // namespace trt_edgellm::plugins
