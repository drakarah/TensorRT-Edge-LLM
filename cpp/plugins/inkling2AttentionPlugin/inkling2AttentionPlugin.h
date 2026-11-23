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

//! Inkling 2 attention visit with a relative-position bias, a per-token sink, query/relative L2 clips, optional
//! log-length query scaling and a sigmoid output gate, over the runtime's paged KV pool. K/V rows are appended in
//! place.
//!
//! Inputs: q_lin [T, nh*128], rel_lin [T, nh*16], sink_lin [T, nh], gate_lin [T, nh*128], k_new [T, nkv*128],
//! v_new [T, nkv*128] (FP16); q_scale [nh*128], rel_scale [nh*16], rel_proj_t [rel_extent, 16] (FP32);
//! past_kv [2, pages, 128, nkv, 128] (FP16); query_start_offsets [B+1], query_lengths [B], past_lengths [B],
//! kv_page_table [B, 2, maxPages] (INT32). Token slots of sequence b are [query_start_offsets[b],
//! query_start_offsets[b+1]); only the first query_lengths[b] are real. Outputs: attention [T, nh*128] (FP16),
//! present_kv (aliases past_kv).
class Inkling2AttentionPlugin : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime
{
public:
    Inkling2AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t relExtent,
        int32_t window, float logAlpha, float logNFloor, float qClip, float relClip, float scaling);

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
    std::string mLayerName;
    std::string mNamespace;
    int32_t mNumQHeads;
    int32_t mNumKVHeads;
    int32_t mRelExtent;
    int32_t mWindow;
    float mLogAlpha;
    float mLogNFloor;
    float mQClip;
    float mRelClip;
    float mScaling;
    std::vector<nvinfer1::PluginField> mData;
    nvinfer1::PluginFieldCollection mFields{};
};

class Inkling2AttentionPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Inkling2AttentionPluginCreator();

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
