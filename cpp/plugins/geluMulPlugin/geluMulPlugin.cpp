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

#include "geluMulPlugin.h"
#include "common/logger.h"
#include "kernels/geluMulKernels/geluMulKernels.h"

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <exception>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
namespace
{
constexpr char const* kPLUGIN_NAME{"GeluTanhMulPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr int32_t kNUM_INPUTS{1};
constexpr int32_t kNUM_OUTPUTS{1};

bool isSupportedRank(int32_t const rank)
{
    return rank == 2 || rank == 3;
}

//! Output keeps the leading dimensions of the input and has half its width (gate and up are the two halves).
bool hasHalfWidthShape(Dims const& input, Dims const& output)
{
    if (!isSupportedRank(input.nbDims) || output.nbDims != input.nbDims)
    {
        return false;
    }
    int32_t const widthDim = input.nbDims - 1;
    for (int32_t dim = 0; dim < widthDim; ++dim)
    {
        if (input.d[dim] <= 0 || input.d[dim] != output.d[dim])
        {
            return false;
        }
    }
    return input.d[widthDim] > 0 && input.d[widthDim] % 2 == 0 && output.d[widthDim] == input.d[widthDim] / 2;
}

bool isFp16Linear(PluginTensorDesc const& desc)
{
    return desc.type == DataType::kHALF && desc.format == PluginFormat::kLINEAR && isSupportedRank(desc.dims.nbDims);
}
} // namespace

REGISTER_TENSORRT_PLUGIN(GeluTanhMulPluginCreator);

GeluTanhMulPlugin::GeluTanhMulPlugin(std::string const& name)
    : mLayerName(name)
{
}

IPluginCapability* GeluTanhMulPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* GeluTanhMulPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new GeluTanhMulPlugin(mLayerName);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* GeluTanhMulPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* GeluTanhMulPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* GeluTanhMulPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GeluTanhMulPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t GeluTanhMulPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t GeluTanhMulPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != kNUM_OUTPUTS || nbInputs != kNUM_INPUTS
        || inputTypes[0] != DataType::kHALF)
    {
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t GeluTanhMulPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */,
    int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS
        || !isSupportedRank(inputs[0].nbDims))
    {
        return -1;
    }
    int32_t const rank = inputs[0].nbDims;
    int32_t const widthDim = rank - 1;
    for (int32_t dim = 0; dim < rank; ++dim)
    {
        if (inputs[0].d[dim] == nullptr)
        {
            return -1;
        }
    }
    outputs[0].nbDims = rank;
    for (int32_t dim = 0; dim < widthDim; ++dim)
    {
        outputs[0].d[dim] = inputs[0].d[dim];
    }
    outputs[0].d[widthDim]
        = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[0].d[widthDim], *exprBuilder.constant(2));
    return 0;
}

bool GeluTanhMulPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    auto const& desc = inOut[pos].desc;
    return desc.type == DataType::kHALF && desc.format == PluginFormat::kLINEAR && isSupportedRank(desc.dims.nbDims);
}

int32_t GeluTanhMulPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        LOG_ERROR("GeluTanhMulPlugin: expected one input and one output");
        return -1;
    }
    if (!isFp16Linear(in[0].desc) || !isFp16Linear(out[0].desc) || !hasHalfWidthShape(in[0].min, out[0].min)
        || !hasHalfWidthShape(in[0].opt, out[0].opt) || !hasHalfWidthShape(in[0].max, out[0].max))
    {
        LOG_ERROR("GeluTanhMulPlugin: input must be rank-2 or rank-3 FP16 with an even last dimension, output has half of it");
        return -1;
    }
    return 0;
}

size_t GeluTanhMulPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t GeluTanhMulPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr || inputs[0] == nullptr
        || outputs[0] == nullptr || !isFp16Linear(inputDesc[0]) || !isFp16Linear(outputDesc[0])
        || !hasHalfWidthShape(inputDesc[0].dims, outputDesc[0].dims))
    {
        LOG_ERROR("GeluTanhMulPlugin: invalid runtime descriptors or null buffers");
        return -1;
    }
    int32_t const rank = inputDesc[0].dims.nbDims;
    int32_t const widthDim = rank - 1;
    int64_t rows = 1;
    for (int32_t dim = 0; dim < widthDim; ++dim)
    {
        rows *= inputDesc[0].dims.d[dim];
    }
    int32_t const width = inputDesc[0].dims.d[widthDim] / 2;
    kernel::launchGeluTanhMul(inputs[0], outputs[0], rows, width, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t GeluTanhMulPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS || in == nullptr || out == nullptr || !isFp16Linear(in[0])
        || !isFp16Linear(out[0]) || !hasHalfWidthShape(in[0].dims, out[0].dims))
    {
        LOG_ERROR("GeluTanhMulPlugin: invalid runtime input or output shape");
        return -1;
    }
    return 0;
}

IPluginV3* GeluTanhMulPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* GeluTanhMulPlugin::getFieldsToSerialize() noexcept
{
    return &mFields;
}

char const* GeluTanhMulPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* GeluTanhMulPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* GeluTanhMulPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* GeluTanhMulPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GeluTanhMulPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* GeluTanhMulPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* /* fc */, TensorRTPhase /* phase */) noexcept
{
    if (name == nullptr)
    {
        LOG_ERROR("GeluTanhMulPluginCreator: plugin name must not be null");
        return nullptr;
    }
    try
    {
        auto* plugin = new GeluTanhMulPlugin(name);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("GeluTanhMulPluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
