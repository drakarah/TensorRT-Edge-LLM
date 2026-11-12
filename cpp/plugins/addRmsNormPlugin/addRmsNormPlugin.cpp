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

#include "addRmsNormPlugin.h"
#include "common/logger.h"
#include "kernels/addRmsNorm/addRmsNorm.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <exception>
#include <stdexcept>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
namespace
{
constexpr char const* kPLUGIN_NAME{"AddRmsNormPlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kFIELD_EPS{"rms_norm_eps"};
constexpr char const* kFIELD_WEIGHT_BEFORE_CAST{"weight_before_cast"};
constexpr int32_t kIN_X{0};
constexpr int32_t kIN_RESIDUAL{1};
constexpr int32_t kIN_GAMMA{2};
constexpr int32_t kNUM_INPUTS{3};
constexpr int32_t kNUM_OUTPUTS{2};

bool isSupportedRank(int32_t const rank)
{
    return rank == 2 || rank == 3;
}

bool isFp16Linear(PluginTensorDesc const& desc)
{
    return desc.type == DataType::kHALF && desc.format == PluginFormat::kLINEAR;
}

bool sameDims(Dims const& a, Dims const& b)
{
    if (a.nbDims != b.nbDims)
    {
        return false;
    }
    for (int32_t dim = 0; dim < a.nbDims; ++dim)
    {
        if (a.d[dim] != b.d[dim])
        {
            return false;
        }
    }
    return true;
}

//! x, residual and both outputs share one shape [.., H]; gamma is [H]; H is a multiple of 8 within the kernel's limit.
//! Leading dimensions may be zero (an empty batch is a no-op).
bool hasValidShapes(Dims const& x, Dims const& residual, Dims const& gamma, Dims const& sum, Dims const& norm)
{
    if (!isSupportedRank(x.nbDims) || !sameDims(x, residual) || !sameDims(x, sum) || !sameDims(x, norm) || gamma.nbDims != 1)
    {
        return false;
    }
    int32_t const hidden = static_cast<int32_t>(x.d[x.nbDims - 1]);
    for (int32_t dim = 0; dim + 1 < x.nbDims; ++dim)
    {
        if (x.d[dim] < 0)
        {
            return false;
        }
    }
    return hidden > 0 && hidden % 8 == 0 && hidden <= kernel::kAddRmsNormMaxHidden && gamma.d[0] == hidden;
}

template <typename T>
T requiredField(PluginFieldCollection const* fields, char const* name, PluginFieldType type)
{
    if (fields == nullptr)
    {
        throw std::invalid_argument("AddRmsNormPlugin: null PluginFieldCollection");
    }
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (field.name != nullptr && std::string(field.name) == name)
        {
            if (field.type != type || field.length != 1 || field.data == nullptr)
            {
                throw std::invalid_argument(std::string("AddRmsNormPlugin: field ") + name + " must be one scalar of its type");
            }
            return *static_cast<T const*>(field.data);
        }
    }
    throw std::invalid_argument(std::string("AddRmsNormPlugin: missing required field ") + name);
}
} // namespace

REGISTER_TENSORRT_PLUGIN(AddRmsNormPluginCreator);

AddRmsNormPlugin::AddRmsNormPlugin(std::string const& name, float rmsNormEps, int32_t weightBeforeCast)
    : mLayerName(name)
    , mRmsNormEps(rmsNormEps)
    , mWeightBeforeCast(weightBeforeCast)
{
    if (!std::isfinite(mRmsNormEps) || mRmsNormEps <= 0.0F || (mWeightBeforeCast != 0 && mWeightBeforeCast != 1))
    {
        throw std::invalid_argument("AddRmsNormPlugin: rms_norm_eps must be finite and positive, weight_before_cast 0 or 1");
    }
}

IPluginCapability* AddRmsNormPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

char const* AddRmsNormPlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* AddRmsNormPlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* AddRmsNormPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AddRmsNormPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t AddRmsNormPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

size_t AddRmsNormPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

IPluginV3* AddRmsNormPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new AddRmsNormPlugin(mLayerName, mRmsNormEps, mWeightBeforeCast);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

int32_t AddRmsNormPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != kNUM_OUTPUTS || nbInputs != kNUM_INPUTS
        || inputTypes[kIN_X] != DataType::kHALF || inputTypes[kIN_RESIDUAL] != DataType::kHALF
        || inputTypes[kIN_GAMMA] != DataType::kHALF)
    {
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    outputTypes[1] = DataType::kHALF;
    return 0;
}

int32_t AddRmsNormPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */,
    int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS
        || !isSupportedRank(inputs[kIN_X].nbDims))
    {
        return -1;
    }
    outputs[0] = inputs[kIN_X];
    outputs[1] = inputs[kIN_X];
    return 0;
}

bool AddRmsNormPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    auto const& desc = inOut[pos].desc;
    if (!isFp16Linear(desc))
    {
        return false;
    }
    return pos == kIN_GAMMA ? desc.dims.nbDims == 1 : isSupportedRank(desc.dims.nbDims);
}

int32_t AddRmsNormPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        LOG_ERROR("AddRmsNormPlugin: expected three inputs and two outputs");
        return -1;
    }
    for (int32_t index = 0; index < kNUM_INPUTS; ++index)
    {
        if (!isFp16Linear(in[index].desc))
        {
            LOG_ERROR("AddRmsNormPlugin: inputs must be FP16 linear");
            return -1;
        }
    }
    if (!hasValidShapes(in[kIN_X].max, in[kIN_RESIDUAL].max, in[kIN_GAMMA].max, out[0].max, out[1].max))
    {
        LOG_ERROR("AddRmsNormPlugin: x, residual and outputs must share a rank-2 or rank-3 shape [.., H], gamma is [H], H %% 8 == 0 and H <= %d",
            kernel::kAddRmsNormMaxHidden);
        return -1;
    }
    return 0;
}

int32_t AddRmsNormPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr || outputs == nullptr
        || !hasValidShapes(inputDesc[kIN_X].dims, inputDesc[kIN_RESIDUAL].dims, inputDesc[kIN_GAMMA].dims, outputDesc[0].dims,
            outputDesc[1].dims))
    {
        LOG_ERROR("AddRmsNormPlugin: invalid runtime descriptors");
        return -1;
    }
    Dims const& dims = inputDesc[kIN_X].dims;
    int64_t rows = 1;
    for (int32_t dim = 0; dim + 1 < dims.nbDims; ++dim)
    {
        rows *= dims.d[dim];
    }
    if (rows == 0)
    {
        return 0;
    }
    if (inputs[kIN_X] == nullptr || inputs[kIN_RESIDUAL] == nullptr || inputs[kIN_GAMMA] == nullptr || outputs[0] == nullptr
        || outputs[1] == nullptr || rows > INT32_MAX)
    {
        LOG_ERROR("AddRmsNormPlugin: null buffers or too many rows");
        return -1;
    }
    kernel::launchAddRmsNorm(static_cast<half const*>(inputs[kIN_X]), static_cast<half const*>(inputs[kIN_RESIDUAL]),
        static_cast<half const*>(inputs[kIN_GAMMA]), static_cast<half*>(outputs[0]), static_cast<half*>(outputs[1]),
        static_cast<int32_t>(rows), static_cast<int32_t>(dims.d[dims.nbDims - 1]), mRmsNormEps, mWeightBeforeCast, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t AddRmsNormPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS || in == nullptr || out == nullptr
        || !hasValidShapes(in[kIN_X].dims, in[kIN_RESIDUAL].dims, in[kIN_GAMMA].dims, out[0].dims, out[1].dims))
    {
        LOG_ERROR("AddRmsNormPlugin: invalid runtime input or output shape");
        return -1;
    }
    return 0;
}

IPluginV3* AddRmsNormPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* AddRmsNormPlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back(kFIELD_EPS, &mRmsNormEps, PluginFieldType::kFLOAT32, 1);
        mDataToSerialize.emplace_back(kFIELD_WEIGHT_BEFORE_CAST, &mWeightBeforeCast, PluginFieldType::kINT32, 1);
        mFields.nbFields = static_cast<int32_t>(mDataToSerialize.size());
        mFields.fields = mDataToSerialize.data();
        return &mFields;
    }
    catch (...)
    {
        return nullptr;
    }
}

AddRmsNormPluginCreator::AddRmsNormPluginCreator()
{
    mAttributes.emplace_back(kFIELD_EPS, nullptr, PluginFieldType::kFLOAT32, 1);
    mAttributes.emplace_back(kFIELD_WEIGHT_BEFORE_CAST, nullptr, PluginFieldType::kINT32, 1);
    mFields.nbFields = static_cast<int32_t>(mAttributes.size());
    mFields.fields = mAttributes.data();
}

char const* AddRmsNormPluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* AddRmsNormPluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* AddRmsNormPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* AddRmsNormPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AddRmsNormPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* AddRmsNormPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fields, TensorRTPhase /* phase */) noexcept
{
    try
    {
        if (name == nullptr)
        {
            throw std::invalid_argument("AddRmsNormPlugin: null layer name");
        }
        auto* plugin = new AddRmsNormPlugin(name, requiredField<float>(fields, kFIELD_EPS, PluginFieldType::kFLOAT32),
            requiredField<int32_t>(fields, kFIELD_WEIGHT_BEFORE_CAST, PluginFieldType::kINT32));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("AddRmsNormPluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
