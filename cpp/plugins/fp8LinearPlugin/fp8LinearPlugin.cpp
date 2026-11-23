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
#include "fp8LinearPlugin.h"
#include "common/logger.h"
#include "kernels/fp8Linear/fp8LinearKernels.h"

#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
namespace
{
constexpr char const* kPluginName{"Fp8LinearPlugin"};
constexpr char const* kPluginVersion{"1"};
constexpr int32_t kNumInputs{3};
constexpr int32_t kInX{0};
constexpr int32_t kInWeight{1};
constexpr int32_t kInScale{2};

bool linear(PluginTensorDesc const& d, DataType type, int32_t rank)
{
    return d.type == type && d.format == PluginFormat::kLINEAR && d.dims.nbDims == rank;
}
} // namespace

REGISTER_TENSORRT_PLUGIN(Fp8LinearPluginCreator);

Fp8LinearPlugin::Fp8LinearPlugin(std::string const& name, int32_t outFeatures, int32_t inFeatures)
    : mLayerName(name)
    , mOutFeatures(outFeatures)
    , mInFeatures(inFeatures)
{
    if (outFeatures <= 0 || outFeatures % 8 != 0 || inFeatures <= 0 || inFeatures % 32 != 0)
    {
        throw std::invalid_argument("Fp8LinearPlugin: out_features must be a positive multiple of 8 and in_features a positive multiple of 32");
    }
}

IPluginCapability* Fp8LinearPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Fp8LinearPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Fp8LinearPlugin(mLayerName, mOutFeatures, mInFeatures);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* Fp8LinearPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Fp8LinearPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Fp8LinearPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Fp8LinearPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t Fp8LinearPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Fp8LinearPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != kNumInputs)
    {
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Fp8LinearPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */,
    DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNumInputs || nbOutputs != 1 || inputs[kInX].nbDims != 2)
    {
        return -1;
    }
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[kInX].d[0];
    outputs[0].d[1] = exprBuilder.constant(mOutFeatures);
    return 0;
}

bool Fp8LinearPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNumInputs || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    PluginTensorDesc const& d = inOut[pos].desc;
    switch (pos)
    {
    case kInX: return linear(d, DataType::kHALF, 2);
    case kInWeight: return linear(d, DataType::kINT32, 2);
    case kInScale: return linear(d, DataType::kHALF, 1) || linear(d, DataType::kFLOAT, 1);
    case kNumInputs: return linear(d, DataType::kHALF, 2);
    default: return false;
    }
}

int32_t Fp8LinearPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNumInputs || nbOutputs != 1)
    {
        LOG_ERROR("Fp8LinearPlugin: expected %d inputs and one output", kNumInputs);
        return -1;
    }
    if (in[kInX].desc.dims.d[1] != mInFeatures || in[kInWeight].desc.dims.d[0] != mOutFeatures || in[kInWeight].desc.dims.d[1] * 4 != mInFeatures
        || in[kInScale].desc.dims.d[0] != mOutFeatures)
    {
        LOG_ERROR("Fp8LinearPlugin %s: shapes do not match out_features %d, in_features %d", mLayerName.c_str(), mOutFeatures, mInFeatures);
        return -1;
    }
    return 0;
}

size_t Fp8LinearPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */, DynamicPluginTensorDesc const* /* outputs */,
    int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t Fp8LinearPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs,
    void* /* workspace */, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr)
    {
        return -1;
    }
    int32_t const tokens = inputDesc[kInX].dims.d[0];
    if (tokens < 0)
    {
        return -1;
    }
    kernel::launchFp8Linear(inputs[kInX], inputs[kInWeight], inputs[kInScale], outputs[0], tokens, mOutFeatures, mInFeatures,
        inputDesc[kInScale].type == DataType::kFLOAT, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t Fp8LinearPlugin::onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return (in != nullptr && out != nullptr && nbInputs == kNumInputs && nbOutputs == 1) ? 0 : -1;
}

IPluginV3* Fp8LinearPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Fp8LinearPlugin::getFieldsToSerialize() noexcept
{
    mData.clear();
    mData.emplace_back("out_features", &mOutFeatures, PluginFieldType::kINT32, 1);
    mData.emplace_back("in_features", &mInFeatures, PluginFieldType::kINT32, 1);
    mFields.nbFields = static_cast<int32_t>(mData.size());
    mFields.fields = mData.data();
    return &mFields;
}

std::vector<PluginField> Fp8LinearPluginCreator::mAttributes;

Fp8LinearPluginCreator::Fp8LinearPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mAttributes.clear();
    for (char const* name : {"out_features", "in_features"})
    {
        mAttributes.emplace_back(name, nullptr, PluginFieldType::kINT32, 1);
    }
    mFields.nbFields = static_cast<int32_t>(mAttributes.size());
    mFields.fields = mAttributes.data();
}

char const* Fp8LinearPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Fp8LinearPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Fp8LinearPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* Fp8LinearPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Fp8LinearPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* Fp8LinearPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    if (name == nullptr || fc == nullptr || (fc->nbFields > 0 && fc->fields == nullptr))
    {
        LOG_ERROR("Fp8LinearPluginCreator: invalid arguments");
        return nullptr;
    }
    try
    {
        std::map<std::string, PluginField const*> byName;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            PluginField const& field = fc->fields[i];
            if (field.name == nullptr || field.data == nullptr || field.length != 1 || field.type != PluginFieldType::kINT32)
            {
                throw std::invalid_argument("Fp8LinearPlugin: every attribute must be a named INT32 scalar");
            }
            if (!byName.emplace(field.name, &field).second)
            {
                throw std::invalid_argument(std::string("Fp8LinearPlugin: duplicate attribute ") + field.name);
            }
        }
        if (byName.size() != 2 || byName.count("out_features") == 0 || byName.count("in_features") == 0)
        {
            throw std::invalid_argument("Fp8LinearPlugin: unknown or missing attributes");
        }
        auto* plugin = new Fp8LinearPlugin(
            name, *static_cast<int32_t const*>(byName["out_features"]->data), *static_cast<int32_t const*>(byName["in_features"]->data));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp8LinearPluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
