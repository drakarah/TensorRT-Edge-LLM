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

#include "inkling2NormRoutePlugin.h"
#include "common/logger.h"
#include "kernels/inkling2NormRoute/inkling2NormRouteKernels.h"

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
namespace
{
constexpr char const* kPluginName{"Inkling2NormRoutePlugin"};
constexpr char const* kPluginVersion{"1"};

bool linearType(PluginTensorDesc const& d, DataType type, int32_t rank)
{
    return d.type == type && d.format == PluginFormat::kLINEAR && d.dims.nbDims == rank;
}
} // namespace

REGISTER_TENSORRT_PLUGIN(Inkling2NormRoutePluginCreator);

Inkling2NormRoutePlugin::Inkling2NormRoutePlugin(std::string const& name, int32_t hidden, int32_t experts, float eps)
    : mLayerName(name)
    , mHidden(hidden)
    , mExperts(experts)
    , mEps(eps)
{
    if (hidden <= 0 || hidden > kernel::kInkling2NormRouteMaxHidden || hidden % 4 != 0 || experts < 0)
    {
        throw std::invalid_argument("Inkling2NormRoutePlugin: hidden a multiple of 4 in [4, 4096] and experts >= 0");
    }
    if (!std::isfinite(eps) || eps <= 0.F)
    {
        throw std::invalid_argument("Inkling2NormRoutePlugin: eps must be finite and positive");
    }
}

IPluginCapability* Inkling2NormRoutePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Inkling2NormRoutePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Inkling2NormRoutePlugin(mLayerName, mHidden, mExperts, mEps);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* Inkling2NormRoutePlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2NormRoutePlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Inkling2NormRoutePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2NormRoutePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t Inkling2NormRoutePlugin::getNbOutputs() const noexcept
{
    return numOutputs();
}

int32_t Inkling2NormRoutePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != numOutputs() || nbInputs != numInputs())
    {
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    if (mExperts > 0)
    {
        outputTypes[1] = DataType::kFLOAT;
    }
    return 0;
}

int32_t Inkling2NormRoutePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != numInputs() || nbOutputs != numOutputs() || inputs[0].nbDims != 2)
    {
        return -1;
    }
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(mHidden);
    if (mExperts > 0)
    {
        outputs[1].nbDims = 2;
        outputs[1].d[0] = inputs[0].d[0];
        outputs[1].d[1] = exprBuilder.constant(mExperts);
    }
    return 0;
}

bool Inkling2NormRoutePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != numInputs() || nbOutputs != numOutputs() || pos < 0 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    if (pos == 0)
    {
        return linearType(inOut[pos].desc, DataType::kFLOAT, 2);
    }
    if (pos == 1)
    {
        return linearType(inOut[pos].desc, DataType::kFLOAT, 1);
    }
    if (pos == 2 && mExperts > 0)
    {
        return linearType(inOut[pos].desc, DataType::kFLOAT, 2);
    }
    if (pos == nbInputs)
    {
        return linearType(inOut[pos].desc, DataType::kHALF, 2);
    }
    return linearType(inOut[pos].desc, DataType::kFLOAT, 2);
}

int32_t Inkling2NormRoutePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != numInputs() || nbOutputs != numOutputs())
    {
        LOG_ERROR("Inkling2NormRoutePlugin: expected %d inputs and %d outputs", numInputs(), numOutputs());
        return -1;
    }
    bool ok = in[0].desc.dims.d[1] == mHidden && in[1].desc.dims.d[0] == mHidden;
    if (mExperts > 0)
    {
        ok = ok && in[2].desc.dims.d[0] == mExperts && in[2].desc.dims.d[1] == mHidden;
    }
    if (!ok)
    {
        LOG_ERROR("Inkling2NormRoutePlugin: tensor shapes do not match hidden size and expert count");
        return -1;
    }
    return 0;
}

size_t Inkling2NormRoutePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs, DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    if (inputs == nullptr || nbInputs < 1 || mExperts == 0)
    {
        return 0;
    }
    int64_t const maxTokens = inputs[0].max.d[0];
    return maxTokens > 0 ? kernel::inkling2NormRouteWorkspaceBytes(maxTokens) : 0;
}

int32_t Inkling2NormRoutePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr)
    {
        LOG_ERROR("Inkling2NormRoutePlugin: null descriptors or buffers");
        return -1;
    }
    for (int32_t i = 0; i < numInputs(); ++i)
    {
        if (inputs[i] == nullptr)
        {
            LOG_ERROR("Inkling2NormRoutePlugin: null input buffer %d", i);
            return -1;
        }
    }
    for (int32_t i = 0; i < numOutputs(); ++i)
    {
        if (outputs[i] == nullptr)
        {
            LOG_ERROR("Inkling2NormRoutePlugin: null output buffer %d", i);
            return -1;
        }
    }
    int64_t const tokens = inputDesc[0].dims.d[0];
    if (mExperts > 0 && tokens > 0 && workspace == nullptr)
    {
        LOG_ERROR("Inkling2NormRoutePlugin: null workspace");
        return -1;
    }
    kernel::launchInkling2NormRoute(static_cast<float const*>(inputs[0]), static_cast<float const*>(inputs[1]),
        mExperts > 0 ? static_cast<float const*>(inputs[2]) : nullptr, static_cast<__half*>(outputs[0]),
        mExperts > 0 ? static_cast<float*>(outputs[1]) : nullptr, tokens, mHidden, mExperts, mEps, workspace, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t Inkling2NormRoutePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return (in != nullptr && out != nullptr && nbInputs == numInputs() && nbOutputs == numOutputs()) ? 0 : -1;
}

IPluginV3* Inkling2NormRoutePlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Inkling2NormRoutePlugin::getFieldsToSerialize() noexcept
{
    mData.clear();
    mData.emplace_back("hidden", &mHidden, PluginFieldType::kINT32, 1);
    mData.emplace_back("experts", &mExperts, PluginFieldType::kINT32, 1);
    mData.emplace_back("eps", &mEps, PluginFieldType::kFLOAT32, 1);
    mFields.nbFields = static_cast<int32_t>(mData.size());
    mFields.fields = mData.data();
    return &mFields;
}

std::vector<PluginField> Inkling2NormRoutePluginCreator::mAttributes;

Inkling2NormRoutePluginCreator::Inkling2NormRoutePluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mAttributes.clear();
    for (char const* name : {"hidden", "experts"})
    {
        mAttributes.emplace_back(name, nullptr, PluginFieldType::kINT32, 1);
    }
    mAttributes.emplace_back("eps", nullptr, PluginFieldType::kFLOAT32, 1);
    mFields.nbFields = static_cast<int32_t>(mAttributes.size());
    mFields.fields = mAttributes.data();
}

char const* Inkling2NormRoutePluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2NormRoutePluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Inkling2NormRoutePluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* Inkling2NormRoutePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2NormRoutePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* Inkling2NormRoutePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    if (name == nullptr || fc == nullptr || (fc->nbFields > 0 && fc->fields == nullptr))
    {
        LOG_ERROR("Inkling2NormRoutePluginCreator: invalid arguments");
        return nullptr;
    }
    try
    {
        std::map<std::string, PluginField const*> byName;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            PluginField const& field = fc->fields[i];
            if (field.name == nullptr || field.data == nullptr || field.length != 1
                || !byName.emplace(field.name, &field).second)
            {
                throw std::invalid_argument("Inkling2NormRoutePlugin: attributes must be unique named scalars");
            }
        }
        if (byName.size() != 3)
        {
            throw std::invalid_argument("Inkling2NormRoutePlugin: unknown or missing attributes");
        }
        auto intField = [&](char const* key) {
            auto it = byName.find(key);
            if (it == byName.end() || it->second->type != PluginFieldType::kINT32)
            {
                throw std::invalid_argument(std::string("Inkling2NormRoutePlugin: missing INT32 attribute ") + key);
            }
            return *static_cast<int32_t const*>(it->second->data);
        };
        auto epsIt = byName.find("eps");
        if (epsIt == byName.end() || epsIt->second->type != PluginFieldType::kFLOAT32)
        {
            throw std::invalid_argument("Inkling2NormRoutePlugin: missing FLOAT32 attribute eps");
        }
        auto* plugin = new Inkling2NormRoutePlugin(name, intField("hidden"), intField("experts"), *static_cast<float const*>(epsIt->second->data));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Inkling2NormRoutePluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
