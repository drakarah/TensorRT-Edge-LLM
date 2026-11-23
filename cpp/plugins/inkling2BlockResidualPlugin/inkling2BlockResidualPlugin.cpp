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
#include "inkling2BlockResidualPlugin.h"
#include "common/logger.h"
#include "kernels/inkling2BlockResidual/inkling2BlockResidualKernels.h"

#include <cmath>
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
constexpr char const* kPluginName{"Inkling2BlockResidualPlugin"};
constexpr char const* kPluginVersion{"1"};

bool linearFloat(PluginTensorDesc const& d, int32_t rank)
{
    return d.type == DataType::kFLOAT && d.format == PluginFormat::kLINEAR && d.dims.nbDims == rank;
}
} // namespace

REGISTER_TENSORRT_PLUGIN(Inkling2BlockResidualPluginCreator);

Inkling2BlockResidualPlugin::Inkling2BlockResidualPlugin(
    std::string const& name, int32_t hidden, int32_t numStates, int32_t numQueries, float eps)
    : mLayerName(name)
    , mHidden(hidden)
    , mNumStates(numStates)
    , mNumQueries(numQueries)
    , mEps(eps)
{
    if (hidden <= 0 || hidden > kernel::kInkling2MaxResidualHidden || numStates < 1 || numStates > kernel::kInkling2MaxResidualStates || numQueries < 1
        || numQueries > kernel::kInkling2MaxResidualQueries)
    {
        throw std::invalid_argument(
            "Inkling2BlockResidualPlugin: hidden in [1, 4096], num_states in [1, 8] and num_queries in [1, 4]");
    }
    if (!std::isfinite(eps) || eps <= 0.F)
    {
        throw std::invalid_argument("Inkling2BlockResidualPlugin: eps must be finite and positive");
    }
}

IPluginCapability* Inkling2BlockResidualPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Inkling2BlockResidualPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Inkling2BlockResidualPlugin(mLayerName, mHidden, mNumStates, mNumQueries, mEps);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* Inkling2BlockResidualPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2BlockResidualPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Inkling2BlockResidualPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2BlockResidualPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t Inkling2BlockResidualPlugin::getNbOutputs() const noexcept
{
    return mNumQueries;
}

int32_t Inkling2BlockResidualPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != mNumQueries || nbInputs != mNumStates + 1)
    {
        return -1;
    }
    for (int32_t i = 0; i < nbOutputs; ++i)
    {
        outputTypes[i] = DataType::kFLOAT;
    }
    return 0;
}

int32_t Inkling2BlockResidualPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != mNumStates + 1 || nbOutputs != mNumQueries
        || inputs[0].nbDims != 2)
    {
        return -1;
    }
    for (int32_t i = 0; i < nbOutputs; ++i)
    {
        outputs[i].nbDims = 2;
        outputs[i].d[0] = inputs[0].d[0];
        outputs[i].d[1] = exprBuilder.constant(mHidden);
    }
    return 0;
}

bool Inkling2BlockResidualPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != mNumStates + 1 || nbOutputs != mNumQueries || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    return linearFloat(inOut[pos].desc, 2);
}

int32_t Inkling2BlockResidualPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != mNumStates + 1 || nbOutputs != mNumQueries)
    {
        LOG_ERROR("Inkling2BlockResidualPlugin: expected %d inputs and %d outputs", mNumStates + 1, mNumQueries);
        return -1;
    }
    bool ok = in[mNumStates].desc.dims.d[0] == mNumQueries && in[mNumStates].desc.dims.d[1] == mHidden;
    for (int32_t i = 0; i < mNumStates; ++i)
    {
        ok = ok && in[i].desc.dims.d[1] == mHidden;
    }
    if (!ok)
    {
        LOG_ERROR("Inkling2BlockResidualPlugin: tensor shapes do not match hidden size and query count");
        return -1;
    }
    return 0;
}

size_t Inkling2BlockResidualPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */,
    int32_t /* nbInputs */, DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t Inkling2BlockResidualPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr)
    {
        LOG_ERROR("Inkling2BlockResidualPlugin: null descriptors or buffers");
        return -1;
    }
    kernel::Inkling2BlockResidualParams params{};
    params.numTokens = inputDesc[0].dims.d[0];
    params.hidden = mHidden;
    params.numStates = mNumStates;
    params.numQueries = mNumQueries;
    params.eps = mEps;
    for (int32_t i = 0; i < mNumStates; ++i)
    {
        if (inputs[i] == nullptr)
        {
            LOG_ERROR("Inkling2BlockResidualPlugin: null input buffer %d", i);
            return -1;
        }
        params.states[i] = static_cast<float const*>(inputs[i]);
    }
    for (int32_t i = 0; i < mNumQueries; ++i)
    {
        if (outputs[i] == nullptr)
        {
            LOG_ERROR("Inkling2BlockResidualPlugin: null output buffer %d", i);
            return -1;
        }
        params.outputs[i] = static_cast<float*>(outputs[i]);
    }
    kernel::launchInkling2BlockResidual(params, static_cast<float const*>(inputs[mNumStates]), stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t Inkling2BlockResidualPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return (in != nullptr && out != nullptr && nbInputs == mNumStates + 1 && nbOutputs == mNumQueries) ? 0 : -1;
}

IPluginV3* Inkling2BlockResidualPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Inkling2BlockResidualPlugin::getFieldsToSerialize() noexcept
{
    mData.clear();
    mData.emplace_back("hidden", &mHidden, PluginFieldType::kINT32, 1);
    mData.emplace_back("num_states", &mNumStates, PluginFieldType::kINT32, 1);
    mData.emplace_back("num_queries", &mNumQueries, PluginFieldType::kINT32, 1);
    mData.emplace_back("eps", &mEps, PluginFieldType::kFLOAT32, 1);
    mFields.nbFields = static_cast<int32_t>(mData.size());
    mFields.fields = mData.data();
    return &mFields;
}

std::vector<PluginField> Inkling2BlockResidualPluginCreator::mAttributes;

Inkling2BlockResidualPluginCreator::Inkling2BlockResidualPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mAttributes.clear();
    for (char const* name : {"hidden", "num_states", "num_queries"})
    {
        mAttributes.emplace_back(name, nullptr, PluginFieldType::kINT32, 1);
    }
    mAttributes.emplace_back("eps", nullptr, PluginFieldType::kFLOAT32, 1);
    mFields.nbFields = static_cast<int32_t>(mAttributes.size());
    mFields.fields = mAttributes.data();
}

char const* Inkling2BlockResidualPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2BlockResidualPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Inkling2BlockResidualPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* Inkling2BlockResidualPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2BlockResidualPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* Inkling2BlockResidualPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    if (name == nullptr || fc == nullptr || (fc->nbFields > 0 && fc->fields == nullptr))
    {
        LOG_ERROR("Inkling2BlockResidualPluginCreator: invalid arguments");
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
                throw std::invalid_argument("Inkling2BlockResidualPlugin: attributes must be unique named scalars");
            }
        }
        if (byName.size() != 4)
        {
            throw std::invalid_argument("Inkling2BlockResidualPlugin: unknown or missing attributes");
        }
        auto intField = [&](char const* key) {
            auto it = byName.find(key);
            if (it == byName.end() || it->second->type != PluginFieldType::kINT32)
            {
                throw std::invalid_argument(std::string("Inkling2BlockResidualPlugin: missing INT32 attribute ") + key);
            }
            return *static_cast<int32_t const*>(it->second->data);
        };
        auto epsIt = byName.find("eps");
        if (epsIt == byName.end() || epsIt->second->type != PluginFieldType::kFLOAT32)
        {
            throw std::invalid_argument("Inkling2BlockResidualPlugin: missing FLOAT32 attribute eps");
        }
        auto* plugin = new Inkling2BlockResidualPlugin(name, intField("hidden"), intField("num_states"),
            intField("num_queries"), *static_cast<float const*>(epsIt->second->data));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Inkling2BlockResidualPluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
