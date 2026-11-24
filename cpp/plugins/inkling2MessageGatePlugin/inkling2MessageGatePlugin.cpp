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
#include "inkling2MessageGatePlugin.h"
#include "common/logger.h"
#include "kernels/inkling2MessageGate/inkling2MessageGateKernels.h"

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
constexpr char const* kPluginName{"Inkling2MessageGatePlugin"};
constexpr char const* kPluginVersion{"1"};

constexpr int32_t kNumInputs{5}; // hidden, message, down, forget, gain

bool linearType(PluginTensorDesc const& d, DataType type, int32_t rank)
{
    return d.type == type && d.format == PluginFormat::kLINEAR && d.dims.nbDims == rank;
}
} // namespace

REGISTER_TENSORRT_PLUGIN(Inkling2MessageGatePluginCreator);

Inkling2MessageGatePlugin::Inkling2MessageGatePlugin(std::string const& name, int32_t hidden, int32_t rank, float eps)
    : mLayerName(name)
    , mHidden(hidden)
    , mRank(rank)
    , mEps(eps)
{
    if (hidden <= 0 || hidden > kernel::kInkling2GateMaxHidden || hidden % 8 != 0 || (rank != 64 && rank != 128))
    {
        throw std::invalid_argument("Inkling2MessageGatePlugin: hidden a multiple of 8 in [8, 4096] and rank 64 or 128");
    }
    if (!std::isfinite(eps) || eps <= 0.F)
    {
        throw std::invalid_argument("Inkling2MessageGatePlugin: eps must be finite and positive");
    }
}

IPluginCapability* Inkling2MessageGatePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Inkling2MessageGatePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Inkling2MessageGatePlugin(mLayerName, mHidden, mRank, mEps);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* Inkling2MessageGatePlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2MessageGatePlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Inkling2MessageGatePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2MessageGatePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t Inkling2MessageGatePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Inkling2MessageGatePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != kNumInputs)
    {
        return -1;
    }
    outputTypes[0] = DataType::kFLOAT;
    return 0;
}

int32_t Inkling2MessageGatePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNumInputs || nbOutputs != 1 || inputs[0].nbDims != 2)
    {
        return -1;
    }
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(mHidden);
    return 0;
}

bool Inkling2MessageGatePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNumInputs || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    bool const weight = pos >= 2 && pos < kNumInputs;
    return linearType(inOut[pos].desc, weight ? DataType::kINT32 : DataType::kFLOAT, 2);
}

int32_t Inkling2MessageGatePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNumInputs || nbOutputs != 1)
    {
        LOG_ERROR("Inkling2MessageGatePlugin: expected %d inputs and one output", kNumInputs);
        return -1;
    }
    bool const ok = in[0].desc.dims.d[1] == mHidden && in[1].desc.dims.d[1] == mHidden && in[2].desc.dims.d[0] == mRank
        && in[2].desc.dims.d[1] == mHidden && in[3].desc.dims.d[0] == mHidden && in[3].desc.dims.d[1] == mRank / 2 && in[4].desc.dims.d[0] == mHidden
        && in[4].desc.dims.d[1] == mRank / 2;
    if (!ok)
    {
        LOG_ERROR("Inkling2MessageGatePlugin: tensor shapes do not match hidden size and rank");
        return -1;
    }
    return 0;
}

size_t Inkling2MessageGatePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs, DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    if (inputs == nullptr || nbInputs < 1)
    {
        return 0;
    }
    int64_t const maxTokens = inputs[0].max.d[0];
    return maxTokens > 0 ? kernel::inkling2MessageGateWorkspaceBytes(maxTokens, mRank) : 0;
}

int32_t Inkling2MessageGatePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr || outputs[0] == nullptr || workspace == nullptr)
    {
        LOG_ERROR("Inkling2MessageGatePlugin: null descriptors or buffers");
        return -1;
    }
    for (int32_t i = 0; i < kNumInputs; ++i)
    {
        if (inputs[i] == nullptr)
        {
            LOG_ERROR("Inkling2MessageGatePlugin: null input buffer %d", i);
            return -1;
        }
    }
    kernel::launchInkling2MessageGate(static_cast<float const*>(inputs[0]), static_cast<float const*>(inputs[1]), static_cast<uint32_t const*>(inputs[2]),
        static_cast<uint32_t const*>(inputs[3]), static_cast<uint32_t const*>(inputs[4]), static_cast<float*>(outputs[0]), inputDesc[0].dims.d[0],
        mHidden, mRank, mEps, workspace, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t Inkling2MessageGatePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return (in != nullptr && out != nullptr && nbInputs == kNumInputs && nbOutputs == 1) ? 0 : -1;
}

IPluginV3* Inkling2MessageGatePlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Inkling2MessageGatePlugin::getFieldsToSerialize() noexcept
{
    mData.clear();
    mData.emplace_back("hidden", &mHidden, PluginFieldType::kINT32, 1);
    mData.emplace_back("rank", &mRank, PluginFieldType::kINT32, 1);
    mData.emplace_back("eps", &mEps, PluginFieldType::kFLOAT32, 1);
    mFields.nbFields = static_cast<int32_t>(mData.size());
    mFields.fields = mData.data();
    return &mFields;
}

std::vector<PluginField> Inkling2MessageGatePluginCreator::mAttributes;

Inkling2MessageGatePluginCreator::Inkling2MessageGatePluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mAttributes.clear();
    for (char const* name : {"hidden", "rank"})
    {
        mAttributes.emplace_back(name, nullptr, PluginFieldType::kINT32, 1);
    }
    mAttributes.emplace_back("eps", nullptr, PluginFieldType::kFLOAT32, 1);
    mFields.nbFields = static_cast<int32_t>(mAttributes.size());
    mFields.fields = mAttributes.data();
}

char const* Inkling2MessageGatePluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2MessageGatePluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Inkling2MessageGatePluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* Inkling2MessageGatePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2MessageGatePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* Inkling2MessageGatePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    if (name == nullptr || fc == nullptr || (fc->nbFields > 0 && fc->fields == nullptr))
    {
        LOG_ERROR("Inkling2MessageGatePluginCreator: invalid arguments");
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
                throw std::invalid_argument("Inkling2MessageGatePlugin: attributes must be unique named scalars");
            }
        }
        if (byName.size() != 3)
        {
            throw std::invalid_argument("Inkling2MessageGatePlugin: unknown or missing attributes");
        }
        auto intField = [&](char const* key) {
            auto it = byName.find(key);
            if (it == byName.end() || it->second->type != PluginFieldType::kINT32)
            {
                throw std::invalid_argument(std::string("Inkling2MessageGatePlugin: missing INT32 attribute ") + key);
            }
            return *static_cast<int32_t const*>(it->second->data);
        };
        auto epsIt = byName.find("eps");
        if (epsIt == byName.end() || epsIt->second->type != PluginFieldType::kFLOAT32)
        {
            throw std::invalid_argument("Inkling2MessageGatePlugin: missing FLOAT32 attribute eps");
        }
        auto* plugin = new Inkling2MessageGatePlugin(name, intField("hidden"), intField("rank"), *static_cast<float const*>(epsIt->second->data));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Inkling2MessageGatePluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
