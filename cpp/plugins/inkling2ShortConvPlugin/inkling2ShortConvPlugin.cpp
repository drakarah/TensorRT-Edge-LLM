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

#include "inkling2ShortConvPlugin.h"
#include "common/logger.h"
#include "kernels/inkling2ShortConv/inkling2ShortConvKernels.h"

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
constexpr char const* kPluginName{"Inkling2ShortConvPlugin"};
constexpr char const* kPluginVersion{"1"};

enum Input : int32_t
{
    kX = 0,
    kTaps,
    kState,
    kQueryStartOffsets,
    kQueryLengths,
    kStateIndices,
    kDyn,
    kMaxInputs
};
constexpr int32_t kOutY{0};
constexpr int32_t kOutState{1};
constexpr int32_t kNumOutputs{2};

bool linear(PluginTensorDesc const& d, DataType type, int32_t rank)
{
    return d.type == type && d.format == PluginFormat::kLINEAR && d.dims.nbDims == rank;
}
} // namespace

REGISTER_TENSORRT_PLUGIN(Inkling2ShortConvPluginCreator);

Inkling2ShortConvPlugin::Inkling2ShortConvPlugin(
    std::string const& name, int32_t channels, int32_t kernelSize, int32_t groups, float scale, int32_t stateOffset)
    : mLayerName(name)
    , mChannels(channels)
    , mKernelSize(kernelSize)
    , mGroups(groups)
    , mScale(scale)
    , mStateOffset(stateOffset)
{
    if (stateOffset < 0)
    {
        throw std::invalid_argument("Inkling2ShortConvPlugin: state_offset must be non-negative");
    }
    if (channels <= 0 || kernelSize < 2 || kernelSize > kernel::kInkling2MaxConvKernel || groups < 0
        || (groups > 0 && channels % groups != 0))
    {
        throw std::invalid_argument(
            "Inkling2ShortConvPlugin: channels must be positive, kernel_size in [2, 8], groups >= 0 and dividing "
            "channels");
    }
    if (!std::isfinite(scale) || scale <= 0.F)
    {
        throw std::invalid_argument("Inkling2ShortConvPlugin: scale must be finite and positive");
    }
}

int32_t Inkling2ShortConvPlugin::numInputs() const noexcept
{
    return mGroups > 0 ? kMaxInputs : kDyn;
}

IPluginCapability* Inkling2ShortConvPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Inkling2ShortConvPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Inkling2ShortConvPlugin(mLayerName, mChannels, mKernelSize, mGroups, mScale, mStateOffset);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* Inkling2ShortConvPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2ShortConvPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Inkling2ShortConvPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2ShortConvPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t Inkling2ShortConvPlugin::getNbOutputs() const noexcept
{
    return kNumOutputs;
}

int32_t Inkling2ShortConvPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != kNumOutputs || nbInputs != numInputs())
    {
        return -1;
    }
    outputTypes[kOutY] = DataType::kHALF;
    outputTypes[kOutState] = inputTypes[kState];
    return 0;
}

int32_t Inkling2ShortConvPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != numInputs() || nbOutputs != kNumOutputs
        || inputs[kX].nbDims != 2 || inputs[kState].nbDims != 3)
    {
        return -1;
    }
    outputs[kOutY].nbDims = 2;
    outputs[kOutY].d[0] = inputs[kX].d[0];
    outputs[kOutY].d[1] = exprBuilder.constant(mChannels);
    outputs[kOutState] = inputs[kState];
    return 0;
}

bool Inkling2ShortConvPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != numInputs() || nbOutputs != kNumOutputs || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    PluginTensorDesc const& d = inOut[pos].desc;
    if (pos == kX || pos == kDyn)
    {
        return linear(d, DataType::kHALF, 2);
    }
    switch (pos)
    {
    case kTaps: return linear(d, DataType::kFLOAT, 2);
    case kState: return linear(d, DataType::kHALF, 3);
    case kQueryStartOffsets:
    case kQueryLengths:
    case kStateIndices: return linear(d, DataType::kINT32, 1);
    default: break;
    }
    if (pos == nbInputs + kOutY)
    {
        return linear(d, DataType::kHALF, 2);
    }
    return pos == nbInputs + kOutState && linear(d, DataType::kHALF, 3);
}

int32_t Inkling2ShortConvPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != numInputs() || nbOutputs != kNumOutputs)
    {
        LOG_ERROR("Inkling2ShortConvPlugin: expected %d inputs and %d outputs", numInputs(), kNumOutputs);
        return -1;
    }
    Dims const& state = in[kState].desc.dims;
    bool ok = in[kX].desc.dims.d[1] == mChannels && in[kTaps].desc.dims.d[0] == mChannels
        && in[kTaps].desc.dims.d[1] == mKernelSize && state.d[1] >= mStateOffset + mChannels
        && state.d[2] == mKernelSize;
    if (mGroups > 0)
    {
        ok = ok && in[kDyn].desc.dims.d[1] == mGroups * mKernelSize;
    }
    if (!ok)
    {
        LOG_ERROR("Inkling2ShortConvPlugin: tensor shapes do not match channels, kernel size and groups");
        return -1;
    }
    return 0;
}

size_t Inkling2ShortConvPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t Inkling2ShortConvPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr || outputs[kOutY] == nullptr)
    {
        LOG_ERROR("Inkling2ShortConvPlugin: null descriptors or buffers");
        return -1;
    }
    for (int32_t i = 0; i < numInputs(); ++i)
    {
        if (inputs[i] == nullptr)
        {
            LOG_ERROR("Inkling2ShortConvPlugin: null input buffer %d", i);
            return -1;
        }
    }
    kernel::Inkling2ShortConvParams params{};
    params.numTokens = inputDesc[kX].dims.d[0];
    params.batchSize = inputDesc[kStateIndices].dims.d[0];
    params.channels = mChannels;
    params.stateStride = inputDesc[kState].dims.d[1];
    params.stateOffset = mStateOffset;
    params.stateRows = inputDesc[kState].dims.d[0];
    params.kernelSize = mKernelSize;
    params.groups = mGroups;
    params.scale = mScale;
    if (inputDesc[kQueryStartOffsets].dims.d[0] != params.batchSize + 1
        || inputDesc[kQueryLengths].dims.d[0] != params.batchSize)
    {
        LOG_ERROR(
            "Inkling2ShortConvPlugin: query_start_offsets needs batch + 1 entries, query_lengths one per sequence");
        return -1;
    }
    // Read and written in place through the input binding: the state is a resident pool, and the convolutions of one
    // visit share it over disjoint channel ranges.
    kernel::launchInkling2ShortConv(params, static_cast<half const*>(inputs[kX]),
        static_cast<float const*>(inputs[kTaps]), mGroups > 0 ? static_cast<half const*>(inputs[kDyn]) : nullptr,
        const_cast<half*>(static_cast<half const*>(inputs[kState])),
        static_cast<int32_t const*>(inputs[kQueryStartOffsets]), static_cast<int32_t const*>(inputs[kQueryLengths]),
        static_cast<int32_t const*>(inputs[kStateIndices]), static_cast<half*>(outputs[kOutY]), stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t Inkling2ShortConvPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return (in != nullptr && out != nullptr && nbInputs == numInputs() && nbOutputs == kNumOutputs) ? 0 : -1;
}

IPluginV3* Inkling2ShortConvPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Inkling2ShortConvPlugin::getFieldsToSerialize() noexcept
{
    mData.clear();
    mData.emplace_back("channels", &mChannels, PluginFieldType::kINT32, 1);
    mData.emplace_back("kernel_size", &mKernelSize, PluginFieldType::kINT32, 1);
    mData.emplace_back("groups", &mGroups, PluginFieldType::kINT32, 1);
    mData.emplace_back("state_offset", &mStateOffset, PluginFieldType::kINT32, 1);
    mData.emplace_back("scale", &mScale, PluginFieldType::kFLOAT32, 1);
    mFields.nbFields = static_cast<int32_t>(mData.size());
    mFields.fields = mData.data();
    return &mFields;
}

std::vector<PluginField> Inkling2ShortConvPluginCreator::mAttributes;

Inkling2ShortConvPluginCreator::Inkling2ShortConvPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mAttributes.clear();
    for (char const* name : {"channels", "kernel_size", "groups", "state_offset"})
    {
        mAttributes.emplace_back(name, nullptr, PluginFieldType::kINT32, 1);
    }
    mAttributes.emplace_back("scale", nullptr, PluginFieldType::kFLOAT32, 1);
    mFields.nbFields = static_cast<int32_t>(mAttributes.size());
    mFields.fields = mAttributes.data();
}

char const* Inkling2ShortConvPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2ShortConvPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Inkling2ShortConvPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* Inkling2ShortConvPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2ShortConvPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* Inkling2ShortConvPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    if (name == nullptr || fc == nullptr || (fc->nbFields > 0 && fc->fields == nullptr))
    {
        LOG_ERROR("Inkling2ShortConvPluginCreator: invalid arguments");
        return nullptr;
    }
    try
    {
        std::map<std::string, PluginField const*> byName;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            PluginField const& field = fc->fields[i];
            if (field.name == nullptr || field.data == nullptr || field.length != 1)
            {
                throw std::invalid_argument("Inkling2ShortConvPlugin: every attribute must be a named scalar");
            }
            if (!byName.emplace(field.name, &field).second)
            {
                throw std::invalid_argument(std::string("Inkling2ShortConvPlugin: duplicate attribute ") + field.name);
            }
        }
        if (byName.size() != 5)
        {
            throw std::invalid_argument("Inkling2ShortConvPlugin: unknown or missing attributes");
        }
        auto intField = [&](char const* key) {
            auto it = byName.find(key);
            if (it == byName.end() || it->second->type != PluginFieldType::kINT32)
            {
                throw std::invalid_argument(std::string("Inkling2ShortConvPlugin: missing INT32 attribute ") + key);
            }
            return *static_cast<int32_t const*>(it->second->data);
        };
        auto scaleIt = byName.find("scale");
        if (scaleIt == byName.end() || scaleIt->second->type != PluginFieldType::kFLOAT32)
        {
            throw std::invalid_argument("Inkling2ShortConvPlugin: missing FLOAT32 attribute scale");
        }
        auto* plugin = new Inkling2ShortConvPlugin(name, intField("channels"), intField("kernel_size"),
            intField("groups"), *static_cast<float const*>(scaleIt->second->data), intField("state_offset"));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Inkling2ShortConvPluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
