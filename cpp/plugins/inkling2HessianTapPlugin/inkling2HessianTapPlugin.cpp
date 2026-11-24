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
#include "inkling2HessianTapPlugin.h"
#include "common/logger.h"
#include "kernels/inkling2HessianTap/inkling2HessianTapKernels.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm::plugins
{
namespace
{
constexpr char const* kPluginName{"Inkling2HessianTapPlugin"};
constexpr char const* kPluginVersion{"1"};

struct Slot
{
    float* device{nullptr};
    int32_t dim{0};
    int64_t tokens{0};
};

std::mutex gMutex;
std::map<int32_t, Slot> gSlots;
std::atomic<int32_t> gAlive{0};

char const* hessianDir()
{
    static char const* dir = std::getenv("EDGELLM_HESSIAN_DIR");
    return dir;
}

void dumpAll()
{
    char const* dir = hessianDir();
    std::lock_guard<std::mutex> lock(gMutex);
    for (auto& [slot, s] : gSlots)
    {
        if (dir != nullptr && s.device != nullptr && s.tokens > 0)
        {
            std::vector<float> host(static_cast<size_t>(s.dim) * s.dim);
            cudaDeviceSynchronize();
            if (cudaMemcpy(host.data(), s.device, host.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess)
            {
                std::string const base = std::string(dir) + "/h" + std::to_string(slot);
                if (FILE* f = std::fopen((base + ".f32").c_str(), "wb"))
                {
                    std::fwrite(host.data(), sizeof(float), host.size(), f);
                    std::fclose(f);
                }
                if (FILE* f = std::fopen((base + ".n").c_str(), "w"))
                {
                    std::fprintf(f, "%lld %d\n", static_cast<long long>(s.tokens), s.dim);
                    std::fclose(f);
                }
            }
        }
        if (s.device != nullptr)
        {
            cudaFree(s.device);
        }
    }
    gSlots.clear();
}
} // namespace

REGISTER_TENSORRT_PLUGIN(Inkling2HessianTapPluginCreator);

Inkling2HessianTapPlugin::Inkling2HessianTapPlugin(std::string const& name, int32_t dim, int32_t slot)
    : mLayerName(name)
    , mDim(dim)
    , mSlot(slot)
{
    if (dim <= 0 || dim % 64 != 0 || slot < 0)
    {
        throw std::invalid_argument("Inkling2HessianTapPlugin: dim a positive multiple of 64 and slot >= 0");
    }
    ++gAlive;
}

Inkling2HessianTapPlugin::~Inkling2HessianTapPlugin()
{
    if (--gAlive == 0)
    {
        dumpAll();
    }
}

IPluginCapability* Inkling2HessianTapPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Inkling2HessianTapPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Inkling2HessianTapPlugin(mLayerName, mDim, mSlot);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* Inkling2HessianTapPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2HessianTapPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Inkling2HessianTapPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2HessianTapPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t Inkling2HessianTapPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Inkling2HessianTapPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != 1)
    {
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Inkling2HessianTapPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbOutputs != 1 || inputs[0].nbDims != 2)
    {
        return -1;
    }
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(mDim);
    return 0;
}

bool Inkling2HessianTapPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != 1 || nbOutputs != 1 || pos < 0 || pos > 1)
    {
        return false;
    }
    auto const& d = inOut[pos].desc;
    return d.type == DataType::kHALF && d.format == PluginFormat::kLINEAR && d.dims.nbDims == 2;
}

int32_t Inkling2HessianTapPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != 1 || nbOutputs != 1 || in[0].desc.dims.d[1] != mDim)
    {
        LOG_ERROR("Inkling2HessianTapPlugin: input must be [T, %d]", mDim);
        return -1;
    }
    return 0;
}

size_t Inkling2HessianTapPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t Inkling2HessianTapPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr)
    {
        return -1;
    }
    int64_t const tokens = inputDesc[0].dims.d[0];
    if (tokens > 0 && cudaMemcpyAsync(outputs[0], inputs[0], static_cast<size_t>(tokens) * mDim * 2, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
    {
        return -1;
    }
    if (hessianDir() != nullptr && tokens > 0)
    {
        float* device = nullptr;
        {
            std::lock_guard<std::mutex> lock(gMutex);
            Slot& s = gSlots[mSlot];
            if (s.device == nullptr)
            {
                if (cudaMalloc(&s.device, static_cast<size_t>(mDim) * mDim * sizeof(float)) != cudaSuccess)
                {
                    return -1;
                }
                cudaMemset(s.device, 0, static_cast<size_t>(mDim) * mDim * sizeof(float));
                s.dim = mDim;
            }
            s.tokens += tokens;
            device = s.device;
        }
        kernel::launchInkling2HessianAccumulate(inputs[0], device, tokens, mDim, stream);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t Inkling2HessianTapPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return (in != nullptr && out != nullptr && nbInputs == 1 && nbOutputs == 1) ? 0 : -1;
}

IPluginV3* Inkling2HessianTapPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Inkling2HessianTapPlugin::getFieldsToSerialize() noexcept
{
    mData.clear();
    mData.emplace_back("dim", &mDim, PluginFieldType::kINT32, 1);
    mData.emplace_back("slot", &mSlot, PluginFieldType::kINT32, 1);
    mFields.nbFields = static_cast<int32_t>(mData.size());
    mFields.fields = mData.data();
    return &mFields;
}

std::vector<PluginField> Inkling2HessianTapPluginCreator::mAttributes;

Inkling2HessianTapPluginCreator::Inkling2HessianTapPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mAttributes.clear();
    for (char const* name : {"dim", "slot"})
    {
        mAttributes.emplace_back(name, nullptr, PluginFieldType::kINT32, 1);
    }
    mFields.nbFields = static_cast<int32_t>(mAttributes.size());
    mFields.fields = mAttributes.data();
}

char const* Inkling2HessianTapPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2HessianTapPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Inkling2HessianTapPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* Inkling2HessianTapPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2HessianTapPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* Inkling2HessianTapPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    if (name == nullptr || fc == nullptr || (fc->nbFields > 0 && fc->fields == nullptr))
    {
        LOG_ERROR("Inkling2HessianTapPluginCreator: invalid arguments");
        return nullptr;
    }
    try
    {
        std::map<std::string, PluginField const*> byName;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            PluginField const& field = fc->fields[i];
            if (field.name == nullptr || field.data == nullptr || field.length != 1 || field.type != PluginFieldType::kINT32
                || !byName.emplace(field.name, &field).second)
            {
                throw std::invalid_argument("Inkling2HessianTapPlugin: attributes must be unique INT32 scalars");
            }
        }
        if (byName.size() != 2 || byName.count("dim") == 0 || byName.count("slot") == 0)
        {
            throw std::invalid_argument("Inkling2HessianTapPlugin: attributes are dim and slot");
        }
        auto* plugin = new Inkling2HessianTapPlugin(name, *static_cast<int32_t const*>(byName["dim"]->data),
            *static_cast<int32_t const*>(byName["slot"]->data));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Inkling2HessianTapPluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
