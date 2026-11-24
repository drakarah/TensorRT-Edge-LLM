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

#include "inkling2AttentionPlugin.h"
#include "common/logger.h"
#include "kernels/inkling2Attention/inkling2AttentionKernels.h"

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
constexpr char const* kPluginName{"Inkling2AttentionPlugin"};
constexpr char const* kPluginVersion{"1"};
constexpr int32_t kHeadDim{kernel::kInkling2HeadDim};
constexpr int32_t kRelDim{kernel::kInkling2RelDim};
//! Fused q/gate/rel/sink row: per query head 128 + 128 + 16 + 1 values.
constexpr int32_t fusedRowFactor()
{
    return 2 * kernel::kInkling2HeadDim + kernel::kInkling2RelDim + 1;
}
constexpr int32_t kTokensPerPage{kernel::kInkling2TokensPerPage};

enum Input : int32_t
{
    kQLin = 0,
    kRelLin,
    kSinkLin,
    kGateLin,
    kKNew,
    kVNew,
    kQScale,
    kRelScale,
    kRelProjT,
    kPastKv,
    kQueryStartOffsets,
    kQueryLengths,
    kPastLengths,
    kPageTable,
    kNumInputs
};
constexpr int32_t kOutAttention{0};
constexpr int32_t kOutKv{1};
constexpr int32_t kNumOutputs{2};

bool linear(PluginTensorDesc const& d, DataType type, int32_t rank)
{
    return d.type == type && d.format == PluginFormat::kLINEAR && d.dims.nbDims == rank;
}

bool allPositive(Dims const& d)
{
    for (int32_t i = 0; i < d.nbDims; ++i)
    {
        if (d.d[i] <= 0)
        {
            return false;
        }
    }
    return true;
}
} // namespace

REGISTER_TENSORRT_PLUGIN(Inkling2AttentionPluginCreator);

Inkling2AttentionPlugin::Inkling2AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads,
    int32_t relExtent, int32_t window, float logAlpha, float logNFloor, float qClip, float relClip, float scaling, int32_t kvInt8, int32_t fusedQrsg)
    : mLayerName(name)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mRelExtent(relExtent)
    , mWindow(window)
    , mLogAlpha(logAlpha)
    , mLogNFloor(logNFloor)
    , mQClip(qClip)
    , mRelClip(relClip)
    , mScaling(scaling)
    , mFusedQrsg(fusedQrsg)
    , mKvInt8(kvInt8)
{
    if (numQHeads <= 0 || numKVHeads <= 0 || numQHeads % numKVHeads != 0)
    {
        throw std::invalid_argument("Inkling2AttentionPlugin: num_q_heads must be a positive multiple of num_kv_heads");
    }
    if (kvInt8 != 0 && kvInt8 != 1)
    {
        throw std::invalid_argument("Inkling2AttentionPlugin: kv_int8 must be 0 or 1");
    }
    if (fusedQrsg != 0 && fusedQrsg != 1)
    {
        throw std::invalid_argument("Inkling2AttentionPlugin: fused_qrsg must be 0 or 1");
    }
    if (relExtent <= 0 || window < 0)
    {
        throw std::invalid_argument("Inkling2AttentionPlugin: rel_extent must be positive and window non-negative");
    }
    if (!std::isfinite(scaling) || scaling <= 0.F || !std::isfinite(qClip) || qClip <= 0.F || !std::isfinite(relClip)
        || relClip <= 0.F || !std::isfinite(logAlpha) || !std::isfinite(logNFloor) || logNFloor <= 0.F)
    {
        throw std::invalid_argument(
            "Inkling2AttentionPlugin: scaling, clips and log_n_floor must be finite and positive");
    }
}

IPluginCapability* Inkling2AttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Inkling2AttentionPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Inkling2AttentionPlugin(mLayerName, mNumQHeads, mNumKVHeads, mRelExtent, mWindow, mLogAlpha,
            mLogNFloor, mQClip, mRelClip, mScaling, mKvInt8, mFusedQrsg);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (...)
    {
        return nullptr;
    }
}

char const* Inkling2AttentionPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2AttentionPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

char const* Inkling2AttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2AttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

int32_t Inkling2AttentionPlugin::getNbOutputs() const noexcept
{
    return kNumOutputs;
}

int32_t Inkling2AttentionPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != kNumOutputs || nbInputs != kNumInputs)
    {
        return -1;
    }
    outputTypes[kOutAttention] = DataType::kHALF;
    outputTypes[kOutKv] = inputTypes[kPastKv];
    return 0;
}

int32_t Inkling2AttentionPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNumInputs || nbOutputs != kNumOutputs
        || inputs[kQLin].nbDims != 2 || inputs[kPastKv].nbDims != 5)
    {
        return -1;
    }
    outputs[kOutAttention].nbDims = 2;
    outputs[kOutAttention].d[0] = inputs[kQLin].d[0];
    outputs[kOutAttention].d[1] = exprBuilder.constant(mNumQHeads * kHeadDim);
    outputs[kOutKv] = inputs[kPastKv];
    return 0;
}

bool Inkling2AttentionPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNumInputs || nbOutputs != kNumOutputs || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    PluginTensorDesc const& d = inOut[pos].desc;
    switch (pos)
    {
    case kQLin:
    case kRelLin:
    case kSinkLin:
    case kGateLin:
    case kKNew:
    case kVNew: return linear(d, DataType::kHALF, 2);
    case kQScale:
    case kRelScale: return linear(d, DataType::kFLOAT, 1);
    case kRelProjT: return linear(d, DataType::kFLOAT, 2);
    case kPastKv: return linear(d, DataType::kHALF, 5) || linear(d, DataType::kFP8, 5) || linear(d, DataType::kINT8, 5);
    case kQueryStartOffsets:
    case kQueryLengths:
    case kPastLengths: return linear(d, DataType::kINT32, 1);
    case kPageTable: return linear(d, DataType::kINT32, 3);
    case kNumInputs + kOutAttention: return linear(d, DataType::kHALF, 2);
    case kNumInputs + kOutKv: return linear(d, inOut[kPastKv].desc.type, 5);
    default: return false;
    }
}

int32_t Inkling2AttentionPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNumInputs || nbOutputs != kNumOutputs)
    {
        LOG_ERROR("Inkling2AttentionPlugin: expected %d inputs and %d outputs", kNumInputs, kNumOutputs);
        return -1;
    }
    Dims const& kv = in[kPastKv].desc.dims;
    struct Check
    {
        char const* what;
        bool ok;
    };
    Check const checks[] = {
        {"past_kv must be [2, pages, 128, num_kv_heads, 128]",
            kv.nbDims == 5 && kv.d[0] == 2 && kv.d[2] == kTokensPerPage && kv.d[3] == mNumKVHeads
                && kv.d[4] == (mKvInt8 != 0 ? kernel::kInkling2Int8RowBytes : kHeadDim)},
        {"INT8 KV rows need a one-byte (INT8 or FP8 typed) pool",
            mKvInt8 == 0 || in[kPastKv].desc.type == DataType::kINT8 || in[kPastKv].desc.type == DataType::kFP8},
        {"q_lin width must be num_q_heads * 128 (num_q_heads * 273 when fused)",
            in[kQLin].desc.dims.d[1] == (mFusedQrsg != 0 ? mNumQHeads * fusedRowFactor() : mNumQHeads * kHeadDim)},
        {"rel_lin width must be num_q_heads * 16 (the fused width when fused)",
            in[kRelLin].desc.dims.d[1] == (mFusedQrsg != 0 ? mNumQHeads * fusedRowFactor() : mNumQHeads * kRelDim)},
        {"sink_lin width must be num_q_heads (the fused width when fused)",
            in[kSinkLin].desc.dims.d[1] == (mFusedQrsg != 0 ? mNumQHeads * fusedRowFactor() : mNumQHeads)},
        {"gate_lin width must be num_q_heads * 128 (the fused width when fused)",
            in[kGateLin].desc.dims.d[1] == (mFusedQrsg != 0 ? mNumQHeads * fusedRowFactor() : mNumQHeads * kHeadDim)},
        {"k_new width must be num_kv_heads * 128", in[kKNew].desc.dims.d[1] == mNumKVHeads * kHeadDim},
        {"v_new width must be num_kv_heads * 128", in[kVNew].desc.dims.d[1] == mNumKVHeads * kHeadDim},
        {"q_scale must have num_q_heads * 128 entries", in[kQScale].desc.dims.d[0] == mNumQHeads * kHeadDim},
        {"rel_scale must have num_q_heads * 16 entries", in[kRelScale].desc.dims.d[0] == mNumQHeads * kRelDim},
        {"rel_proj_t must be [rel_extent, 16]",
            in[kRelProjT].desc.dims.d[0] == mRelExtent && in[kRelProjT].desc.dims.d[1] == kRelDim},
        {"kv_page_table must be [batch, 2, pages]", in[kPageTable].desc.dims.d[1] == 2},
    };
    for (Check const& check : checks)
    {
        if (!check.ok)
        {
            LOG_ERROR("Inkling2AttentionPlugin: %s", check.what);
            return -1;
        }
    }
    return 0;
}

size_t Inkling2AttentionPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    if (inputs == nullptr || nbInputs != kNumInputs || inputs[kQLin].max.nbDims != 2)
    {
        return 0;
    }
    // partial results of the key splits of the tiled attention kernels
    return kernel::inkling2AttentionWorkspaceBytes(inputs[kQLin].max.d[0], mNumQHeads);
}

int32_t Inkling2AttentionPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr || outputs[kOutAttention] == nullptr)
    {
        LOG_ERROR("Inkling2AttentionPlugin: null descriptors or buffers");
        return -1;
    }
    for (int32_t i = 0; i < kNumInputs; ++i)
    {
        if (inputs[i] == nullptr && i != kPastKv)
        {
            LOG_ERROR("Inkling2AttentionPlugin: null input buffer %d", i);
            return -1;
        }
        if (!allPositive(inputDesc[i].dims))
        {
            LOG_ERROR("Inkling2AttentionPlugin: input %d has an empty dimension", i);
            return -1;
        }
    }
    kernel::Inkling2AttentionParams params{};
    params.numTokens = inputDesc[kQLin].dims.d[0];
    params.batchSize = inputDesc[kPastLengths].dims.d[0];
    params.numQHeads = mNumQHeads;
    params.numKVHeads = mNumKVHeads;
    params.relExtent = mRelExtent;
    params.window = mWindow;
    params.maxPagesPerSeq = inputDesc[kPageTable].dims.d[2];
    params.kvFp8 = inputDesc[kPastKv].type == DataType::kFP8 && mKvInt8 == 0 ? 1 : 0;
    params.kvInt8 = mKvInt8;
    params.scaling = mScaling;
    params.logAlpha = mLogAlpha;
    params.logNFloor = mLogNFloor;
    params.qClip = mQClip;
    params.relClip = mRelClip;
    half const* qPtr = static_cast<half const*>(inputs[kQLin]);
    half const* relPtr = static_cast<half const*>(inputs[kRelLin]);
    half const* sinkPtr = static_cast<half const*>(inputs[kSinkLin]);
    half const* gatePtr = static_cast<half const*>(inputs[kGateLin]);
    if (mFusedQrsg != 0)
    {
        // one [T, nh * 273] tensor laid out [q | gate | rel | sink]
        int32_t const ld = mNumQHeads * fusedRowFactor();
        params.qLd = params.relLd = params.sinkLd = params.gateLd = ld;
        gatePtr = qPtr + mNumQHeads * kHeadDim;
        relPtr = qPtr + 2 * mNumQHeads * kHeadDim;
        sinkPtr = relPtr + mNumQHeads * kRelDim;
    }
    if (inputDesc[kQueryStartOffsets].dims.d[0] != params.batchSize + 1
        || inputDesc[kQueryLengths].dims.d[0] != params.batchSize
        || inputDesc[kPageTable].dims.d[0] != params.batchSize)
    {
        LOG_ERROR(
            "Inkling2AttentionPlugin: query_start_offsets needs batch + 1 entries, query_lengths and the page table "
            "one per sequence");
        return -1;
    }
    // The KV pool is aliased: the plugin reads and writes the output buffer, which is the input buffer.
    void* kvPool = outputs[kOutKv];
    kernel::launchInkling2Attention(params, qPtr, relPtr, sinkPtr, gatePtr, static_cast<float const*>(inputs[kQScale]),
        static_cast<float const*>(inputs[kRelScale]), static_cast<float const*>(inputs[kRelProjT]),
        static_cast<half const*>(inputs[kKNew]), static_cast<half const*>(inputs[kVNew]), kvPool,
        static_cast<int32_t const*>(inputs[kQueryStartOffsets]), static_cast<int32_t const*>(inputs[kQueryLengths]),
        static_cast<int32_t const*>(inputs[kPastLengths]), static_cast<int32_t const*>(inputs[kPageTable]),
        static_cast<half*>(outputs[kOutAttention]), workspace,
        workspace != nullptr ? kernel::inkling2AttentionWorkspaceBytes(params.numTokens, mNumQHeads) : 0, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int32_t Inkling2AttentionPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kNumInputs || nbOutputs != kNumOutputs)
    {
        return -1;
    }
    return 0;
}

IPluginV3* Inkling2AttentionPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Inkling2AttentionPlugin::getFieldsToSerialize() noexcept
{
    mData.clear();
    mData.emplace_back("num_q_heads", &mNumQHeads, PluginFieldType::kINT32, 1);
    mData.emplace_back("num_kv_heads", &mNumKVHeads, PluginFieldType::kINT32, 1);
    mData.emplace_back("rel_extent", &mRelExtent, PluginFieldType::kINT32, 1);
    mData.emplace_back("window", &mWindow, PluginFieldType::kINT32, 1);
    mData.emplace_back("log_alpha", &mLogAlpha, PluginFieldType::kFLOAT32, 1);
    mData.emplace_back("log_n_floor", &mLogNFloor, PluginFieldType::kFLOAT32, 1);
    mData.emplace_back("q_clip", &mQClip, PluginFieldType::kFLOAT32, 1);
    mData.emplace_back("rel_clip", &mRelClip, PluginFieldType::kFLOAT32, 1);
    mData.emplace_back("scaling", &mScaling, PluginFieldType::kFLOAT32, 1);
    mData.emplace_back("kv_int8", &mKvInt8, PluginFieldType::kINT32, 1);
    mData.emplace_back("fused_qrsg", &mFusedQrsg, PluginFieldType::kINT32, 1);
    mFields.nbFields = static_cast<int32_t>(mData.size());
    mFields.fields = mData.data();
    return &mFields;
}

std::vector<PluginField> Inkling2AttentionPluginCreator::mAttributes;

Inkling2AttentionPluginCreator::Inkling2AttentionPluginCreator()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    mAttributes.clear();
    for (char const* name : {"num_q_heads", "num_kv_heads", "rel_extent", "window", "kv_int8", "fused_qrsg"})
    {
        mAttributes.emplace_back(name, nullptr, PluginFieldType::kINT32, 1);
    }
    for (char const* name : {"log_alpha", "log_n_floor", "q_clip", "rel_clip", "scaling"})
    {
        mAttributes.emplace_back(name, nullptr, PluginFieldType::kFLOAT32, 1);
    }
    mFields.nbFields = static_cast<int32_t>(mAttributes.size());
    mFields.fields = mAttributes.data();
}

char const* Inkling2AttentionPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* Inkling2AttentionPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* Inkling2AttentionPluginCreator::getFieldNames() noexcept
{
    return &mFields;
}

char const* Inkling2AttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Inkling2AttentionPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace != nullptr ? pluginNamespace : "";
}

IPluginV3* Inkling2AttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    if (name == nullptr || fc == nullptr || (fc->nbFields > 0 && fc->fields == nullptr))
    {
        LOG_ERROR("Inkling2AttentionPluginCreator: invalid arguments");
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
                throw std::invalid_argument("Inkling2AttentionPlugin: every attribute must be a named scalar");
            }
            if (!byName.emplace(field.name, &field).second)
            {
                throw std::invalid_argument(std::string("Inkling2AttentionPlugin: duplicate attribute ") + field.name);
            }
        }
        auto intField = [&](char const* key) {
            auto it = byName.find(key);
            if (it == byName.end() || it->second->type != PluginFieldType::kINT32)
            {
                throw std::invalid_argument(std::string("Inkling2AttentionPlugin: missing INT32 attribute ") + key);
            }
            return *static_cast<int32_t const*>(it->second->data);
        };
        auto floatField = [&](char const* key) {
            auto it = byName.find(key);
            if (it == byName.end() || it->second->type != PluginFieldType::kFLOAT32)
            {
                throw std::invalid_argument(std::string("Inkling2AttentionPlugin: missing FLOAT32 attribute ") + key);
            }
            return *static_cast<float const*>(it->second->data);
        };
        // kv_int8 and fused_qrsg are optional (absent in engines built before them)
        if (byName.size() != 9 + byName.count("kv_int8") + byName.count("fused_qrsg"))
        {
            throw std::invalid_argument("Inkling2AttentionPlugin: unknown or missing attributes");
        }
        auto* plugin = new Inkling2AttentionPlugin(name, intField("num_q_heads"), intField("num_kv_heads"),
            intField("rel_extent"), intField("window"), floatField("log_alpha"), floatField("log_n_floor"),
            floatField("q_clip"), floatField("rel_clip"), floatField("scaling"), byName.count("kv_int8") != 0 ? intField("kv_int8") : 0,
            byName.count("fused_qrsg") != 0 ? intField("fused_qrsg") : 0);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Inkling2AttentionPluginCreator: creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace trt_edgellm::plugins
