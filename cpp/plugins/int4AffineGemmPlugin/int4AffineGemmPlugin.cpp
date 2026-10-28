/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "int4AffineGemmPlugin.h"
#include "common/cudaUtils.h"
#include "cuteDslInt4AffineGemm.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>
#include <string>
#include <vector>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kINT4_AFFINE_GEMM_PLUGIN_VERSION{"1"};
constexpr char const* kINT4_AFFINE_GEMM_PLUGIN_NAME{"Int4AffineGroupwiseGemmPlugin"};
constexpr char const* kMAX_LOCK_WORKSPACE_BYTES_FIELD{"max_lock_workspace_bytes"};
// Worst-case lock sizing: bM=16 is the smallest M tile and bN=128 is fixed across variants.
constexpr int32_t kWORST_CASE_LOCK_TILE_M{16};
constexpr int32_t kLOCK_TILE_N{128};
constexpr int32_t kGROUP_SIZE{64};
constexpr int32_t kFRAG_TILE_N{128};
constexpr int32_t kFRAG_TILE_K{64};

// Fragment weight buffer rows for a (N, K) problem (bN=128, bK=64, kn=8): input[1] is INT8 [rows, 512].
inline int64_t fragmentRows(int32_t N, int32_t K)
{
    return static_cast<int64_t>(divUp(N, kFRAG_TILE_N)) * (K / kFRAG_TILE_K) * 8;
}

inline int64_t leadingVolume(Dims const& dims)
{
    int64_t volume{1};
    for (int32_t i = 0; i < dims.nbDims - 1; ++i)
    {
        volume *= dims.d[i];
    }
    return volume;
}

#ifdef CUTE_DSL_INT4_AFFINE_GEMM_ENABLED
// Tactic ids: 1..nVariants = GEMM variant table index + 1; nVariants + 1 = decode GEMV (M <= 8).
inline int32_t gemvTacticId()
{
    return cuteDslInt4AffineNumVariants() + 1;
}

// Default GEMM variant (64x128x64, 3 stages, no split-K) when no tactic is selected or a GEMV tactic meets M > 8.
inline int32_t defaultGemmIndex()
{
    for (int32_t i = 0; i < cuteDslInt4AffineNumVariants(); ++i)
    {
        auto const& v = cuteDslInt4AffineVariantAt(i);
        if (v.bM == 64 && v.stages == 3 && v.splitK == 1)
        {
            return i;
        }
    }
    return 0;
}

inline int64_t computeMaxLockWorkspaceBytes(
    DynamicPluginTensorDesc const* inputs, int32_t nbInputs, int32_t N, int32_t K)
{
    if (inputs == nullptr || nbInputs < 1 || N <= 0 || K <= 0 || K % kGROUP_SIZE != 0)
    {
        return 0;
    }
    auto const& mx = inputs[0].max;
    if (mx.nbDims < 2)
    {
        return 0;
    }
    int64_t const mMax = leadingVolume(mx);
    if (mMax <= 0)
    {
        return 0;
    }
    int64_t const locks
        = cuteDslInt4AffineLockCount(static_cast<int32_t>(mMax), N, kWORST_CASE_LOCK_TILE_M, kLOCK_TILE_N);
    return locks * static_cast<int64_t>(sizeof(int32_t));
}

// Per-shape tactic filter shared by getNbTactics()/getValidTactics(). A GEMM variant is autotuned iff it can
// serve (N, K) (splitK | K/64) and its CTA tile is plausible for the profile's opt M (M >= 256 drops bM=16,
// otherwise bM=128 is dropped: the same pruning as the symmetric plugin). The GEMV tactic is offered when the opt
// M is unknown or <= 8.
inline bool gemmTacticSelectable(CuteDslInt4AffineVariant const& v, int32_t N, int32_t K, int32_t autotuneM)
{
    if (!cuteDslInt4AffineVariantValid(v, N, K))
    {
        return false;
    }
    if (autotuneM <= 0)
    {
        return true;
    }
    return (autotuneM >= 256) ? (v.bM != 16) : (v.bM != 128);
}

inline bool gemvTacticSelectable(int32_t N, int32_t K, int32_t autotuneM)
{
    return N > 0 && K > 0 && K % kGROUP_SIZE == 0 && (autotuneM <= 0 || autotuneM <= kInt4AffineGemvMaxM);
}
#endif
} // namespace

// Static class fields initialization
PluginFieldCollection Int4AffineGroupwiseGemmPluginCreator::mFieldCollection{};
std::vector<PluginField> Int4AffineGroupwiseGemmPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Int4AffineGroupwiseGemmPluginCreator);

Int4AffineGroupwiseGemmPlugin::Int4AffineGroupwiseGemmPlugin(
    std::string const& name, int32_t N, int32_t K, int32_t groupSize)
    : mLayerName(name)
    , mGemmN(N)
    , mGemmK(K)
    , mGroupSize(groupSize)
{
}

Int4AffineGroupwiseGemmPlugin::Int4AffineGroupwiseGemmPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "gemm_n")
        {
            mGemmN = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "gemm_k")
        {
            mGemmK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "group_size")
        {
            mGroupSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == kMAX_LOCK_WORKSPACE_BYTES_FIELD)
        {
            mMaxLockWorkspaceBytes = *static_cast<int64_t const*>(fc->fields[i].data);
        }
    }
}

Int4AffineGroupwiseGemmPlugin::~Int4AffineGroupwiseGemmPlugin()
{
    releaseLockWorkspace();
}

bool Int4AffineGroupwiseGemmPlugin::hasValidGeometry() const noexcept
{
    return mGemmN > 0 && mGemmN % 8 == 0 && mGemmK > 0 && mGemmK % kGROUP_SIZE == 0 && mGroupSize == kGROUP_SIZE;
}

IPluginCapability* Int4AffineGroupwiseGemmPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
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
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* Int4AffineGroupwiseGemmPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Int4AffineGroupwiseGemmPlugin(mLayerName, mGemmN, mGemmK, mGroupSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->mTactic = mTactic;
        plugin->mAutotuneM = mAutotuneM;
        plugin->mMaxLockWorkspaceBytes = mMaxLockWorkspaceBytes;
        return plugin;
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

char const* Int4AffineGroupwiseGemmPlugin::getPluginName() const noexcept
{
    return kINT4_AFFINE_GEMM_PLUGIN_NAME;
}

char const* Int4AffineGroupwiseGemmPlugin::getPluginVersion() const noexcept
{
    return kINT4_AFFINE_GEMM_PLUGIN_VERSION;
}

char const* Int4AffineGroupwiseGemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Int4AffineGroupwiseGemmPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

int32_t Int4AffineGroupwiseGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Int4AffineGroupwiseGemmPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* /* inputTypes */, int32_t /* nbInputs */) const noexcept
{
    try
    {
        assert(nbOutputs == 1);
        outputTypes[0] = DataType::kHALF;
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t Int4AffineGroupwiseGemmPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        if (nbInputs != 4 || nbOutputs != 1)
        {
            return -1;
        }
        if (inputs[0].nbDims != 2 && inputs[0].nbDims != 3)
        {
            return -1;
        }
        outputs[0].nbDims = inputs[0].nbDims;
        for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i)
        {
            outputs[0].d[i] = inputs[0].d[i];
        }
        outputs[0].d[inputs[0].nbDims - 1] = exprBuilder.constant(mGemmN);
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool Int4AffineGroupwiseGemmPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    try
    {
        assert(nbInputs == 4 && nbOutputs == 1);
        assert(pos < (nbInputs + nbOutputs));
        auto const& tensorDesc = inOut[pos].desc;
        bool status{true};
        status &= tensorDesc.format == PluginFormat::kLINEAR;

        switch (pos)
        {
        case 0:
        case 4:
        {
            // Activation (pos 0, last dim K) and output (pos 4, last dim N): fp16, rank 2 or 3.
            status &= tensorDesc.type == DataType::kHALF;
            bool const supportedRank = tensorDesc.dims.nbDims == 2 || tensorDesc.dims.nbDims == 3;
            status &= supportedRank;
            status &= supportedRank && tensorDesc.dims.d[tensorDesc.dims.nbDims - 1] == (pos == 0 ? mGemmK : mGemmN);
            break;
        }
        case 1:
        {
            // Fragment-layout weights: INT8 [rows, 512], rows = ceil(N/128) * (K/64) * 8.
            status &= tensorDesc.type == DataType::kINT8;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == fragmentRows(mGemmN, mGemmK);
            status &= tensorDesc.dims.d[1] == 512;
            break;
        }
        case 2:
        case 3:
        {
            // Scales and shifted bias: fp16 [K/64, N].
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == mGemmK / mGroupSize;
            status &= tensorDesc.dims.d[1] == mGemmN;
            break;
        }
        default: break;
        }
        return status;
    }
    catch (std::exception const& e)
    {
        return false;
    }
}

int32_t Int4AffineGroupwiseGemmPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    if (nbInputs >= 1)
    {
        auto const& opt = in[0].opt;
        if (opt.nbDims >= 2)
        {
            int64_t const m = leadingVolume(opt);
            mAutotuneM = (m > 0) ? static_cast<int32_t>(m) : 0;
        }
    }
#ifdef CUTE_DSL_INT4_AFFINE_GEMM_ENABLED
    int64_t const maxLockWorkspaceBytes = computeMaxLockWorkspaceBytes(in, nbInputs, mGemmN, mGemmK);
    if (maxLockWorkspaceBytes > mMaxLockWorkspaceBytes)
    {
        mMaxLockWorkspaceBytes = maxLockWorkspaceBytes;
    }
#endif
    return 0;
}

size_t Int4AffineGroupwiseGemmPlugin::getWorkspaceSize([[maybe_unused]] DynamicPluginTensorDesc const* inputs,
    int32_t /* nbInputs */, DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

bool Int4AffineGroupwiseGemmPlugin::ensureLockWorkspace(size_t requiredBytes) noexcept
{
#ifdef CUTE_DSL_INT4_AFFINE_GEMM_ENABLED
    if (requiredBytes <= mLockWorkspaceBytes)
    {
        return true;
    }

    void* newWorkspace{nullptr};
    cudaError_t const allocError = cudaMalloc(&newWorkspace, requiredBytes);
    if (allocError != cudaSuccess)
    {
        LOG_ERROR("Int4AffineGroupwiseGemmPlugin: cudaMalloc for serial split-K locks failed: %s",
            cudaGetErrorString(allocError));
        return false;
    }

    releaseLockWorkspace();
    mLockWorkspace = newWorkspace;
    mLockWorkspaceBytes = requiredBytes;
    return true;
#else
    return true;
#endif
}

void Int4AffineGroupwiseGemmPlugin::releaseLockWorkspace() noexcept
{
    if (mLockWorkspace == nullptr)
    {
        return;
    }
    cudaError_t const freeError = cudaFree(mLockWorkspace);
    if (freeError != cudaSuccess)
    {
        LOG_ERROR("Int4AffineGroupwiseGemmPlugin: cudaFree for serial split-K locks failed: %s",
            cudaGetErrorString(freeError));
    }
    mLockWorkspace = nullptr;
    mLockWorkspaceBytes = 0;
}

int32_t Int4AffineGroupwiseGemmPlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /* outputDesc */, void const* const* inputs, void* const* outputs,
    [[maybe_unused]] void* workspace, cudaStream_t stream) noexcept
{
    try
    {
#ifdef CUTE_DSL_INT4_AFFINE_GEMM_ENABLED
        int32_t const M = static_cast<int32_t>(leadingVolume(inputDesc[0].dims));
        if (M <= 0)
        {
            return 0; // empty batch
        }
        if (mGemmN <= 0 || mGemmN % 8 != 0 || mGemmK <= 0 || mGemmK % kGROUP_SIZE != 0)
        {
            return -1;
        }
        if (!cuteDslInt4AffineLoadModules())
        {
            return -1;
        }

        // EDGELLM_INT4_AFFINE_FORCE_TACTIC=<id>: run this tactic id (1..16 GEMM variant, 17 GEMV) instead of the
        // engine's stored choice whenever it is valid for the shape (test / A-B tool, applies to every instance).
        static int32_t const kForcedTactic = []() {
            char const* value = std::getenv("EDGELLM_INT4_AFFINE_FORCE_TACTIC");
            return value != nullptr ? std::atoi(value) : 0;
        }();
        int32_t tactic = mTactic;
        if (kForcedTactic > 0)
        {
            bool const valid = kForcedTactic == gemvTacticId()
                || (kForcedTactic <= cuteDslInt4AffineNumVariants()
                    && cuteDslInt4AffineVariantValid(cuteDslInt4AffineVariantAt(kForcedTactic - 1), mGemmN, mGemmK));
            if (valid)
            {
                tactic = kForcedTactic;
            }
        }

        // Tactic 0 (TensorRT default): GEMV for decode-size M, GEMM default variant otherwise.
        bool const wantGemv = (tactic == 0 || tactic == gemvTacticId());
        if (wantGemv && M <= kInt4AffineGemvMaxM)
        {
            return cuteDslInt4AffineGemvLaunch(M, inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], mGemmN, mGemmK, stream);
        }
        int32_t const idx = (tactic >= 1 && tactic <= cuteDslInt4AffineNumVariants()) ? (tactic - 1) : defaultGemmIndex();
        CuteDslInt4AffineVariant const v = cuteDslInt4AffineVariantAt(idx);
        if (!cuteDslInt4AffineVariantValid(v, mGemmN, mGemmK))
        {
            return -1;
        }
        if (v.splitK > 1)
        {
            int64_t const nLocks = cuteDslInt4AffineLockCount(M, mGemmN, v.bM, v.bN);
            size_t const lockBytes = static_cast<size_t>(nLocks) * sizeof(int32_t);
            if (mLockWorkspace == nullptr || lockBytes > mLockWorkspaceBytes)
            {
                LOG_ERROR("Int4AffineGroupwiseGemmPlugin: serial split-K lock buffer was not allocated for this shape.");
                return -1;
            }
            cudaError_t const memsetError = cudaMemsetAsync(mLockWorkspace, 0, lockBytes, stream);
            if (memsetError != cudaSuccess)
            {
                LOG_ERROR("Int4AffineGroupwiseGemmPlugin: cudaMemsetAsync for serial split-K locks failed: %s",
                    cudaGetErrorString(memsetError));
                return -1;
            }
        }
        return cuteDslInt4AffineGemmLaunch(v, inputs[0], inputs[1], inputs[2], inputs[3], outputs[0],
            (v.splitK > 1) ? mLockWorkspace : nullptr, M, mGemmN, mGemmK, stream);
#else
        return -1;
#endif
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t Int4AffineGroupwiseGemmPlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
#ifdef CUTE_DSL_INT4_AFFINE_GEMM_ENABLED
    try
    {
        if (mMaxLockWorkspaceBytes <= 0)
        {
            return -1;
        }
        return ensureLockWorkspace(static_cast<size_t>(mMaxLockWorkspaceBytes)) ? 0 : -1;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
#endif
    return 0;
}

IPluginV3* Int4AffineGroupwiseGemmPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

int32_t Int4AffineGroupwiseGemmPlugin::getNbTactics() noexcept
{
#ifdef CUTE_DSL_INT4_AFFINE_GEMM_ENABLED
    int32_t count = gemvTacticSelectable(mGemmN, mGemmK, mAutotuneM) ? 1 : 0;
    for (int32_t i = 0; i < cuteDslInt4AffineNumVariants(); ++i)
    {
        if (gemmTacticSelectable(cuteDslInt4AffineVariantAt(i), mGemmN, mGemmK, mAutotuneM))
        {
            ++count;
        }
    }
    return count;
#else
    return 0;
#endif
}

int32_t Int4AffineGroupwiseGemmPlugin::getValidTactics(
    [[maybe_unused]] int32_t* tactics, [[maybe_unused]] int32_t nbTactics) noexcept
{
#ifdef CUTE_DSL_INT4_AFFINE_GEMM_ENABLED
    // Tactic id = GEMM variant table index + 1, then the GEMV tactic (positive; 0 is TRT's reserved default).
    int32_t j = 0;
    for (int32_t i = 0; i < cuteDslInt4AffineNumVariants() && j < nbTactics; ++i)
    {
        if (gemmTacticSelectable(cuteDslInt4AffineVariantAt(i), mGemmN, mGemmK, mAutotuneM))
        {
            tactics[j++] = i + 1;
        }
    }
    if (j < nbTactics && gemvTacticSelectable(mGemmN, mGemmK, mAutotuneM))
    {
        tactics[j++] = gemvTacticId();
    }
#endif
    return 0;
}

int32_t Int4AffineGroupwiseGemmPlugin::setTactic(int32_t tactic) noexcept
{
#ifdef CUTE_DSL_INT4_AFFINE_GEMM_ENABLED
    if (tactic > 0 && tactic != gemvTacticId())
    {
        int32_t const idx = tactic - 1;
        if (idx >= cuteDslInt4AffineNumVariants()
            || !cuteDslInt4AffineVariantValid(cuteDslInt4AffineVariantAt(idx), mGemmN, mGemmK))
        {
            return -1;
        }
    }
    else if (tactic > gemvTacticId())
    {
        return -1;
    }
#endif
    mTactic = tactic;
    return 0;
}

char const* Int4AffineGroupwiseGemmPlugin::getTimingCacheID() noexcept
{
    // Opt into TRT's build timing cache: the plugin's whole state (N, K, group_size) is carried by the I/O tensor
    // shapes, which TRT hashes as the cache key, so identical-shape layers are autotuned once.
    return "";
}

PluginFieldCollection const* Int4AffineGroupwiseGemmPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("gemm_n", &mGemmN, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("gemm_k", &mGemmK, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("group_size", &mGroupSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(kMAX_LOCK_WORKSPACE_BYTES_FIELD, &mMaxLockWorkspaceBytes, PluginFieldType::kINT64, 1);

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

Int4AffineGroupwiseGemmPluginCreator::Int4AffineGroupwiseGemmPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("gemm_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("gemm_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Int4AffineGroupwiseGemmPluginCreator::getPluginName() const noexcept
{
    return kINT4_AFFINE_GEMM_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* Int4AffineGroupwiseGemmPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Int4AffineGroupwiseGemmPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Int4AffineGroupwiseGemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Int4AffineGroupwiseGemmPluginCreator::getPluginVersion() const noexcept
{
    return kINT4_AFFINE_GEMM_PLUGIN_VERSION;
}

IPluginV3* Int4AffineGroupwiseGemmPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new Int4AffineGroupwiseGemmPlugin(std::string(name), fc);
        if (!plugin->hasValidGeometry())
        {
            LOG_ERROR("Int4AffineGroupwiseGemmPlugin: unsupported geometry (need N %% 8 == 0, K %% 64 == 0, group_size == 64).");
            delete plugin;
            return nullptr;
        }
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
