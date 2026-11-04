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
#pragma once

#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{
/*!
 * @brief TensorRT plugin for affine INT4 group-wise quantised (W4A16) GEMM, CuTe DSL backend.
 *
 * Serves checkpoints quantised with MLX-style affine INT4 (group size 64): w = scale * q + bias with q in
 * [0, 15] (not symmetric). The kernels dequantise (q - 8) and apply fma(., scale, bias') with
 * bias' = fp16(bias + 8 * scale), which the export step precomputes. Weights are consumed in the CuTe DSL
 * fragment layout (bN=128, bK=64, raw nibbles). Tactics are the 16 AOT GEMM variants (CTA tile x stages x
 * serial split-K, split-K only where it divides K/64) plus one "decode GEMV" tactic that runs the per-M
 * GEMV for M <= 8 (and falls back to a default GEMM variant for larger M).
 *
 * Inputs: [0] activations [.., K] fp16; [1] fragment weights [rows, 512] int8
 * (rows = ceil(N/128) * (K/64) * 8); [2] scales [K/64, N] fp16; [3] bias' [K/64, N] fp16.
 * Output: [.., N] fp16.
 */
class Int4AffineGroupwiseGemmPlugin : public nvinfer1::IPluginV3,
                                      public nvinfer1::IPluginV3OneCore,
                                      public nvinfer1::IPluginV3OneBuild,
                                      public nvinfer1::IPluginV3OneRuntime
{
public:
    /*!
     * @brief Construct the plugin
     * @param name Layer name
     * @param N Output features
     * @param K Input features
     * @param groupSize Quantisation group size (must be 64)
     */
    Int4AffineGroupwiseGemmPlugin(std::string const& name, int32_t N, int32_t K, int32_t groupSize);

    //! @brief Construct from a field collection
    Int4AffineGroupwiseGemmPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    Int4AffineGroupwiseGemmPlugin() = delete;
    Int4AffineGroupwiseGemmPlugin(Int4AffineGroupwiseGemmPlugin const&) = delete;
    ~Int4AffineGroupwiseGemmPlugin() override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    //! @brief Captures the profile's optimum token count M for tactic pruning and the max M for lock sizing.
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    //! @brief Allocates the plugin-owned serial split-K lock buffer before enqueue.
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    //! @brief True when (N, K, group size) fit the kernels: N % 8 == 0, K % 64 == 0, group size 64.
    bool hasValidGeometry() const noexcept;

    //! @brief Number of valid custom tactics for this (N, K) and profile opt M.
    int32_t getNbTactics() noexcept override;
    //! @brief Valid tactic ids: 1..16 = GEMM variant table index + 1; 17 = decode GEMV (M <= 8).
    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override;
    int32_t setTactic(int32_t tactic) noexcept override;
    char const* getTimingCacheID() noexcept override;

private:
    std::string mLayerName;
    std::string mNamespace;

    int32_t mGemmN{};
    int32_t mGemmK{};
    int32_t mGroupSize{};

    //! Selected tactic (0 = default heuristic).
    int32_t mTactic{0};

    //! Profile opt token count (M = leading volume), captured in configurePlugin. 0 = unknown.
    int32_t mAutotuneM{0};

    //! Serial split-K semaphore storage, plugin-owned so it cannot alias TensorRT temporary workspace.
    int64_t mMaxLockWorkspaceBytes{};
    void* mLockWorkspace{};
    size_t mLockWorkspaceBytes{};

    bool ensureLockWorkspace(size_t requiredBytes) noexcept;
    void releaseLockWorkspace() noexcept;

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

//! @brief Factory for Int4AffineGroupwiseGemmPlugin instances.
class Int4AffineGroupwiseGemmPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Int4AffineGroupwiseGemmPluginCreator();
    ~Int4AffineGroupwiseGemmPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
