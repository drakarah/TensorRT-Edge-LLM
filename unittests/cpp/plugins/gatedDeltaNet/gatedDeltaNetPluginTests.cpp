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

#include "kernels/gdnKernels/gdnTreeChunkKernels.h"
#include "testPluginLoader.h"

#include <NvInferRuntime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <optional>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{

constexpr char const* kNAME{"gated_delta_net"};
constexpr char const* kVERSION{"1"};

struct Attrs
{
    int32_t kDim{128};
    int32_t vDim{128};
    int32_t useMtp{0};
    int32_t useDdtree{1};
    int32_t useDiffusionState{0};
    //! Unset leaves the field out, as an engine built before tree_scratch_rows has it.
    std::optional<int32_t> treeScratchRows;
};

IPluginCreatorV3One* getCreator()
{
    if (test::loadPluginLibrary() == nullptr)
    {
        return nullptr;
    }
    auto* creator = getPluginRegistry()->getCreator(kNAME, kVERSION, "");
    return creator == nullptr ? nullptr : static_cast<IPluginCreatorV3One*>(creator);
}

IPluginV3* create(Attrs const& attrs)
{
    IPluginCreatorV3One* creator = getCreator();
    if (creator == nullptr)
    {
        return nullptr;
    }
    std::vector<PluginField> fields{
        PluginField("k_dim", &attrs.kDim, PluginFieldType::kINT32, 1),
        PluginField("v_dim", &attrs.vDim, PluginFieldType::kINT32, 1),
        PluginField("use_mtp", &attrs.useMtp, PluginFieldType::kINT32, 1),
        PluginField("use_ddtree", &attrs.useDdtree, PluginFieldType::kINT32, 1),
        PluginField("use_diffusion_state", &attrs.useDiffusionState, PluginFieldType::kINT32, 1),
    };
    if (attrs.treeScratchRows.has_value())
    {
        fields.emplace_back("tree_scratch_rows", &*attrs.treeScratchRows, PluginFieldType::kINT32, 1);
    }
    PluginFieldCollection fc{static_cast<int32_t>(fields.size()), fields.data()};
    return creator->createPlugin("gdn", &fc, TensorRTPhase::kRUNTIME);
}

PluginFieldCollection const* serializedFields(IPluginV3* plugin)
{
    auto* runtime = static_cast<IPluginV3OneRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    return runtime == nullptr ? nullptr : runtime->getFieldsToSerialize();
}

int32_t serializedInt(IPluginV3* plugin, char const* name)
{
    PluginFieldCollection const* fc = serializedFields(plugin);
    for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i)
    {
        if (std::strcmp(fc->fields[i].name, name) == 0)
        {
            return *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
    return INT32_MIN;
}

//! The input descriptors configurePlugin validates, for h k-heads and hv v-heads (dk = dv = 128), DDTree inputs on.
std::vector<DynamicPluginTensorDesc> makeInputs(int32_t h, int32_t hv)
{
    std::vector<DynamicPluginTensorDesc> in(15);
    auto set = [&](int32_t index, DataType type, std::vector<int64_t> dims) {
        in[index].desc.type = type;
        in[index].desc.format = TensorFormat::kLINEAR;
        in[index].desc.dims.nbDims = static_cast<int32_t>(dims.size());
        for (size_t d = 0; d < dims.size(); ++d)
        {
            in[index].desc.dims.d[d] = dims[d];
        }
    };
    set(0, DataType::kHALF, {-1, h, 128});   // q
    set(1, DataType::kHALF, {-1, h, 128});   // k
    set(2, DataType::kHALF, {-1, hv, 128});  // v
    set(3, DataType::kHALF, {-1, hv});       // a
    set(4, DataType::kHALF, {-1, hv});       // b
    set(5, DataType::kFLOAT, {hv});          // a_log
    set(6, DataType::kHALF, {hv});           // dt_bias
    set(7, DataType::kFLOAT, {8, hv, 128, 128}); // h0 source (state pool)
    for (int32_t index : {8, 9, 10, 11, 12, 13, 14})
    {
        set(index, DataType::kINT32, {-1});
    }
    return in;
}

int32_t configure(IPluginV3* plugin, std::vector<DynamicPluginTensorDesc> const& in)
{
    auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
    std::vector<DynamicPluginTensorDesc> out(3);
    return build->configurePlugin(in.data(), static_cast<int32_t>(in.size()), out.data(), 3);
}

class GatedDeltaNetPluginTreeScratchTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (getCreator() == nullptr)
        {
            GTEST_SKIP() << "the gated_delta_net plugin is not available in this build";
        }
    }
};

} // namespace

TEST_F(GatedDeltaNetPluginTreeScratchTest, RowsSurviveSerializationAndClone)
{
    Attrs attrs;
    attrs.treeScratchRows = 2;
    IPluginV3* plugin = create(attrs);
    ASSERT_NE(plugin, nullptr);
    EXPECT_EQ(serializedInt(plugin, "tree_scratch_rows"), 2);
    IPluginV3* restored = getCreator()->createPlugin("restored", serializedFields(plugin), TensorRTPhase::kRUNTIME);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(serializedInt(restored, "tree_scratch_rows"), 2);
    IPluginV3* cloned = plugin->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(serializedInt(cloned, "tree_scratch_rows"), 2);
    EXPECT_EQ(serializedInt(cloned, "use_ddtree"), 1);
    delete cloned;
    delete restored;
    delete plugin;
}

TEST_F(GatedDeltaNetPluginTreeScratchTest, AnEngineWithoutTheFieldKeepsTheTokenMajorLayout)
{
    Attrs attrs; // treeScratchRows unset: field absent
    IPluginV3* plugin = create(attrs);
    ASSERT_NE(plugin, nullptr);
    EXPECT_EQ(serializedInt(plugin, "tree_scratch_rows"), 0);
    delete plugin;
}

TEST_F(GatedDeltaNetPluginTreeScratchTest, RowsApplyToDdtreeOnly)
{
    Attrs attrs;
    attrs.useDdtree = 0;
    attrs.useMtp = 1;
    attrs.treeScratchRows = 2;
    IPluginV3* plugin = create(attrs);
    ASSERT_NE(plugin, nullptr);
    EXPECT_EQ(serializedInt(plugin, "tree_scratch_rows"), 0) << "linear MTP verify needs a row per token";
    delete plugin;
}

TEST_F(GatedDeltaNetPluginTreeScratchTest, NegativeRowsReadAsZero)
{
    Attrs attrs;
    attrs.treeScratchRows = -5;
    // The plugin clamps it to the legacy layout.
    IPluginV3* plugin = create(attrs);
    ASSERT_NE(plugin, nullptr);
    EXPECT_EQ(serializedInt(plugin, "tree_scratch_rows"), 0);
    delete plugin;
}

TEST_F(GatedDeltaNetPluginTreeScratchTest, ConfigureRefusesRowsBelowTheTreeVerifyScratchAndAcceptsTheMinimum)
{
    struct Shape
    {
        int32_t h;
        int32_t hv;
    };
    for (Shape const shape : {Shape{16, 48}, Shape{2, 6}, Shape{1, 2}, Shape{4, 32}})
    {
        size_t const rowBytes = static_cast<size_t>(shape.hv) * 128 * 128 * sizeof(float);
        size_t const needed = kernel::gdnTreeScratchBytes(shape.h, shape.hv);
        int32_t const minimumRows = static_cast<int32_t>((needed + rowBytes - 1) / rowBytes);
        auto const inputs = makeInputs(shape.h, shape.hv);
        for (int32_t rows : {minimumRows - 1, minimumRows, minimumRows + 3})
        {
            Attrs attrs;
            attrs.treeScratchRows = rows;
            IPluginV3* plugin = create(attrs);
            ASSERT_NE(plugin, nullptr);
            int32_t const status = configure(plugin, inputs);
            if (rows < minimumRows)
            {
                // rows == 0 is the legacy layout and always configures; any positive undersized count is refused.
                EXPECT_EQ(status, rows <= 0 ? 0 : -1) << "h=" << shape.h << " hv=" << shape.hv << " rows=" << rows;
            }
            else
            {
                EXPECT_EQ(status, 0) << "h=" << shape.h << " hv=" << shape.hv << " rows=" << rows;
            }
            delete plugin;
        }
    }
}

TEST_F(GatedDeltaNetPluginTreeScratchTest, TheQwen38ShapeNeedsExactlyTwoRows)
{
    EXPECT_EQ(kernel::gdnTreeScratchBytes(16, 48), 6116096U);
    size_t const rowBytes = size_t{48} * 128 * 128 * sizeof(float);
    EXPECT_GT(kernel::gdnTreeScratchBytes(16, 48), rowBytes);
    EXPECT_LE(kernel::gdnTreeScratchBytes(16, 48), 2 * rowBytes);
}

TEST_F(GatedDeltaNetPluginTreeScratchTest, LegacyLayoutConfiguresWithoutTheCheck)
{
    Attrs attrs;
    attrs.treeScratchRows = 0;
    IPluginV3* plugin = create(attrs);
    ASSERT_NE(plugin, nullptr);
    EXPECT_EQ(configure(plugin, makeInputs(16, 48)), 0);
    delete plugin;
}
