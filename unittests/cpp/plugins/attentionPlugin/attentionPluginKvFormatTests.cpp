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

// KV cache format of the attention plugin: enable_int8_kv_cache is 0 (FP16), 1 (INT8 K and V) or 2 (INT8 K with V packed to
// 4 bits). The tests go through the plugin registry, like the engine builder and runtime do.

#include "kernels/decodeAttentionKernels/decoderXQAJitCompiler.h"
#include "testPluginLoader.h"

#include <NvInferRuntime.h>
#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{
constexpr char const* kNAME = "AttentionPlugin";
constexpr char const* kVERSION = "1";

struct Attrs
{
    int32_t numQHeads{16};
    int32_t numKVHeads{2};
    int32_t headSize{256};
    int32_t int8KvCache{0};
    int32_t fp8KvCache{0};
    int32_t treeAttention{0};
    int32_t visionBlock{0};
    int32_t slidingWindow{-1};
    int32_t boundedKvCache{0};
    std::vector<float> qkvScales{1.0F, 0.05F, 0.04F};
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

IPluginV3* create(Attrs const& attrs, TensorRTPhase phase = TensorRTPhase::kRUNTIME)
{
    IPluginCreatorV3One* creator = getCreator();
    if (creator == nullptr)
    {
        return nullptr;
    }
    std::vector<PluginField> fields{
        PluginField("num_q_heads", &attrs.numQHeads, PluginFieldType::kINT32, 1),
        PluginField("num_kv_heads", &attrs.numKVHeads, PluginFieldType::kINT32, 1),
        PluginField("head_size", &attrs.headSize, PluginFieldType::kINT32, 1),
        PluginField("enable_int8_kv_cache", &attrs.int8KvCache, PluginFieldType::kINT32, 1),
        PluginField("enable_fp8_kv_cache", &attrs.fp8KvCache, PluginFieldType::kINT32, 1),
        PluginField("enable_tree_attention", &attrs.treeAttention, PluginFieldType::kINT32, 1),
        PluginField("enable_vision_block_attention", &attrs.visionBlock, PluginFieldType::kINT32, 1),
        PluginField("sliding_window_size", &attrs.slidingWindow, PluginFieldType::kINT32, 1),
        PluginField("supports_bounded_kv_cache", &attrs.boundedKvCache, PluginFieldType::kINT32, 1),
        PluginField("qkv_scales", attrs.qkvScales.data(), PluginFieldType::kFLOAT32,
            static_cast<int32_t>(attrs.qkvScales.size())),
    };
    PluginFieldCollection fc{static_cast<int32_t>(fields.size()), fields.data()};
    return creator->createPlugin("attention", &fc, phase);
}

PluginFieldCollection const* serializedFields(IPluginV3* plugin)
{
    auto* runtime = static_cast<IPluginV3OneRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
    return runtime == nullptr ? nullptr : runtime->getFieldsToSerialize();
}

PluginField const* findField(PluginFieldCollection const* fc, char const* name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        if (std::strcmp(fc->fields[i].name, name) == 0)
        {
            return &fc->fields[i];
        }
    }
    return nullptr;
}

int32_t serializedInt(IPluginV3* plugin, char const* name)
{
    PluginField const* field = findField(serializedFields(plugin), name);
    EXPECT_NE(field, nullptr) << name;
    return field == nullptr ? -1 : *static_cast<int32_t const*>(field->data);
}

class AttentionPluginKvFormatTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (getCreator() == nullptr)
        {
            GTEST_SKIP() << "AttentionPlugin creator is not registered (plugin library not found)";
        }
    }
};
} // namespace

TEST_F(AttentionPluginKvFormatTest, AcceptsEachKvFormatAndSerializesIt)
{
    for (int32_t format : {0, 1, 2})
    {
        SCOPED_TRACE("enable_int8_kv_cache=" + std::to_string(format));
        Attrs attrs;
        attrs.int8KvCache = format;
        IPluginV3* plugin = create(attrs);
        ASSERT_NE(plugin, nullptr);
        EXPECT_EQ(serializedInt(plugin, "enable_int8_kv_cache"), format);
        delete plugin;
    }
}

TEST_F(AttentionPluginKvFormatTest, FormatSurvivesSerializationAndClone)
{
    Attrs attrs;
    attrs.int8KvCache = 2;
    IPluginV3* plugin = create(attrs);
    ASSERT_NE(plugin, nullptr);

    // An engine stores the serialized fields and the runtime builds the plugin back from them.
    PluginFieldCollection const* serialized = serializedFields(plugin);
    ASSERT_NE(serialized, nullptr);
    IPluginV3* restored = getCreator()->createPlugin("restored", serialized, TensorRTPhase::kRUNTIME);
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(serializedInt(restored, "enable_int8_kv_cache"), 2);
    PluginField const* scales = findField(serializedFields(restored), "qkv_scales");
    ASSERT_NE(scales, nullptr);
    ASSERT_EQ(scales->length, 3);
    EXPECT_FLOAT_EQ(static_cast<float const*>(scales->data)[1], 0.05F);

    IPluginV3* cloned = plugin->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(serializedInt(cloned, "enable_int8_kv_cache"), 2);
    delete cloned;
    delete restored;
    delete plugin;
}

TEST_F(AttentionPluginKvFormatTest, RejectsValuesOutsideZeroToTwo)
{
    for (int32_t format : {3, 4, -1})
    {
        SCOPED_TRACE("enable_int8_kv_cache=" + std::to_string(format));
        Attrs attrs;
        attrs.int8KvCache = format;
        EXPECT_EQ(create(attrs), nullptr);
    }
}

TEST_F(AttentionPluginKvFormatTest, PackedValueNeedsQkvScalesLikeInt8)
{
    Attrs attrs;
    attrs.int8KvCache = 2;
    attrs.qkvScales = {1.0F, 0.05F};
    EXPECT_EQ(create(attrs), nullptr);
}

TEST_F(AttentionPluginKvFormatTest, PackedValueRejectsUnsupportedConfigurations)
{
    {
        Attrs attrs;
        attrs.int8KvCache = 2;
        attrs.headSize = 128;
        EXPECT_EQ(create(attrs), nullptr) << "head_dim 128";
        attrs.int8KvCache = 1;
        IPluginV3* control = create(attrs);
        EXPECT_NE(control, nullptr) << "plain INT8 must stay available for head_dim 128";
        delete control;
    }
    {
        Attrs attrs;
        attrs.int8KvCache = 2;
        attrs.headSize = 512;
        attrs.numQHeads = 16;
        EXPECT_EQ(create(attrs), nullptr) << "head_dim 512";
    }
    {
        Attrs attrs;
        attrs.int8KvCache = 2;
        attrs.slidingWindow = 1024;
        EXPECT_EQ(create(attrs), nullptr) << "sliding window";
    }
    {
        Attrs attrs;
        attrs.int8KvCache = 2;
        attrs.treeAttention = 1;
        EXPECT_EQ(create(attrs), nullptr) << "tree attention";
    }
    {
        Attrs attrs;
        attrs.int8KvCache = 2;
        attrs.visionBlock = 1;
        EXPECT_EQ(create(attrs), nullptr) << "vision block attention";
    }
    {
        Attrs attrs;
        attrs.int8KvCache = 2;
        attrs.slidingWindow = 512;
        attrs.boundedKvCache = 1;
        EXPECT_EQ(create(attrs), nullptr) << "bounded KV cache";
    }
    {
        Attrs attrs;
        attrs.int8KvCache = 2;
        attrs.fp8KvCache = 1;
        EXPECT_EQ(create(attrs), nullptr) << "FP8 together with a packed INT8 cache";
    }
}

TEST_F(AttentionPluginKvFormatTest, BuildCompilesTheDecodeKernelForTheCacheFormat)
{
    struct Case
    {
        int32_t format;
        DataType expected;
    };
    for (Case const& c : {Case{1, DataType::kINT8}, Case{2, DataType::kINT4}})
    {
        SCOPED_TRACE("enable_int8_kv_cache=" + std::to_string(c.format));
        Attrs attrs;
        attrs.int8KvCache = c.format;
        IPluginV3* plugin = create(attrs, TensorRTPhase::kBUILD);
        ASSERT_NE(plugin, nullptr);
        PluginField const* blob = findField(serializedFields(plugin), "xqa_jit_kernels");
        ASSERT_NE(blob, nullptr);
        ASSERT_GT(blob->length, 0) << "no XQA kernel was compiled at build";
        auto const kernels = deserializeXQAJitKernels(static_cast<uint8_t const*>(blob->data), blob->length);
        ASSERT_FALSE(kernels.empty());
        for (auto const& kernel : kernels)
        {
            EXPECT_EQ(kernel.key.kvDataType, c.expected);
            EXPECT_EQ(kernel.key.headSize, 256);
            EXPECT_EQ(kernel.key.qHeadsPerKv, 8);
            EXPECT_EQ(kernel.key.tokensPerPage, 128);
            EXPECT_FALSE(kernel.key.specDecode);
            EXPECT_FALSE(kernel.cubin.empty());
        }
        // The runtime path registers the serialized kernel for this key.
        IPluginV3* restored = getCreator()->createPlugin("restored", serializedFields(plugin), TensorRTPhase::kRUNTIME);
        EXPECT_NE(restored, nullptr);
        delete restored;
        delete plugin;
    }
}
