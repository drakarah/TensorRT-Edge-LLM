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

#include "runtime/state/externalWeightManager.h"

#include "common/checkMacros.h"
#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "scratchDir.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fstream>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using Json = nlohmann::json;

namespace
{

//! Two FP16 tensors in one checkpoint: "a" (4x8) and "b" (4x8) with distinct ramps.
class ExternalWeightManagerTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
        mDir = makeScratchDir("externalWeightManagerTests");
        std::vector<Tensor> tensors;
        for (std::string const name : {"a.weight", "b.weight"})
        {
            Tensor tensor({4, 8}, DeviceType::kCPU, nvinfer1::DataType::kHALF, name);
            auto* data = static_cast<half*>(tensor.rawPointer());
            for (int32_t i = 0; i < 32; ++i)
            {
                data[i] = __float2half(static_cast<float>(i) + (name == "a.weight" ? 1.0F : 100.0F));
            }
            tensors.push_back(std::move(tensor));
        }
        ASSERT_TRUE(safetensors::saveSafetensors(mDir / "model.safetensors", tensors, mStream));
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
        CUDA_CHECK(cudaStreamDestroy(mStream));
    }

    static Json binding(std::string const& name, std::string const& key, std::optional<std::string> source,
        std::string const& dtype = "F16")
    {
        Json b;
        b["engine_name"] = name;
        b["checkpoint_keys"] = Json::array({key});
        b["source_layout"] = "plugin";
        b["dtype"] = dtype;
        b["shape"] = Json::array({4, 8});
        if (source.has_value())
        {
            b["checkpoint_source"] = *source;
        }
        return b;
    }

    std::filesystem::path writeConfig(std::string const& fileName, std::vector<Json> const& bindings)
    {
        Json config;
        config["checkpoint_weight_bindings"] = bindings;
        config["checkpoint_dir"] = mDir.string();
        // The identity the builder records for the checkpoint: file size and mtime, and each tensor's dtype and shape.
        auto const file = mDir / "model.safetensors";
        Json source;
        source["build_source"] = mDir.string();
        source["files"]["model.safetensors"]["bytes"] = std::filesystem::file_size(file);
        source["files"]["model.safetensors"]["mtime_ns"]
            = std::filesystem::last_write_time(file).time_since_epoch().count();
        for (std::string const name : {"a.weight", "b.weight"})
        {
            source["tensors"][name] = {{"dtype", "F16"}, {"shape", Json::array({4, 8})}, {"bytes", 64},
                {"samples", Json::array()}};
        }
        config["checkpoint_identity"] = {{"version", 1}, {"sources", {{"component", source}, {"target", source}}}};
        std::filesystem::path const path = mDir / fileName;
        std::ofstream out(path);
        out << config.dump();
        return path;
    }

    std::vector<half> readBack(Tensor const& tensor) const
    {
        std::vector<half> host(32);
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.rawPointer(), host.size() * sizeof(half), cudaMemcpyDeviceToHost));
        return host;
    }

    static Tensor const& weightNamed(ExternalWeightManager const& manager, std::string const& name)
    {
        Tensor const* const found = manager.findSharedWeight(name, ExternalWeightManagerTests::lastContract(name));
        EXPECT_NE(found, nullptr);
        return *found;
    }

    //! The contract text a binding without checkpoint_source dumps to (what findSharedWeight compares).
    static std::string lastContract(std::string const& name)
    {
        Json contract = binding(name, name + ".weight", std::nullopt);
        return contract.dump();
    }

    cudaStream_t mStream{};
    std::filesystem::path mDir;
};

} // namespace

TEST_F(ExternalWeightManagerTests, UnloadedManagerSharesNothing)
{
    ExternalWeightManager manager;
    EXPECT_EQ(manager.findSharedWeight("a", lastContract("a")), nullptr);
}

TEST_F(ExternalWeightManagerTests, FindSharedWeightRequiresTheExactNameAndContract)
{
    ExternalWeightManager base;
    base.load(mDir, writeConfig("base.json", {binding("a", "a.weight", std::nullopt)}), mStream, mDir);
    ASSERT_NE(base.findSharedWeight("a", lastContract("a")), nullptr);
    EXPECT_EQ(base.findSharedWeight("missing", lastContract("a")), nullptr);
    EXPECT_EQ(base.findSharedWeight("a", lastContract("b")), nullptr) << "a different checkpoint key is a different contract";
    EXPECT_EQ(base.findSharedWeight("a", ""), nullptr);
}

TEST_F(ExternalWeightManagerTests, DraftBindingsOnTheTargetCheckpointShareTheLoadedTensor)
{
    ExternalWeightManager base;
    base.load(mDir,
        writeConfig("base.json", {binding("a", "a.weight", std::nullopt), binding("b", "b.weight", std::nullopt)}),
        mStream, mDir);
    ExternalWeightManager draft;
    draft.load(mDir, writeConfig("draft.json", {binding("a", "a.weight", "target")}), mStream, mDir, mDir,
        std::nullopt, std::nullopt, &base);

    Tensor const& baseA = weightNamed(base, "a");
    // The draft's tensor is the same device memory, not a second arena copy.
    Tensor const* const draftA = draft.findSharedWeight("a", lastContract("a"));
    ASSERT_NE(draftA, nullptr);
    EXPECT_EQ(draftA->rawPointer(), baseA.rawPointer());
    auto const values = readBack(*draftA);
    for (int32_t i = 0; i < 32; ++i)
    {
        EXPECT_EQ(__half2float(values[static_cast<size_t>(i)]), static_cast<float>(i) + 1.0F);
    }
}

TEST_F(ExternalWeightManagerTests, WithoutAPeerTheDraftLoadsItsOwnCopy)
{
    ExternalWeightManager base;
    base.load(mDir, writeConfig("base.json", {binding("a", "a.weight", std::nullopt)}), mStream, mDir);
    ExternalWeightManager draft;
    draft.load(mDir, writeConfig("draft.json", {binding("a", "a.weight", "target")}), mStream, mDir, mDir);
    EXPECT_NE(weightNamed(draft, "a").rawPointer(), weightNamed(base, "a").rawPointer());
    EXPECT_EQ(readBack(weightNamed(draft, "a")).size(), 32U);
}

TEST_F(ExternalWeightManagerTests, OnlyTargetSourcedBindingsWithAnIdenticalContractAreShared)
{
    ExternalWeightManager base;
    base.load(mDir,
        writeConfig("base.json", {binding("a", "a.weight", std::nullopt), binding("b", "b.weight", std::nullopt)}),
        mStream, mDir);
    // "a" comes from the draft's own (component) checkpoint, "b" names another key than the base's "b".
    ExternalWeightManager draft;
    draft.load(mDir,
        writeConfig("draft.json", {binding("a", "a.weight", "component"), binding("b", "a.weight", "target")}), mStream,
        mDir, mDir, std::nullopt, std::nullopt, &base);
    EXPECT_NE(weightNamed(draft, "a").rawPointer(), weightNamed(base, "a").rawPointer());
    Tensor const* const draftB = draft.findSharedWeight("b", binding("b", "a.weight", std::nullopt).dump());
    ASSERT_NE(draftB, nullptr);
    EXPECT_NE(draftB->rawPointer(), weightNamed(base, "b").rawPointer());
    // The draft's own "b" holds the "a" values, not the base's "b" values.
    EXPECT_EQ(__half2float(readBack(*draftB)[0]), 1.0F);
}

TEST_F(ExternalWeightManagerTests, AnyDifferenceInTheBindingContractPreventsSharing)
{
    ExternalWeightManager base;
    base.load(mDir, writeConfig("base.json", {binding("a", "a.weight", std::nullopt)}), mStream, mDir);
    // Same name, key, dtype and shape, but a different materialization parameter: not the same bytes in general.
    Json changed = binding("a", "a.weight", "target");
    changed["zero_point_offset"] = 8;
    ExternalWeightManager draft;
    draft.load(mDir, writeConfig("draft.json", {changed}), mStream, mDir, mDir, std::nullopt, std::nullopt, &base);
    Json contract = changed;
    contract.erase("checkpoint_source");
    Tensor const* const draftA = draft.findSharedWeight("a", contract.dump());
    ASSERT_NE(draftA, nullptr);
    EXPECT_NE(draftA->rawPointer(), weightNamed(base, "a").rawPointer());
}

TEST_F(ExternalWeightManagerTests, ADraftWithNothingToAllocateOfItsOwnLoads)
{
    // Every draft binding is shared, so the draft's own arena is empty.
    ExternalWeightManager base;
    base.load(mDir, writeConfig("base.json", {binding("a", "a.weight", std::nullopt)}), mStream, mDir);
    ExternalWeightManager draft;
    EXPECT_NO_THROW(draft.load(mDir, writeConfig("draft.json", {binding("a", "a.weight", "target")}), mStream, mDir,
        mDir, std::nullopt, std::nullopt, &base));
    EXPECT_EQ(draft.size(), 1U);
}

TEST_F(ExternalWeightManagerTests, SharedAndOwnedWeightsSurviveMovingTheManager)
{
    ExternalWeightManager base;
    base.load(mDir, writeConfig("base.json", {binding("a", "a.weight", std::nullopt)}), mStream, mDir);
    void const* const address = weightNamed(base, "a").rawPointer();
    ExternalWeightManager moved(std::move(base));
    EXPECT_EQ(weightNamed(moved, "a").rawPointer(), address);
    ExternalWeightManager draft;
    draft.load(mDir, writeConfig("draft.json", {binding("a", "a.weight", "target")}), mStream, mDir, mDir,
        std::nullopt, std::nullopt, &moved);
    EXPECT_EQ(weightNamed(draft, "a").rawPointer(), address);
    ExternalWeightManager movedDraft;
    movedDraft = std::move(draft);
    EXPECT_EQ(weightNamed(movedDraft, "a").rawPointer(), address);
}
