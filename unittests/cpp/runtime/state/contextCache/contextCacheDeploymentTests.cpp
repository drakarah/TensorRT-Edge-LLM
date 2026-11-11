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

#include "runtime/state/contextCache/contextCacheDeployment.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"

#include <gtest/gtest.h>

#include <array>

using namespace trt_edgellm::rt;

namespace
{

LLMEngineConfig makeAttentionConfig(int32_t attentionLayers = 4)
{
    LLMEngineConfig config;
    config.hiddenSize = 1024;
    config.numDecoderLayers = attentionLayers;
    config.numAttentionLayers = attentionLayers;
    config.numKVHeads = 8;
    config.headDim = 128;
    config.rotaryDim = 128;
    config.maxSupportedBatchSize = 2;
    config.maxKVCacheCapacity = 512;
    int64_t const minimumActivePages
        = computeMinimumKvPoolPages(config.maxSupportedBatchSize, config.maxKVCacheCapacity);
    ELLM_CHECK(minimumActivePages <= kMAX_KV_POOL_PAGES, "Test KV pool page count must fit int32.");
    config.kvPoolPages = static_cast<int32_t>(minimumActivePages);
    config.kvCacheDtype = nvinfer1::DataType::kHALF;
    config.layerTypes.assign(attentionLayers, HybridCacheManager::LayerType::kAttention);
    config.kvLayerConfigs.assign(attentionLayers, KVLayerConfig{config.numKVHeads, config.headDim});
    return config;
}

LLMEngineConfig makeHybridConfig(int32_t attentionLayers = 2, int32_t recurrentLayers = 2)
{
    LLMEngineConfig config = makeAttentionConfig(attentionLayers);
    config.numDecoderLayers = attentionLayers + recurrentLayers;
    config.numLinearAttnLayers = recurrentLayers;
    config.layerTypes.insert(config.layerTypes.end(), recurrentLayers, HybridCacheManager::LayerType::kMamba);
    config.recurrentStateNumHeads = 16;
    config.recurrentStateHeadDim = 32;
    config.recurrentStateSize = 64;
    config.convDim = 256;
    config.convKernel = 4;
    config.recurrentStateDtype = nvinfer1::DataType::kHALF;
    config.convStateDtype = nvinfer1::DataType::kHALF;
    return config;
}

DeploymentConfig makeEagleDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeAttentionConfig();
    deployment.base.specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.base.isSpecDecodeBase = true;
    deployment.base.specTargetLayerIds = {0, 2, 3};
    deployment.draft = makeAttentionConfig(/*attentionLayers=*/2);
    deployment.draft->specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.draft->baseModelHiddenSize = deployment.base.hiddenSize * 3;
    SpecDecodeConfig specConfig;
    specConfig.draftingTopK = 1;
    specConfig.draftingStep = 2;
    specConfig.verifySize = 4;
    deployment.specConfig = specConfig;
    return deployment;
}

DeploymentConfig makeGemma4MTPDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeAttentionConfig();
    deployment.base.specDecodeType = SpecDecodeMode::kGemma4MTP;
    deployment.base.isSpecDecodeBase = true;
    deployment.draft = makeAttentionConfig();
    deployment.draft->specDecodeType = SpecDecodeMode::kGemma4MTP;
    deployment.draft->hasOwnKVCache = false;
    deployment.draft->sharesTargetKV = true;
    SpecDecodeConfig specConfig;
    specConfig.draftingTopK = 1;
    specConfig.draftingStep = 4;
    specConfig.verifySize = 5;
    deployment.specConfig = specConfig;
    return deployment;
}

DeploymentConfig makeHybridMtpDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeHybridConfig();
    deployment.base.specDecodeType = SpecDecodeMode::kMTP;
    deployment.base.isSpecDecodeBase = true;
    deployment.draft = makeAttentionConfig(/*attentionLayers=*/2);
    deployment.draft->specDecodeType = SpecDecodeMode::kMTP;
    deployment.draft->hasOwnKVCache = true;
    deployment.draft->sharesTargetKV = false;
    deployment.draft->baseModelHiddenSize = deployment.base.hiddenSize;
    SpecDecodeConfig specConfig;
    specConfig.draftingTopK = 2;
    specConfig.draftingStep = 3;
    specConfig.verifySize = 5;
    deployment.specConfig = specConfig;
    return deployment;
}

} // namespace

TEST(ContextCacheDeploymentTests, ClassifiesSupportedVanillaHybridAndPureRecurrentDeployments)
{
    DeploymentConfig vanilla{makeAttentionConfig(), std::nullopt, std::nullopt};
    ContextCacheDeploymentProfile const vanillaProfile = validateContextCacheDeployment(vanilla);
    EXPECT_EQ(vanillaProfile.baseStateKind, ContextCacheModelStateKind::kAttentionOnly);
    EXPECT_FALSE(vanillaProfile.specReuseContract.has_value());

    DeploymentConfig hybrid{makeHybridConfig(), std::nullopt, std::nullopt};
    ContextCacheDeploymentProfile const hybridProfile = validateContextCacheDeployment(hybrid);
    EXPECT_EQ(hybridProfile.baseStateKind, ContextCacheModelStateKind::kHybrid);
    EXPECT_FALSE(hybridProfile.specReuseContract.has_value());

    LLMEngineConfig recurrent = makeHybridConfig(/*attentionLayers=*/0, /*recurrentLayers=*/3);
    DeploymentConfig pureRecurrent{recurrent, std::nullopt, std::nullopt};
    ContextCacheDeploymentProfile const recurrentProfile = validateContextCacheDeployment(pureRecurrent);
    EXPECT_EQ(recurrentProfile.baseStateKind, ContextCacheModelStateKind::kPureRecurrent);
    EXPECT_FALSE(recurrentProfile.specReuseContract.has_value());
}

TEST(ContextCacheDeploymentTests, AdmitsPromptOnlyBlockDiffusionAndRejectsUnsupportedVariants)
{
    DeploymentConfig deployment{makeAttentionConfig(), std::nullopt, std::nullopt};
    deployment.base.isDiffusionBackbone = true;

    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(deployment);
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kAttentionOnly);
    EXPECT_FALSE(profile.isSpeculative());

    deployment.base.numLinearAttnLayers = 1;
    deployment.base.numDecoderLayers += 1;
    deployment.base.layerTypes.push_back(HybridCacheManager::LayerType::kMamba);
    deployment.base.recurrentStateNumHeads = 16;
    deployment.base.recurrentStateHeadDim = 32;
    deployment.base.recurrentStateSize = 64;
    deployment.base.convDim = 256;
    deployment.base.convKernel = 4;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment = DeploymentConfig{makeAttentionConfig(), std::nullopt, std::nullopt};
    deployment.base.isDiffusionBackbone = true;
    deployment.base.specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.base.isSpecDecodeBase = true;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, SupportsFp8AndRejectsUnsupportedDtypeAndVisionAttention)
{
    DeploymentConfig deployment{makeAttentionConfig(), std::nullopt, std::nullopt};

    deployment.base.kvCacheDtype = nvinfer1::DataType::kFP8;
    EXPECT_NO_THROW(validateContextCacheDeployment(deployment));

    deployment.base = makeAttentionConfig();
    deployment.base.kvCacheDtype = nvinfer1::DataType::kBF16;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment.base = makeAttentionConfig();
    deployment.base.useVisionBidirectionalAttention = true;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, AllowsNonStatefulDecoderLayers)
{
    DeploymentConfig deployment{makeHybridConfig(), std::nullopt, std::nullopt};
    deployment.base.numDecoderLayers += 2;
    EXPECT_NO_THROW(validateContextCacheDeployment(deployment));
}

TEST(ContextCacheDeploymentTests, RejectsMalformedDeploymentTuples)
{
    DeploymentConfig malformedVanilla{makeAttentionConfig(), std::nullopt, SpecDecodeConfig{}};
    EXPECT_THROW(validateContextCacheDeployment(malformedVanilla), std::runtime_error);

    DeploymentConfig malformedEagle = makeEagleDeployment();
    malformedEagle.draft.reset();
    EXPECT_THROW(validateContextCacheDeployment(malformedEagle), std::runtime_error);

    malformedEagle = makeEagleDeployment();
    malformedEagle.base.isSpecDecodeBase = false;
    EXPECT_THROW(validateContextCacheDeployment(malformedEagle), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, RejectsUnsafeKvDonorGraphsAndLayouts)
{
    DeploymentConfig deployment{makeAttentionConfig(/*attentionLayers=*/3), std::nullopt, std::nullopt};
    deployment.base.kvSharingDonors = {-1, 0, 1};
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment.base.kvSharingDonors = {-1, 0, -1};
    deployment.base.kvLayerConfigs[1].headDim += 16;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, AdmitsPageTableAwareSpecModesAndRejectsHybridEagle)
{
    std::array<SpecDecodeMode, 3> const supportedModes{
        SpecDecodeMode::kDFlash,
        SpecDecodeMode::kJetSpec,
        SpecDecodeMode::kDSpark,
    };
    for (SpecDecodeMode const mode : supportedModes)
    {
        DeploymentConfig deployment = makeEagleDeployment();
        deployment.base.specDecodeType = mode;
        deployment.draft->specDecodeType = mode;
        if (mode == SpecDecodeMode::kDSpark)
        {
            deployment.draft->specDraftBlockSize = 4;
        }
        else
        {
            deployment.specConfig->dflashBlockSize = 4;
        }
        ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(deployment);
        ASSERT_TRUE(profile.specReuseContract.has_value());
    }

    DeploymentConfig hybridEagle = makeEagleDeployment();
    hybridEagle.base = makeHybridConfig();
    hybridEagle.base.specDecodeType = SpecDecodeMode::kEAGLE;
    hybridEagle.base.isSpecDecodeBase = true;
    EXPECT_THROW(validateContextCacheDeployment(hybridEagle), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, AdmitsHybridMtpAndRejectsAttentionOnlyOrSharedKvMtp)
{
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(makeHybridMtpDeployment());
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kHybrid);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_TRUE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 1);

    DeploymentConfig attentionOnly = makeHybridMtpDeployment();
    attentionOnly.base = makeAttentionConfig();
    attentionOnly.base.specDecodeType = SpecDecodeMode::kMTP;
    attentionOnly.base.isSpecDecodeBase = true;
    EXPECT_THROW(validateContextCacheDeployment(attentionOnly), std::runtime_error);

    DeploymentConfig sharedKv = makeHybridMtpDeployment();
    sharedKv.draft->hasOwnKVCache = false;
    sharedKv.draft->sharesTargetKV = true;
    EXPECT_THROW(validateContextCacheDeployment(sharedKv), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, AdmitsHybridDFlashAndRejectsOtherHybridBlockDrafters)
{
    DeploymentConfig dflash = makeHybridMtpDeployment();
    dflash.base.specDecodeType = SpecDecodeMode::kDFlash;
    dflash.draft->specDecodeType = SpecDecodeMode::kDFlash;
    dflash.specConfig->dflashBlockSize = 8;
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(dflash);
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kHybrid);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_TRUE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 0);

    // The other block drafters keep their single-pass, attention-only contract.
    for (SpecDecodeMode const mode : {SpecDecodeMode::kJetSpec, SpecDecodeMode::kDSpark})
    {
        DeploymentConfig other = dflash;
        other.base.specDecodeType = mode;
        other.draft->specDecodeType = mode;
        EXPECT_THROW(validateContextCacheDeployment(other), std::runtime_error);
    }

    // A draft without its own KV cache cannot be paired with the hybrid snapshot.
    DeploymentConfig sharedKv = dflash;
    sharedKv.draft->hasOwnKVCache = false;
    sharedKv.draft->sharesTargetKV = true;
    EXPECT_THROW(validateContextCacheDeployment(sharedKv), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, RingedDFlashDraftIsAPlainHybridWithASnapshottedRing)
{
    DeploymentConfig dflash = makeHybridMtpDeployment();
    dflash.base.specDecodeType = SpecDecodeMode::kDFlash;
    dflash.draft->specDecodeType = SpecDecodeMode::kDFlash;
    dflash.specConfig->dflashBlockSize = 8;
    ContextCacheDeploymentProfile const paged = validateContextCacheDeployment(dflash);
    EXPECT_FALSE(paged.ringedDraftKv());
    EXPECT_TRUE(paged.ownsPagedSpecState());

    // The builder's --draft-kv-ring-pages: a draft pool smaller than its capacity needs, at least the 2,048-token
    // window (16 pages) plus one prefill pass plus a margin.
    DeploymentConfig ringed = dflash;
    ringed.draft->maxSupportedBatchSize = 1;
    ringed.draft->maxSupportedInputLength = 1024;
    ringed.draft->maxKVCacheCapacity = 262144;
    ringed.draft->kvPoolPages = 32;
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(ringed);
    EXPECT_TRUE(profile.ringedDraftKv());
    EXPECT_FALSE(profile.ownsPagedSpecState());
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 0);

    // A pool below the window cannot be a ring.
    DeploymentConfig tooSmall = ringed;
    tooSmall.draft->kvPoolPages = 17;
    EXPECT_THROW(validateContextCacheDeployment(tooSmall), std::runtime_error);

    // Only DFlash drafts may declare a ring.
    DeploymentConfig mtp = makeHybridMtpDeployment();
    mtp.draft->kvPoolPages = 4;
    EXPECT_THROW(validateContextCacheDeployment(mtp), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, RingDetectionBoundaries)
{
    DeploymentConfig dflash = makeHybridMtpDeployment();
    dflash.base.specDecodeType = SpecDecodeMode::kDFlash;
    dflash.draft->specDecodeType = SpecDecodeMode::kDFlash;
    dflash.specConfig->dflashBlockSize = 8;
    dflash.draft->maxSupportedBatchSize = 1;
    dflash.draft->maxSupportedInputLength = 1024;
    dflash.draft->maxKVCacheCapacity = 262144;

    // Exactly the floor (26) is a ring, one page less is refused, the full pool and above are plain paged drafts.
    dflash.draft->kvPoolPages = 26;
    EXPECT_TRUE(validateContextCacheDeployment(dflash).ringedDraftKv());
    dflash.draft->kvPoolPages = 25;
    EXPECT_THROW(validateContextCacheDeployment(dflash), std::runtime_error);
    dflash.draft->kvPoolPages = 2047;
    EXPECT_TRUE(validateContextCacheDeployment(dflash).ringedDraftKv());
    dflash.draft->kvPoolPages = 2048;
    EXPECT_FALSE(validateContextCacheDeployment(dflash).ringedDraftKv());
    dflash.draft->kvPoolPages = 4096;
    EXPECT_FALSE(validateContextCacheDeployment(dflash).ringedDraftKv());

    // The floor follows the input length in whole pages.
    dflash.draft->maxSupportedInputLength = 1025;
    dflash.draft->kvPoolPages = 26;
    EXPECT_THROW(validateContextCacheDeployment(dflash), std::runtime_error);
    dflash.draft->kvPoolPages = 27;
    EXPECT_TRUE(validateContextCacheDeployment(dflash).ringedDraftKv());
}

TEST(ContextCacheDeploymentTests, RingedProfileHelpersAreFalseWithoutASpeculativeContract)
{
    ContextCacheDeploymentProfile const vanilla{ContextCacheModelStateKind::kHybrid, std::nullopt};
    EXPECT_FALSE(vanilla.ringedDraftKv());
    EXPECT_FALSE(vanilla.ownsPagedSpecState());
}

TEST(ContextCacheDeploymentTests, ResolvesPerMethodReuseContract)
{
    DeploymentConfig eagle = makeEagleDeployment();
    SpecReuseContract const eagleContract = *validateContextCacheDeployment(eagle).specReuseContract;
    EXPECT_TRUE(eagleContract.ownsPagedSpecState);
    EXPECT_EQ(eagleContract.futureDependencyTokens, 1);

    for (SpecDecodeMode const mode : {SpecDecodeMode::kDFlash, SpecDecodeMode::kJetSpec})
    {
        DeploymentConfig blockDraft = makeEagleDeployment();
        blockDraft.base.specDecodeType = mode;
        blockDraft.draft->specDecodeType = mode;
        blockDraft.specConfig->dflashBlockSize = 7;
        SpecReuseContract const contract = *validateContextCacheDeployment(blockDraft).specReuseContract;
        EXPECT_TRUE(contract.ownsPagedSpecState);
        EXPECT_EQ(contract.futureDependencyTokens, 0);
    }

    DeploymentConfig dspark = makeEagleDeployment();
    dspark.base.specDecodeType = SpecDecodeMode::kDSpark;
    dspark.draft->specDecodeType = SpecDecodeMode::kDSpark;
    dspark.draft->specDraftBlockSize = 9;
    SpecReuseContract const dsparkContract = *validateContextCacheDeployment(dspark).specReuseContract;
    EXPECT_TRUE(dsparkContract.ownsPagedSpecState);
    EXPECT_EQ(dsparkContract.futureDependencyTokens, 0);

    SpecReuseContract const gemmaContract
        = *validateContextCacheDeployment(makeGemma4MTPDeployment()).specReuseContract;
    EXPECT_FALSE(gemmaContract.ownsPagedSpecState);
    EXPECT_EQ(gemmaContract.futureDependencyTokens, 0);
}

TEST(ContextCacheDeploymentTests, ClassifiesSupportedEagle)
{
    DeploymentConfig deployment = makeEagleDeployment();
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(deployment);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kAttentionOnly);
    EXPECT_TRUE(profile.specReuseContract->ownsPagedSpecState);
    EXPECT_EQ(profile.specReuseContract->futureDependencyTokens, 1);
}

TEST(ContextCacheDeploymentTests, AdmitsEagleTreeConfiguration)
{
    DeploymentConfig deployment = makeEagleDeployment();
    deployment.specConfig->draftingTopK = 2;

    EXPECT_NO_THROW(validateContextCacheDeployment(deployment));
}

TEST(ContextCacheDeploymentTests, AdmitsAttentionOnlyGemma4MTPWithoutPagedSpecState)
{
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(makeGemma4MTPDeployment());

    EXPECT_EQ(profile.baseStateKind, ContextCacheModelStateKind::kAttentionOnly);
    ASSERT_TRUE(profile.specReuseContract.has_value());
    EXPECT_FALSE(profile.specReuseContract->ownsPagedSpecState);
}
