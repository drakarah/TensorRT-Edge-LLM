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
#include "runtime/state/contextCache/contextCacheCoordinator.h"

#include "common/cudaMacros.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/contextCache/hybridSnapshotStorage.h"
#include "runtime/state/kvPageTable.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kMAX_SEQUENCE_LENGTH{4096}; // 32 pages
constexpr int32_t kMAX_INPUT_LENGTH{128};
constexpr int32_t kHIDDEN_SIZE{64};
constexpr int32_t kRING_PAGES{20}; // 16 window pages + 1 prefill pass page + 2 margin, below the 32-page full pool

std::vector<int32_t> makeTokens(int32_t count)
{
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    std::iota(tokens.begin(), tokens.end(), 1);
    return tokens;
}

LLMEngineConfig makeBaseConfig()
{
    LLMEngineConfig config;
    config.modelType = "ring-coordinator-base";
    config.hiddenSize = kHIDDEN_SIZE;
    config.numDecoderLayers = 2;
    config.numAttentionLayers = 1;
    config.numLinearAttnLayers = 1;
    config.numKVHeads = 1;
    config.headDim = 8;
    config.rotaryDim = 8;
    config.maxSupportedBatchSize = 1;
    config.maxSupportedInputLength = kMAX_INPUT_LENGTH;
    config.maxKVCacheCapacity = kMAX_SEQUENCE_LENGTH;
    config.kvPoolPages = computeMinimumKvPoolPages(1, kMAX_SEQUENCE_LENGTH);
    config.kvCacheDtype = DataType::kHALF;
    config.layerTypes = {HybridCacheManager::LayerType::kAttention, HybridCacheManager::LayerType::kMamba};
    config.kvLayerConfigs = {KVLayerConfig{config.numKVHeads, config.headDim}};
    config.recurrentStateNumHeads = 2;
    config.recurrentStateHeadDim = 3;
    config.recurrentStateSize = 4;
    config.convDim = 5;
    config.convKernel = 6;
    config.recurrentStateDtype = DataType::kHALF;
    config.convStateDtype = DataType::kHALF;
    config.specDecodeType = SpecDecodeMode::kDFlash;
    config.isSpecDecodeBase = true;
    return config;
}

LLMEngineConfig makeDraftConfig(int32_t poolPages)
{
    LLMEngineConfig config;
    config.modelType = "ring-coordinator-draft";
    config.hiddenSize = kHIDDEN_SIZE;
    config.numDecoderLayers = 1;
    config.numAttentionLayers = 1;
    config.numLinearAttnLayers = 0;
    config.numKVHeads = 1;
    config.headDim = 8;
    config.rotaryDim = 8;
    config.maxSupportedBatchSize = 1;
    config.maxSupportedInputLength = kMAX_INPUT_LENGTH;
    config.maxKVCacheCapacity = kMAX_SEQUENCE_LENGTH;
    config.kvPoolPages = poolPages;
    config.kvCacheDtype = DataType::kHALF;
    config.layerTypes = {HybridCacheManager::LayerType::kAttention};
    config.kvLayerConfigs = {KVLayerConfig{config.numKVHeads, config.headDim}};
    config.specDecodeType = SpecDecodeMode::kDFlash;
    config.isSpecDecodeBase = false;
    return config;
}

DeploymentConfig makeDeployment(int32_t draftPoolPages)
{
    DeploymentConfig deployment;
    deployment.base = makeBaseConfig();
    deployment.draft = makeDraftConfig(draftPoolPages);
    SpecDecodeConfig spec;
    spec.draftingTopK = 1;
    spec.draftingStep = 1;
    spec.verifySize = 8;
    spec.dflashBlockSize = 8;
    deployment.specConfig = spec;
    return deployment;
}

HybridCacheManager::Config makeCacheConfig(LLMEngineConfig const& engine, int32_t ringPages)
{
    KVCacheManager::Config kvConfig{engine.numAttentionLayers, engine.maxSupportedBatchSize, engine.maxKVCacheCapacity,
        engine.kvLayerConfigs, engine.kvCacheDtype, engine.kvPoolPages};
    if (ringPages > 0)
    {
        kvConfig.ringPages = ringPages;
        kvConfig.useBoundedSwaKVCache = false;
    }
    MambaCacheManager::Config mambaConfig = engine.numLinearAttnLayers > 0
        ? MambaCacheManager::Config{engine.numLinearAttnLayers, engine.maxSupportedBatchSize,
              engine.recurrentStateNumHeads, engine.recurrentStateHeadDim, engine.recurrentStateSize, engine.convDim,
              engine.convKernel, 0, engine.recurrentStateDtype, engine.convStateDtype}
        : MambaCacheManager::Config{/*.numRecurrentLayers=*/0, engine.maxSupportedBatchSize};
    return HybridCacheManager::Config{
        engine.layerTypes, std::move(kvConfig), std::move(mambaConfig), engine.maxSupportedBatchSize};
}

//! A DFlash hybrid deployment whose draft KV is a 20-page static ring.
class ContextCacheRingDraftCoordinatorTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mDeployment = makeDeployment(kRING_PAGES);
        mBaseCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mDeployment.base, 0), mStream);
        mDraftCache = std::make_unique<HybridCacheManager>(makeCacheConfig(*mDeployment.draft, kRING_PAGES), mStream);
        KVCacheManager const& baseKv = mBaseCache->getKVCacheManager();
        mBasePageTable = std::make_unique<KVPageTable>(1, pagesPerSlot(baseKv.maxCapPadded()), baseKv.numPages());
        mBasePageTable->setIdentity();
        mBasePageTable->upload(mStream);
        KVCacheManager const& draftKv = mDraftCache->getKVCacheManager();
        mDraftPageTable = std::make_unique<KVPageTable>(1, pagesPerSlot(draftKv.maxCapPadded()), draftKv.numPages());
        mDraftPageTable->setRingPages(kRING_PAGES);
        mDraftPageTable->setIdentity();
        mDraftPageTable->upload(mStream);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        createCoordinator();
    }

    void TearDown() override
    {
        if (mCoordinator != nullptr)
        {
            EXPECT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
            mCoordinator.reset();
        }
        mDraftPageTable.reset();
        mBasePageTable.reset();
        mDraftCache.reset();
        mBaseCache.reset();
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    //! One checkpoint: recurrent + conv state, the boundary hidden row (every speculative hybrid slot reserves it) and
    //! the whole draft ring.
    size_t recurrentBytesPerSlot() const
    {
        return HybridSnapshotStorage::recurrentBytesPerSlot(mBaseCache->getMambaCacheManager().getConfig())
            + HybridSnapshotStorage::boundaryHiddenBytesPerSlot(kHIDDEN_SIZE, DataType::kHALF)
            + HybridSnapshotStorage::draftRingBytesPerSlot(mDraftCache->getKVCacheManager());
    }

    size_t partialKvBytesPerSlot() const
    {
        // The ring carries the draft state, so the partial-KV axis holds the base page only.
        return HybridSnapshotStorage::partialKvBytesPerSlot(mBaseCache->getKVCacheManager().getConfig());
    }

    void createCoordinator(int32_t snapshotSlots = 4)
    {
        createCoordinatorWithBudgets(static_cast<int64_t>(static_cast<size_t>(snapshotSlots) * recurrentBytesPerSlot()),
            static_cast<int64_t>(static_cast<size_t>(snapshotSlots) * partialKvBytesPerSlot()));
    }

    void createCoordinatorWithBudgets(int64_t recurrentPoolBytes, int64_t partialKvPoolBytes)
    {
        ContextCachePhysicalResources resources{*mBaseCache, *mBasePageTable, mDraftCache.get(), mDraftPageTable.get()};
        ContextCacheConfig config{/*.enabled=*/true, /*.maxRecords=*/16, recurrentPoolBytes, partialKvPoolBytes};
        mCoordinator = std::make_unique<ContextCacheCoordinator>(
            config, mDeployment, validateContextCacheDeployment(mDeployment), resources, mStream);
    }

    ContextCacheCoordinator::AdmissionResult begin(std::vector<int32_t> tokens,
        ContextCacheLookupPolicy lookupPolicy = ContextCacheLookupPolicy::kUseCache, bool speculativeRequest = true)
    {
        ContextCacheBatchAdmission admission;
        admission.speculativeRequest = speculativeRequest;
        // The DFlash decoder asks for draft headroom; with a ring there is no page pool to reserve it in.
        DecodingKvHeadroom const headroom = speculativeRequest ? DecodingKvHeadroom{8, 8} : DecodingKvHeadroom{1, 0};
        admission.lookupPolicy = lookupPolicy;
        admission.sequences.push_back(ContextCacheSequenceAdmission{std::move(tokens), {}, {}, ResidentRef{0, 1}});
        ContextCacheCoordinator::BeginRequestResult result = mCoordinator->beginRequest(admission, headroom, mStream);
        EXPECT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
        EXPECT_TRUE(result.admission.has_value());
        return std::move(*result.admission);
    }

    void seedRecurrent(uint8_t recurrentPattern, uint8_t convPattern)
    {
        MambaCacheManager& mamba = mBaseCache->getMambaCacheManager();
        MambaCacheManager::Config const& config = mamba.getConfig();
        size_t const recurrentBytes = static_cast<size_t>(config.recurrentStateNumHeads * config.recurrentStateHeadDim
                                          * config.recurrentStateSize)
            * sizeof(half);
        size_t const convBytes = static_cast<size_t>(config.convDim * config.convKernel) * sizeof(half);
        CUDA_CHECK(cudaMemsetAsync(mamba.getRecurrentState(0).rawPointer(), recurrentPattern, recurrentBytes, mStream));
        CUDA_CHECK(cudaMemsetAsync(mamba.getConvState(0).rawPointer(), convPattern, convBytes, mStream));
    }

    void expectRecurrent(uint8_t recurrentPattern)
    {
        MambaCacheManager& mamba = mBaseCache->getMambaCacheManager();
        MambaCacheManager::Config const& config = mamba.getConfig();
        size_t const bytes = static_cast<size_t>(config.recurrentStateNumHeads * config.recurrentStateHeadDim
                                 * config.recurrentStateSize)
            * sizeof(half);
        std::vector<uint8_t> host(bytes);
        ASSERT_EQ(cudaMemcpy(host.data(), mamba.getRecurrentState(0).rawPointer(), bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_TRUE(
            std::all_of(host.begin(), host.end(), [recurrentPattern](uint8_t value) { return value == recurrentPattern; }));
    }

    size_t ringBytes() const
    {
        return HybridSnapshotStorage::draftRingBytesPerSlot(mDraftCache->getKVCacheManager());
    }

    void seedRing(uint8_t pattern)
    {
        CUDA_CHECK(cudaMemsetAsync(
            mDraftCache->getKVCacheManager().getCombinedKVCache(0).rawPointer(), pattern, ringBytes(), mStream));
    }

    void expectRing(uint8_t pattern)
    {
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<uint8_t> host(ringBytes());
        ASSERT_EQ(cudaMemcpy(host.data(), mDraftCache->getKVCacheManager().getCombinedKVCache(0).rawPointer(),
                      host.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_TRUE(std::all_of(host.begin(), host.end(), [pattern](uint8_t value) { return value == pattern; }));
    }

    int32_t draftLength()
    {
        int32_t length{-1};
        EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(&length, mDraftCache->getKVCacheLengths().rawPointer(), sizeof(int32_t),
                      cudaMemcpyDeviceToHost),
            cudaSuccess);
        return length;
    }

    //! The draft page table must stay the static ring through every request.
    void expectStaticRingTable()
    {
        int32_t const maxPages = mDraftPageTable->maxPagesPerSeq();
        for (int32_t j = 0; j < maxPages; ++j)
        {
            ASSERT_EQ(mDraftPageTable->hostRow(0)[j], j % kRING_PAGES) << "logical page " << j;
        }
    }

    //! Publish an exact checkpoint of `length` resident tokens for `tokens` and finish the producer.
    void publishCheckpoint(std::vector<int32_t> tokens, int32_t length, uint8_t recurrent, uint8_t ring)
    {
        auto producer = begin(std::move(tokens));
        ASSERT_EQ(mCoordinator->preparePrefill(producer.request), ContextCacheCoordinatorStatus::kOk);
        seedRecurrent(recurrent, 0x61);
        seedRing(ring);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        ASSERT_EQ(mCoordinator->publishHybridEndpoint(producer.request, 0, length), ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);
    }

    cudaStream_t mStream{};
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mBaseCache;
    std::unique_ptr<HybridCacheManager> mDraftCache;
    std::unique_ptr<KVPageTable> mBasePageTable;
    std::unique_ptr<KVPageTable> mDraftPageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

} // namespace

TEST_F(ContextCacheRingDraftCoordinatorTests, ProfileIsAPlainHybridWithARingedDraft)
{
    ContextCacheDeploymentProfile const profile = validateContextCacheDeployment(mDeployment);
    EXPECT_TRUE(profile.ringedDraftKv());
    EXPECT_FALSE(profile.ownsPagedSpecState());
    EXPECT_EQ(mCoordinator->manager().pools().capacity(ResourceType::kDraftKvPage), 0)
        << "a ring leases no draft pages";
    EXPECT_EQ(mCoordinator->manager().pools().capacity(ResourceType::kRecurrentSnapshot), 4);
}

TEST_F(ContextCacheRingDraftCoordinatorTests, ARingedRequestAcceptsDraftHeadroomWithoutLeasingDraftPages)
{
    auto request = begin(makeTokens(kTOKENS_PER_PAGE + 1));
    ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->finish(request.request), ContextCacheCoordinatorStatus::kOk);
    expectStaticRingTable();
}

TEST_F(ContextCacheRingDraftCoordinatorTests, DecodeStepWithDraftHeadroomGrowsOnlyBasePages)
{
    auto request = begin(makeTokens(kTOKENS_PER_PAGE - 1));
    ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    ASSERT_EQ(mCoordinator->enqueuePrefillCaptures(request.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    int32_t const firstToken = 1;
    std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&firstToken, 1, kTOKENS_PER_PAGE - 1}};
    ASSERT_EQ(mCoordinator->finalizePrefillPublication(request.request, progress), ContextCacheCoordinatorStatus::kOk);
    // The working set crosses a page boundary: the base lease grows, the draft ring does not change.
    EXPECT_EQ(mCoordinator->prepareDecodeStep(request.request, DecodingKvHeadroom{8, 8}),
        ContextCacheCoordinatorStatus::kOk);
    expectStaticRingTable();
    EXPECT_EQ(mCoordinator->finish(request.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheRingDraftCoordinatorTests, HitRestoresRecurrentStateBasePartialPageAndTheWholeDraftRing)
{
    constexpr int32_t kLength{kTOKENS_PER_PAGE + 5};
    publishCheckpoint(makeTokens(kLength), kLength, 0x31, 0x4A);
    ASSERT_EQ(mCoordinator->manager().records().size(), 1U);

    // Scrub the live state so only the checkpoint can bring it back.
    seedRecurrent(0, 0);
    seedRing(0x01);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    auto consumer = begin(makeTokens(kLength + 3));
    ASSERT_EQ(consumer.prefillStarts, std::vector<int32_t>{kLength});
    ASSERT_EQ(mCoordinator->preparePrefill(consumer.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    expectRecurrent(0x31);
    expectRing(0x4A);
    EXPECT_EQ(draftLength(), kLength) << "the draft resumes at the restored length";
    expectStaticRingTable();
    EXPECT_EQ(mCoordinator->metrics().hybridRestores, 1U);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheRingDraftCoordinatorTests, EachCheckpointRestoresItsOwnRing)
{
    constexpr int32_t kShort{kTOKENS_PER_PAGE + 5};
    constexpr int32_t kLong{3 * kTOKENS_PER_PAGE + 9};
    // Two unrelated prompts (different first token) so both checkpoints stay.
    auto promptA = makeTokens(kShort);
    auto promptB = makeTokens(kLong);
    for (auto& token : promptB)
    {
        token += 100000;
    }
    publishCheckpoint(promptA, kShort, 0x31, 0x4A);
    publishCheckpoint(promptB, kLong, 0x32, 0x5B);
    ASSERT_EQ(mCoordinator->manager().records().size(), 2U);

    for (auto const& [prompt, length, recurrent, ring] :
        {std::tuple{promptA, kShort, 0x31, 0x4A}, std::tuple{promptB, kLong, 0x32, 0x5B}})
    {
        seedRecurrent(0, 0);
        seedRing(0x01);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        auto extended = prompt;
        extended.push_back(7);
        auto consumer = begin(std::move(extended));
        ASSERT_EQ(consumer.prefillStarts, std::vector<int32_t>{length});
        ASSERT_EQ(mCoordinator->preparePrefill(consumer.request), ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        expectRecurrent(static_cast<uint8_t>(recurrent));
        expectRing(static_cast<uint8_t>(ring));
        EXPECT_EQ(draftLength(), length);
        ASSERT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
    }
}

TEST_F(ContextCacheRingDraftCoordinatorTests, MissAndBypassRestoreNothingAndStartTheDraftAtZero)
{
    constexpr int32_t kLength{kTOKENS_PER_PAGE + 5};
    publishCheckpoint(makeTokens(kLength), kLength, 0x31, 0x4A);
    seedRing(0x01);
    seedRecurrent(0x09, 0x09);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);

    // A bypass request never looks up the record.
    auto bypass = begin(makeTokens(kLength + 3), ContextCacheLookupPolicy::kBypass);
    ASSERT_EQ(bypass.prefillStarts, std::vector<int32_t>{0});
    ASSERT_EQ(mCoordinator->preparePrefill(bypass.request), ContextCacheCoordinatorStatus::kOk);
    expectRing(0x01);
    EXPECT_EQ(draftLength(), 0);
    ASSERT_EQ(mCoordinator->finish(bypass.request), ContextCacheCoordinatorStatus::kOk);

    // A different prompt is a miss: the live ring is left alone, the draft starts empty.
    auto miss = makeTokens(kLength + 3);
    miss[0] = 999999;
    auto request = begin(std::move(miss));
    ASSERT_EQ(request.prefillStarts, std::vector<int32_t>{0});
    ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
    expectRing(0x01);
    EXPECT_EQ(draftLength(), 0);
    EXPECT_EQ(mCoordinator->finish(request.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheRingDraftCoordinatorTests, PublishGuardsSkipEmptyBypassAndRepeatedBoundaries)
{
    auto request = begin(makeTokens(kTOKENS_PER_PAGE + 5));
    ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    EXPECT_EQ(mCoordinator->publishHybridEndpoint(request.request, 0, 0), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);
    EXPECT_EQ(mCoordinator->publishHybridEndpoint(request.request, 0, kTOKENS_PER_PAGE + 5),
        ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->manager().records().size(), 1U);
    // The same boundary again is idempotent.
    EXPECT_EQ(mCoordinator->publishHybridEndpoint(request.request, 0, kTOKENS_PER_PAGE + 5),
        ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->manager().records().size(), 1U);
    EXPECT_EQ(mCoordinator->finish(request.request), ContextCacheCoordinatorStatus::kOk);

    auto bypass = begin(makeTokens(2 * kTOKENS_PER_PAGE), ContextCacheLookupPolicy::kBypass);
    ASSERT_EQ(mCoordinator->preparePrefill(bypass.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->publishHybridEndpoint(bypass.request, 0, 2 * kTOKENS_PER_PAGE),
        ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(mCoordinator->manager().records().size(), 1U);
    EXPECT_EQ(mCoordinator->finish(bypass.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheRingDraftCoordinatorTests, PageAlignedBoundaryNeedsNoPartialPageAndStillRestoresTheRing)
{
    constexpr int32_t kLength{2 * kTOKENS_PER_PAGE};
    publishCheckpoint(makeTokens(kLength), kLength, 0x33, 0x6C);
    seedRecurrent(0, 0);
    seedRing(0x01);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    auto consumer = begin(makeTokens(kLength + 1));
    ASSERT_EQ(consumer.prefillStarts, std::vector<int32_t>{kLength});
    ASSERT_EQ(mCoordinator->preparePrefill(consumer.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    expectRecurrent(0x33);
    expectRing(0x6C);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheRingDraftCoordinatorTests, VanillaRequestOnARingedDeploymentStillRoundTrips)
{
    constexpr int32_t kLength{kTOKENS_PER_PAGE + 5};
    publishCheckpoint(makeTokens(kLength), kLength, 0x31, 0x4A);
    auto vanilla = begin(makeTokens(kLength + 1), ContextCacheLookupPolicy::kUseCache, /*speculativeRequest=*/false);
    ASSERT_EQ(vanilla.prefillStarts, std::vector<int32_t>{kLength});
    EXPECT_EQ(mCoordinator->preparePrefill(vanilla.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    expectRecurrent(0x31);
    EXPECT_EQ(mCoordinator->finish(vanilla.request), ContextCacheCoordinatorStatus::kOk);
}

// The slot budget prices the ring: a budget one byte under a slot gives no slot (construction refuses), exactly one
// slot's bytes gives one slot.
TEST_F(ContextCacheRingDraftCoordinatorTests, RecurrentBudgetCountsTheRingBytesPerSlot)
{
    mCoordinator->shutdown();
    mCoordinator.reset();
    int64_t const slot = static_cast<int64_t>(recurrentBytesPerSlot());
    int64_t const partial = static_cast<int64_t>(partialKvBytesPerSlot());
    EXPECT_THROW(createCoordinatorWithBudgets(slot - 1, partial), std::runtime_error);
    createCoordinatorWithBudgets(slot, partial);
    EXPECT_EQ(mCoordinator->manager().pools().capacity(ResourceType::kRecurrentSnapshot), 1);
    mCoordinator->shutdown();
    mCoordinator.reset();
    createCoordinatorWithBudgets(2 * slot - 1, partial);
    EXPECT_EQ(mCoordinator->manager().pools().capacity(ResourceType::kRecurrentSnapshot), 1);
    createCoordinator(); // TearDown expects a coordinator to shut down
}

TEST_F(ContextCacheRingDraftCoordinatorTests, RingedDeploymentNeedsTheDraftCacheBound)
{
    mCoordinator->shutdown();
    mCoordinator.reset();
    ContextCachePhysicalResources withoutDraft{*mBaseCache, *mBasePageTable, nullptr, nullptr};
    ContextCacheConfig config{/*.enabled=*/true, /*.maxRecords=*/16, static_cast<int64_t>(recurrentBytesPerSlot()),
        static_cast<int64_t>(partialKvBytesPerSlot())};
    EXPECT_THROW(ContextCacheCoordinator(config, mDeployment, validateContextCacheDeployment(mDeployment),
                     withoutDraft, mStream),
        std::runtime_error);
    createCoordinator();
}
