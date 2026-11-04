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

#pragma once

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

//! Device memory the runtime manages itself, captured at request boundaries (the live structures are not safe to read
//! from another thread). Fields that do not apply stay at their defaults.
struct RuntimeMemoryMetrics
{
    bool valid{false};
    bool kvGrowable{false};
    int64_t kvMappedTokens{};
    int64_t kvCapacityTokens{};
    int64_t kvMappedBytes{};
    int64_t kvCapacityBytes{};
    //! Context-cache base KV pages: usable now (mapped range), free of those, and the highest page still held.
    int32_t kvPagesUsable{-1};
    int32_t kvPagesFree{-1};
    int32_t kvPagesHighWater{-1};
    int64_t contextRecords{-1};
    int32_t recurrentSnapshotsFree{-1};
    int32_t recurrentSnapshotsCapacity{-1};
    int32_t partialKvSnapshotsFree{-1};
    int32_t partialKvSnapshotsCapacity{-1};
    //! Expert cache of the MoE plugin: layers, slots per layer (the smallest over the layers), the most a layer can
    //! hold, device bytes of one slot across all layers, and device bytes of the whole cache.
    int32_t expertLayers{};
    int32_t expertSlotsPerLayer{};
    int32_t expertMaxSlotsPerLayer{};
    int64_t expertBytesPerSlotAllLayers{};
    int64_t expertCacheBytes{};
    //! Expert cache allocations that failed since start (the layer then streams through staging buffers).
    int64_t expertCacheAllocFailures{};
    //! Pinned host memory the runtime holds, by use: routed-expert weights, the token embedding table, the recurrent and
    //! partial-KV snapshot pools, and the host KV tier (capacity and the part holding sessions).
    int64_t hostExpertWeightsBytes{};
    int64_t hostEmbeddingBytes{};
    int64_t hostSnapshotBytes{};
    int64_t hostKvTierBytes{};
    int64_t hostKvTierUsedBytes{};
};

} // namespace rt
} // namespace trt_edgellm
