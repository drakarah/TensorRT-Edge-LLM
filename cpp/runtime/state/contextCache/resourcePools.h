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

#include "runtime/state/contextCache/contextCacheTypes.h"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Host-side allocation and reference accounting for all context-cache storage types.
//!
//! Each ResourceType has an independent fixed-capacity pool and circular free list (or, once switched with
//! setLowestFirst, lowest-ID-first allocation below a usable limit). ResourcePools manages only IDs and
//! reference counts; the corresponding GPU buffers are allocated and bound by the runtime. An active reference pins a
//! resource for an in-flight request, while a cache reference retains it for a published CacheRecord. A resource
//! returns to its free list only when both counts reach zero.
//!
//! Multi-pool allocation is all-or-nothing. Valid releases do not allocate and cannot fail after their preconditions
//! have been checked.
class ResourcePools
{
public:
    explicit ResourcePools(ResourceDemand capacities);

    //! Atomically allocate the full typed demand with active=1 and cache=0, or return nullopt without mutation.
    std::optional<std::vector<ResourceId>> allocate(ResourceDemand const& demand);
    //! Fill caller-owned storage and commit without dynamic memory allocation.
    //! The output vector must already match the total demand; insufficient capacity returns false without mutation.
    bool allocateInto(ResourceDemand const& demand, std::vector<ResourceId>& resources);

    //! Pin an allocated resource for an active request.
    void addActiveRef(ResourceId const& resource);
    //! Drop an active pin, returning the resource to the free list only when both reference counts reach zero.
    void releaseActiveRef(ResourceId const& resource);
    //! Add published-record ownership to an allocated resource.
    void addCacheRef(ResourceId const& resource);
    //! Drop published ownership, returning the resource to the free list only when both reference counts reach zero.
    void releaseCacheRef(ResourceId const& resource);

    int32_t activeRefCount(ResourceId const& resource) const;
    int32_t cacheRefCount(ResourceId const& resource) const;

    //! Free resources that can be allocated now: below the usable limit for a lowest-first pool.
    int32_t freeCount(ResourceType type) const noexcept;
    int32_t capacity(ResourceType type) const noexcept;
    bool canAllocate(ResourceDemand const& demand) const noexcept;

    //! Switch a pool that has nothing allocated to lowest-ID-first allocation restricted to IDs below @p usableLimit
    //! (clamped to the capacity), instead of the circular free list. Used when the storage behind the IDs is mapped on
    //! demand, so that every allocated ID lies in the mapped range.
    //! @throws std::runtime_error if a resource of this type is allocated
    void setLowestFirst(ResourceType type, int32_t usableLimit);
    //! Move the usable limit of a lowest-first pool; every allocated ID must stay below it.
    //! @throws std::runtime_error if the pool is not lowest-first or an allocated ID would lie at or above the limit
    void setUsableLimit(ResourceType type, int32_t usableLimit);
    //! False for an ID at or above the usable limit of a lowest-first pool (it cannot be allocated even when free).
    bool isUsable(ResourceId const& resource) const noexcept;
    //! One past the highest allocated ID of a pool (active or cache reference), 0 when none is allocated.
    int32_t highWater(ResourceType type) const noexcept;

private:
    struct Slot
    {
        int32_t activeRefCount{};
        int32_t cacheRefCount{};
        bool onFreeList{true};
    };

    struct Pool
    {
        std::vector<Slot> slots;
        std::vector<int32_t> freeList;
        size_t freeListHead{};
        //! Free resources; the circular list is only used when the pool is not lowest-first.
        size_t freeListCount{};
        bool lowestFirst{false};
        int32_t usableLimit{0};
    };

    Pool& pool(ResourceType type) noexcept;
    Pool const& pool(ResourceType type) const noexcept;
    Slot& slot(ResourceId const& resource);
    Slot const& slot(ResourceId const& resource) const;
    void releaseIfUnused(ResourceId const& resource, Slot& resourceSlot) noexcept;
    static size_t usableFree(Pool const& resourcePool) noexcept;

    std::array<Pool, 4> mPools;
};

} // namespace rt
} // namespace trt_edgellm
