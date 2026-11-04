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

#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/contextCache/cacheRecord.h"

#include <cstdint>
#include <list>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! One device KV page paired with one host slot, for a copy the runtime has to enqueue.
struct HostKvTransfer
{
    PageId page{};
    int32_t slot{};
    //! True: device page to host slot (demotion). False: host slot to device page (restore).
    bool toHost{true};
};

//! One block of a demoted record and the host slot that holds its KV page.
struct HostKvBlock
{
    BlockHash hash{};
    int32_t slot{};
};

enum class HostKvPlanStatus : uint8_t
{
    //! The entry can be added; victims (possibly none) must be evicted first.
    kReady,
    //! An entry with the same key is already stored.
    kAlreadyPresent,
    //! The entry cannot fit even after evicting every unpinned entry, or it needs more slots than the store has.
    kInfeasible,
    //! The entry has no blocks.
    kEmpty,
};

//! What demoting one record would do. Built by planDemotion() and consumed by commitDemotion().
struct HostKvDemotionPlan
{
    HostKvPlanStatus status{HostKvPlanStatus::kEmpty};
    //! Entries to evict, least recently used first.
    std::vector<CacheRecordKey> victims;
    //! Blocks that need a new slot, in prefix order, with the slot each one gets.
    std::vector<HostKvBlock> newBlocks;
    //! Index version the plan was made against; commit rejects a plan from an older version.
    uint64_t version{};
};

//! A host entry that covers a prefix of a request.
struct HostKvHit
{
    CacheRecordKey key;
    //! One slot per block of the entry, in prefix order.
    std::vector<int32_t> slots;
};

//! CPU-only bookkeeping for the host-RAM KV tier: which block hashes live in which host slot.
//!
//! Entries mirror evicted cache records (a chain of block hashes ending in a terminal hash). Blocks are shared between
//! entries that share a prefix: one slot per distinct block hash, reference counted by the entries that contain it.
//! Slots are handed out lowest-first. Eviction is least recently used and skips pinned entries. The class owns no
//! memory and does no copies; HostKvPageStore moves the bytes. Not thread-safe; callers serialize access.
class HostKvIndex
{
public:
    //! @p slotCount may be zero, which makes every demotion infeasible.
    explicit HostKvIndex(int32_t slotCount);

    //! Decide how to store a record whose blocks are @p blockHashes (prefix order, last one is the terminal hash).
    //! Does not change the index.
    HostKvDemotionPlan planDemotion(std::vector<BlockHash> const& blockHashes) const;

    //! Apply a plan made against the current version: evict the victims, assign the slots, and insert the entry as
    //! most recently used. Returns false and changes nothing when the plan is not kReady or the index changed since.
    bool commitDemotion(std::vector<BlockHash> const& blockHashes, HostKvDemotionPlan const& plan);

    //! Longest stored entry whose blocks equal the first blocks of @p requestHashes. Does not change the index.
    std::optional<HostKvHit> lookupPrefix(std::vector<BlockHash> const& requestHashes) const;

    //! Mark an entry most recently used. Returns false if it does not exist.
    bool touch(CacheRecordKey const& key);
    //! A pinned entry cannot be evicted or erased (a restore copy is reading it). Pins nest.
    bool pin(CacheRecordKey const& key);
    //! Returns false if the entry does not exist or is not pinned.
    bool unpin(CacheRecordKey const& key);
    //! Remove an entry and free the slots no other entry uses. Returns false if missing or pinned.
    bool erase(CacheRecordKey const& key);

    bool contains(CacheRecordKey const& key) const;
    int32_t slotCount() const noexcept;
    int32_t usedSlots() const noexcept;
    int32_t freeSlots() const noexcept;
    size_t entryCount() const noexcept;
    //! Entries evicted to make room since construction.
    uint64_t evictedEntries() const noexcept;

private:
    struct BlockInfo
    {
        int32_t slot{};
        int32_t refs{};
    };

    struct Entry
    {
        std::vector<BlockHash> hashes;
        int32_t pins{};
        std::list<CacheRecordKey>::iterator lruPos;
    };

    void removeEntry(std::unordered_map<CacheRecordKey, Entry>::iterator it);

    int32_t mSlotCount{};
    std::set<int32_t> mFreeSlots;
    std::unordered_map<BlockHash, BlockInfo> mBlocks;
    std::unordered_map<CacheRecordKey, Entry> mEntries;
    //! Front is least recently used.
    std::list<CacheRecordKey> mLru;
    uint64_t mVersion{};
    uint64_t mEvicted{};
};

} // namespace rt
} // namespace trt_edgellm
