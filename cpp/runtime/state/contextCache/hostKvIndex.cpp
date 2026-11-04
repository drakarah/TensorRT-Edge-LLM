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


#include "runtime/state/contextCache/hostKvIndex.h"

#include <algorithm>
#include <unordered_set>

namespace trt_edgellm
{
namespace rt
{

namespace
{

//! The hashes of @p hashes without repeats, first occurrence kept.
std::vector<BlockHash> distinctHashes(std::vector<BlockHash> const& hashes)
{
    std::vector<BlockHash> result;
    std::unordered_set<BlockHash> seen;
    for (auto const& hash : hashes)
    {
        if (seen.insert(hash).second)
        {
            result.push_back(hash);
        }
    }
    return result;
}

} // namespace

HostKvIndex::HostKvIndex(int32_t slotCount)
    : mSlotCount(std::max<int32_t>(slotCount, 0))
{
    for (int32_t slot = 0; slot < mSlotCount; ++slot)
    {
        mFreeSlots.insert(slot);
    }
}

HostKvDemotionPlan HostKvIndex::planDemotion(std::vector<BlockHash> const& blockHashes) const
{
    HostKvDemotionPlan plan;
    plan.version = mVersion;
    if (blockHashes.empty())
    {
        plan.status = HostKvPlanStatus::kEmpty;
        return plan;
    }
    CacheRecordKey const key{blockHashes.back(), static_cast<int32_t>(blockHashes.size())};
    if (mEntries.count(key) != 0)
    {
        plan.status = HostKvPlanStatus::kAlreadyPresent;
        return plan;
    }
    auto const needed = distinctHashes(blockHashes);
    if (static_cast<int64_t>(needed.size()) > mSlotCount)
    {
        plan.status = HostKvPlanStatus::kInfeasible;
        return plan;
    }

    // Simulate evicting victims: a block frees its slot when every entry that holds it is gone.
    std::unordered_map<BlockHash, int32_t> dropped;
    std::set<int32_t> freeSim = mFreeSlots;
    auto const isStored = [&](BlockHash const& hash)
    {
        auto const it = mBlocks.find(hash);
        if (it == mBlocks.end())
        {
            return false;
        }
        auto const drop = dropped.find(hash);
        return it->second.refs - (drop == dropped.end() ? 0 : drop->second) > 0;
    };
    auto const countMissing = [&]()
    {
        size_t missing = 0;
        for (auto const& hash : needed)
        {
            missing += isStored(hash) ? 0 : 1;
        }
        return missing;
    };

    size_t missing = countMissing();
    auto victim = mLru.begin();
    while (missing > freeSim.size())
    {
        while (victim != mLru.end() && mEntries.at(*victim).pins > 0)
        {
            ++victim;
        }
        if (victim == mLru.end())
        {
            plan.victims.clear();
            plan.status = HostKvPlanStatus::kInfeasible;
            return plan;
        }
        for (auto const& hash : distinctHashes(mEntries.at(*victim).hashes))
        {
            int32_t const drop = ++dropped[hash];
            auto const& info = mBlocks.at(hash);
            if (info.refs - drop == 0)
            {
                freeSim.insert(info.slot);
            }
        }
        plan.victims.push_back(*victim);
        ++victim;
        missing = countMissing();
    }

    for (auto const& hash : needed)
    {
        if (!isStored(hash))
        {
            plan.newBlocks.push_back({hash, *freeSim.begin()});
            freeSim.erase(freeSim.begin());
        }
    }
    plan.status = HostKvPlanStatus::kReady;
    return plan;
}

bool HostKvIndex::commitDemotion(std::vector<BlockHash> const& blockHashes, HostKvDemotionPlan const& plan)
{
    if (plan.status != HostKvPlanStatus::kReady || plan.version != mVersion || blockHashes.empty())
    {
        return false;
    }
    for (auto const& victim : plan.victims)
    {
        mEvicted += 1;
        removeEntry(mEntries.find(victim));
    }
    for (auto const& block : plan.newBlocks)
    {
        mFreeSlots.erase(block.slot);
        mBlocks[block.hash] = BlockInfo{block.slot, 0};
    }
    for (auto const& hash : distinctHashes(blockHashes))
    {
        mBlocks.at(hash).refs += 1;
    }
    CacheRecordKey const key{blockHashes.back(), static_cast<int32_t>(blockHashes.size())};
    mLru.push_back(key);
    Entry entry;
    entry.hashes = blockHashes;
    entry.lruPos = std::prev(mLru.end());
    mEntries.emplace(key, std::move(entry));
    ++mVersion;
    return true;
}

std::optional<HostKvHit> HostKvIndex::lookupPrefix(std::vector<BlockHash> const& requestHashes) const
{
    for (size_t count = requestHashes.size(); count > 0; --count)
    {
        CacheRecordKey const key{requestHashes[count - 1], static_cast<int32_t>(count)};
        auto const it = mEntries.find(key);
        if (it == mEntries.end() || !std::equal(it->second.hashes.begin(), it->second.hashes.end(), requestHashes.begin()))
        {
            continue;
        }
        HostKvHit hit;
        hit.key = key;
        for (auto const& hash : it->second.hashes)
        {
            hit.slots.push_back(mBlocks.at(hash).slot);
        }
        return hit;
    }
    return std::nullopt;
}

bool HostKvIndex::touch(CacheRecordKey const& key)
{
    auto const it = mEntries.find(key);
    if (it == mEntries.end())
    {
        return false;
    }
    mLru.splice(mLru.end(), mLru, it->second.lruPos);
    ++mVersion;
    return true;
}

bool HostKvIndex::pin(CacheRecordKey const& key)
{
    auto const it = mEntries.find(key);
    if (it == mEntries.end())
    {
        return false;
    }
    it->second.pins += 1;
    ++mVersion;
    return true;
}

bool HostKvIndex::unpin(CacheRecordKey const& key)
{
    auto const it = mEntries.find(key);
    if (it == mEntries.end() || it->second.pins == 0)
    {
        return false;
    }
    it->second.pins -= 1;
    ++mVersion;
    return true;
}

bool HostKvIndex::erase(CacheRecordKey const& key)
{
    auto const it = mEntries.find(key);
    if (it == mEntries.end() || it->second.pins > 0)
    {
        return false;
    }
    removeEntry(it);
    return true;
}

void HostKvIndex::removeEntry(std::unordered_map<CacheRecordKey, Entry>::iterator it)
{
    for (auto const& hash : distinctHashes(it->second.hashes))
    {
        auto const block = mBlocks.find(hash);
        if (--block->second.refs == 0)
        {
            mFreeSlots.insert(block->second.slot);
            mBlocks.erase(block);
        }
    }
    mLru.erase(it->second.lruPos);
    mEntries.erase(it);
    ++mVersion;
}

bool HostKvIndex::contains(CacheRecordKey const& key) const
{
    return mEntries.count(key) != 0;
}

int32_t HostKvIndex::slotCount() const noexcept
{
    return mSlotCount;
}

int32_t HostKvIndex::usedSlots() const noexcept
{
    return mSlotCount - static_cast<int32_t>(mFreeSlots.size());
}

int32_t HostKvIndex::freeSlots() const noexcept
{
    return static_cast<int32_t>(mFreeSlots.size());
}

size_t HostKvIndex::entryCount() const noexcept
{
    return mEntries.size();
}

uint64_t HostKvIndex::evictedEntries() const noexcept
{
    return mEvicted;
}

} // namespace rt
} // namespace trt_edgellm
