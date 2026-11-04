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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

BlockHash blockHash(uint64_t id)
{
    return BlockHash{id, id * 7 + 1};
}

//! A chain of @p count blocks: the first @p shared blocks come from family 0, the rest from @p family.
std::vector<BlockHash> chain(uint64_t family, int32_t count, int32_t shared = 0)
{
    std::vector<BlockHash> hashes;
    for (int32_t i = 0; i < count; ++i)
    {
        uint64_t const owner = i < shared ? 0 : family;
        hashes.push_back(blockHash(owner * 1000 + static_cast<uint64_t>(i) + 1));
    }
    return hashes;
}

CacheRecordKey keyOf(std::vector<BlockHash> const& hashes)
{
    return CacheRecordKey{hashes.back(), static_cast<int32_t>(hashes.size())};
}

//! Plan and commit; returns the plan so tests can inspect it.
HostKvDemotionPlan demote(HostKvIndex& index, std::vector<BlockHash> const& hashes)
{
    auto const plan = index.planDemotion(hashes);
    EXPECT_EQ(plan.status, HostKvPlanStatus::kReady);
    EXPECT_TRUE(index.commitDemotion(hashes, plan));
    return plan;
}

} // namespace

TEST(HostKvIndexTest, ZeroSlotsRejectsEverything)
{
    HostKvIndex index(0);
    EXPECT_EQ(index.planDemotion(chain(1, 1)).status, HostKvPlanStatus::kInfeasible);
    EXPECT_EQ(index.slotCount(), 0);
}

TEST(HostKvIndexTest, NegativeSlotCountIsZero)
{
    HostKvIndex index(-5);
    EXPECT_EQ(index.slotCount(), 0);
    EXPECT_EQ(index.freeSlots(), 0);
}

TEST(HostKvIndexTest, EmptyEntryIsRejected)
{
    HostKvIndex index(4);
    auto const plan = index.planDemotion({});
    EXPECT_EQ(plan.status, HostKvPlanStatus::kEmpty);
    EXPECT_FALSE(index.commitDemotion({}, plan));
}

TEST(HostKvIndexTest, EmptyIndexAssignsLowestSlotsFirst)
{
    HostKvIndex index(8);
    auto const hashes = chain(1, 3);
    auto const plan = demote(index, hashes);
    EXPECT_TRUE(plan.victims.empty());
    ASSERT_EQ(plan.newBlocks.size(), 3U);
    for (int32_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(plan.newBlocks[i].slot, i);
        EXPECT_EQ(plan.newBlocks[i].hash, hashes[i]);
    }
    EXPECT_EQ(index.usedSlots(), 3);
    EXPECT_EQ(index.freeSlots(), 5);
    EXPECT_EQ(index.entryCount(), 1U);
    EXPECT_TRUE(index.contains(keyOf(hashes)));
}

TEST(HostKvIndexTest, PlanDoesNotChangeTheIndex)
{
    HostKvIndex index(4);
    auto const plan = index.planDemotion(chain(1, 2));
    EXPECT_EQ(plan.status, HostKvPlanStatus::kReady);
    EXPECT_EQ(index.entryCount(), 0U);
    EXPECT_EQ(index.usedSlots(), 0);
}

TEST(HostKvIndexTest, AlreadyPresentKeyIsNotAddedTwice)
{
    HostKvIndex index(4);
    auto const hashes = chain(1, 2);
    demote(index, hashes);
    auto const plan = index.planDemotion(hashes);
    EXPECT_EQ(plan.status, HostKvPlanStatus::kAlreadyPresent);
    EXPECT_FALSE(index.commitDemotion(hashes, plan));
    EXPECT_EQ(index.usedSlots(), 2);
}

TEST(HostKvIndexTest, SharedPrefixBlocksUseOneSlot)
{
    HostKvIndex index(8);
    auto const a = chain(1, 4, 2);
    auto const b = chain(2, 5, 2);
    demote(index, a);
    auto const plan = demote(index, b);
    // b shares two blocks with a, so only its three own blocks need slots.
    EXPECT_EQ(plan.newBlocks.size(), 3U);
    EXPECT_EQ(index.usedSlots(), 7);
    // Dropping a keeps the shared blocks alive for b.
    EXPECT_TRUE(index.erase(keyOf(a)));
    EXPECT_EQ(index.usedSlots(), 5);
    // Dropping b frees the rest.
    EXPECT_TRUE(index.erase(keyOf(b)));
    EXPECT_EQ(index.usedSlots(), 0);
    EXPECT_EQ(index.entryCount(), 0U);
}

TEST(HostKvIndexTest, EntryThatIsAPrefixOfAnotherSharesAllItsBlocks)
{
    HostKvIndex index(8);
    auto const longer = chain(1, 5);
    auto const shorter = chain(1, 3);
    demote(index, longer);
    auto const plan = demote(index, shorter);
    EXPECT_TRUE(plan.newBlocks.empty());
    EXPECT_EQ(index.usedSlots(), 5);
    EXPECT_EQ(index.entryCount(), 2U);
}

TEST(HostKvIndexTest, DuplicateHashesInOneEntryCountOnce)
{
    HostKvIndex index(4);
    std::vector<BlockHash> const hashes{blockHash(1), blockHash(2), blockHash(1)};
    auto const plan = demote(index, hashes);
    EXPECT_EQ(plan.newBlocks.size(), 2U);
    EXPECT_EQ(index.usedSlots(), 2);
    EXPECT_TRUE(index.erase(keyOf(hashes)));
    EXPECT_EQ(index.usedSlots(), 0);
}

TEST(HostKvIndexTest, EvictsLeastRecentlyUsedWhenFull)
{
    HostKvIndex index(4);
    auto const a = chain(1, 2);
    auto const b = chain(2, 2);
    auto const c = chain(3, 2);
    demote(index, a);
    demote(index, b);
    auto const plan = demote(index, c);
    ASSERT_EQ(plan.victims.size(), 1U);
    EXPECT_EQ(plan.victims[0], keyOf(a));
    EXPECT_FALSE(index.contains(keyOf(a)));
    EXPECT_TRUE(index.contains(keyOf(b)));
    EXPECT_TRUE(index.contains(keyOf(c)));
    EXPECT_EQ(index.evictedEntries(), 1U);
    // c reuses the slots a freed, lowest first.
    EXPECT_EQ(plan.newBlocks[0].slot, 0);
    EXPECT_EQ(plan.newBlocks[1].slot, 1);
}

TEST(HostKvIndexTest, TouchChangesTheVictim)
{
    HostKvIndex index(4);
    auto const a = chain(1, 2);
    auto const b = chain(2, 2);
    demote(index, a);
    demote(index, b);
    EXPECT_TRUE(index.touch(keyOf(a)));
    auto const plan = demote(index, chain(3, 2));
    ASSERT_EQ(plan.victims.size(), 1U);
    EXPECT_EQ(plan.victims[0], keyOf(b));
}

TEST(HostKvIndexTest, VictimsAreOnlyAsManyAsNeeded)
{
    HostKvIndex index(6);
    demote(index, chain(1, 2));
    demote(index, chain(2, 2));
    demote(index, chain(3, 2));
    auto const plan = demote(index, chain(4, 3));
    EXPECT_EQ(plan.victims.size(), 2U);
    EXPECT_EQ(index.entryCount(), 2U);
}

TEST(HostKvIndexTest, EvictingAnEntryWithSharedBlocksMayFreeNothing)
{
    HostKvIndex index(5);
    auto const a = chain(1, 3, 2);
    auto const b = chain(2, 3, 2);
    demote(index, a); // slots 0..2
    demote(index, b); // shares two blocks, one new -> 4 used
    // c needs 2 new blocks with one free slot: evicting a frees only its own tail block.
    auto const plan = demote(index, chain(3, 2));
    EXPECT_EQ(plan.victims.size(), 1U);
    EXPECT_EQ(plan.victims[0], keyOf(a));
    EXPECT_TRUE(index.contains(keyOf(b)));
    EXPECT_EQ(index.usedSlots(), 5);
}

TEST(HostKvIndexTest, PinnedEntriesAreSkippedAsVictims)
{
    HostKvIndex index(4);
    auto const a = chain(1, 2);
    auto const b = chain(2, 2);
    demote(index, a);
    demote(index, b);
    EXPECT_TRUE(index.pin(keyOf(a)));
    auto const plan = demote(index, chain(3, 2));
    ASSERT_EQ(plan.victims.size(), 1U);
    EXPECT_EQ(plan.victims[0], keyOf(b));
    EXPECT_TRUE(index.contains(keyOf(a)));
}

TEST(HostKvIndexTest, AllPinnedIsInfeasibleAndEvictsNothing)
{
    HostKvIndex index(4);
    auto const a = chain(1, 2);
    auto const b = chain(2, 2);
    demote(index, a);
    demote(index, b);
    index.pin(keyOf(a));
    index.pin(keyOf(b));
    auto const plan = index.planDemotion(chain(3, 2));
    EXPECT_EQ(plan.status, HostKvPlanStatus::kInfeasible);
    EXPECT_TRUE(plan.victims.empty());
    EXPECT_FALSE(index.commitDemotion(chain(3, 2), plan));
    EXPECT_EQ(index.entryCount(), 2U);
}

TEST(HostKvIndexTest, EntryLargerThanTheStoreIsInfeasibleWithoutEvicting)
{
    HostKvIndex index(3);
    demote(index, chain(1, 2));
    auto const plan = index.planDemotion(chain(2, 4));
    EXPECT_EQ(plan.status, HostKvPlanStatus::kInfeasible);
    EXPECT_TRUE(plan.victims.empty());
    EXPECT_EQ(index.entryCount(), 1U);
}

TEST(HostKvIndexTest, VictimHoldingBlocksTheNewEntryNeedsStillFits)
{
    HostKvIndex index(3);
    auto const a = chain(1, 3);
    demote(index, a);
    // The new entry extends a's first two blocks; evicting a frees them, and the new entry then takes them again.
    auto extended = chain(1, 2);
    extended.push_back(blockHash(9001));
    extended.push_back(blockHash(9002));
    EXPECT_EQ(index.planDemotion(extended).status, HostKvPlanStatus::kInfeasible);
    auto fits = chain(1, 2);
    fits.push_back(blockHash(9001));
    auto const plan = demote(index, fits);
    EXPECT_EQ(plan.victims.size(), 1U);
    EXPECT_TRUE(index.contains(keyOf(fits)));
    EXPECT_EQ(index.usedSlots(), 3);
    EXPECT_EQ(plan.newBlocks.size(), 3U);
}

TEST(HostKvIndexTest, StalePlanIsRejected)
{
    HostKvIndex index(4);
    auto const first = chain(1, 2);
    auto const second = chain(2, 2);
    auto const plan = index.planDemotion(first);
    demote(index, second); // changes the index after the plan was made
    EXPECT_FALSE(index.commitDemotion(first, plan));
    EXPECT_FALSE(index.contains(keyOf(first)));
}

TEST(HostKvIndexTest, EveryMutationInvalidatesAnOutstandingPlan)
{
    HostKvIndex index(8);
    auto const a = chain(1, 2);
    auto const b = chain(2, 2);
    demote(index, a);
    demote(index, b);
    auto const fresh = chain(3, 1);

    auto plan = index.planDemotion(fresh);
    index.touch(keyOf(a));
    EXPECT_FALSE(index.commitDemotion(fresh, plan));

    plan = index.planDemotion(fresh);
    index.pin(keyOf(a));
    EXPECT_FALSE(index.commitDemotion(fresh, plan));

    plan = index.planDemotion(fresh);
    index.unpin(keyOf(a));
    EXPECT_FALSE(index.commitDemotion(fresh, plan));

    plan = index.planDemotion(fresh);
    index.erase(keyOf(a));
    EXPECT_FALSE(index.commitDemotion(fresh, plan));

    plan = index.planDemotion(fresh);
    EXPECT_TRUE(index.commitDemotion(fresh, plan));
}

TEST(HostKvIndexTest, CommitRejectsNonReadyPlans)
{
    HostKvIndex index(4);
    HostKvDemotionPlan plan;
    plan.status = HostKvPlanStatus::kInfeasible;
    EXPECT_FALSE(index.commitDemotion(chain(1, 1), plan));
}

TEST(HostKvIndexTest, LookupFindsTheLongestMatchingEntry)
{
    HostKvIndex index(16);
    auto const shortEntry = chain(1, 2);
    auto const longEntry = chain(1, 5);
    demote(index, shortEntry);
    demote(index, longEntry);
    auto const request = chain(1, 7);
    auto const hit = index.lookupPrefix(request);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->key, keyOf(longEntry));
    ASSERT_EQ(hit->slots.size(), 5U);
}

TEST(HostKvIndexTest, LookupSlotsFollowTheSharedBlocks)
{
    HostKvIndex index(8);
    auto const a = chain(1, 3, 2);
    auto const b = chain(2, 3, 2);
    demote(index, a);
    demote(index, b);
    auto const hitA = index.lookupPrefix(a);
    auto const hitB = index.lookupPrefix(b);
    ASSERT_TRUE(hitA.has_value());
    ASSERT_TRUE(hitB.has_value());
    EXPECT_EQ(hitA->slots[0], hitB->slots[0]);
    EXPECT_EQ(hitA->slots[1], hitB->slots[1]);
    EXPECT_NE(hitA->slots[2], hitB->slots[2]);
}

TEST(HostKvIndexTest, LookupShorterRequestDoesNotMatchLongerEntry)
{
    HostKvIndex index(8);
    demote(index, chain(1, 5));
    EXPECT_FALSE(index.lookupPrefix(chain(1, 3)).has_value());
}

TEST(HostKvIndexTest, LookupDivergentChainDoesNotMatch)
{
    HostKvIndex index(8);
    demote(index, chain(1, 4));
    // Same length and shared head, different tail: terminal hash differs.
    EXPECT_FALSE(index.lookupPrefix(chain(2, 4, 2)).has_value());
}

TEST(HostKvIndexTest, LookupRejectsAKeyCollisionWithDifferentBlocks)
{
    HostKvIndex index(8);
    auto const stored = chain(1, 3);
    demote(index, stored);
    // Same terminal hash and length as the stored entry, different first block.
    auto forged = stored;
    forged[0] = blockHash(777);
    EXPECT_FALSE(index.lookupPrefix(forged).has_value());
}

TEST(HostKvIndexTest, LookupOnEmptyInputsReturnsNothing)
{
    HostKvIndex index(8);
    EXPECT_FALSE(index.lookupPrefix({}).has_value());
    demote(index, chain(1, 2));
    EXPECT_FALSE(index.lookupPrefix({}).has_value());
}

TEST(HostKvIndexTest, LookupDoesNotChangeRecency)
{
    HostKvIndex index(4);
    auto const a = chain(1, 2);
    auto const b = chain(2, 2);
    demote(index, a);
    demote(index, b);
    index.lookupPrefix(a);
    auto const plan = demote(index, chain(3, 2));
    ASSERT_EQ(plan.victims.size(), 1U);
    EXPECT_EQ(plan.victims[0], keyOf(a));
}

TEST(HostKvIndexTest, PinsNestAndUnpinNeedsAPin)
{
    HostKvIndex index(4);
    auto const a = chain(1, 2);
    demote(index, a);
    EXPECT_FALSE(index.unpin(keyOf(a)));
    EXPECT_TRUE(index.pin(keyOf(a)));
    EXPECT_TRUE(index.pin(keyOf(a)));
    EXPECT_FALSE(index.erase(keyOf(a)));
    EXPECT_TRUE(index.unpin(keyOf(a)));
    EXPECT_FALSE(index.erase(keyOf(a)));
    EXPECT_TRUE(index.unpin(keyOf(a)));
    EXPECT_TRUE(index.erase(keyOf(a)));
}

TEST(HostKvIndexTest, MissingKeysAreReportedByEveryOperation)
{
    HostKvIndex index(4);
    CacheRecordKey const missing{blockHash(42), 3};
    EXPECT_FALSE(index.touch(missing));
    EXPECT_FALSE(index.pin(missing));
    EXPECT_FALSE(index.unpin(missing));
    EXPECT_FALSE(index.erase(missing));
    EXPECT_FALSE(index.contains(missing));
}

TEST(HostKvIndexTest, FreedSlotsAreReusedLowestFirst)
{
    HostKvIndex index(6);
    auto const a = chain(1, 2); // slots 0,1
    auto const b = chain(2, 2); // slots 2,3
    demote(index, a);
    demote(index, b);
    index.erase(keyOf(a));
    auto const plan = demote(index, chain(3, 3));
    ASSERT_EQ(plan.newBlocks.size(), 3U);
    EXPECT_EQ(plan.newBlocks[0].slot, 0);
    EXPECT_EQ(plan.newBlocks[1].slot, 1);
    EXPECT_EQ(plan.newBlocks[2].slot, 4);
}

TEST(HostKvIndexTest, RandomOperationsKeepTheInvariants)
{
    constexpr int32_t kSLOTS = 24;
    HostKvIndex index(kSLOTS);
    std::mt19937 rng(1234);
    std::vector<std::vector<BlockHash>> known;
    for (int step = 0; step < 4000; ++step)
    {
        int const op = static_cast<int>(rng() % 5);
        if (op <= 1 || known.empty())
        {
            int32_t const family = static_cast<int32_t>(rng() % 6) + 1;
            int32_t const count = static_cast<int32_t>(rng() % 6) + 1;
            int32_t const shared = static_cast<int32_t>(rng() % 3);
            auto hashes = chain(static_cast<uint64_t>(family), count, std::min(shared, count));
            auto const plan = index.planDemotion(hashes);
            if (plan.status == HostKvPlanStatus::kReady)
            {
                ASSERT_TRUE(index.commitDemotion(hashes, plan));
                known.push_back(hashes);
            }
        }
        else if (op == 2)
        {
            index.touch(keyOf(known[rng() % known.size()]));
        }
        else if (op == 3)
        {
            index.erase(keyOf(known[rng() % known.size()]));
        }
        else
        {
            index.lookupPrefix(known[rng() % known.size()]);
        }

        // Invariants: the slots of every live entry are distinct per block hash, within range, and counted once.
        std::map<BlockHash, int32_t, bool (*)(BlockHash const&, BlockHash const&)> slotOf(
            [](BlockHash const& l, BlockHash const& r) { return l.hi != r.hi ? l.hi < r.hi : l.lo < r.lo; });
        std::set<int32_t> slots;
        for (auto const& hashes : known)
        {
            auto const hit = index.lookupPrefix(hashes);
            if (!hit.has_value() || !(hit->key == keyOf(hashes)))
            {
                continue;
            }
            for (size_t i = 0; i < hashes.size(); ++i)
            {
                int32_t const slot = hit->slots[i];
                ASSERT_GE(slot, 0);
                ASSERT_LT(slot, kSLOTS);
                auto const [it, inserted] = slotOf.emplace(hashes[i], slot);
                ASSERT_EQ(it->second, slot);
                slots.insert(slot);
            }
        }
        ASSERT_LE(static_cast<int32_t>(slots.size()), index.usedSlots());
        ASSERT_EQ(index.usedSlots() + index.freeSlots(), kSLOTS);
        // Distinct hashes map to distinct slots.
        std::set<int32_t> distinctSlots;
        for (auto const& [hash, slot] : slotOf)
        {
            ASSERT_TRUE(distinctSlots.insert(slot).second);
        }
    }
}
