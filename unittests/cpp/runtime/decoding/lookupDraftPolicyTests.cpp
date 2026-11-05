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

#include "runtime/decoding/lookupDraftPolicy.h"

#include <gtest/gtest.h>

using trt_edgellm::rt::LookupDraftPolicy;

namespace
{

//! Feed the policy rounds of one window size with a fixed cost.
void feedDrafter(LookupDraftPolicy& policy, int32_t window, int32_t accepted, double ms, int32_t rounds)
{
    for (int32_t i = 0; i < rounds; ++i)
    {
        policy.observe(false, window, accepted, 0, ms);
    }
}

void feedLookup(LookupDraftPolicy& policy, int32_t window, int32_t accepted, int32_t match, double ms, int32_t rounds)
{
    for (int32_t i = 0; i < rounds; ++i)
    {
        policy.observe(true, window, accepted, match, ms);
    }
}

} // namespace

TEST(LookupDraftPolicy, NoLookupProposalKeepsTheDrafterWindow)
{
    LookupDraftPolicy policy(8);
    auto const pick = policy.choose(3, 0, 0);
    EXPECT_FALSE(pick.lookup);
    EXPECT_EQ(pick.window, 3);
}

TEST(LookupDraftPolicy, DrafterWindowIsClampedToTheEngineMaximum)
{
    LookupDraftPolicy policy(4);
    EXPECT_EQ(policy.choose(9, 0, 0).window, 4);
    EXPECT_EQ(policy.choose(0, 0, 0).window, 1);
}

TEST(LookupDraftPolicy, LongReliableMatchOnAFlatCostTakesALongWindow)
{
    // A dense model: a verify of 3 and of 8 rows cost almost the same. A long match that is accepted almost always
    // must get a wide window.
    LookupDraftPolicy policy(8);
    feedDrafter(policy, 3, 1, 30.0, 50);        // drafter window: 2 tokens per 30 ms
    feedLookup(policy, 8, 7, 30, 32.0, 40);     // full lookup windows were accepted in full
    feedLookup(policy, 5, 4, 30, 31.0, 20);
    auto const pick = policy.choose(3, 7, 30);
    EXPECT_TRUE(pick.lookup);
    EXPECT_GE(pick.window, 6);
}

TEST(LookupDraftPolicy, SteepCostAndWeakAcceptanceKeepTheDrafter)
{
    // A mixture of experts where each extra row costs a lot, and lookups that are rarely right.
    LookupDraftPolicy policy(8);
    feedDrafter(policy, 3, 2, 30.0, 50);
    feedLookup(policy, 4, 0, 4, 55.0, 30);
    feedLookup(policy, 5, 0, 4, 70.0, 30);
    feedLookup(policy, 8, 0, 4, 110.0, 30);
    auto const pick = policy.choose(3, 7, 4);
    EXPECT_FALSE(pick.lookup);
    EXPECT_EQ(pick.window, 3);
}

TEST(LookupDraftPolicy, LearnsToStopTrustingAMatchLengthThatKeepsFailing)
{
    LookupDraftPolicy policy(8);
    feedDrafter(policy, 3, 2, 30.0, 50);
    feedLookup(policy, 4, 3, 5, 31.0, 5); // the prior is still trusted at first
    EXPECT_TRUE(policy.choose(3, 6, 5).lookup);
    feedLookup(policy, 4, 0, 5, 31.0, 80); // then 80 rounds in which the first lookup token was wrong
    EXPECT_FALSE(policy.choose(3, 6, 5).lookup);
}

TEST(LookupDraftPolicy, AcceptanceIsTrackedPerMatchLength)
{
    LookupDraftPolicy policy(8);
    feedLookup(policy, 4, 0, 4, 31.0, 80);  // short matches fail
    feedLookup(policy, 4, 3, 40, 31.0, 80); // long matches are accepted in full
    EXPECT_LT(policy.lookupRate(4), 0.2);
    EXPECT_GT(policy.lookupRate(40), 0.9);
}

TEST(LookupDraftPolicy, AnUnmeasuredWindowSizeIsScaledFromTheMeasuredOnes)
{
    LookupDraftPolicy policy(8);
    feedDrafter(policy, 3, 1, 30.0, 10);
    // Never measured: scaled by the prior shape from the window of 3 (1.5): 30 * 2.0 / 1.5.
    EXPECT_NEAR(policy.costMs(5), 40.0, 1e-9);
    // Measured sizes are returned as measured.
    EXPECT_NEAR(policy.costMs(3), 30.0, 1e-9);
}

TEST(LookupDraftPolicy, ProbesAConfidentWindowOnlyAFewTimes)
{
    // The measured cost of the widest window is prohibitive but it has never been measured: a confident match tries
    // it, so a guessed cost cannot veto it forever, and only until it has been measured a few times.
    LookupDraftPolicy policy(8);
    feedDrafter(policy, 3, 2, 10.0, 50); // cheap, good drafter
    feedLookup(policy, 5, 4, 40, 30.0, 5); // a window of 5 measured at three times the cost: not worth it
    auto const first = policy.choose(3, 7, 40);
    EXPECT_TRUE(first.lookup);
    EXPECT_EQ(first.window, 8);
    feedLookup(policy, 8, 7, 40, 200.0, 3); // measured: much too slow
    EXPECT_FALSE(policy.choose(3, 7, 40).lookup || policy.choose(3, 7, 40).window == 8);
}

TEST(LookupDraftPolicy, AStaleWindowCostIsMeasuredAgain)
{
    LookupDraftPolicy policy(8);
    feedDrafter(policy, 3, 2, 30.0, 10);
    feedLookup(policy, 8, 7, 40, 500.0, 5); // one slow stretch priced the widest window out
    feedDrafter(policy, 3, 2, 30.0, 400);   // long afterwards
    feedLookup(policy, 5, 4, 40, 31.0, 20);
    auto const pick = policy.choose(3, 7, 40);
    EXPECT_TRUE(pick.lookup);
    EXPECT_EQ(pick.window, 8);
}

TEST(LookupDraftPolicy, ProposalLongerThanTheEngineWindowIsCapped)
{
    LookupDraftPolicy policy(4);
    feedDrafter(policy, 3, 0, 30.0, 50);
    feedLookup(policy, 4, 3, 40, 31.0, 50);
    auto const pick = policy.choose(3, 50, 40);
    EXPECT_LE(pick.window, 4);
}
