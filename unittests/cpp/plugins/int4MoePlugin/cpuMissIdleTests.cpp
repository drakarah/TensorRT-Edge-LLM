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

#include "plugins/int4MoePlugin/cpuMissIdle.h"
#include <chrono>
#include <gtest/gtest.h>

using namespace trt_edgellm::plugins::cpumiss;
using std::chrono::milliseconds;

TEST(CpuMissIdleTest, SleepAfterDefaultsWhenUnsetEmptyOrUnparsable)
{
    EXPECT_EQ(parseSleepAfter(nullptr), kDefaultSleepAfter);
    EXPECT_EQ(parseSleepAfter(""), kDefaultSleepAfter);
    EXPECT_EQ(parseSleepAfter("abc"), kDefaultSleepAfter);
    EXPECT_EQ(parseSleepAfter("12x"), kDefaultSleepAfter);
    EXPECT_EQ(parseSleepAfter("1.5"), kDefaultSleepAfter);
}

TEST(CpuMissIdleTest, SleepAfterAcceptsZeroAndPositiveValues)
{
    EXPECT_EQ(parseSleepAfter("0"), milliseconds{0});
    EXPECT_EQ(parseSleepAfter("250"), milliseconds{250});
    EXPECT_EQ(parseSleepAfter("86400000"), milliseconds{86400000});
}

TEST(CpuMissIdleTest, SleepAfterOutsideItsRangeFallsBack)
{
    EXPECT_EQ(parseSleepAfter("-1"), kDefaultSleepAfter);
    EXPECT_EQ(parseSleepAfter("86400001"), kDefaultSleepAfter);
    EXPECT_EQ(parseSleepAfter("99999999999999999999"), kDefaultSleepAfter);
}

TEST(CpuMissIdleTest, StageFollowsTheIdleTime)
{
    milliseconds const sleepAfter{1000};
    EXPECT_EQ(idleStage(milliseconds{0}, sleepAfter), IdleStage::kSpin);
    EXPECT_EQ(idleStage(kPollAfter, sleepAfter), IdleStage::kSpin);
    EXPECT_EQ(idleStage(kPollAfter + milliseconds{1}, sleepAfter), IdleStage::kPoll);
    EXPECT_EQ(idleStage(sleepAfter, sleepAfter), IdleStage::kPoll);
    EXPECT_EQ(idleStage(sleepAfter + milliseconds{1}, sleepAfter), IdleStage::kSleep);
}

TEST(CpuMissIdleTest, ZeroSleepAfterNeverSleeps)
{
    EXPECT_EQ(idleStage(std::chrono::hours{48}, milliseconds{0}), IdleStage::kPoll);
    EXPECT_EQ(idleStage(milliseconds{5}, milliseconds{0}), IdleStage::kSpin);
}

TEST(CpuMissIdleTest, SleepAfterBelowThePollThresholdSleepsFirst)
{
    milliseconds const sleepAfter{5};
    EXPECT_EQ(idleStage(milliseconds{4}, sleepAfter), IdleStage::kSpin);
    EXPECT_EQ(idleStage(milliseconds{6}, sleepAfter), IdleStage::kSleep);
}
