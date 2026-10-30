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

#include "runtime/decoding/reasoningBudget.h"

#include <gtest/gtest.h>

#include <vector>

using namespace trt_edgellm::rt;
using State = ReasoningBudget::State;

namespace
{
constexpr int32_t kThinkStart = 1;
constexpr int32_t kThinkEnd = 2;
constexpr int32_t kWord = 10;
constexpr int32_t kPartialChar = 11;
constexpr int32_t kMessageA = 7;
constexpr int32_t kMessageB = 8;

ReasoningBudget makeBudget(int32_t budget, std::vector<int32_t> forced = {kThinkEnd}, std::vector<uint8_t> utf8 = {})
{
    return ReasoningBudget({kThinkStart}, {{kThinkEnd}}, std::move(forced), budget, std::move(utf8));
}

void feed(ReasoningBudget& budget, std::vector<int32_t> const& tokens)
{
    for (int32_t const token : tokens)
    {
        budget.accept(token);
    }
}
} // namespace

TEST(ReasoningBudgetTest, DisabledWhenNothingIsConfigured)
{
    EXPECT_FALSE(ReasoningBudget{}.enabled());
    EXPECT_FALSE(makeBudget(-1).enabled());
    EXPECT_FALSE(makeBudget(4, {}).enabled());
    EXPECT_FALSE(ReasoningBudget({}, {{kThinkEnd}}, {kThinkEnd}, 4).enabled());
    EXPECT_FALSE(ReasoningBudget({kThinkStart}, {}, {kThinkEnd}, 4).enabled());

    ReasoningBudget off;
    feed(off, {kThinkStart, kWord, kWord});
    EXPECT_EQ(off.state(), State::kIdle);
    EXPECT_FALSE(off.forcedToken().has_value());
}

TEST(ReasoningBudgetTest, IdleUntilTheStartSequence)
{
    auto budget = makeBudget(3);
    feed(budget, {kWord, kWord});
    EXPECT_EQ(budget.state(), State::kIdle);
    budget.accept(kThinkStart);
    EXPECT_EQ(budget.state(), State::kCounting);
}

TEST(ReasoningBudgetTest, ForcesTheEndSequenceWhenTheBudgetRunsOut)
{
    auto budget = makeBudget(3, {kMessageA, kMessageB, kThinkEnd});
    budget.accept(kThinkStart);
    feed(budget, {kWord, kWord});
    EXPECT_EQ(budget.state(), State::kCounting);
    EXPECT_FALSE(budget.forcedToken().has_value());
    budget.accept(kWord);
    ASSERT_EQ(budget.state(), State::kForcing);
    EXPECT_EQ(budget.forcedToken(), kMessageA);
    budget.accept(kMessageA);
    EXPECT_EQ(budget.forcedToken(), kMessageB);
    budget.accept(kMessageB);
    EXPECT_EQ(budget.forcedToken(), kThinkEnd);
    budget.accept(kThinkEnd);
    EXPECT_EQ(budget.state(), State::kDone);
    EXPECT_FALSE(budget.forcedToken().has_value());
}

TEST(ReasoningBudgetTest, NaturalEndBeforeTheBudgetIsDone)
{
    auto budget = makeBudget(5);
    feed(budget, {kThinkStart, kWord, kThinkEnd});
    EXPECT_EQ(budget.state(), State::kDone);
    feed(budget, {kWord, kWord, kWord, kWord, kWord, kWord});
    EXPECT_EQ(budget.state(), State::kDone);
    EXPECT_FALSE(budget.forcedToken().has_value());
}

TEST(ReasoningBudgetTest, BudgetOfZeroForcesImmediately)
{
    auto budget = makeBudget(0);
    budget.accept(kThinkStart);
    ASSERT_EQ(budget.state(), State::kForcing);
    EXPECT_EQ(budget.forcedToken(), kThinkEnd);
}

TEST(ReasoningBudgetTest, BudgetOfOneAllowsASingleToken)
{
    auto budget = makeBudget(1);
    budget.accept(kThinkStart);
    EXPECT_EQ(budget.state(), State::kCounting);
    budget.accept(kWord);
    EXPECT_EQ(budget.state(), State::kForcing);
}

TEST(ReasoningBudgetTest, WaitsForAMultiByteCharacterToFinish)
{
    std::vector<uint8_t> utf8(16, 1);
    utf8[kPartialChar] = 0;
    auto budget = makeBudget(1, {kThinkEnd}, utf8);
    budget.accept(kThinkStart);
    budget.accept(kPartialChar);
    EXPECT_EQ(budget.state(), State::kWaitingUtf8);
    EXPECT_FALSE(budget.forcedToken().has_value());
    budget.accept(kPartialChar);
    EXPECT_EQ(budget.state(), State::kWaitingUtf8);
    budget.accept(kWord);
    EXPECT_EQ(budget.state(), State::kForcing);
}

TEST(ReasoningBudgetTest, NaturalEndWhileWaitingForUtf8)
{
    std::vector<uint8_t> utf8(16, 1);
    utf8[kPartialChar] = 0;
    auto budget = makeBudget(1, {kThinkEnd}, utf8);
    feed(budget, {kThinkStart, kPartialChar, kThinkEnd});
    EXPECT_EQ(budget.state(), State::kDone);
}

TEST(ReasoningBudgetTest, TokensOutsideTheUtf8TableCountAsComplete)
{
    std::vector<uint8_t> utf8(4, 1);
    auto budget = makeBudget(1, {kThinkEnd}, utf8);
    feed(budget, {kThinkStart, kWord});
    EXPECT_EQ(budget.state(), State::kForcing);
}

TEST(ReasoningBudgetTest, ASecondBlockGetsAFreshBudget)
{
    auto budget = makeBudget(2);
    feed(budget, {kThinkStart, kWord, kThinkEnd});
    ASSERT_EQ(budget.state(), State::kDone);
    budget.accept(kThinkStart);
    EXPECT_EQ(budget.state(), State::kCounting);
    feed(budget, {kWord, kWord});
    EXPECT_EQ(budget.state(), State::kForcing);
}

TEST(ReasoningBudgetTest, EndTokenInsideTheForcedSequenceEndsForcing)
{
    auto budget = makeBudget(1, {kMessageA, kThinkEnd, kMessageB});
    feed(budget, {kThinkStart, kWord});
    ASSERT_EQ(budget.state(), State::kForcing);
    feed(budget, {kMessageA, kThinkEnd});
    EXPECT_EQ(budget.state(), State::kDone);
}

TEST(ReasoningBudgetTest, MultiTokenStartAndEndSequences)
{
    ReasoningBudget budget({20, 21}, {{30, 31}, {32}}, {30, 31}, 3);
    feed(budget, {20, kWord, 21});
    EXPECT_EQ(budget.state(), State::kIdle);
    feed(budget, {20, 21});
    EXPECT_EQ(budget.state(), State::kCounting);
    feed(budget, {kWord, 30});
    EXPECT_EQ(budget.state(), State::kCounting);
    budget.accept(31);
    EXPECT_EQ(budget.state(), State::kDone);

    ReasoningBudget second({20, 21}, {{30, 31}, {32}}, {30, 31}, 5);
    feed(second, {20, 21, kWord, 32});
    EXPECT_EQ(second.state(), State::kDone);
}

TEST(ReasoningBudgetTest, SyncFindsABlockTheTemplateOpenedInThePrompt)
{
    auto budget = makeBudget(3);
    // prompt ends in an open block ("<think>\n"), then the first generated token: two tokens of the budget are used
    budget.sync({kWord, kWord, kThinkStart, kWord, kWord});
    EXPECT_EQ(budget.state(), State::kCounting);
    budget.sync({kWord, kWord, kThinkStart, kWord, kWord, kWord});
    EXPECT_EQ(budget.state(), State::kForcing);

    auto open = makeBudget(10);
    open.sync({kWord, kThinkStart, kWord});
    EXPECT_EQ(open.state(), State::kCounting);
    open.sync({kWord, kThinkStart, kWord, kWord, kThinkEnd});
    EXPECT_EQ(open.state(), State::kDone);
}

TEST(ReasoningBudgetTest, SyncIgnoresBlocksAlreadyClosedInTheHistory)
{
    auto budget = makeBudget(2);
    budget.sync({kThinkStart, kWord, kThinkEnd, kWord, kWord, kWord});
    EXPECT_EQ(budget.state(), State::kDone);
    EXPECT_FALSE(budget.forcedToken().has_value());
}

TEST(ReasoningBudgetTest, SyncWithoutAStartSequenceStaysIdle)
{
    auto budget = makeBudget(2);
    budget.sync({kWord, kWord, kWord});
    EXPECT_EQ(budget.state(), State::kIdle);
    budget.sync({kWord, kWord, kWord, kThinkStart});
    EXPECT_EQ(budget.state(), State::kCounting);
}

TEST(ReasoningBudgetTest, SyncFeedsOnlyTheNewTokens)
{
    auto budget = makeBudget(3);
    std::vector<int32_t> tokens{kWord, kThinkStart};
    budget.sync(tokens);
    for (int step = 0; step < 2; ++step)
    {
        tokens.push_back(kWord);
        budget.sync(tokens);
        budget.sync(tokens);
    }
    EXPECT_EQ(budget.state(), State::kCounting);
    tokens.push_back(kWord);
    budget.sync(tokens);
    EXPECT_EQ(budget.state(), State::kForcing);
}

TEST(ReasoningBudgetTest, SyncOfAnEmptyOrShortListIsHarmless)
{
    ReasoningBudget budget({20, 21}, {{30}}, {30}, 3);
    budget.sync({});
    budget.sync({20});
    EXPECT_EQ(budget.state(), State::kIdle);
    budget.sync({20, 21});
    EXPECT_EQ(budget.state(), State::kCounting);
}
