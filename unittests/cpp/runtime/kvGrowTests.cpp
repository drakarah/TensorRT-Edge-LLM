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
#include "runtime/kvGrow.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

//! A pool that can map tokens only from the released bytes the device can actually use, like a fragmented one.
struct FakePool
{
    int32_t mapped{1000};
    int64_t bytesPerToken{1024};
    int64_t releaseLimit{INT64_MAX};
    int64_t usable{0};
    std::vector<int64_t> asked;

    bool ensure(int32_t tokens)
    {
        if (tokens <= mapped)
        {
            return true;
        }
        int64_t const need = static_cast<int64_t>(tokens - mapped) * bytesPerToken;
        if (usable >= need)
        {
            usable -= need;
            mapped = tokens;
            return true;
        }
        return false;
    }

    int64_t bytes(int32_t tokens) const
    {
        return tokens <= mapped ? 0 : static_cast<int64_t>(tokens - mapped) * bytesPerToken;
    }

    //! Frees what was asked, but only a fraction of it is usable by the pool.
    int64_t release(int64_t ask, double usableFraction)
    {
        asked.push_back(ask);
        int64_t const give = std::min(ask, releaseLimit);
        releaseLimit -= give;
        usable += static_cast<int64_t>(give * usableFraction);
        return give;
    }
};

KvGrowResult run(FakePool& pool, double const fraction, int32_t const wanted, int32_t const minimum, int64_t const extra = 1 << 20, int32_t const maxExtra = 4)
{
    return growKvWithRelease([&](int32_t t) { return pool.ensure(t); }, [&](int32_t t) { return pool.bytes(t); },
        [&](int64_t b) { return pool.release(b, fraction); }, wanted, minimum, extra, maxExtra);
}

} // namespace

TEST(KvGrowTest, AlreadyMappedReleasesNothing)
{
    FakePool pool;
    auto const r = run(pool, 1.0, 900, 900);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kFull);
    EXPECT_EQ(r.releases, 0);
    EXPECT_TRUE(pool.asked.empty());
}

TEST(KvGrowTest, OneReleaseIsEnough)
{
    FakePool pool;
    auto const r = run(pool, 1.0, 1100, 1100);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kFull);
    EXPECT_EQ(r.releases, 1);
    ASSERT_EQ(pool.asked.size(), 1u);
    EXPECT_EQ(pool.asked[0], 100 * 1024);
    EXPECT_EQ(pool.mapped, 1100);
}

TEST(KvGrowTest, UnusableFragmentsNeedFurtherReleases)
{
    FakePool pool;
    auto const r = run(pool, 0.9, 1100, 1100, 20 * 1024);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kFull);
    EXPECT_EQ(r.releases, 2);
    EXPECT_EQ(pool.asked.back(), 20 * 1024);
    EXPECT_EQ(r.released, 100 * 1024 + 20 * 1024);
}

TEST(KvGrowTest, RetriesAreBounded)
{
    FakePool pool;
    auto const r = run(pool, 0.0, 1100, 1100, 1024, 3);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kFailed);
    EXPECT_EQ(r.releases, 4);
    EXPECT_EQ(pool.mapped, 1000);
}

TEST(KvGrowTest, NothingToReleaseStopsAtOnce)
{
    FakePool pool;
    pool.releaseLimit = 0;
    auto const r = run(pool, 1.0, 1100, 1100);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kFailed);
    EXPECT_EQ(r.releases, 1);
    EXPECT_EQ(r.released, 0);
}

TEST(KvGrowTest, ReleaseRunsDryDuringRetries)
{
    FakePool pool;
    pool.releaseLimit = 100 * 1024 + 4096;
    auto const r = run(pool, 0.5, 1100, 1100, 4096, 4);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kFailed);
    EXPECT_EQ(r.releases, 3);
    EXPECT_EQ(r.released, 100 * 1024 + 4096);
}

TEST(KvGrowTest, FallsBackToTheMinimum)
{
    FakePool pool;
    pool.releaseLimit = 60 * 1024;
    auto const r = run(pool, 1.0, 1100, 1040, 1024, 2);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kMinimum);
    EXPECT_EQ(pool.mapped, 1040);
}

TEST(KvGrowTest, MinimumEqualToWantedIsNotTriedTwice)
{
    FakePool pool;
    pool.releaseLimit = 0;
    auto const r = run(pool, 1.0, 1100, 1100);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kFailed);
}

TEST(KvGrowTest, NoExtraReleasesWhenDisabled)
{
    FakePool pool;
    auto const r = run(pool, 0.5, 1100, 1100, 0, 4);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kFailed);
    EXPECT_EQ(r.releases, 1);
}

TEST(KvGrowTest, NegativeReleaseResultIsNotCounted)
{
    int32_t const mapped = 10;
    auto const r = growKvWithRelease([&](int32_t t) { return t <= mapped; }, [](int32_t) { return int64_t{100}; },
        [](int64_t) { return int64_t{-5}; }, 20, 20, 10, 3);
    EXPECT_EQ(r.outcome, KvGrowOutcome::kFailed);
    EXPECT_EQ(r.released, 0);
    EXPECT_EQ(r.releases, 1);
}
