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


#include "common/hostMemoryAccounting.h"

#include <gtest/gtest.h>

#include <utility>

using namespace trt_edgellm;

namespace
{

// The counters are process-wide, so each test measures a difference from its own start.
class HostMemoryAccountingTests : public ::testing::Test
{
protected:
    int64_t baseline(HostMemoryCategory category) const
    {
        return HostMemoryAccounting::bytes(category);
    }
};

} // namespace

TEST_F(HostMemoryAccountingTests, AddAndReleaseMoveTheCounter)
{
    int64_t const before = baseline(HostMemoryCategory::kExpertWeights);
    HostMemoryAccounting::add(HostMemoryCategory::kExpertWeights, 100);
    EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kExpertWeights), before + 100);
    HostMemoryAccounting::add(HostMemoryCategory::kExpertWeights, -100);
    EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kExpertWeights), before);
}

TEST_F(HostMemoryAccountingTests, CategoriesAreIndependent)
{
    int64_t const snapshots = baseline(HostMemoryCategory::kSnapshots);
    int64_t const tier = baseline(HostMemoryCategory::kKvTier);
    HostMemoryAccounting::add(HostMemoryCategory::kSnapshots, 7);
    EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kSnapshots), snapshots + 7);
    EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kKvTier), tier);
    HostMemoryAccounting::add(HostMemoryCategory::kSnapshots, -7);
}

TEST_F(HostMemoryAccountingTests, ReservationIsHeldForItsLifetime)
{
    int64_t const before = baseline(HostMemoryCategory::kKvTier);
    {
        HostMemoryReservation reservation(HostMemoryCategory::kKvTier, 4096);
        EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kKvTier), before + 4096);
    }
    EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kKvTier), before);
}

TEST_F(HostMemoryAccountingTests, MovedReservationReleasesOnce)
{
    int64_t const before = baseline(HostMemoryCategory::kEmbeddingTable);
    {
        HostMemoryReservation first(HostMemoryCategory::kEmbeddingTable, 10);
        HostMemoryReservation second(std::move(first));
        EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kEmbeddingTable), before + 10);
        HostMemoryReservation third(HostMemoryCategory::kEmbeddingTable, 5);
        EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kEmbeddingTable), before + 15);
        third = std::move(second);
        EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kEmbeddingTable), before + 10)
            << "assigning over a reservation releases the old bytes";
    }
    EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kEmbeddingTable), before);
}

TEST_F(HostMemoryAccountingTests, DefaultReservationAndSelfMoveAreHarmless)
{
    HostMemoryReservation empty;
    HostMemoryReservation held(HostMemoryCategory::kSnapshots, 3);
    int64_t const during = HostMemoryAccounting::bytes(HostMemoryCategory::kSnapshots);
    HostMemoryReservation& alias = held;
    held = std::move(alias);
    EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kSnapshots), during);
}

TEST_F(HostMemoryAccountingTests, UnknownCategoryIsIgnored)
{
    HostMemoryAccounting::add(HostMemoryCategory::kCount, 99);
    EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kCount), 0);
    HostMemoryReservation ignored(HostMemoryCategory::kCount, 99);
    EXPECT_EQ(HostMemoryAccounting::bytes(HostMemoryCategory::kCount), 0);
}
