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

#include "runtime/decoding/suffixDraftCache.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using trt_edgellm::rt::SuffixDraftCache;

namespace
{

void appendAll(SuffixDraftCache& cache, std::vector<int32_t> const& tokens)
{
    for (int32_t token : tokens)
    {
        cache.append(token);
    }
}

std::vector<int32_t> propose(SuffixDraftCache const& cache, std::vector<int32_t> const& context, int32_t maxLen)
{
    return cache.propose(context.data(), static_cast<int32_t>(context.size()), maxLen);
}

} // namespace

TEST(SuffixDraftCache, ProposesContinuationOfEarlierRequest)
{
    SuffixDraftCache cache;
    appendAll(cache, {10, 11, 12, 13, 14, 15, 16});
    cache.beginSequence();
    appendAll(cache, {7, 11, 12});

    // Last 2-gram (11, 12) occurred in the first request; its continuation is proposed.
    EXPECT_EQ(propose(cache, {7, 11, 12}, 3), (std::vector<int32_t>{13, 14, 15}));
    EXPECT_EQ(propose(cache, {7, 11, 12}, 16), (std::vector<int32_t>{13, 14, 15, 16}));
}

TEST(SuffixDraftCache, PrefersLongestNgramAndMostRecentOccurrence)
{
    SuffixDraftCache cache;
    appendAll(cache, {1, 2, 3, 4, 50});
    cache.beginSequence();
    appendAll(cache, {9, 3, 4, 60});
    cache.beginSequence();
    appendAll(cache, {2, 3, 4});

    // (3, 4) last continued with 60, but the 3-gram (2, 3, 4) only occurred before 50: longest wins.
    EXPECT_EQ(propose(cache, {2, 3, 4}, 1), (std::vector<int32_t>{50}));
    // A context that only matches the 2-gram gets the most recent continuation.
    EXPECT_EQ(propose(cache, {8, 3, 4}, 1), (std::vector<int32_t>{60}));
}

TEST(SuffixDraftCache, DoesNotProposeFromItsOwnEndOrAcrossRequests)
{
    SuffixDraftCache cache;
    appendAll(cache, {5, 6});
    cache.beginSequence();
    appendAll(cache, {1, 2, 1, 2});

    // The context's own end is skipped; the earlier (1, 2) within this request continues with 1, 2.
    EXPECT_EQ(propose(cache, {1, 2, 1, 2}, 4), (std::vector<int32_t>{1, 2}));
    // The first request's continuation stops at the separator.
    EXPECT_TRUE(propose(cache, {4, 4, 5, 6}, 4).empty());
    // Unknown context: nothing.
    EXPECT_TRUE(propose(cache, {77, 78}, 4).empty());
}

TEST(SuffixDraftCache, StartsOverWhenFull)
{
    SuffixDraftCache cache(6);
    appendAll(cache, {1, 2, 3, 4});
    cache.beginSequence();
    EXPECT_EQ(propose(cache, {1, 2}, 2), (std::vector<int32_t>{3, 4}));

    // The 7th token exceeds the cap: everything before it is dropped.
    appendAll(cache, {8, 9});
    EXPECT_TRUE(propose(cache, {1, 2}, 2).empty());
    // Indexing continues from the kept token.
    appendAll(cache, {10, 7});
    EXPECT_EQ(propose(cache, {9, 10}, 2), (std::vector<int32_t>{7}));
}
