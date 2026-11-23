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
#include "common/stringUtils.h"

#include <gtest/gtest.h>

using trt_edgellm::format::containsAnySubstring;

TEST(ContainsAnySubstring, SingleSubstring)
{
    EXPECT_TRUE(containsAnySubstring("model.llm.layers.3.mlp.experts.fc1", ".mlp.experts."));
    EXPECT_FALSE(containsAnySubstring("model.llm.layers.3.attn.wq", ".mlp.experts."));
    EXPECT_TRUE(containsAnySubstring("__embedding__", "__embedding__"));
}

TEST(ContainsAnySubstring, AnyOfSeveral)
{
    std::string const patterns = ".mlp.experts.,__embedding__";
    EXPECT_TRUE(containsAnySubstring("model.llm.layers.0.mlp.experts.fc", patterns));
    EXPECT_TRUE(containsAnySubstring("__embedding__", patterns));
    EXPECT_FALSE(containsAnySubstring("model.llm.layers.0.attn.wq", patterns));
    EXPECT_FALSE(containsAnySubstring("embedding", patterns));
}

TEST(ContainsAnySubstring, EmptyEntriesNeverMatch)
{
    EXPECT_FALSE(containsAnySubstring("anything", ""));
    EXPECT_FALSE(containsAnySubstring("anything", ","));
    EXPECT_FALSE(containsAnySubstring("anything", ",,,"));
    EXPECT_TRUE(containsAnySubstring("anything", ",thing"));
    EXPECT_TRUE(containsAnySubstring("anything", "thing,"));
    EXPECT_TRUE(containsAnySubstring("anything", "xyz,,any"));
}

TEST(ContainsAnySubstring, EmptyTextAndLongPatterns)
{
    EXPECT_FALSE(containsAnySubstring("", "a"));
    EXPECT_FALSE(containsAnySubstring("", ""));
    EXPECT_FALSE(containsAnySubstring("ab", "abc"));
    EXPECT_TRUE(containsAnySubstring("abc", "abc"));
}

TEST(ContainsAnySubstring, CaseSensitiveAndNoTrimming)
{
    EXPECT_FALSE(containsAnySubstring("Embedding", "embedding"));
    EXPECT_TRUE(containsAnySubstring("ab", "a,b"));
    EXPECT_FALSE(containsAnySubstring("c", "a,b"));
    EXPECT_FALSE(containsAnySubstring("embedding", " embedding"));
    EXPECT_TRUE(containsAnySubstring(" embedding", " embedding"));
}
