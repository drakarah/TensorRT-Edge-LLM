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

//! Encodes real prompts with a real model tokenizer and writes the ids, for comparison against the
//! Hugging Face reference outside the test (the reference is not available to the C++ suite).
//! Skipped unless EDGELLM_TOKENIZER_DIR (directory with tokenizer.json), EDGELLM_TOKENIZER_PROMPTS
//! (JSON array of strings) and EDGELLM_TOKENIZER_OUT (output JSON path) are set.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>

#include "tokenizer/tokenizer.h"

using namespace trt_edgellm;

TEST(TokenizerRealModel, EncodePromptsFromEnv)
{
    char const* dir = std::getenv("EDGELLM_TOKENIZER_DIR");
    char const* promptsPath = std::getenv("EDGELLM_TOKENIZER_PROMPTS");
    char const* outPath = std::getenv("EDGELLM_TOKENIZER_OUT");
    if (dir == nullptr || promptsPath == nullptr || outPath == nullptr)
    {
        GTEST_SKIP() << "set EDGELLM_TOKENIZER_DIR, EDGELLM_TOKENIZER_PROMPTS and EDGELLM_TOKENIZER_OUT";
    }
    tokenizer::Tokenizer tok;
    ASSERT_TRUE(tok.loadFromHF(dir));
    std::ifstream in(promptsPath);
    nlohmann::json const prompts = nlohmann::json::parse(in);
    nlohmann::json out = nlohmann::json::array();
    double totalMs = 0.0;
    size_t totalChars = 0;
    for (auto const& prompt : prompts)
    {
        std::string const text = prompt.get<std::string>();
        auto const start = std::chrono::steady_clock::now();
        std::vector<tokenizer::Rank> const ids = tok.encode(text, false);
        totalMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        totalChars += text.size();
        out.push_back(ids);
    }
    std::ofstream(outPath) << out.dump();
    std::cout << "TOKENIZER prompts " << prompts.size() << " chars " << totalChars << " encode ms total " << totalMs
              << " mean " << totalMs / std::max<size_t>(1, prompts.size()) << std::endl;
}

//! SentencePiece-style BPE (explicit merges, byte fallback) on long pieces: equal-priority merges
//! resolve leftmost first, and a long run merges to the same result as the pairwise reference.
TEST(TokenizerBpe, MergePriorityAndLeftmostTieBreakOnLongPieces)
{
    auto const dir = std::filesystem::temp_directory_path() / "edgellm_bpe_tiebreak";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "byte_fallback": true,
            "vocab": {"a": 0, "b": 1, "aa": 2, "aaaa": 3, "ab": 4, "<0x63>": 5},
            "merges": ["a a", "aa aa", "a b"]},
  "added_tokens": []
})JSON";
    std::ofstream(dir / "tokenizer_config.json") << "{}";
    tokenizer::Tokenizer tok;
    ASSERT_TRUE(tok.loadFromHF(dir));
    // "a a" (priority 0) beats "a b" (2); five a's: leftmost pairs first -> aaaa + a.
    EXPECT_EQ(tok.encode("aaaaa", false), (std::vector<tokenizer::Rank>{3, 0}));
    EXPECT_EQ(tok.encode("aaab", false), (std::vector<tokenizer::Rank>{2, 4}));
    // Unknown byte falls back to <0xNN>.
    EXPECT_EQ(tok.encode("ac", false), (std::vector<tokenizer::Rank>{0, 5}));
    // Long run: 4001 a's -> 1000 x aaaa + a.
    std::string const run(4001, 'a');
    std::vector<tokenizer::Rank> expected(1000, 3);
    expected.push_back(0);
    EXPECT_EQ(tok.encode(run, false), expected);
}
