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

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief Token budget for the reasoning block of one request.
 *
 * Follows the reasoning-budget sampler of llama.cpp (common/reasoning-budget.cpp, MIT license, the ggml authors):
 * the same five states and transitions, so a budget means the same thing on both runtimes. It is a host-side
 * state machine; the decoder reads forcedToken() before sampling a step and feeds every sampled token to accept().
 *
 *   kIdle        watching for a start sequence
 *   kCounting    counting the tokens of the block, watching for a natural end sequence
 *   kWaitingUtf8 budget used up, letting the current multi-byte character finish
 *   kForcing     emitting the forced tokens (message and end sequence) one per step
 *   kDone        passthrough until a new start sequence re-arms the budget
 */
class ReasoningBudget
{
public:
    enum class State
    {
        kIdle,
        kCounting,
        kWaitingUtf8,
        kForcing,
        kDone,
    };

    ReasoningBudget() = default;

    /*!
     * @param startSequence Tokens that open the block
     * @param endSequences Tokens that close it naturally
     * @param forcedTokens Tokens forced when the budget runs out (an optional message, then the end sequence)
     * @param budget Tokens allowed inside the block; 0 ends the block as soon as it opens
     * @param endsUtf8 Per token id: its bytes leave no unfinished UTF-8 character; empty means every token does
     */
    ReasoningBudget(std::vector<int32_t> startSequence, std::vector<std::vector<int32_t>> endSequences,
        std::vector<int32_t> forcedTokens, int32_t budget, std::vector<uint8_t> endsUtf8 = {})
        : mStart(std::move(startSequence))
        , mEnds(std::move(endSequences))
        , mForced(std::move(forcedTokens))
        , mEndsUtf8(std::move(endsUtf8))
        , mBudget(budget)
        , mRemaining(budget)
        , mEnabled(!mStart.empty() && !mEnds.empty() && !mForced.empty() && budget >= 0)
    {
    }

    bool enabled() const noexcept
    {
        return mEnabled;
    }

    State state() const noexcept
    {
        return mState;
    }

    //! The token the next step has to emit, if the budget is forcing one.
    std::optional<int32_t> forcedToken() const noexcept
    {
        if (mEnabled && mState == State::kForcing && mForcePos < mForced.size())
        {
            return mForced[mForcePos];
        }
        return std::nullopt;
    }

    /*!
     * @brief Brings the machine up to date with the full token list of the request (prompt, then generated tokens).
     *
     * The first call replays the list from its last start sequence, which is how a block the chat template already
     * opened in the prompt is found; later calls feed only the tokens appended since. Tokens generated before the
     * first call (the one the prefill sampled) were out of the budget's reach, so they do not consume forced tokens.
     *
     * @param generated Number of trailing entries of `tokens` that were generated, not part of the prompt
     */
    void sync(std::vector<int32_t> const& tokens, size_t generated = 0)
    {
        if (!mEnabled)
        {
            return;
        }
        size_t from = mSeen;
        if (mSeen == 0 && tokens.size() >= mStart.size())
        {
            from = tokens.size();
            for (size_t i = tokens.size() - mStart.size() + 1; i-- > 0;)
            {
                if (std::equal(mStart.begin(), mStart.end(), tokens.begin() + static_cast<std::ptrdiff_t>(i)))
                {
                    from = i;
                    break;
                }
            }
        }
        size_t const promptEnd = tokens.size() - std::min(generated, tokens.size());
        bool const priming = mSeen == 0;
        for (size_t i = from; i < tokens.size(); ++i)
        {
            if (priming && i >= promptEnd && mState == State::kForcing)
            {
                continue;
            }
            accept(tokens[i]);
        }
        mSeen = tokens.size();
    }

    //! Feeds one token of the prompt tail or of the generation, in order.
    void accept(int32_t token)
    {
        if (!mEnabled)
        {
            return;
        }
        switch (mState)
        {
        case State::kIdle:
            if (advance(mStart, mStartMatch, token))
            {
                arm();
            }
            break;
        case State::kCounting:
        case State::kWaitingUtf8:
        {
            if (advanceAny(token))
            {
                mState = State::kDone;
                break;
            }
            bool const complete = endsUtf8(token);
            if (mState == State::kWaitingUtf8)
            {
                if (complete)
                {
                    startForcing();
                }
            }
            else if (--mRemaining <= 0)
            {
                if (complete)
                {
                    startForcing();
                }
                else
                {
                    mState = State::kWaitingUtf8;
                    mEndMatch.clear();
                }
            }
            break;
        }
        case State::kForcing:
            ++mForcePos;
            if (advanceAny(token) || mForcePos >= mForced.size())
            {
                mState = State::kDone;
            }
            break;
        case State::kDone:
            if (advance(mStart, mStartMatch, token))
            {
                mEndMatch.clear();
                arm();
            }
            break;
        }
    }

private:
    void arm()
    {
        mState = State::kCounting;
        mRemaining = mBudget;
        if (mRemaining <= 0)
        {
            startForcing();
        }
    }

    void startForcing()
    {
        mState = State::kForcing;
        mForcePos = 0;
        mEndMatch.clear();
    }

    bool endsUtf8(int32_t token) const
    {
        return mEndsUtf8.empty() || token < 0 || static_cast<size_t>(token) >= mEndsUtf8.size()
            || mEndsUtf8[static_cast<size_t>(token)] != 0;
    }

    //! True when `sequence` ends at the newest token; the match window restarts after a hit.
    static bool advance(std::vector<int32_t> const& sequence, std::vector<int32_t>& window, int32_t token)
    {
        window.push_back(token);
        if (window.size() > sequence.size())
        {
            window.erase(window.begin());
        }
        if (window == sequence)
        {
            window.clear();
            return true;
        }
        return false;
    }

    bool advanceAny(int32_t token)
    {
        mEndMatch.push_back(token);
        size_t longest = 0;
        for (auto const& end : mEnds)
        {
            longest = std::max(longest, end.size());
        }
        if (mEndMatch.size() > longest)
        {
            mEndMatch.erase(mEndMatch.begin(), mEndMatch.end() - static_cast<std::ptrdiff_t>(longest));
        }
        for (auto const& end : mEnds)
        {
            if (mEndMatch.size() >= end.size()
                && std::equal(end.begin(), end.end(), mEndMatch.end() - static_cast<std::ptrdiff_t>(end.size())))
            {
                mEndMatch.clear();
                return true;
            }
        }
        return false;
    }

    std::vector<int32_t> mStart;
    std::vector<std::vector<int32_t>> mEnds;
    std::vector<int32_t> mForced;
    std::vector<uint8_t> mEndsUtf8;
    std::vector<int32_t> mStartMatch;
    std::vector<int32_t> mEndMatch;
    int32_t mBudget{-1};
    int32_t mRemaining{-1};
    size_t mForcePos{0};
    size_t mSeen{0};
    State mState{State::kIdle};
    bool mEnabled{false};
};

} // namespace rt
} // namespace trt_edgellm
