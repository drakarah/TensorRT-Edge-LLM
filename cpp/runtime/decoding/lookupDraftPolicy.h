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
#include <array>
#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

//! Per-round choice between the drafter's own window and a longer window filled from an n-gram (suffix) lookup.
//!
//! A lookup window is only worth its verify cost when enough of it is accepted: taken whenever the lookup proposes
//! more than the drafter, it loses on ordinary text, because a long window costs more to verify than the drafter's
//! usual 3-4 rows (a mixture-of-experts model reads more experts per extra row). The policy therefore compares the
//! expected committed tokens per millisecond of both sides, learned online:
//!
//!   drafter  E = mean tokens a window of that size has committed (moving average), at the measured cost of that size
//!   lookup   E(k) = 1 + q + q^2 + ... + q^k for a proposal of k tokens, at the measured cost of k + 1 rows, where q is
//!            the acceptance rate of earlier lookup drafts whose match was about as long (four match-length buckets,
//!            decayed counts, a prior worth a few observations)
//!
//! and takes the lookup window only when its best tokens per millisecond beats the drafter's by `margin`.
//! Costs are measured round times per window size (moving average); sizes not seen yet are scaled from the seen ones
//! by a prior shape. The policy only chooses which draft is verified; verification keeps the output unchanged.
//! The idea of pricing both sources by measured tokens per millisecond follows PolyStrata's draft policy (MIT).
class LookupDraftPolicy
{
public:
    static constexpr int32_t kMaxWindow = 8;
    static constexpr int32_t kBuckets = 4;

    struct Pick
    {
        bool lookup{false}; //!< fill the window from the lookup
        int32_t window{1};  //!< window size in rows (the committed token plus its drafts)
    };

    //! \param maxWindow largest window the engine can verify (<= kMaxWindow).
    //! \param margin required relative gain of the lookup window over the drafter's.
    explicit LookupDraftPolicy(int32_t maxWindow, double margin = 0.03)
        : mMaxWindow(std::clamp(maxWindow, 1, kMaxWindow))
        , mMargin(margin)
    {
    }

    //! Bucket of a match length (tokens of context that matched): <6, <12, <24, longer.
    static int32_t bucket(int32_t matchLength)
    {
        return matchLength < 6 ? 0 : matchLength < 12 ? 1 : matchLength < 24 ? 2 : 3;
    }

    //! Acceptance rate of lookup drafts for a match length.
    double lookupRate(int32_t matchLength) const
    {
        int32_t const b = bucket(matchLength);
        return (mOk[b] + kPriorWeight * kPriorRate[b]) / (mOk[b] + mBad[b] + kPriorWeight);
    }

    //! Cost in ms of a round with a window of t rows: measured, or scaled from the measured sizes.
    double costMs(int32_t t) const
    {
        t = std::clamp(t, 1, kMaxWindow);
        if (mCostCount[t] > 0)
        {
            return mCost[t];
        }
        double num = 0.0;
        double den = 0.0;
        for (int32_t u = 1; u <= kMaxWindow; ++u)
        {
            if (mCostCount[u] > 0)
            {
                double const w = std::min(mCostCount[u], 20.0);
                num += w * mCost[u] * kShape[t] / kShape[u];
                den += w;
            }
        }
        return den > 0.0 ? num / den : kShape[t];
    }

    //! Mean tokens committed per round by the drafter's window of t rows (a typical acceptance before any data).
    double drafterTokens(int32_t t) const
    {
        t = std::clamp(t, 1, kMaxWindow);
        return mDrafterCount[t] > 0.0 ? mDrafterTokens[t] : 1.0 + 0.7 * (t - 1);
    }

    //! \param drafterWindow rows of the drafter's own window this round.
    //! \param lookupLength tokens the lookup proposes (0 = none).
    //! \param matchLength context tokens that matched for that proposal.
    Pick choose(int32_t drafterWindow, int32_t lookupLength, int32_t matchLength) const
    {
        Pick pick;
        pick.window = std::clamp(drafterWindow, 1, mMaxWindow);
        if (lookupLength <= 0)
        {
            return pick;
        }
        int32_t const maxK = std::min(lookupLength, mMaxWindow - 1);
        double const base = drafterTokens(pick.window) / costMs(pick.window);
        double const q = lookupRate(matchLength);
        double expected = 1.0;
        double power = 1.0;
        double best = 0.0;
        int32_t bestWindow = 0;
        for (int32_t k = 1; k <= maxK; ++k)
        {
            power *= q;
            expected += power;
            double const rate = expected / costMs(k + 1);
            if (rate > best)
            {
                best = rate;
                bestWindow = k + 1;
            }
        }
        int32_t const fullWindow = maxK + 1;
        if (bestWindow > 0 && best > base * (1.0 + mMargin))
        {
            pick.lookup = true;
            pick.window = bestWindow;
            // A wider size whose cost has gone stale is re-measured: one slow stretch must not price it out for good.
            if (bestWindow < fullWindow && q >= kConfident && isStale(fullWindow))
            {
                pick.window = fullWindow;
            }
            return pick;
        }
        // A size never measured has only a guessed cost, and a guessed cost could veto it forever: a confident
        // lookup tries it a few times (the output is unaffected, only that round's speed is at stake).
        if (fullWindow > pick.window && (mCostCount[fullWindow] < kProbes || isStale(fullWindow)) && q >= kConfident)
        {
            pick.lookup = true;
            pick.window = fullWindow;
        }
        return pick;
    }

    //! After a round. \param accepted draft tokens accepted (not counting the committed token).
    void observe(bool lookup, int32_t window, int32_t accepted, int32_t matchLength, double roundMs)
    {
        window = std::clamp(window, 1, kMaxWindow);
        if (roundMs > 0.0)
        {
            mCost[window] = mCostCount[window] > 0.0 ? (1.0 - kCostAlpha) * mCost[window] + kCostAlpha * roundMs : roundMs;
            mCostCount[window] += 1.0;
            mLastRound[window] = mRounds;
        }
        mRounds += 1.0;
        if (lookup)
        {
            int32_t const b = bucket(matchLength);
            mOk[b] = kDecay * mOk[b] + accepted;
            mBad[b] = kDecay * mBad[b] + (accepted < window - 1 ? 1.0 : 0.0);
        }
        else
        {
            double const got = accepted + 1.0;
            mDrafterTokens[window] = mDrafterCount[window] > 0.0
                ? (1.0 - kTokenAlpha) * mDrafterTokens[window] + kTokenAlpha * got
                : got;
            mDrafterCount[window] += 1.0;
        }
    }

    int32_t maxWindow() const
    {
        return mMaxWindow;
    }

private:
    bool isStale(int32_t window) const
    {
        return mCostCount[window] > 0.0 && mRounds - mLastRound[window] > kStaleRounds;
    }

    // Cost of a round by window size relative to one row, used only for sizes not measured yet. Flat for a dense
    // model whose verify is bound by reading the weights once, steeper for a mixture of experts; real round times
    // replace it as they arrive.
    static constexpr std::array<double, kMaxWindow + 1> kShape{0.0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75};
    static constexpr double kCostAlpha = 0.1;
    static constexpr double kTokenAlpha = 0.05;
    static constexpr double kDecay = 0.97;
    static constexpr double kPriorWeight = 4.0;
    static constexpr std::array<double, kBuckets> kPriorRate{0.75, 0.88, 0.93, 0.96};
    static constexpr double kConfident = 0.85;
    static constexpr double kProbes = 3.0;
    static constexpr double kStaleRounds = 300.0;

    int32_t mMaxWindow;
    double mMargin;
    double mRounds{0.0};
    std::array<double, kMaxWindow + 1> mCost{};
    std::array<double, kMaxWindow + 1> mCostCount{};
    std::array<double, kMaxWindow + 1> mLastRound{};
    std::array<double, kMaxWindow + 1> mDrafterTokens{};
    std::array<double, kMaxWindow + 1> mDrafterCount{};
    std::array<double, kBuckets> mOk{};
    std::array<double, kBuckets> mBad{};
};

} // namespace rt
} // namespace trt_edgellm
