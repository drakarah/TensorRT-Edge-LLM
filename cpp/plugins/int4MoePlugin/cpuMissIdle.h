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

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace trt_edgellm
{
namespace plugins
{
namespace cpumiss
{

//! Wait states of an idle miss worker: spin on the mailbox, poll it every 50 us, or sleep between polls.
enum class IdleStage
{
    kSpin,
    kPoll,
    kSleep
};

constexpr std::chrono::milliseconds kPollAfter{20};
constexpr std::chrono::milliseconds kDefaultSleepAfter{1000};
constexpr std::chrono::milliseconds kSleepPeriod{1};
constexpr int64_t kMaxSleepAfterMs = 24LL * 3600 * 1000;

//! EDGELLM_MOE_CPU_IDLE_MS: idle time after which the workers sleep (first doorbell afterwards waits up to kSleepPeriod).
//! Unset, empty or unparsable: the default; 0: never sleep; negative or above a day: the default.
inline std::chrono::milliseconds parseSleepAfter(char const* value)
{
    if (value == nullptr || value[0] == '\0')
    {
        return kDefaultSleepAfter;
    }
    char* end = nullptr;
    long long const parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > kMaxSleepAfterMs)
    {
        return kDefaultSleepAfter;
    }
    return std::chrono::milliseconds{parsed};
}

//! A zero sleepAfter keeps the workers in the poll stage for good.
inline IdleStage idleStage(std::chrono::nanoseconds idle, std::chrono::milliseconds sleepAfter)
{
    if (sleepAfter.count() > 0 && idle > sleepAfter)
    {
        return IdleStage::kSleep;
    }
    return idle > kPollAfter ? IdleStage::kPoll : IdleStage::kSpin;
}

} // namespace cpumiss
} // namespace plugins
} // namespace trt_edgellm
