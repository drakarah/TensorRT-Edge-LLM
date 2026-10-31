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

#include "runtime/kvNeedWindow.h"
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

using namespace trt_edgellm::rt;
using Clock = KvNeedWindow::Clock;
using std::chrono::seconds;

namespace
{

class EnvGuard
{
public:
    EnvGuard(char const* name, char const* value)
        : mName(name)
    {
        char const* old = std::getenv(name);
        mHadValue = old != nullptr;
        if (mHadValue)
        {
            mOld = old;
        }
        if (value != nullptr)
        {
            setenv(name, value, 1);
        }
        else
        {
            unsetenv(name);
        }
    }

    ~EnvGuard()
    {
        if (mHadValue)
        {
            setenv(mName, mOld.c_str(), 1);
        }
        else
        {
            unsetenv(mName);
        }
    }

private:
    char const* mName;
    bool mHadValue{false};
    std::string mOld;
};

} // namespace

TEST(KvNeedWindowTest, EmptyWindowHasNoNeed)
{
    KvNeedWindow window(4, seconds(60));
    EXPECT_EQ(window.peak(), 0);
    EXPECT_EQ(window.size(), 0u);
}

TEST(KvNeedWindowTest, ASingleRequestIsItsOwnPeak)
{
    KvNeedWindow window(4, seconds(60));
    EXPECT_EQ(window.record(Clock::now(), 5000), 5000);
}

TEST(KvNeedWindowTest, AShortRequestBetweenLongTurnsKeepsThePeak)
{
    KvNeedWindow window(32, seconds(900));
    auto const t0 = Clock::now();
    EXPECT_EQ(window.record(t0, 120000), 120000);
    EXPECT_EQ(window.record(t0 + seconds(5), 2000), 120000) << "a title request must not lower the target";
    EXPECT_EQ(window.record(t0 + seconds(10), 121000), 121000);
    EXPECT_EQ(window.record(t0 + seconds(15), 1500), 121000);
}

TEST(KvNeedWindowTest, PeakDropsOnceTheLongRequestLeavesTheCountWindow)
{
    KvNeedWindow window(3, seconds(900));
    auto const t0 = Clock::now();
    window.record(t0, 100000);
    EXPECT_EQ(window.record(t0, 1000), 100000);
    EXPECT_EQ(window.record(t0, 1000), 100000);
    EXPECT_EQ(window.record(t0, 1000), 1000) << "the long request is the fourth most recent";
    EXPECT_EQ(window.size(), 3u);
}

TEST(KvNeedWindowTest, PeakDropsOnceTheLongRequestIsTooOld)
{
    KvNeedWindow window(32, seconds(60));
    auto const t0 = Clock::now();
    window.record(t0, 100000);
    EXPECT_EQ(window.record(t0 + seconds(60), 1000), 100000) << "exactly the maximum age is still inside";
    EXPECT_EQ(window.record(t0 + seconds(61), 1000), 1000);
}

TEST(KvNeedWindowTest, ASingleEntryWindowFollowsEveryRequest)
{
    KvNeedWindow zero(0, seconds(60));
    KvNeedWindow one(1, seconds(60));
    auto const t0 = Clock::now();
    EXPECT_EQ(zero.record(t0, 90000), 90000);
    EXPECT_EQ(zero.record(t0, 100), 100) << "a size of 0 behaves as 1";
    EXPECT_EQ(one.record(t0, 90000), 90000);
    EXPECT_EQ(one.record(t0, 100), 100);
}

TEST(KvNeedWindowTest, TheNewestEntryIsKeptEvenAfterALongIdle)
{
    KvNeedWindow window(32, seconds(0));
    auto const t0 = Clock::now();
    window.record(t0, 100000);
    EXPECT_EQ(window.record(t0 + seconds(3600), 700), 700);
    EXPECT_EQ(window.size(), 1u);
}

TEST(KvNeedWindowTest, NegativeNeedsCountAsZero)
{
    KvNeedWindow window(4, seconds(60));
    EXPECT_EQ(window.record(Clock::now(), -50), 0);
    EXPECT_EQ(window.record(Clock::now(), 300), 300);
}

TEST(KvEnvIntegerTest, ParsesAndFallsBack)
{
    {
        EnvGuard guard("KV_TEST_VALUE", nullptr);
        EXPECT_EQ(kvEnvInteger("KV_TEST_VALUE", 7, 0), 7);
    }
    {
        EnvGuard guard("KV_TEST_VALUE", "");
        EXPECT_EQ(kvEnvInteger("KV_TEST_VALUE", 7, 0), 7);
    }
    {
        EnvGuard guard("KV_TEST_VALUE", "250");
        EXPECT_EQ(kvEnvInteger("KV_TEST_VALUE", 7, 0), 250);
    }
    {
        EnvGuard guard("KV_TEST_VALUE", "12abc");
        EXPECT_EQ(kvEnvInteger("KV_TEST_VALUE", 7, 0), 7) << "trailing garbage";
    }
    {
        EnvGuard guard("KV_TEST_VALUE", "abc");
        EXPECT_EQ(kvEnvInteger("KV_TEST_VALUE", 7, 0), 7);
    }
    {
        EnvGuard guard("KV_TEST_VALUE", "-3");
        EXPECT_EQ(kvEnvInteger("KV_TEST_VALUE", 7, 0), 7) << "below the minimum";
        EXPECT_EQ(kvEnvInteger("KV_TEST_VALUE", 7, -10), -3);
    }
    {
        EnvGuard guard("KV_TEST_VALUE", "0");
        EXPECT_EQ(kvEnvInteger("KV_TEST_VALUE", 7, 0), 0);
        EXPECT_EQ(kvEnvInteger("KV_TEST_VALUE", 7, 1), 7);
    }
}

TEST(KvNeedWindowTest, EnvironmentConfiguresTheWindow)
{
    EnvGuard requests("EDGELLM_KV_SHRINK_WINDOW_REQUESTS", "2");
    EnvGuard age("EDGELLM_KV_SHRINK_WINDOW_SECONDS", "900");
    KvNeedWindow window = makeKvNeedWindowFromEnv();
    auto const t0 = Clock::now();
    window.record(t0, 9000);
    window.record(t0, 10);
    EXPECT_EQ(window.record(t0, 10), 10);
}

TEST(KvNeedWindowTest, DefaultsAreFourRequestsAndFiveMinutes)
{
    EnvGuard requests("EDGELLM_KV_SHRINK_WINDOW_REQUESTS", nullptr);
    EnvGuard age("EDGELLM_KV_SHRINK_WINDOW_SECONDS", nullptr);
    KvNeedWindow window = makeKvNeedWindowFromEnv();
    auto const t0 = Clock::now();
    window.record(t0, 9000);
    for (int i = 1; i <= 3; ++i)
    {
        EXPECT_EQ(window.record(t0 + seconds(i), 10), 9000);
    }
    EXPECT_EQ(window.record(t0 + seconds(4), 10), 10) << "the fifth most recent request is gone";

    KvNeedWindow aged = makeKvNeedWindowFromEnv();
    aged.record(t0, 9000);
    EXPECT_EQ(aged.record(t0 + seconds(300), 10), 9000);
    EXPECT_EQ(aged.record(t0 + seconds(301), 10), 10);
}
