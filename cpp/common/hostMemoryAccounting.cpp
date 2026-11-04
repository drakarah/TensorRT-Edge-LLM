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

#include <array>
#include <cstddef>
#include <atomic>
#include <utility>

namespace trt_edgellm
{
namespace
{

std::array<std::atomic<int64_t>, static_cast<size_t>(HostMemoryCategory::kCount)>& counters() noexcept
{
    static std::array<std::atomic<int64_t>, static_cast<size_t>(HostMemoryCategory::kCount)> values{};
    return values;
}

bool valid(HostMemoryCategory category) noexcept
{
    return category >= HostMemoryCategory::kExpertWeights && category < HostMemoryCategory::kCount;
}

} // namespace

void HostMemoryAccounting::add(HostMemoryCategory category, int64_t bytes) noexcept
{
    if (valid(category))
    {
        counters()[static_cast<size_t>(category)].fetch_add(bytes, std::memory_order_relaxed);
    }
}

int64_t HostMemoryAccounting::bytes(HostMemoryCategory category) noexcept
{
    return valid(category) ? counters()[static_cast<size_t>(category)].load(std::memory_order_relaxed) : 0;
}

HostMemoryReservation::HostMemoryReservation(HostMemoryCategory category, int64_t bytes) noexcept
    : mCategory(category)
    , mBytes(bytes)
{
    HostMemoryAccounting::add(mCategory, mBytes);
}

HostMemoryReservation::~HostMemoryReservation() noexcept
{
    release();
}

HostMemoryReservation::HostMemoryReservation(HostMemoryReservation&& other) noexcept
    : mCategory(other.mCategory)
    , mBytes(std::exchange(other.mBytes, 0))
{
}

HostMemoryReservation& HostMemoryReservation::operator=(HostMemoryReservation&& other) noexcept
{
    if (this != &other)
    {
        release();
        mCategory = other.mCategory;
        mBytes = std::exchange(other.mBytes, 0);
    }
    return *this;
}

void HostMemoryReservation::release() noexcept
{
    if (mBytes != 0)
    {
        HostMemoryAccounting::add(mCategory, -mBytes);
        mBytes = 0;
    }
}

} // namespace trt_edgellm
