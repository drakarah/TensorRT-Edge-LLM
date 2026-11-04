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

#include <cstdint>

namespace trt_edgellm
{

//! Large pinned host allocations the runtime reports on its memory page.
enum class HostMemoryCategory : int32_t
{
    kExpertWeights,
    kEmbeddingTable,
    kSnapshots,
    kKvTier,
    kCount,
};

//! Process-wide byte counters for the categories above. Counts what the owners register, not what the OS reports.
class HostMemoryAccounting
{
public:
    //! @p bytes may be negative to release.
    static void add(HostMemoryCategory category, int64_t bytes) noexcept;
    static int64_t bytes(HostMemoryCategory category) noexcept;
};

//! Registers bytes for its lifetime; move-only.
class HostMemoryReservation
{
public:
    HostMemoryReservation() noexcept = default;
    HostMemoryReservation(HostMemoryCategory category, int64_t bytes) noexcept;
    ~HostMemoryReservation() noexcept;
    HostMemoryReservation(HostMemoryReservation&& other) noexcept;
    HostMemoryReservation& operator=(HostMemoryReservation&& other) noexcept;
    HostMemoryReservation(HostMemoryReservation const&) = delete;
    HostMemoryReservation& operator=(HostMemoryReservation const&) = delete;

private:
    void release() noexcept;

    HostMemoryCategory mCategory{HostMemoryCategory::kCount};
    int64_t mBytes{};
};

} // namespace trt_edgellm
