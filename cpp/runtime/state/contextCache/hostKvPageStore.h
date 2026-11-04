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

#include "common/hostMemoryAccounting.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/kvCacheManager.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace rt
{

//! One device KV page paired with one host slot.
struct HostKvPageCopy
{
    PageId page{};
    int32_t slot{};
};

//! Pinned host memory that holds whole KV pages (every attention layer, K and V) copied out of the device pool.
//!
//! Slot ownership and lookup live elsewhere; this class only moves bytes. K and V of a layer are separate slabs
//! indexed by slot, so runs of consecutive pages that map to consecutive slots are copied with one call per layer
//! and half. Copies are stream-ordered; the caller synchronises before the device pages are reused or the host
//! slots are read.
class HostKvPageStore
{
public:
    HostKvPageStore(KVCacheManager const& kvCache, int32_t slotCount);
    HostKvPageStore(HostKvPageStore const&) = delete;
    HostKvPageStore& operator=(HostKvPageStore const&) = delete;

    //! Pinned host bytes one slot occupies for @p config (K and V of every attention layer).
    static size_t bytesPerSlot(KVCacheManager::Config const& config);

    int32_t slotCount() const noexcept;

    //! Device pages to host slots.
    void capture(std::vector<HostKvPageCopy> const& copies, cudaStream_t stream);
    //! Host slots to device pages.
    void restore(std::vector<HostKvPageCopy> const& copies, cudaStream_t stream);

private:
    void copy(std::vector<HostKvPageCopy> const& copies, cudaMemcpyKind kind, cudaStream_t stream);

    KVCacheManager const& mKvCache;
    int32_t mSlotCount{};
    std::vector<size_t> mPageBytes;
    std::vector<Tensor> mKeys;
    std::vector<Tensor> mValues;
    HostMemoryReservation mHostReservation;
};

} // namespace rt
} // namespace trt_edgellm
