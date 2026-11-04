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

#include <cstddef>
#include <vector>

#include <cuda.h>

namespace trt_edgellm
{
namespace rt
{

//! Device buffer made of two halves in one reserved virtual range; the second half starts right after the first. Physical
//! memory is mapped to a prefix of each half on demand, so the base pointer and the halves' offsets never change when the
//! buffer grows.
class VmmRegion
{
public:
    //! @brief Reserve virtual space for two halves of @p halfBytes each. Nothing is mapped yet.
    //! @param halfBytes Must be a multiple of granularity().
    //! @throws std::runtime_error if the reservation fails.
    VmmRegion(size_t halfBytes, int device);

    //! @brief Reserve virtual space for a first half of @p firstHalfBytes and a second half that holds at least
    //! @p secondHalfBytes (rounded up to the granularity). The second half starts at firstHalfBytes.
    //! @param firstHalfBytes Must be a multiple of granularity().
    //! @throws std::runtime_error if a size is zero, the first half is misaligned, or the reservation fails.
    VmmRegion(size_t firstHalfBytes, size_t secondHalfBytes, int device);
    ~VmmRegion() noexcept;
    VmmRegion(VmmRegion const&) = delete;
    VmmRegion& operator=(VmmRegion const&) = delete;

    //! Minimum mapping size for the device.
    static size_t granularity(int device);

    //! @brief Map physical memory so that each half has at least @p bytesPerHalf accessible bytes.
    //! @return false, with nothing newly mapped, if the device cannot supply the memory.
    bool growTo(size_t bytesPerHalf) noexcept;

    //! As growTo(size_t), with a separate target for each half. A target above its half's size is clamped.
    bool growTo(size_t firstBytes, size_t secondBytes) noexcept;

    //! Unmap and release granules from the end until @p bytesPerHalf (rounded up to the granularity) remain mapped.
    void shrinkTo(size_t bytesPerHalf) noexcept;

    //! As shrinkTo(size_t), with a separate target for each half.
    void shrinkTo(size_t firstBytes, size_t secondBytes) noexcept;

    void* base() const noexcept
    {
        return reinterpret_cast<void*>(mBase);
    }

    //! Size of the first half; with the single-size constructor, of both halves.
    size_t halfBytes() const noexcept
    {
        return mHalfBytes[0];
    }

    size_t secondHalfBytes() const noexcept
    {
        return mHalfBytes[1];
    }

    size_t granuleBytes() const noexcept
    {
        return mGranularity;
    }

    //! Mapped bytes of the first half.
    size_t mappedPerHalf() const noexcept
    {
        return mMapped[0];
    }

    size_t mappedBytes(int half) const noexcept
    {
        return mMapped[half];
    }

    //! Mapped bytes of both halves.
    size_t mappedTotal() const noexcept
    {
        return mMapped[0] + mMapped[1];
    }

private:
    CUdeviceptr mBase{};
    size_t mHalfBytes[2]{};
    size_t mGranularity{};
    size_t mMapped[2]{};
    int mDevice{};
    std::vector<CUmemGenericAllocationHandle> mHandles[2];
};

} // namespace rt
} // namespace trt_edgellm
