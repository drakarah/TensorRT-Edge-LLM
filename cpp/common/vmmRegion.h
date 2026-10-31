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

//! Device buffer made of two equally sized halves in one reserved virtual range. Physical memory is mapped to a prefix
//! of each half on demand, so the base pointer and the halves' offsets never change when the buffer grows.
class VmmRegion
{
public:
    //! @brief Reserve virtual space for two halves of @p halfBytes each. Nothing is mapped yet.
    //! @param halfBytes Must be a multiple of granularity().
    //! @throws std::runtime_error if the reservation fails.
    VmmRegion(size_t halfBytes, int device);
    ~VmmRegion() noexcept;
    VmmRegion(VmmRegion const&) = delete;
    VmmRegion& operator=(VmmRegion const&) = delete;

    //! Minimum mapping size for the device.
    static size_t granularity(int device);

    //! @brief Map physical memory so that each half has at least @p bytesPerHalf accessible bytes.
    //! @return false, with nothing newly mapped, if the device cannot supply the memory.
    bool growTo(size_t bytesPerHalf) noexcept;

    //! Release the most recently mapped chunks while at least @p bytesPerHalf (rounded up to the granularity) stays mapped.
    void shrinkTo(size_t bytesPerHalf) noexcept;

    void* base() const noexcept
    {
        return reinterpret_cast<void*>(mBase);
    }

    size_t halfBytes() const noexcept
    {
        return mHalfBytes;
    }

    size_t mappedPerHalf() const noexcept
    {
        return mMapped;
    }

private:
    CUdeviceptr mBase{};
    size_t mHalfBytes{};
    size_t mGranularity{};
    size_t mMapped{};
    int mDevice{};
    std::vector<CUmemGenericAllocationHandle> mHandles[2];
    std::vector<size_t> mChunkSizes;
};

} // namespace rt
} // namespace trt_edgellm
