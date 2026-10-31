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

#include "common/vmmRegion.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace rt
{

namespace
{
CUmemAllocationProp allocationProp(int device)
{
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    return prop;
}
} // namespace

size_t VmmRegion::granularity(int device)
{
    cudaFree(nullptr);
    int supported = 0;
    CUdevice handle{};
    if (cuDeviceGet(&handle, device) != CUDA_SUCCESS
        || cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, handle) != CUDA_SUCCESS
        || supported == 0)
    {
        throw std::runtime_error("VmmRegion: the device does not support virtual memory management.");
    }
    CUmemAllocationProp const prop = allocationProp(device);
    size_t granule = 0;
    if (cuMemGetAllocationGranularity(&granule, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || granule == 0)
    {
        throw std::runtime_error("VmmRegion: cannot query the allocation granularity.");
    }
    return granule;
}

VmmRegion::VmmRegion(size_t halfBytes, int device)
    : mHalfBytes(halfBytes)
    , mGranularity(granularity(device))
    , mDevice(device)
{
    if (halfBytes == 0 || halfBytes % mGranularity != 0)
    {
        throw std::runtime_error("VmmRegion: half size must be a non-zero multiple of the granularity.");
    }
    if (cuMemAddressReserve(&mBase, 2 * halfBytes, mGranularity, 0, 0) != CUDA_SUCCESS)
    {
        throw std::runtime_error("VmmRegion: virtual address reservation failed.");
    }
}

VmmRegion::~VmmRegion() noexcept
{
    shrinkTo(0);
    if (mBase != 0)
    {
        cuMemAddressFree(mBase, 2 * mHalfBytes);
    }
}

bool VmmRegion::growTo(size_t bytesPerHalf) noexcept
{
    size_t const target = std::min(mHalfBytes, (bytesPerHalf + mGranularity - 1) / mGranularity * mGranularity);
    if (target <= mMapped)
    {
        return true;
    }
    size_t const chunk = target - mMapped;
    CUmemAllocationProp const prop = allocationProp(mDevice);
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUmemGenericAllocationHandle handles[2]{};
    bool ok = true;
    int created = 0;
    for (; created < 2 && ok; ++created)
    {
        ok = cuMemCreate(&handles[created], chunk, &prop, 0) == CUDA_SUCCESS;
    }
    int mapped = 0;
    if (ok)
    {
        for (; mapped < 2 && ok; ++mapped)
        {
            CUdeviceptr const at = mBase + static_cast<size_t>(mapped) * mHalfBytes + mMapped;
            ok = cuMemMap(at, chunk, 0, handles[mapped], 0) == CUDA_SUCCESS
                && cuMemSetAccess(at, chunk, &access, 1) == CUDA_SUCCESS;
        }
    }
    if (!ok)
    {
        for (int h = 0; h < mapped; ++h)
        {
            cuMemUnmap(mBase + static_cast<size_t>(h) * mHalfBytes + mMapped, chunk);
        }
        for (int h = 0; h < 2; ++h)
        {
            if (handles[h] != 0)
            {
                cuMemRelease(handles[h]);
            }
        }
        return false;
    }
    for (int h = 0; h < 2; ++h)
    {
        mHandles[h].push_back(handles[h]);
    }
    mChunkSizes.push_back(chunk);
    mMapped = target;
    return true;
}

void VmmRegion::shrinkTo(size_t bytesPerHalf) noexcept
{
    size_t const keep = (bytesPerHalf + mGranularity - 1) / mGranularity * mGranularity;
    while (!mChunkSizes.empty() && mMapped - mChunkSizes.back() >= keep)
    {
        size_t const chunk = mChunkSizes.back();
        for (int h = 0; h < 2; ++h)
        {
            cuMemUnmap(mBase + static_cast<size_t>(h) * mHalfBytes + mMapped - chunk, chunk);
            cuMemRelease(mHandles[h].back());
            mHandles[h].pop_back();
        }
        mMapped -= chunk;
        mChunkSizes.pop_back();
    }
}

} // namespace rt
} // namespace trt_edgellm
