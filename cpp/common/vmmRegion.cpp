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
    : VmmRegion(halfBytes, halfBytes, device)
{
}

VmmRegion::VmmRegion(size_t firstHalfBytes, size_t secondHalfBytes, int device)
    : mGranularity(granularity(device))
    , mDevice(device)
{
    if (firstHalfBytes == 0 || secondHalfBytes == 0 || firstHalfBytes % mGranularity != 0)
    {
        throw std::runtime_error("VmmRegion: half sizes must be non-zero and the first a multiple of the granularity.");
    }
    mHalfBytes[0] = firstHalfBytes;
    mHalfBytes[1] = (secondHalfBytes + mGranularity - 1) / mGranularity * mGranularity;
    if (cuMemAddressReserve(&mBase, mHalfBytes[0] + mHalfBytes[1], mGranularity, 0, 0) != CUDA_SUCCESS)
    {
        throw std::runtime_error("VmmRegion: virtual address reservation failed.");
    }
}

VmmRegion::~VmmRegion() noexcept
{
    shrinkTo(0);
    if (mBase != 0)
    {
        cuMemAddressFree(mBase, mHalfBytes[0] + mHalfBytes[1]);
    }
}

bool VmmRegion::growTo(size_t bytesPerHalf) noexcept
{
    return growTo(bytesPerHalf, bytesPerHalf);
}

bool VmmRegion::growTo(size_t firstBytes, size_t secondBytes) noexcept
{
    size_t target[2];
    size_t const request[2] = {firstBytes, secondBytes};
    for (int h = 0; h < 2; ++h)
    {
        target[h] = std::max(mMapped[h], std::min(mHalfBytes[h], (request[h] + mGranularity - 1) / mGranularity * mGranularity));
    }
    CUmemAllocationProp const prop = allocationProp(mDevice);
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    size_t const before[2] = {mHandles[0].size(), mHandles[1].size()};
    bool ok = true;
    for (int h = 0; h < 2 && ok; ++h)
    {
        size_t const first = mMapped[h] / mGranularity;
        size_t const last = target[h] / mGranularity;
        CUdeviceptr const halfBase = mBase + (h == 0 ? 0 : mHalfBytes[0]);
        for (size_t g = first; g < last && ok; ++g)
        {
            CUmemGenericAllocationHandle handle{};
            CUdeviceptr const at = halfBase + g * mGranularity;
            if (cuMemCreate(&handle, mGranularity, &prop, 0) != CUDA_SUCCESS)
            {
                ok = false;
            }
            else if (cuMemMap(at, mGranularity, 0, handle, 0) != CUDA_SUCCESS)
            {
                cuMemRelease(handle);
                ok = false;
            }
            else
            {
                mHandles[h].push_back(handle);
            }
        }
        if (ok && last > first)
        {
            ok = cuMemSetAccess(halfBase + first * mGranularity, (last - first) * mGranularity, &access, 1)
                == CUDA_SUCCESS;
        }
    }
    if (!ok)
    {
        for (int h = 0; h < 2; ++h)
        {
            CUdeviceptr const halfBase = mBase + (h == 0 ? 0 : mHalfBytes[0]);
            while (mHandles[h].size() > before[h])
            {
                size_t const g = mHandles[h].size() - 1;
                cuMemUnmap(halfBase + g * mGranularity, mGranularity);
                cuMemRelease(mHandles[h].back());
                mHandles[h].pop_back();
            }
        }
        return false;
    }
    mMapped[0] = target[0];
    mMapped[1] = target[1];
    return true;
}

void VmmRegion::shrinkTo(size_t bytesPerHalf) noexcept
{
    shrinkTo(bytesPerHalf, bytesPerHalf);
}

void VmmRegion::shrinkTo(size_t firstBytes, size_t secondBytes) noexcept
{
    size_t const request[2] = {firstBytes, secondBytes};
    size_t keep[2];
    bool unmapping = false;
    for (int h = 0; h < 2; ++h)
    {
        keep[h] = (request[h] + mGranularity - 1) / mGranularity * mGranularity;
        unmapping = unmapping || mMapped[h] > keep[h];
    }
    if (unmapping)
    {
        // Unmapping requires that no outstanding work touches the range.
        cudaDeviceSynchronize();
    }
    for (int h = 0; h < 2; ++h)
    {
        CUdeviceptr const halfBase = mBase + (h == 0 ? 0 : mHalfBytes[0]);
        while (mHandles[h].size() * mGranularity > keep[h])
        {
            size_t const g = mHandles[h].size() - 1;
            cuMemUnmap(halfBase + g * mGranularity, mGranularity);
            cuMemRelease(mHandles[h].back());
            mHandles[h].pop_back();
        }
        mMapped[h] = std::min(mMapped[h], keep[h]);
    }
}

} // namespace rt
} // namespace trt_edgellm
