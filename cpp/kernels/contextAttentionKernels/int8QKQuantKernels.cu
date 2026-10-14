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

#include "int8QKQuantKernels.h"

#include "common/checkMacros.h"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kVEC = 8; // halves per 16-byte load

__device__ __forceinline__ float warpMax(float v)
{
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset /= 2)
    {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ int32_t quantize4(half2 a, half2 b, float inv)
{
    float2 const fa = __half22float2(a);
    float2 const fb = __half22float2(b);
    auto q = [inv](float x) { return static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(x * inv)))); };
    return static_cast<int32_t>(q(fa.x) | (q(fa.y) << 8) | (q(fb.x) << 16) | (q(fb.y) << 24));
}

//! Quantise kVEC halves held in a uint4 to 8 int8 values (uint2).
__device__ __forceinline__ uint2 quantize8(uint4 v, float inv)
{
    half2 const* h = reinterpret_cast<half2 const*>(&v);
    return make_uint2(static_cast<uint32_t>(quantize4(h[0], h[1], inv)), static_cast<uint32_t>(quantize4(h[2], h[3], inv)));
}

__device__ __forceinline__ float absMax8(uint4 v)
{
    half2 const* h = reinterpret_cast<half2 const*>(&v);
    float m = 0.F;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        float2 const f = __half22float2(__habs2(h[i]));
        m = fmaxf(m, fmaxf(f.x, f.y));
    }
    return m;
}

//! One warp per (b, s, h) row; each lane holds D / 256 16-byte vectors.
template <int32_t kD>
__global__ void quantizeQPerRowKernel(
    half const* q, int8_t* qOut, float* qScale, int32_t batchSize, int32_t seqLen, int32_t numHeads)
{
    constexpr int32_t kVecPerLane = kD / (32 * kVEC);
    int64_t const row = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    int64_t const numRows = static_cast<int64_t>(batchSize) * seqLen * numHeads;
    if (row >= numRows)
    {
        return;
    }
    int32_t const lane = threadIdx.x % 32;
    uint4 const* src = reinterpret_cast<uint4 const*>(q + row * kD);
    uint4 v[kVecPerLane];
    float amax = 0.F;
#pragma unroll
    for (int32_t i = 0; i < kVecPerLane; ++i)
    {
        v[i] = src[i * 32 + lane];
        amax = fmaxf(amax, absMax8(v[i]));
    }
    amax = fmaxf(warpMax(amax), 1e-8F);
    float const inv = 127.F / amax;
    uint2* dst = reinterpret_cast<uint2*>(qOut + row * kD);
#pragma unroll
    for (int32_t i = 0; i < kVecPerLane; ++i)
    {
        dst[i * 32 + lane] = quantize8(v[i], inv);
    }
    if (lane == 0)
    {
        // row = (b * S + s) * H + h  ->  qScale[b, h, s]
        int32_t const h = static_cast<int32_t>(row % numHeads);
        int64_t const bs = row / numHeads;
        int32_t const s = static_cast<int32_t>(bs % seqLen);
        int64_t const b = bs / seqLen;
        qScale[(b * numHeads + h) * seqLen + s] = amax / 127.F;
    }
}

//! One block (256 threads) per (b, 32-key tile, h): thread t holds row t / 8, columns (t % 8) * D / 8 ...
template <int32_t kD>
__global__ void quantizeKPerTileKernel(half const* k, int8_t* kOut, float* kScale, int32_t seqLen, int32_t numHeads)
{
    constexpr int32_t kColsPerThread = kD / 8;
    constexpr int32_t kVecPerThread = kColsPerThread / kVEC;
    int32_t const numTiles = (seqLen + kINT8_QK_KEY_TILE - 1) / kINT8_QK_KEY_TILE;
    int32_t const h = blockIdx.x % numHeads;
    int32_t const tile = (blockIdx.x / numHeads) % numTiles;
    int64_t const b = blockIdx.x / (numHeads * numTiles);
    int32_t const r = threadIdx.x / 8;
    int32_t const c0 = (threadIdx.x % 8) * kColsPerThread;
    int32_t const s = tile * kINT8_QK_KEY_TILE + r;
    bool const valid = s < seqLen;
    int64_t const rowOffset = ((b * seqLen + s) * numHeads + h) * kD + c0;

    uint4 v[kVecPerThread];
    float amax = 0.F;
#pragma unroll
    for (int32_t i = 0; i < kVecPerThread; ++i)
    {
        v[i] = valid ? reinterpret_cast<uint4 const*>(k + rowOffset)[i] : make_uint4(0, 0, 0, 0);
        amax = fmaxf(amax, absMax8(v[i]));
    }
    __shared__ float sMax[8];
    amax = warpMax(amax);
    if (threadIdx.x % 32 == 0)
    {
        sMax[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    amax = sMax[0];
#pragma unroll
    for (int32_t w = 1; w < 8; ++w)
    {
        amax = fmaxf(amax, sMax[w]);
    }
    amax = fmaxf(amax, 1e-8F);
    float const inv = 127.F / amax;
    if (valid)
    {
        uint2* dst = reinterpret_cast<uint2*>(kOut + rowOffset);
#pragma unroll
        for (int32_t i = 0; i < kVecPerThread; ++i)
        {
            dst[i] = quantize8(v[i], inv);
        }
    }
    if (threadIdx.x == 0)
    {
        kScale[(b * numHeads + h) * numTiles + tile] = amax / 127.F;
    }
}

} // namespace

void quantizeQPerRowInt8(half const* q, int8_t* qOut, float* qScale, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headDim, cudaStream_t stream)
{
    int64_t const numRows = static_cast<int64_t>(batchSize) * seqLen * numHeads;
    if (numRows == 0)
    {
        return;
    }
    constexpr int32_t kWarps = 8;
    dim3 const grid(static_cast<uint32_t>((numRows + kWarps - 1) / kWarps));
    switch (headDim)
    {
    case 256:
        quantizeQPerRowKernel<256><<<grid, kWarps * 32, 0, stream>>>(q, qOut, qScale, batchSize, seqLen, numHeads);
        break;
    case 512:
        quantizeQPerRowKernel<512><<<grid, kWarps * 32, 0, stream>>>(q, qOut, qScale, batchSize, seqLen, numHeads);
        break;
    default: check::check(false, "quantizeQPerRowInt8: head dim must be 256 or 512");
    }
    CUDA_CHECK(cudaGetLastError());
}

void quantizeKPerTileInt8(half const* k, int8_t* kOut, float* kScale, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headDim, cudaStream_t stream)
{
    int32_t const numTiles = (seqLen + kINT8_QK_KEY_TILE - 1) / kINT8_QK_KEY_TILE;
    if (batchSize == 0 || numTiles == 0)
    {
        return;
    }
    dim3 const grid(static_cast<uint32_t>(batchSize * numTiles * numHeads));
    switch (headDim)
    {
    case 256: quantizeKPerTileKernel<256><<<grid, 256, 0, stream>>>(k, kOut, kScale, seqLen, numHeads); break;
    case 512: quantizeKPerTileKernel<512><<<grid, 256, 0, stream>>>(k, kOut, kScale, seqLen, numHeads); break;
    default: check::check(false, "quantizeKPerTileInt8: head dim must be 256 or 512");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
