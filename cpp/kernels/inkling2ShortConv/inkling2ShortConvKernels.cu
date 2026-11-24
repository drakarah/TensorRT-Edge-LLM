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

#include "inkling2ShortConvKernels.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <string>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
//! Sequence and local index of slot t; false when t is padding.
__device__ __forceinline__ bool locate(
    int32_t t, int32_t const* offsets, int32_t const* lengths, int32_t batchSize, int32_t& seq, int32_t& local)
{
    seq = 0;
    while (seq + 1 < batchSize && t >= offsets[seq + 1])
    {
        ++seq;
    }
    local = t - offsets[seq];
    return local >= 0 && local < lengths[seq];
}

__global__ void shortConvKernel(Inkling2ShortConvParams p, half const* __restrict__ x, float const* __restrict__ taps,
    half const* __restrict__ dyn, half const* __restrict__ state, int32_t const* __restrict__ offsets,
    int32_t const* __restrict__ lengths, int32_t const* __restrict__ stateIndices, half* __restrict__ y)
{
    int32_t const c = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const t = blockIdx.y;
    if (c >= p.channels)
    {
        return;
    }
    int32_t seq, local;
    int64_t const outIdx = int64_t(t) * p.channels + c;
    if (!locate(t, offsets, lengths, p.batchSize, seq, local))
    {
        y[outIdx] = __float2half(0.F);
        return;
    }
    int32_t const row = stateIndices[seq];
    int32_t const group = p.groups > 0 ? c / (p.channels / p.groups) : 0;
    float acc = 0.F;
    for (int32_t j = 0; j < p.kernelSize; ++j)
    {
        float xv;
        if (j <= local)
        {
            xv = __half2float(x[int64_t(t - j) * p.channels + c]);
        }
        else if (row >= 0 && row < p.stateRows)
        {
            xv = __half2float(state[(int64_t(row) * p.stateStride + p.stateOffset + c) * p.kernelSize
                + (p.kernelSize - 1 - (j - local))]);
        }
        else
        {
            xv = 0.F;
        }
        float tap = taps[int64_t(c) * p.kernelSize + j];
        if (p.groups > 0)
        {
            tap += __half2float(dyn[int64_t(t) * p.groups * p.kernelSize + group * p.kernelSize + j]);
        }
        acc += tap * xv;
    }
    acc += __half2float(x[outIdx]);
    y[outIdx] = __float2half(acc * p.scale);
}

//! One thread per (sequence, channel): reads the old state and the input tail into registers, then rewrites the state.
__global__ void stateUpdateKernel(Inkling2ShortConvParams p, half const* __restrict__ x, half* __restrict__ state,
    int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths, int32_t const* __restrict__ stateIndices)
{
    int32_t const c = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const seq = blockIdx.y;
    if (c >= p.channels)
    {
        return;
    }
    int32_t const row = stateIndices[seq];
    int32_t const length = lengths[seq];
    if (row < 0 || row >= p.stateRows || length <= 0)
    {
        return;
    }
    int32_t const slots = p.kernelSize - 1;
    half old[kInkling2MaxConvKernel], next[kInkling2MaxConvKernel];
    half* rowPtr = state + (int64_t(row) * p.stateStride + p.stateOffset + c) * p.kernelSize;
    for (int32_t m = 0; m < slots; ++m)
    {
        old[m] = rowPtr[m];
    }
    for (int32_t m = 0; m < slots; ++m)
    {
        int32_t const idx = length - slots + m;
        next[m] = idx >= 0 ? x[int64_t(offsets[seq] + idx) * p.channels + c] : old[length + m];
    }
    for (int32_t m = 0; m < slots; ++m)
    {
        rowPtr[m] = next[m];
    }
}
//! Single token of a single sequence (decode): the convolution and the state update in one kernel, one thread per channel. The thread reads the old state into
//! registers, writes the output and then rewrites the state, which only its own channel owns, so no other thread can observe a half-updated state.
__global__ void shortConvDecodeKernel(Inkling2ShortConvParams p, half const* __restrict__ x, float const* __restrict__ taps,
    half const* __restrict__ dyn, half* __restrict__ state, int32_t const* __restrict__ offsets, int32_t const* __restrict__ lengths,
    int32_t const* __restrict__ stateIndices, half* __restrict__ y)
{
    int32_t const c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= p.channels)
    {
        return;
    }
    if (offsets[0] != 0 || lengths[0] <= 0)
    {
        y[c] = __float2half(0.F); // padding slot: zero output, state untouched
        return;
    }
    int32_t const row = stateIndices[0];
    bool const haveState = row >= 0 && row < p.stateRows;
    int32_t const slots = p.kernelSize - 1;
    half* rowPtr = haveState ? state + (int64_t(row) * p.stateStride + p.stateOffset + c) * p.kernelSize : nullptr;
    half old[kInkling2MaxConvKernel];
    for (int32_t m = 0; m < slots; ++m)
    {
        old[m] = haveState ? rowPtr[m] : __float2half(0.F);
    }
    int32_t const group = p.groups > 0 ? c / (p.channels / p.groups) : 0;
    float const xCur = __half2float(x[c]);
    float acc = 0.F;
    for (int32_t j = 0; j < p.kernelSize; ++j)
    {
        float const xv = j == 0 ? xCur : __half2float(old[slots - j]); // local index 0: lag j reads state slot k - 1 - j
        float tap = taps[int64_t(c) * p.kernelSize + j];
        if (p.groups > 0)
        {
            tap += __half2float(dyn[group * p.kernelSize + j]);
        }
        acc += tap * xv;
    }
    acc += xCur;
    y[c] = __float2half(acc * p.scale);
    if (haveState)
    {
        // length 1: the new state is the old one shifted by one with the current input appended
        for (int32_t m = 0; m < slots - 1; ++m)
        {
            rowPtr[m] = old[m + 1];
        }
        rowPtr[slots - 1] = x[c];
    }
}
} // namespace

void launchInkling2ShortConv(Inkling2ShortConvParams const& params, half const* x, float const* taps, half const* dyn,
    half* state, int32_t const* queryStartOffsets, int32_t const* queryLengths, int32_t const* stateIndices, half* y,
    cudaStream_t stream)
{
    if (params.numTokens <= 0)
    {
        return;
    }
    constexpr int32_t kThreads = 128;
    static bool const fusedEnabled = []
    {
        char const* const value = std::getenv("EDGELLM_INKLING_FUSED_CONV");
        return value == nullptr || std::string(value) != "0";
    }();
    if (fusedEnabled && params.numTokens == 1 && params.batchSize == 1)
    {
        shortConvDecodeKernel<<<(params.channels + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            params, x, taps, dyn, state, queryStartOffsets, queryLengths, stateIndices, y);
        return;
    }
    dim3 const grid((params.channels + kThreads - 1) / kThreads, params.numTokens);
    shortConvKernel<<<grid, kThreads, 0, stream>>>(
        params, x, taps, dyn, state, queryStartOffsets, queryLengths, stateIndices, y);
    dim3 const updateGrid((params.channels + kThreads - 1) / kThreads, params.batchSize);
    stateUpdateKernel<<<updateGrid, kThreads, 0, stream>>>(
        params, x, state, queryStartOffsets, queryLengths, stateIndices);
}

} // namespace kernel
} // namespace trt_edgellm
