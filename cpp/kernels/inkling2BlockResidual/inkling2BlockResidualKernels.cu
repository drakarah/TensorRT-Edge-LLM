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
#include "inkling2BlockResidualKernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{
namespace
{
constexpr int32_t kThreads = 1024;
constexpr int32_t kWarps = kThreads / 32;
constexpr int32_t kMaxElems = kInkling2MaxResidualHidden / kThreads;
constexpr int32_t kMaxReduce = kInkling2MaxResidualStates * (kInkling2MaxResidualQueries + 1);

// One block per token; every thread keeps its slice of all states in registers across both phases.
__global__ void __launch_bounds__(kThreads) blockResidualKernel(Inkling2BlockResidualParams const p, float const* queries)
{
    __shared__ float partial[kWarps][kMaxReduce];
    __shared__ float total[kMaxReduce];

    int64_t const token = blockIdx.x;
    int32_t const hidden = p.hidden;
    int32_t const numStates = p.numStates;
    int32_t const numQueries = p.numQueries;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;

    float vals[kInkling2MaxResidualStates][kMaxElems];
    float q[kInkling2MaxResidualQueries][kMaxElems];
    for (int32_t e = 0; e < kMaxElems; ++e)
    {
        int32_t const h = threadIdx.x + e * kThreads;
        for (int32_t j = 0; j < kInkling2MaxResidualStates; ++j)
        {
            vals[j][e] = (j < numStates && h < hidden) ? p.states[j][token * hidden + h] : 0.F;
        }
        for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
        {
            q[i][e] = (i < numQueries && h < hidden) ? queries[static_cast<int64_t>(i) * hidden + h] : 0.F;
        }
    }

    // reduce[j] = sum of squares of state j, reduce[numStates + i * numStates + j] = <state_j, query_i>
    float reduce[kMaxReduce];
    for (int32_t j = 0; j < numStates; ++j)
    {
        float sq = 0.F;
        for (int32_t e = 0; e < kMaxElems; ++e)
        {
            sq += vals[j][e] * vals[j][e];
        }
        reduce[j] = sq;
        for (int32_t i = 0; i < numQueries; ++i)
        {
            float d = 0.F;
            for (int32_t e = 0; e < kMaxElems; ++e)
            {
                d += vals[j][e] * q[i][e];
            }
            reduce[numStates + i * numStates + j] = d;
        }
    }
    int32_t const count = numStates + numStates * numQueries;
    for (int32_t c = 0; c < count; ++c)
    {
        float v = reduce[c];
        for (int32_t off = 16; off > 0; off >>= 1)
        {
            v += __shfl_xor_sync(0xffffffffU, v, off);
        }
        if (lane == 0)
        {
            partial[warp][c] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < count)
    {
        float v = 0.F;
        for (int32_t w = 0; w < kWarps; ++w)
        {
            v += partial[w][threadIdx.x];
        }
        total[threadIdx.x] = v;
    }
    __syncthreads();

    float inv[kInkling2MaxResidualStates];
    for (int32_t j = 0; j < numStates; ++j)
    {
        inv[j] = 1.F / sqrtf(total[j] / static_cast<float>(hidden) + p.eps);
    }
    // weight[i][j] = softmax_j(inv_j * dot_ij * hidden^-0.5) * inv_j
    float weight[kInkling2MaxResidualQueries][kInkling2MaxResidualStates];
    float const scale = rsqrtf(static_cast<float>(hidden));
    for (int32_t i = 0; i < numQueries; ++i)
    {
        float score[kInkling2MaxResidualStates];
        float top = -INFINITY;
        for (int32_t j = 0; j < numStates; ++j)
        {
            score[j] = inv[j] * total[numStates + i * numStates + j] * scale;
            top = fmaxf(top, score[j]);
        }
        float denom = 0.F;
        for (int32_t j = 0; j < numStates; ++j)
        {
            score[j] = expf(score[j] - top);
            denom += score[j];
        }
        for (int32_t j = 0; j < numStates; ++j)
        {
            weight[i][j] = score[j] / denom * inv[j];
        }
    }
    for (int32_t e = 0; e < kMaxElems; ++e)
    {
        int32_t const h = threadIdx.x + e * kThreads;
        if (h >= hidden)
        {
            break;
        }
        for (int32_t i = 0; i < numQueries; ++i)
        {
            float out = 0.F;
            for (int32_t j = 0; j < numStates; ++j)
            {
                out += weight[i][j] * vals[j][e];
            }
            p.outputs[i][token * hidden + h] = out;
        }
    }
}

constexpr int32_t kThreadsV2 = 256;

//! One block of 8 warps per token. Phase 1: warp j reduces state j (sum of squares and the dot products with every query row) with float4 loads;
//! phase 2: one thread per query turns the scores into softmax weights (scaled by the state's inverse RMS) in shared memory; phase 3: all threads
//! write the weighted sums, re-reading the states from L1. No per-thread arrays indexed by runtime bounds, so nothing spills (the 1,024-thread
//! kernel above is limited to 64 registers and kept about 90 live floats per thread).
__global__ void __launch_bounds__(kThreadsV2) blockResidualKernelV2(Inkling2BlockResidualParams const p, float const* queries)
{
    __shared__ float total[kMaxReduce];
    __shared__ float weight[kInkling2MaxResidualQueries][kInkling2MaxResidualStates];

    int64_t const token = blockIdx.x;
    int32_t const hidden = p.hidden;
    int32_t const hidden4 = hidden >> 2;
    int32_t const numStates = p.numStates;
    int32_t const numQueries = p.numQueries;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;

    if (warp < numStates)
    {
        float4 const* state = reinterpret_cast<float4 const*>(p.states[warp] + token * hidden);
        float sq = 0.F;
        float dot[kInkling2MaxResidualQueries] = {0.F, 0.F, 0.F, 0.F};
        for (int32_t h = lane; h < hidden4; h += 32)
        {
            float4 const v = state[h];
            sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
            for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
            {
                if (i < numQueries)
                {
                    float4 const qv = reinterpret_cast<float4 const*>(queries + static_cast<int64_t>(i) * hidden)[h];
                    dot[i] += v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
                }
            }
        }
#pragma unroll
        for (int32_t off = 16; off > 0; off >>= 1)
        {
            sq += __shfl_xor_sync(0xffffffffU, sq, off);
#pragma unroll
            for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
            {
                dot[i] += __shfl_xor_sync(0xffffffffU, dot[i], off);
            }
        }
        if (lane == 0)
        {
            total[warp] = sq;
#pragma unroll
            for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
            {
                if (i < numQueries)
                {
                    total[numStates + i * numStates + warp] = dot[i];
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < numQueries)
    {
        int32_t const i = threadIdx.x;
        float const scale = rsqrtf(static_cast<float>(hidden));
        float score[kInkling2MaxResidualStates];
        float inv[kInkling2MaxResidualStates];
        float top = -INFINITY;
#pragma unroll
        for (int32_t j = 0; j < kInkling2MaxResidualStates; ++j)
        {
            if (j < numStates)
            {
                inv[j] = 1.F / sqrtf(total[j] / static_cast<float>(hidden) + p.eps);
                score[j] = inv[j] * total[numStates + i * numStates + j] * scale;
                top = fmaxf(top, score[j]);
            }
        }
        float denom = 0.F;
#pragma unroll
        for (int32_t j = 0; j < kInkling2MaxResidualStates; ++j)
        {
            if (j < numStates)
            {
                score[j] = expf(score[j] - top);
                denom += score[j];
            }
        }
#pragma unroll
        for (int32_t j = 0; j < kInkling2MaxResidualStates; ++j)
        {
            if (j < numStates)
            {
                weight[i][j] = score[j] / denom * inv[j];
            }
        }
    }
    __syncthreads();
    for (int32_t h = threadIdx.x; h < hidden4; h += kThreadsV2)
    {
        float4 acc[kInkling2MaxResidualQueries];
#pragma unroll
        for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
        {
            acc[i] = make_float4(0.F, 0.F, 0.F, 0.F);
        }
#pragma unroll
        for (int32_t j = 0; j < kInkling2MaxResidualStates; ++j)
        {
            if (j < numStates)
            {
                float4 const v = reinterpret_cast<float4 const*>(p.states[j] + token * hidden)[h];
#pragma unroll
                for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
                {
                    float const w = weight[i][j];
                    acc[i].x += w * v.x;
                    acc[i].y += w * v.y;
                    acc[i].z += w * v.z;
                    acc[i].w += w * v.w;
                }
            }
        }
#pragma unroll
        for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
        {
            if (i < numQueries)
            {
                reinterpret_cast<float4*>(p.outputs[i] + token * hidden)[h] = acc[i];
            }
        }
    }
}

constexpr int32_t kSplit = kInkling2ResidualSplit;
constexpr int32_t kMaxSlice4 = 128;   //!< float4 per block slice the staged query tile can hold
constexpr int32_t kApplyThreads = 64; //!< apply kernel block: one float4 per thread of a 64-float4 slice

//! Split version, stage 1: grid (kSplit, tokens). Block s of a token reduces its slice of the hidden dimension: warp j handles state j (sum of squares and the
//! dot products with every query) and writes the partial sums to the workspace.
__global__ void __launch_bounds__(kThreadsV2) blockResidualPartialKernel(
    Inkling2BlockResidualParams const p, float const* queries, float* partials)
{
    int64_t const token = blockIdx.y;
    int32_t const slice = blockIdx.x;
    int32_t const hidden = p.hidden;
    int32_t const hidden4 = hidden >> 2;
    int32_t const perSlice = (hidden4 + kSplit - 1) / kSplit;
    int32_t const lo = slice * perSlice;
    int32_t const hi = min(hidden4, lo + perSlice);
    int32_t const numStates = p.numStates;
    int32_t const numQueries = p.numQueries;
    int32_t const warp = threadIdx.x >> 5;
    int32_t const lane = threadIdx.x & 31;
    // the query slice of this block is shared by all state warps: load it once (the block has one warp per state)
    __shared__ float4 queryTile[kInkling2MaxResidualQueries][kMaxSlice4];
    for (int32_t idx = threadIdx.x; idx < numQueries * perSlice; idx += blockDim.x)
    {
        int32_t const i = idx / perSlice;
        int32_t const hh = idx % perSlice;
        queryTile[i][hh] = lo + hh < hi ? reinterpret_cast<float4 const*>(queries + static_cast<int64_t>(i) * hidden)[lo + hh] : make_float4(0.F, 0.F, 0.F, 0.F);
    }
    __syncthreads();
    float4 const* state = reinterpret_cast<float4 const*>(p.states[warp] + token * hidden);
    float sq = 0.F;
    float dot[kInkling2MaxResidualQueries] = {0.F, 0.F, 0.F, 0.F};
    for (int32_t h = lo + lane; h < hi; h += 32)
    {
        float4 const v = state[h];
        sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
#pragma unroll
        for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
        {
            if (i < numQueries)
            {
                float4 const qv = queryTile[i][h - lo];
                dot[i] += v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
            }
        }
    }
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1)
    {
        sq += __shfl_xor_sync(0xffffffffU, sq, off);
#pragma unroll
        for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
        {
            dot[i] += __shfl_xor_sync(0xffffffffU, dot[i], off);
        }
    }
    if (lane == 0)
    {
        float* out = partials + (token * kSplit + slice) * kMaxReduce;
        out[warp] = sq;
#pragma unroll
        for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
        {
            if (i < numQueries)
            {
                out[numStates + i * numStates + warp] = dot[i];
            }
        }
    }
}

//! Stage 2: grid (kSplit, tokens). Every block sums the kSplit partials of its token, derives the softmax weights and writes its slice of the outputs.
__global__ void __launch_bounds__(kApplyThreads) blockResidualApplyKernel(
    Inkling2BlockResidualParams const p, float const* partials)
{
    __shared__ float total[kMaxReduce];
    __shared__ float weight[kInkling2MaxResidualQueries][kInkling2MaxResidualStates];

    int64_t const token = blockIdx.y;
    int32_t const slice = blockIdx.x;
    int32_t const hidden = p.hidden;
    int32_t const hidden4 = hidden >> 2;
    int32_t const numStates = p.numStates;
    int32_t const numQueries = p.numQueries;
    int32_t const count = numStates + numStates * numQueries;
    if (threadIdx.x < count)
    {
        float v = 0.F;
#pragma unroll
        for (int32_t s = 0; s < kSplit; ++s)
        {
            v += partials[(token * kSplit + s) * kMaxReduce + threadIdx.x];
        }
        total[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x < numQueries)
    {
        int32_t const i = threadIdx.x;
        float const scale = rsqrtf(static_cast<float>(hidden));
        float score[kInkling2MaxResidualStates];
        float inv[kInkling2MaxResidualStates];
        float top = -INFINITY;
#pragma unroll
        for (int32_t j = 0; j < kInkling2MaxResidualStates; ++j)
        {
            if (j < numStates)
            {
                inv[j] = 1.F / sqrtf(total[j] / static_cast<float>(hidden) + p.eps);
                score[j] = inv[j] * total[numStates + i * numStates + j] * scale;
                top = fmaxf(top, score[j]);
            }
        }
        float denom = 0.F;
#pragma unroll
        for (int32_t j = 0; j < kInkling2MaxResidualStates; ++j)
        {
            if (j < numStates)
            {
                score[j] = expf(score[j] - top);
                denom += score[j];
            }
        }
#pragma unroll
        for (int32_t j = 0; j < kInkling2MaxResidualStates; ++j)
        {
            if (j < numStates)
            {
                weight[i][j] = score[j] / denom * inv[j];
            }
        }
    }
    __syncthreads();
    int32_t const perSlice = (hidden4 + kSplit - 1) / kSplit;
    int32_t const lo = slice * perSlice;
    int32_t const hi = min(hidden4, lo + perSlice);
    for (int32_t h = lo + threadIdx.x; h < hi; h += kApplyThreads)
    {
        float4 acc[kInkling2MaxResidualQueries];
#pragma unroll
        for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
        {
            acc[i] = make_float4(0.F, 0.F, 0.F, 0.F);
        }
#pragma unroll
        for (int32_t j = 0; j < kInkling2MaxResidualStates; ++j)
        {
            if (j < numStates)
            {
                float4 const v = reinterpret_cast<float4 const*>(p.states[j] + token * hidden)[h];
#pragma unroll
                for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
                {
                    float const w = weight[i][j];
                    acc[i].x += w * v.x;
                    acc[i].y += w * v.y;
                    acc[i].z += w * v.z;
                    acc[i].w += w * v.w;
                }
            }
        }
#pragma unroll
        for (int32_t i = 0; i < kInkling2MaxResidualQueries; ++i)
        {
            if (i < numQueries)
            {
                reinterpret_cast<float4*>(p.outputs[i] + token * hidden)[h] = acc[i];
            }
        }
    }
}

bool aligned16(void const* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 15U) == 0U;
}

bool canUseV2(Inkling2BlockResidualParams const& p, float const* queries)
{
    if (p.hidden % 4 != 0 || p.numStates > kThreadsV2 / 32 || !aligned16(queries) || ((p.hidden / 4) + kSplit - 1) / kSplit > kMaxSlice4)
    {
        return false;
    }
    for (int32_t j = 0; j < p.numStates; ++j)
    {
        if (!aligned16(p.states[j]))
        {
            return false;
        }
    }
    for (int32_t i = 0; i < p.numQueries; ++i)
    {
        if (!aligned16(p.outputs[i]))
        {
            return false;
        }
    }
    return true;
}
} // namespace

size_t inkling2BlockResidualWorkspaceBytes(int64_t numTokens)
{
    return static_cast<size_t>(numTokens) * kSplit * kMaxReduce * sizeof(float);
}

void launchInkling2BlockResidual(Inkling2BlockResidualParams const& params, float const* queries, cudaStream_t stream, void* workspace)
{
    if (params.numTokens <= 0)
    {
        return;
    }
    if (workspace != nullptr && canUseV2(params, queries) && params.numTokens <= 0x7fffffffLL && params.numTokens <= 65535)
    {
        dim3 const grid(kSplit, static_cast<unsigned int>(params.numTokens));
        blockResidualPartialKernel<<<grid, 32 * std::max(1, params.numStates), 0, stream>>>(params, queries, static_cast<float*>(workspace));
        blockResidualApplyKernel<<<grid, kApplyThreads, 0, stream>>>(params, static_cast<float const*>(workspace));
        return;
    }
    if (canUseV2(params, queries))
    {
        blockResidualKernelV2<<<static_cast<unsigned int>(params.numTokens), kThreadsV2, 0, stream>>>(params, queries);
        return;
    }
    blockResidualKernel<<<static_cast<unsigned int>(params.numTokens), kThreads, 0, stream>>>(params, queries);
}

namespace
{
//! Debug capture (EDGELLM_TAP_TRACE): copies `rows` rows of `hidden` floats to dst at the device-side row counter, then a second kernel advances the counter and logs the step size.
__global__ void tapCopyKernel(float const* src, float* dst, int32_t const* counter, int32_t hidden, int32_t capacityRows)
{
    int64_t const row = blockIdx.x;
    int64_t const at = static_cast<int64_t>(counter[0]) + row;
    if (at >= capacityRows)
    {
        return;
    }
    for (int32_t i = threadIdx.x; i < hidden; i += blockDim.x)
    {
        dst[at * hidden + i] = src[row * hidden + i];
    }
}

__global__ void tapAdvanceKernel(int32_t* counter, int32_t* stepLog, int32_t* stepCounter, int32_t rows, int32_t capacityRows, int32_t capacitySteps)
{
    if (stepLog != nullptr)
    {
        int32_t const step = stepCounter[0];
        if (step < capacitySteps)
        {
            stepLog[step] = rows;
        }
        stepCounter[0] = step + 1;
    }
    counter[0] = min(counter[0] + rows, capacityRows);
}
} // namespace

void launchTapCapture(float const* src, int64_t rows, int32_t hidden, float* dst, int32_t* counter, int32_t capacityRows, int32_t* stepLog, int32_t* stepCounter,
    int32_t capacitySteps, cudaStream_t stream)
{
    if (rows <= 0)
    {
        return;
    }
    tapCopyKernel<<<static_cast<unsigned int>(rows), 256, 0, stream>>>(src, dst, counter, hidden, capacityRows);
    tapAdvanceKernel<<<1, 1, 0, stream>>>(counter, stepLog, stepCounter, static_cast<int32_t>(rows), capacityRows, capacitySteps);
}

} // namespace kernel
} // namespace trt_edgellm
