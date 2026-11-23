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

#include <cmath>

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
} // namespace

void launchInkling2BlockResidual(Inkling2BlockResidualParams const& params, float const* queries, cudaStream_t stream)
{
    if (params.numTokens <= 0)
    {
        return;
    }
    blockResidualKernel<<<static_cast<unsigned int>(params.numTokens), kThreads, 0, stream>>>(params, queries);
}

} // namespace kernel
} // namespace trt_edgellm
