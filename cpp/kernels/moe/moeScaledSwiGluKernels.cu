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

#include "moeScaledSwiGluKernels.h"

#include "common/checkMacros.h"
#include "common/stringUtils.h"

namespace trt_edgellm
{
namespace kernel
{

namespace
{
constexpr int32_t kVec = 8;

struct alignas(16) Half8
{
    half v[kVec];
};

__global__ void scaledSwiGluKernel(Half8 const* __restrict__ gateUp, Half8* __restrict__ output,
    int64_t vecsPerRow, float scale)
{
    int64_t const row = blockIdx.x;
    Half8 const* const gateRow = gateUp + row * 2 * vecsPerRow;
    Half8 const* const upRow = gateRow + vecsPerRow;
    Half8* const outRow = output + row * vecsPerRow;
    for (int64_t i = threadIdx.x; i < vecsPerRow; i += blockDim.x)
    {
        Half8 const g = gateRow[i];
        Half8 const u = upRow[i];
        Half8 o;
#pragma unroll
        for (int32_t j = 0; j < kVec; ++j)
        {
            float const gf = __half2float(g.v[j]);
            float const silu = gf / (1.0F + __expf(-gf));
            o.v[j] = __float2half_rn(silu * __half2float(u.v[j]) * scale);
        }
        outRow[i] = o;
    }
}
} // namespace

void moeScaledSwiGlu(
    half const* gateUp, half* output, int64_t numRows, int64_t intermediateDim, float scale, cudaStream_t stream)
{
    check::check(intermediateDim > 0 && intermediateDim % kVec == 0,
        format::fmtstr("moeScaledSwiGlu: intermediateDim %ld must be a multiple of %d", intermediateDim, kVec));
    check::check(reinterpret_cast<uintptr_t>(gateUp) % 16 == 0 && reinterpret_cast<uintptr_t>(output) % 16 == 0,
        "moeScaledSwiGlu: pointers must be 16-byte aligned");
    if (numRows <= 0)
    {
        return;
    }
    int64_t const vecsPerRow = intermediateDim / kVec;
    int32_t const threads = static_cast<int32_t>(vecsPerRow < 256 ? vecsPerRow : 256);
    scaledSwiGluKernel<<<static_cast<unsigned int>(numRows), threads, 0, stream>>>(
        reinterpret_cast<Half8 const*>(gateUp), reinterpret_cast<Half8*>(output), vecsPerRow, scale);
}

} // namespace kernel
} // namespace trt_edgellm
