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
#include "inkling2HessianTapKernels.h"

#include <cuda_fp16.h>

namespace trt_edgellm::kernel
{
namespace
{
constexpr int32_t kTile = 64;
constexpr int32_t kRowsPerStep = 16;

// One 64x64 output tile per block (16x16 threads, 4x4 outputs each); the T rows stream through shared memory 16 at a time.
// grid = (K / 64, K / 64); blocks above the diagonal exit at once.
__global__ void hessianKernel(__half const* x, float* h, int64_t tokens, int32_t dim)
{
    int32_t const bx = blockIdx.x, by = blockIdx.y;
    if (by < bx)
    {
        return;
    }
    __shared__ float sa[kRowsPerStep][kTile + 1];
    __shared__ float sb[kRowsPerStep][kTile + 1];
    int32_t const tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float acc[4][4] = {};
    for (int64_t t0 = 0; t0 < tokens; t0 += kRowsPerStep)
    {
        for (int32_t i = threadIdx.x; i < kRowsPerStep * kTile; i += 256)
        {
            int32_t const r = i / kTile, c = i % kTile;
            int64_t const row = t0 + r;
            bool const in = row < tokens;
            sa[r][c] = in ? __half2float(x[row * dim + by * kTile + c]) : 0.F;
            sb[r][c] = in ? __half2float(x[row * dim + bx * kTile + c]) : 0.F;
        }
        __syncthreads();
#pragma unroll
        for (int32_t r = 0; r < kRowsPerStep; ++r)
        {
            float a[4], b[4];
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
                a[i] = sa[r][ty * 4 + i];
                b[i] = sb[r][tx * 4 + i];
            }
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
#pragma unroll
                for (int32_t j = 0; j < 4; ++j)
                {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
#pragma unroll
        for (int32_t j = 0; j < 4; ++j)
        {
            h[static_cast<int64_t>(by * kTile + ty * 4 + i) * dim + bx * kTile + tx * 4 + j] += acc[i][j];
        }
    }
}
} // namespace

void launchInkling2HessianAccumulate(void const* x, float* hessian, int64_t tokens, int32_t dim, cudaStream_t stream)
{
    if (tokens <= 0)
    {
        return;
    }
    dim3 const grid(dim / kTile, dim / kTile);
    hessianKernel<<<grid, 256, 0, stream>>>(static_cast<__half const*>(x), hessian, tokens, dim);
}
} // namespace trt_edgellm::kernel
