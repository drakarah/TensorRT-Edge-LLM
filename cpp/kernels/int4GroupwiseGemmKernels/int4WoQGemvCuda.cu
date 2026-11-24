/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * Copyright (c) 2023 MIT HAN Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * reference: https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization_new/gemv/gemv_cuda.cu
 */

#include "common/checkMacros.h"
#include "dequantize.cuh"
#include <cstdlib>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128

namespace trt_edgellm
{
namespace kernel
{

// Reduce sum within the warp using the tree reduction algorithm.
template <int Num>
__device__ __forceinline__ static void warp_reduce(half* psum, float (*out_smem)[Num * 4])
{
    // kInterleave = 4
    float fpsum[Num];
#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
        fpsum[i] = static_cast<float>(psum[i]);
    }

#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
        // T0 + T1 + T8 + T9 + T16 + T17 + T24 + T25 (kInterleave = 4)
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 16);
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 8);
        fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 1);
    }
    __syncthreads();
    int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if (lane == 0 || lane == 2 || lane == 4 || lane == 6)
    {
#pragma unroll
        for (int i = 0; i < Num; ++i)
        {
            out_smem[warp][i * 4 + lane / 2] = fpsum[i];
        }
    }
    __syncthreads();
};

// The block size is a launch parameter (a multiple of 32, at most 1024): the per-row arithmetic does not depend on it except for how the K range is split over the
// threads (fp16 partial sums per thread, fp32 across threads and warps), so 256 threads with 2 rows reproduces the original MIT-AWQ kernel bit for bit.
// NPerBlock rows per thread pair (2 or 4): 4 keeps twice as many weight loads in flight per thread.
template <int NPerBlock, int Batch>
__global__ void __launch_bounds__(1024) gemv_kernel(
    half const* inputs, uint32_t const* weight, half const* scales, half* outputs, int const IC, int const OC)
{
    constexpr int GroupSize = 128;
    int const BlockSize = blockDim.x;
    int const kStride = 64;
    int const kElemsPerThread = MEM_ACCESS_SIZE / 4;
    int const kThreadsNumPerTile = kStride / kElemsPerThread;
    // assert(MEM_ACCESS_SIZE == 128);

    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 4;
    static constexpr int kShuffleStrided = 4;

    constexpr int Num = NPerBlock * Batch;
    constexpr int kInterleave = 4;

    half local_inputs[kElemsPerThread];
    uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
    half half_weight_buffer[kElemsPerThread];
    half dequantized_weight[kElemsPerThread * NPerBlock];
    half local_scale[NPerBlock];

    half psum[Num];
    for (int i = 0; i < Num; ++i)
        psum[i] = static_cast<half>(0.f);

    __shared__ float out_smem[1024 / WARP_SIZE][Num * kInterleave];

    int const blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
    int const thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
    int const act_k_offset = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride
        + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    int const group_offset = act_k_offset / GroupSize;
    uint32_t const* blk_weight_ptr = weight + static_cast<size_t>(blk_row_offset) * IC / PACK_FACTOR;
    half const* scale_ptr = scales + blk_row_offset + thd_row_offset + static_cast<size_t>(group_offset) * OC;
    half const* inputs_ptr = inputs + act_k_offset;

    int const act_forward_step = BlockSize * kElemsPerThread / kInterleave;
    int const scale_forward_step = act_forward_step / GroupSize * OC;

    // Main loop iteration, each block completes the outputs for several OCs
    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
    {
// Load qweight, scales and scaled_zeros
#pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx)
        {
            // use float4 to load weights, each thread load 32 int4 numbers (1 x float4, 128 bit)
            *((float4*) (local_qweights))
                = *((float4*) (blk_weight_ptr + (static_cast<size_t>(idx) * kInterleave * IC + kk) / PACK_FACTOR));
            local_scale[idx] = *(scale_ptr + idx * kInterleave);

// Map int4 qweight to fp format
#pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
            {
                // Converts 32 bits (8 x int4) to 8 fp16
                dequantize_s4_to_fp16x2(*reinterpret_cast<half2*>(local_qweights + i),
                    reinterpret_cast<uint4*>(half_weight_buffer + i * PACK_FACTOR));
            }

// Dequantize (apply s/z) and shuffle elements to match the weight packing format
#pragma unroll
            for (int i = 0; i < kShuffleContinous; ++i)
            {
#pragma unroll
                for (int j = 0; j < kShuffleStrided; ++j)
                {
                    half2 w = *reinterpret_cast<half2*>(
                        half_weight_buffer + (i + j * kShuffleContinous) * kShuffleBasicTile);
                    w = __hmul2(w, __half2half2(local_scale[idx]));
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0) * NPerBlock + idx] = w.x;
                    dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1) * NPerBlock + idx] = w.y;
                }
            }
        }
#pragma unroll
        for (int batch_idx = 0; batch_idx < Batch; ++batch_idx)
        {
            half const* local_inputs_ptr = inputs_ptr + batch_idx * IC;
#pragma unroll
            for (int idx = 0; idx < kElemsPerThread / 8; ++idx)
            {
                // load activation, 8 halves (128 bits) / step.
                *((float4*) (local_inputs + idx * 8)) = *((float4*) (local_inputs_ptr + idx * 8));
            }
// Perform the MACs
#pragma unroll
            for (int x = 0; x < NPerBlock / 2; ++x)
            {
#pragma unroll
                for (int y = 0; y < kElemsPerThread; ++y)
                {
                    *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2) = __hfma2(
                        *reinterpret_cast<half2*>(dequantized_weight + y * NPerBlock + x * 2),
                        __half2half2(local_inputs[y]), *reinterpret_cast<half2*>(psum + batch_idx * NPerBlock + x * 2));
                }
            }
        }
        inputs_ptr += act_forward_step;
        scale_ptr += scale_forward_step;
    }

    warp_reduce<Num>(psum, out_smem);

    // Num * Interleave = batch * NPerBlock * Interleave -> 1 thread_block write back num
    for (int i = threadIdx.x; i < Num * kInterleave; i += BlockSize)
    {
        int batch_idx = i / (NPerBlock * kInterleave);
        int oc_idx = i % (NPerBlock * kInterleave);
        float acc = 0.f;
        for (int j = 0; j < BlockSize / WARP_SIZE; ++j)
        {
            acc += out_smem[j][i];
        }
        outputs[static_cast<size_t>(batch_idx) * OC + blk_row_offset + oc_idx] = static_cast<half>(acc);
    }
}

namespace
{
//! EDGELLM_INT4_GEMV_LEGACY=1 restores the original launch configuration (256 threads, 2 rows per thread pair) for every shape.
bool legacyGemvConfig()
{
    static bool const legacy = []
    {
        char const* const value = std::getenv("EDGELLM_INT4_GEMV_LEGACY");
        return value != nullptr && std::string(value) == "1";
    }();
    return legacy;
}

template <int NPerBlock, int Batch>
void launchGemv(half const* in_feats, uint32_t const* weights, half const* scaling_factors, half* out_feats, int n, int k,
    int block, cudaStream_t stream)
{
    static constexpr int K_INTERLEAVE = 4;
    gemv_kernel<NPerBlock, Batch><<<n / NPerBlock / K_INTERLEAVE, block, 0, stream>>>(
        in_feats, weights, scaling_factors, out_feats, k, n);
}
} // namespace

//! Launch configuration (measured on the Orin, METRICS 196, 11 shapes of Inkling 2, Qwen3.8 and Gemma 4 at M = 1 and 3): blocks of 128 threads beat 256 for every
//! shape with N >= 1024 (3-18%), and 4 rows per thread pair help at M = 1 when N is a multiple of 16; small N (a router or sink projection) stays on the original
//! configuration, which is already at the launch-latency floor.
void gemv_forward_cuda_new(half const* in_feats, int8_t const* weights_device, half const* scaling_factors,
    half* out_feats, int m, int n, int k, int group_size, cudaStream_t stream)
{
    auto kernel = reinterpret_cast<uint32_t const*>(weights_device);
    ELLM_CHECK(group_size == 128,
        "Unsupported group size " + std::to_string(group_size) + " for gemv kernel; only 128 is supported.");

    bool const tuned = !legacyGemvConfig() && n >= 1024;
    int const block = tuned ? 128 : 256;
    bool const fourRows = tuned && m == 1 && n % 16 == 0;
    switch (m)
    {
    case 1:
        if (fourRows)
        {
            launchGemv<4, 1>(in_feats, kernel, scaling_factors, out_feats, n, k, block, stream);
        }
        else
        {
            launchGemv<2, 1>(in_feats, kernel, scaling_factors, out_feats, n, k, block, stream);
        }
        break;
    case 2: launchGemv<2, 2>(in_feats, kernel, scaling_factors, out_feats, n, k, block, stream); break;
    case 3: launchGemv<2, 3>(in_feats, kernel, scaling_factors, out_feats, n, k, block, stream); break;
    case 4: launchGemv<2, 4>(in_feats, kernel, scaling_factors, out_feats, n, k, block, stream); break;
    case 5: launchGemv<2, 5>(in_feats, kernel, scaling_factors, out_feats, n, k, block, stream); break;
    case 6: launchGemv<2, 6>(in_feats, kernel, scaling_factors, out_feats, n, k, block, stream); break;
    default: throw std::runtime_error("Unsupported batch size for gemv kernel.\n");
    }
}

} // namespace kernel
} // namespace trt_edgellm