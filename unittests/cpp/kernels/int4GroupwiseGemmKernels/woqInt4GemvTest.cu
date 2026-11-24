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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/int4GroupwiseGemmKernels/int4GroupwiseGemm.h"
#include "naiveGemmReference.h"
#include "references.h"
#include "testUtils.h"

using namespace trt_edgellm;

/* TestInt4GroupwiseGemvAccuracy
 * -----------------------------------------------
 * Purpose:
 *   Sanity-check the numeric accuracy of our INT4, per-group dequantized
 *   GEMV CUDA kernels against a CPU reference.
 *
 * Inputs:
 *   m, n, k       : GEMV shapes (A: m×k, W: k×n, C: m×n) where m = 1.
 *   group_size    : Number of input channels per scale (per-group quantization), default = 128.
 *
 * Descriptions:
 * Randomly initialize input activations, weights and scales. For INT4 GEMV kernel, we pakced the weights according
 * to python reference implementation https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/qmodule.py#L26. For
 * the CPU reference, we use the unpacked weights in FP16, scaled them and compute reference results.
 */
void TestInt4GroupwiseGemvAccuracy(int m, int n, int k, int group_size)
{
    // Initialize input_arr, input_weights, input_scales
    std::vector<half> input_arr(static_cast<size_t>(m) * k);
    std::vector<int16_t> input_weights(static_cast<size_t>(k) * n);
    std::vector<half> input_scales(static_cast<size_t>(k / group_size) * n);

    uniformFloatInitialization(input_arr, 0.2f, 0.5f);
    uniformIntInitialization(input_weights, -5, 5);
    uniformFloatInitialization(input_scales, 0.2f, 0.5f);

    std::vector<half> scaled_weights;
    scaledWeightsReference(input_weights.data(), input_scales.data(), k, n, group_size, scaled_weights);
    std::vector<int16_t> W_kn_i8(input_weights.begin(), input_weights.end());
    std::vector<int16_t> Wpacked_Ndiv4xK((n / 4) * k);
    awqPackReference(W_kn_i8.data(), n, k, Wpacked_Ndiv4xK.data());
    std::vector<half> gpu_results(static_cast<size_t>(m) * n, __float2half(0.0f));
    std::vector<half> gpu_results_ref(static_cast<size_t>(m) * n, __float2half(0.0f));

    cudaStream_t stream{nullptr};
    half *d_act, *d_scales, *d_out, *d_out_ref, *d_weight_ref;
    int8_t* d_weight;

    // For WOQ kernel
    CUDA_CHECK(cudaMallocAsync(&d_act, (size_t) m * k * sizeof(__half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_weight, ((size_t) k * n / 2) * sizeof(int8_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scales, (size_t) (k / group_size) * n * sizeof(__half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_out, (size_t) m * n * sizeof(__half), stream));

    CUDA_CHECK(
        cudaMemcpyAsync(d_act, input_arr.data(), (size_t) m * k * sizeof(__half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weight, reinterpret_cast<int8_t const*>(Wpacked_Ndiv4xK.data()),
        ((size_t) k * n / 2) * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        d_scales, input_scales.data(), (size_t) (k / group_size) * n * sizeof(__half), cudaMemcpyHostToDevice, stream));

    // For GPU reference
    CUDA_CHECK(cudaMallocAsync(&d_weight_ref, ((size_t) k * n) * sizeof(__half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_out_ref, (size_t) m * n * sizeof(__half), stream));

    CUDA_CHECK(cudaMemcpyAsync(
        d_weight_ref, scaled_weights.data(), ((size_t) k * n) * sizeof(__half), cudaMemcpyHostToDevice, stream));

    // WOQ kernel inference
    trt_edgellm::kernel::gemv_forward_cuda_new(d_act, d_weight, d_scales, d_out, m, n, k, group_size, stream);

    // Reference GEMM inference
    naive_gemm_forward(d_act, d_weight_ref, d_out_ref, m, n, k, stream);

    CUDA_CHECK(
        cudaMemcpyAsync(gpu_results.data(), d_out, (size_t) m * n * sizeof(__half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        gpu_results_ref.data(), d_out_ref, (size_t) m * n * sizeof(__half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_act, stream));
    CUDA_CHECK(cudaFreeAsync(d_weight, stream));
    CUDA_CHECK(cudaFreeAsync(d_scales, stream));
    CUDA_CHECK(cudaFreeAsync(d_out, stream));
    CUDA_CHECK(cudaFreeAsync(d_out_ref, stream));

    // Compare (convert GPU fp16 to CPU result)
    int const total = m * n;
    float max_abs = 0;
    for (int i = 0; i < total; ++i)
    {
        float o = __half2float(gpu_results_ref.data()[i]);
        float r = __half2float(gpu_results.data()[i]);
        float ad = std::fabs(o - r);
        max_abs = std::max(max_abs, ad);
        EXPECT_TRUE(isclose(o, r, 1e-1, 1e-1));
    }
    std::printf("Largest abs error = %g\n", max_abs);
}

TEST(WOQInt4GemvTest, accuracyGemv)
{
    TestInt4GroupwiseGemvAccuracy(1, 384, 896, 128);
    TestInt4GroupwiseGemvAccuracy(1, 512, 896, 128);
    TestInt4GroupwiseGemvAccuracy(1, 640, 1024, 128);
}

// Small-N coverage: when N < CTA_N (=128) the plugin dispatcher falls back to GEMV
// even for M > 1 (split into <=6-row chunks).  Qwen3.5-4B's GDN in_proj_a / in_proj_b
// project hidden=2560 -> num_recurrent_heads=32, so N=32, K=2560 is the production
// case.  Exercise the kernel directly across M=1..6 (the template-instantiated range)
// to keep the fallback path covered.
TEST(WOQInt4GemvTest, accuracySmallNGemv)
{
    for (int m = 1; m <= 6; ++m)
    {
        TestInt4GroupwiseGemvAccuracy(m, 32, 2560, 128);
    }
}

// Shapes of the tuned launch configuration (METRICS 196): N >= 1024 runs 128-thread blocks, M = 1 with N % 16 == 0 runs 4 rows per thread pair; small N keeps 256 threads.
// The check is against a float64 CPU reference on the unpacked weights, with the error bounded relative to the output scale (the fp16 accumulation of the kernel is exact to about
// 1e-3 of the output RMS; the fp16 naive GEMM of the tests above is not a usable reference once K reaches a few thousand).
void TestInt4GemvAgainstDouble(int m, int n, int k)
{
    int const group_size = 128;
    std::vector<half> input_arr(static_cast<size_t>(m) * k);
    std::vector<int16_t> input_weights(static_cast<size_t>(k) * n);
    std::vector<half> input_scales(static_cast<size_t>(k / group_size) * n);
    uniformFloatInitialization(input_arr, -0.5f, 0.5f);
    uniformIntInitialization(input_weights, -5, 5);
    uniformFloatInitialization(input_scales, 0.2f, 0.5f);

    std::vector<int16_t> Wpacked_Ndiv4xK((n / 4) * k);
    awqPackReference(input_weights.data(), n, k, Wpacked_Ndiv4xK.data());

    std::vector<double> ref(static_cast<size_t>(m) * n, 0.0);
    double sumSq = 0.0;
    for (int r = 0; r < m; ++r)
    {
        for (int c = 0; c < n; ++c)
        {
            double acc = 0.0;
            for (int kk = 0; kk < k; ++kk)
            {
                acc += static_cast<double>(__half2float(input_arr[static_cast<size_t>(r) * k + kk]))
                    * input_weights[static_cast<size_t>(kk) * n + c]
                    * static_cast<double>(__half2float(input_scales[static_cast<size_t>(kk / group_size) * n + c]));
            }
            ref[static_cast<size_t>(r) * n + c] = acc;
            sumSq += acc * acc;
        }
    }
    double const rms = std::sqrt(sumSq / ref.size());

    cudaStream_t stream{nullptr};
    half *d_act, *d_scales, *d_out;
    int8_t* d_weight;
    CUDA_CHECK(cudaMalloc(&d_act, (size_t) m * k * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_weight, ((size_t) k * n / 2) * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_scales, (size_t) (k / group_size) * n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t) m * n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_act, input_arr.data(), (size_t) m * k * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, reinterpret_cast<int8_t const*>(Wpacked_Ndiv4xK.data()), ((size_t) k * n / 2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, input_scales.data(), (size_t) (k / group_size) * n * sizeof(__half), cudaMemcpyHostToDevice));
    trt_edgellm::kernel::gemv_forward_cuda_new(d_act, d_weight, d_scales, d_out, m, n, k, group_size, stream);
    std::vector<half> gpu(static_cast<size_t>(m) * n);
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_out, gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_act));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_out));

    double maxErr = 0.0, errSq = 0.0;
    for (size_t i = 0; i < gpu.size(); ++i)
    {
        double const e = std::fabs(static_cast<double>(__half2float(gpu[i])) - ref[i]);
        maxErr = std::max(maxErr, e);
        errSq += e * e;
    }
    double const relRms = std::sqrt(errSq / gpu.size()) / rms;
    std::printf("m=%d n=%d k=%d: output rms %.3f, max |err| %.4f (%.2e of rms), relative rms error %.2e\n", m, n, k, rms, maxErr, maxErr / rms, relRms);
    EXPECT_LT(relRms, 3e-3) << "m=" << m << " n=" << n << " k=" << k;
    EXPECT_LT(maxErr, 0.03 * rms) << "m=" << m << " n=" << n << " k=" << k;
}

// Inkling 2 attention projections (K = 2304 and 3072), Gemma-sized and Qwen-sized projections, M = 1..3 (M = 1 takes the 4-row path, M = 2 and 3 the 2-row path with 128 threads).
// N must be a multiple of 32 here: the reference packing of the tests does not handle other N (the 24-row sink projection of Inkling 2 runs in the engine and is covered by the KL checks).
TEST(WOQInt4GemvTest, accuracyTunedLaunchConfig)
{
    for (int m = 1; m <= 3; ++m)
    {
        TestInt4GemvAgainstDouble(m, 3072, 2304);
        TestInt4GemvAgainstDouble(m, 2304, 3072);
        TestInt4GemvAgainstDouble(m, 1024, 2304);
        TestInt4GemvAgainstDouble(m, 2560, 2560);
        TestInt4GemvAgainstDouble(m, 5120, 5120);
    }
}
