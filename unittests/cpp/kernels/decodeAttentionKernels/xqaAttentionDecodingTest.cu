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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "references.h"
#include "testUtils.h"
#include "xqaJitTestUtils.h"

#include <algorithm>
#include <cmath>
#include <optional>

using namespace nvinfer1;
using namespace trt_edgellm;

//! Split-KV (multi-block) count for the next TestXQAAttentionDecodingAccuracy call; 1 = single block.
uint32_t gMultiBlockSplits = 1;

void TestXQAAttentionDecodingAccuracy(int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t kvCacheCapacity, bool useFp8Cache = false, int32_t slidingWindowSize = 0,
    std::optional<float> attentionScale = std::nullopt, int32_t fixedContextLen = 0,
    std::optional<std::vector<float>> const& attentionSinks = std::nullopt)
{
    ASSERT_FALSE(attentionSinks.has_value() && attentionSinks->size() != static_cast<size_t>(numQHeads));
    float const resolvedAttentionScale = attentionScale.value_or(1.0F / std::sqrt(static_cast<float>(headSize)));
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (useFp8Cache && smVersion < 89)
    {
        GTEST_SKIP() << "Skipping FP8 KV cache tests: requires SM >= 89, but got SM " << smVersion;
    }
    // Decoding attention length always set qSequenceLength to 1
    constexpr int qSequenceLength = 1;

    std::vector<int32_t> kvCacheLengths(batchSize);
    uniformIntInitialization(kvCacheLengths, kvCacheCapacity / 4, kvCacheCapacity);
    if (fixedContextLen > 0)
    {
        // Deterministic context length to reproduce reported failure shapes.
        ASSERT_LE(fixedContextLen, kvCacheCapacity);
        std::fill(kvCacheLengths.begin(), kvCacheLengths.end(), fixedContextLen);
    }
    if (slidingWindowSize > 0)
    {
        // Cover KV lengths greater than, equal to, and smaller than the sliding window.
        for (int32_t i = 0; i < batchSize; ++i)
        {
            int32_t targetLength = kvCacheCapacity;
            if (i % 3 == 0)
            {
                targetLength = slidingWindowSize + 37;
            }
            else if (i % 3 == 1)
            {
                targetLength = slidingWindowSize;
            }
            else
            {
                targetLength = std::max(1, slidingWindowSize / 2);
            }
            kvCacheLengths[i] = std::min(kvCacheCapacity, targetLength);
        }
    }

    std::vector<half> qInput;
    // Initialize KVCahce buffer to full capacity.
    std::vector<half> kvInput(batchSize * 2 * numKVHeads * kvCacheCapacity * headSize, 0.F);
    std::vector<half> outReference;

    for (int32_t i = 0; i < batchSize; i++)
    {
        int32_t kvLength = kvCacheLengths[i];
        std::vector<half> qi(numQHeads * headSize * qSequenceLength);
        std::vector<half> ki(numKVHeads * headSize * kvLength);
        std::vector<half> vi(numKVHeads * headSize * kvLength);
        if (attentionSinks.has_value())
        {
            constexpr int32_t kMODULUS = 257;
            constexpr float kSCALE = 1.0F / 128.0F;
            for (size_t idx = 0; idx < qi.size(); ++idx)
            {
                qi[idx] = __float2half(
                    static_cast<float>((static_cast<int32_t>(idx % kMODULUS) * 37 + i * 17) % kMODULUS - 128) * kSCALE);
            }
            for (size_t idx = 0; idx < ki.size(); ++idx)
            {
                ki[idx] = __float2half(
                    static_cast<float>((static_cast<int32_t>(idx % kMODULUS) * 29 + i * 19) % kMODULUS - 128) * kSCALE);
                vi[idx] = __float2half(
                    static_cast<float>((static_cast<int32_t>(idx % kMODULUS) * 31 + i * 23) % kMODULUS - 128) * kSCALE);
            }
        }
        else
        {
            uniformFloatInitialization(qi, -1.0F, 1.0F);
            uniformFloatInitialization(ki, -1.0F, 1.0F);
            uniformFloatInitialization(vi, -1.0F, 1.0F);
        }

        int32_t const attentionLength = slidingWindowSize > 0 ? std::min(kvLength, slidingWindowSize) : kvLength;
        auto kiRef = sliceKVWindow(ki, numKVHeads, headSize, kvLength, slidingWindowSize);
        auto viRef = sliceKVWindow(vi, numKVHeads, headSize, kvLength, slidingWindowSize);
        auto ref = casualAttentionRef<half>(qi, kiRef, viRef, qSequenceLength, attentionLength, numQHeads, numKVHeads,
            headSize, resolvedAttentionScale, std::nullopt, 1.0F, 1.0F, 0, /*contiguousQuerySwa=*/false,
            attentionSinks);

        // Add data from batch to input Tensors
        qInput.insert(qInput.end(), qi.begin(), qi.end());

        // Add KV data to KVCache buffer, layout assumed to be [B, 2, Hkv, S, D]
        int32_t const batchOffset = i * 2 * numKVHeads * kvCacheCapacity * headSize;
        int32_t const vOffset = numKVHeads * kvCacheCapacity * headSize;
        for (int32_t hkv = 0; hkv < numKVHeads; hkv++)
        {
            for (int32_t skv = 0; skv < kvLength; skv++)
            {
                for (int32_t d = 0; d < headSize; d++)
                {
                    kvInput[batchOffset + hkv * kvCacheCapacity * headSize + skv * headSize + d]
                        = ki[hkv * kvLength * headSize + skv * headSize + d];
                    kvInput[batchOffset + vOffset + hkv * kvCacheCapacity * headSize + skv * headSize + d]
                        = vi[hkv * kvLength * headSize + skv * headSize + d];
                }
            }
        }
        outReference.insert(outReference.end(), ref.begin(), ref.end());
    }
    // Prepare device memory for kernel execution.
    thrust::device_vector<half> qInputDevice(qInput);
    thrust::device_vector<half> kvInputDevice(kvInput);
    thrust::device_vector<half> outDevice(outReference.size(), 0.0F);
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLengths);
    thrust::device_vector<float> attentionSinksDevice(attentionSinks.value_or(std::vector<float>{}));

    EXPECT_TRUE(
        trt_edgellm::canCompileXQAKernel(numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kHALF));
    ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(
        smVersion, DataType::kHALF, DataType::kHALF, headSize, numQHeads, numKVHeads, slidingWindowSize > 0, false));
    trt_edgellm::DecoderXQARunner runner(
        DataType::kHALF, DataType::kHALF, batchSize, numQHeads, numKVHeads, headSize, smVersion);
    auto params = runner.initXQAParams();
    params.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvInputDevice.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    params.kvCache.capacity = kvCacheCapacity;
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.attentionSinks
        = attentionSinks.has_value() ? thrust::raw_pointer_cast(attentionSinksDevice.data()) : nullptr;
    params.attentionScale = resolvedAttentionScale;
    params.slidingWinSize = slidingWindowSize > 0 ? static_cast<uint32_t>(slidingWindowSize) : 0U;
    thrust::device_vector<int32_t> semaphoresDevice(gMultiBlockSplits > 1 ? batchSize * numKVHeads : 0, 0);
    size_t const scratchBytes = gMultiBlockSplits > 1
        ? trt_edgellm::xqaMultiBlockScratchUpperBound(headSize, static_cast<size_t>(batchSize) * numKVHeads * gMultiBlockSplits)
        : 0;
    thrust::device_vector<int8_t> scratchDevice(scratchBytes, 0);
    if (gMultiBlockSplits > 1)
    {
        params.nbSubSeqPerSeq = gMultiBlockSplits;
        params.semaphores = thrust::raw_pointer_cast(semaphoresDevice.data());
        params.scratch = thrust::raw_pointer_cast(scratchDevice.data());
        params.scratchBytes = scratchBytes;
    }

    // Use default stream .
    cudaStream_t stream{nullptr};
    runner.dispatchXQAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Check accuracy.
    thrust::host_vector<half> outHost(outDevice.size());
    thrust::copy(outDevice.begin(), outDevice.end(), outHost.begin());

    bool NanValueDetected = false;
    int32_t numErrorWithin1E_3 = 0;
    for (int32_t i = 0; i < batchSize * numQHeads * headSize; ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outReference[i], 1e-2, 1e-2));
        if (isclose(outHost[i], outReference[i], 1e-3, 1e-3))
        {
            numErrorWithin1E_3++;
        }
        if (isnan(__half2float(outHost[i])) || isinf(__half2float(outHost[i])))
        {
            NanValueDetected = true;
        }
    }
    float passRate1E_3 = static_cast<float>(numErrorWithin1E_3) / (batchSize * numQHeads * headSize);

    std::cout << "XQA Attention Decoding test. [FP16 KV cache] batch_size: " << batchSize
              << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
              << " sliding_window: " << slidingWindowSize << " kvcache lengths: " << kvCacheLengths
              << " attention_scale: " << resolvedAttentionScale << " pass_rate_1e-3: " << passRate1E_3 << std::endl;
    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);

#if SUPPORTS_FP8
    if (useFp8Cache)
    {
        // Compute FP8 amax-based scale for KV cache
        float kAmax = 0.0F;
        float vAmax = 0.0F;
        int32_t const kvStrideHalf = numKVHeads * kvCacheCapacity * headSize; // elements per K or V per batch
        for (int32_t b = 0; b < batchSize; ++b)
        {
            size_t const batchBase = static_cast<size_t>(b) * 2 * kvStrideHalf;
            size_t const vBase = batchBase + kvStrideHalf;
            for (int32_t idx = 0; idx < kvStrideHalf; ++idx)
            {
                kAmax = std::max(kAmax, std::fabs(__half2float(kvInput[batchBase + idx])));
                vAmax = std::max(vAmax, std::fabs(__half2float(kvInput[vBase + idx])));
            }
        }

        // FP8 E4M3 max finite value
        constexpr float FP8_E4M3_MAX = 448.0F;
        assert(kAmax > 0.0F && vAmax > 0.0F);
        float const kScaleQuantOrig = kAmax / FP8_E4M3_MAX;
        float const vScaleQuantOrig = vAmax / FP8_E4M3_MAX;
        float const kScaleOrigQuant = 1.0F / kScaleQuantOrig;
        float const vScaleOrigQuant = 1.0F / vScaleQuantOrig;

        // FP8 decode path: quantize KV cache to FP8 using computed scale and compare against FP16 decoding outputs.
        std::vector<__nv_fp8_e4m3> kvInputFp8(kvInput.size());
        for (int32_t b = 0; b < batchSize; ++b)
        {
            size_t const batchBase = static_cast<size_t>(b) * 2 * kvStrideHalf;
            size_t const vBase = batchBase + kvStrideHalf;
            for (int32_t idx = 0; idx < kvStrideHalf; ++idx)
            {
                kvInputFp8[batchBase + idx] = __nv_fp8_e4m3(__half2float(kvInput[batchBase + idx]) * kScaleOrigQuant);
                kvInputFp8[vBase + idx] = __nv_fp8_e4m3(__half2float(kvInput[vBase + idx]) * vScaleOrigQuant);
            }
        }

        std::vector<half> outReferenceFp8;
        int32_t const qOffset = numQHeads * headSize * qSequenceLength;
        int32_t const batchKvStride = 2 * numKVHeads * kvCacheCapacity * headSize;
        int32_t const vRegionOffset = numKVHeads * kvCacheCapacity * headSize;
        for (int32_t i = 0; i < batchSize; i++)
        {
            int32_t kvLength = kvCacheLengths[i];

            // Reconstruct qi from the flattened qInput buffer.
            std::vector<half> qi(numQHeads * headSize * qSequenceLength);
            std::copy_n(qInput.begin() + i * qOffset, qOffset, qi.begin());

            // Reconstruct compact K/V (shape [Hkv, kvLength, D]) from KV cache layout
            // kvInput layout per batch: [2, Hkv, S=kvCacheCapacity, D]
            std::vector<__nv_fp8_e4m3> ki(numKVHeads * headSize * kvLength);
            std::vector<__nv_fp8_e4m3> vi(numKVHeads * headSize * kvLength);

            int32_t const batchOffset = i * batchKvStride;
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                for (int32_t skv = 0; skv < kvLength; ++skv)
                {
                    for (int32_t d = 0; d < headSize; ++d)
                    {
                        int32_t const compactIdx = hkv * kvLength * headSize + skv * headSize + d;

                        int32_t const kCacheIdx = batchOffset + hkv * kvCacheCapacity * headSize + skv * headSize + d;
                        int32_t const vCacheIdx
                            = batchOffset + vRegionOffset + hkv * kvCacheCapacity * headSize + skv * headSize + d;

                        ki[compactIdx] = kvInputFp8[kCacheIdx];
                        vi[compactIdx] = kvInputFp8[vCacheIdx];
                    }
                }
            }

            int32_t const attentionLength = slidingWindowSize > 0 ? std::min(kvLength, slidingWindowSize) : kvLength;
            auto kiRef = sliceKVWindow(ki, numKVHeads, headSize, kvLength, slidingWindowSize);
            auto viRef = sliceKVWindow(vi, numKVHeads, headSize, kvLength, slidingWindowSize);
            auto ref = casualAttentionRef<__nv_fp8_e4m3>(qi, kiRef, viRef, qSequenceLength, attentionLength, numQHeads,
                numKVHeads, headSize, resolvedAttentionScale, std::nullopt, kScaleQuantOrig, vScaleQuantOrig, 0,
                /*contiguousQuerySwa=*/false, attentionSinks);
            outReferenceFp8.insert(outReferenceFp8.end(), ref.begin(), ref.end());
        }

        thrust::device_vector<__nv_fp8_e4m3> kvInputFp8Device(kvInputFp8);
        thrust::device_vector<half> outFp8Device(batchSize * numQHeads * headSize, __float2half(0.0F));
        EXPECT_TRUE(trt_edgellm::canCompileXQAKernel(
            numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kFP8));
        ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(
            smVersion, DataType::kHALF, DataType::kFP8, headSize, numQHeads, numKVHeads, slidingWindowSize > 0, false));
        trt_edgellm::DecoderXQARunner runnerFp8(
            DataType::kHALF, DataType::kFP8, batchSize, numQHeads, numKVHeads, headSize, smVersion);
        auto paramsFp8 = runnerFp8.initXQAParams();
        paramsFp8.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
        paramsFp8.kvCache.data = thrust::raw_pointer_cast(kvInputFp8Device.data());
        paramsFp8.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
        paramsFp8.kvCache.capacity = kvCacheCapacity;
        paramsFp8.output = thrust::raw_pointer_cast(outFp8Device.data());
        paramsFp8.attentionSinks
            = attentionSinks.has_value() ? thrust::raw_pointer_cast(attentionSinksDevice.data()) : nullptr;
        paramsFp8.attentionScale = resolvedAttentionScale;
        paramsFp8.kScale = kScaleQuantOrig;
        paramsFp8.vScale = vScaleQuantOrig;
        paramsFp8.slidingWinSize = slidingWindowSize > 0 ? static_cast<uint32_t>(slidingWindowSize) : 0U;

        // Reuse the same stream used for FP16 decoding.
        cudaStream_t stream{nullptr};
        runnerFp8.dispatchXQAKernel(paramsFp8, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());

        thrust::host_vector<half> outFp8Host(outFp8Device.size());
        thrust::copy(outFp8Device.begin(), outFp8Device.end(), outFp8Host.begin());

        // Compare FP8 vs FP16 decoding outputs.
        ASSERT_EQ(outReferenceFp8.size(), outFp8Host.size());
        int32_t numClose = 0;
        float maxAbsDiff = 0.0F;
        bool NanValueDetectedFp8 = false;
        for (int32_t i = 0; i < static_cast<int32_t>(outReferenceFp8.size()); ++i)
        {
            float const v16 = __half2float(outReferenceFp8[i]);
            float const v8 = __half2float(outFp8Host[i]);
            float const absDiff = std::fabs(v16 - v8);
            maxAbsDiff = std::max(maxAbsDiff, absDiff);

            if (isclose(outFp8Host[i], outReferenceFp8[i], 1e-3, 1e-3))
            {
                numClose++;
            }
            if (!std::isfinite(__half2float(outFp8Host[i])))
            {
                NanValueDetectedFp8 = true;
            }
            EXPECT_TRUE(std::isfinite(__half2float(outFp8Host[i])));
        }
        float const matchRate = static_cast<float>(numClose) / static_cast<float>(outHost.size());
        std::cout << "XQA Attention Decoding test. [FP8 KV cache] batch_size: " << batchSize
                  << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
                  << " sliding_window: " << slidingWindowSize << " kvcache lengths: " << kvCacheLengths
                  << " attention_scale: " << resolvedAttentionScale << " pass_rate_1e-3: " << passRate1E_3 << std::endl;
        EXPECT_GT(matchRate, 0.9);
        EXPECT_FALSE(NanValueDetectedFp8);
    }
#else
    (void) useFp8Cache;
#endif
}

//! INT8 KV cache decode: K/V are quantized per tensor with scale = amax / 127. The kernel output is checked
//! (1) against the reference on the dequantized INT8 K/V with the FP16-path tolerances, which isolates the
//! kernel's scale handling, and (2) against the FP16 reference on the original K/V within one V quantization
//! step: the V rounding error is at most vScale / 2 per element and the softmax weights sum to one, while the
//! K rounding error only perturbs the logits by O(attentionScale * kScale).
void TestXQAAttentionDecodingInt8Accuracy(
    int32_t batchSize, int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t kvCacheCapacity)
{
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(headSize));
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    constexpr int32_t qSequenceLength = 1;

    std::vector<int32_t> kvCacheLengths(batchSize);
    uniformIntInitialization(kvCacheLengths, kvCacheCapacity / 4, kvCacheCapacity);

    // KV cache layout [B, 2, Hkv, S, D]; slots past each sequence length stay zero.
    int32_t const kvStride = numKVHeads * kvCacheCapacity * headSize;
    std::vector<half> qInput(static_cast<size_t>(batchSize) * numQHeads * headSize);
    std::vector<half> kvInput(static_cast<size_t>(batchSize) * 2 * kvStride, __float2half(0.0F));
    uniformFloatInitialization(qInput, -1.0F, 1.0F);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t hkv = 0; hkv < 2 * numKVHeads; ++hkv)
        {
            std::vector<half> values(static_cast<size_t>(kvCacheLengths[b]) * headSize);
            uniformFloatInitialization(values, -1.0F, 1.0F);
            std::copy(values.begin(), values.end(),
                kvInput.begin() + static_cast<size_t>(b) * 2 * kvStride
                    + static_cast<size_t>(hkv) * kvCacheCapacity * headSize);
        }
    }

    float kAmax = 0.0F;
    float vAmax = 0.0F;
    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t idx = 0; idx < kvStride; ++idx)
        {
            kAmax = std::max(kAmax, std::fabs(__half2float(kvInput[static_cast<size_t>(b) * 2 * kvStride + idx])));
            vAmax = std::max(
                vAmax, std::fabs(__half2float(kvInput[static_cast<size_t>(b) * 2 * kvStride + kvStride + idx])));
        }
    }
    ASSERT_GT(kAmax, 0.0F);
    ASSERT_GT(vAmax, 0.0F);
    float const kScaleQuantOrig = kAmax / 127.0F;
    float const vScaleQuantOrig = vAmax / 127.0F;
    std::vector<int8_t> kvInputInt8(kvInput.size());
    for (size_t idx = 0; idx < kvInput.size(); ++idx)
    {
        bool const isV = (idx / kvStride) % 2 == 1;
        kvInputInt8[idx]
            = quantizeInt8Symmetric(__half2float(kvInput[idx]), 1.0F / (isV ? vScaleQuantOrig : kScaleQuantOrig));
    }

    std::vector<half> outReference;
    std::vector<half> outReferenceInt8;
    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t const kvLength = kvCacheLengths[b];
        std::vector<half> qi(qInput.begin() + static_cast<size_t>(b) * numQHeads * headSize,
            qInput.begin() + static_cast<size_t>(b + 1) * numQHeads * headSize);
        // Compact [Hkv, kvLength, D] views of the cache for the references.
        std::vector<half> ki(static_cast<size_t>(numKVHeads) * kvLength * headSize);
        std::vector<half> vi(ki.size());
        std::vector<int8_t> kiInt8(ki.size());
        std::vector<int8_t> viInt8(ki.size());
        for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
        {
            for (int32_t skv = 0; skv < kvLength; ++skv)
            {
                for (int32_t d = 0; d < headSize; ++d)
                {
                    size_t const compactIdx = (static_cast<size_t>(hkv) * kvLength + skv) * headSize + d;
                    size_t const kIdx = static_cast<size_t>(b) * 2 * kvStride
                        + (static_cast<size_t>(hkv) * kvCacheCapacity + skv) * headSize + d;
                    size_t const vIdx = kIdx + kvStride;
                    ki[compactIdx] = kvInput[kIdx];
                    vi[compactIdx] = kvInput[vIdx];
                    kiInt8[compactIdx] = kvInputInt8[kIdx];
                    viInt8[compactIdx] = kvInputInt8[vIdx];
                }
            }
        }
        auto const ref = casualAttentionRef<half>(
            qi, ki, vi, qSequenceLength, kvLength, numQHeads, numKVHeads, headSize, attentionScale);
        auto const refInt8 = casualAttentionRef<int8_t>(qi, kiInt8, viInt8, qSequenceLength, kvLength, numQHeads,
            numKVHeads, headSize, attentionScale, std::nullopt, kScaleQuantOrig, vScaleQuantOrig);
        outReference.insert(outReference.end(), ref.begin(), ref.end());
        outReferenceInt8.insert(outReferenceInt8.end(), refInt8.begin(), refInt8.end());
    }

    thrust::device_vector<half> qInputDevice(qInput);
    thrust::device_vector<int8_t> kvInputInt8Device(kvInputInt8);
    thrust::device_vector<half> outDevice(outReference.size(), __float2half(0.0F));
    thrust::device_vector<int32_t> kvCacheLengthDevice(kvCacheLengths);

    ASSERT_TRUE(
        trt_edgellm::canCompileXQAKernel(numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, DataType::kINT8));
    ASSERT_TRUE(trt_edgellm::loadXQAJitKernelForTest(
        smVersion, DataType::kHALF, DataType::kINT8, headSize, numQHeads, numKVHeads, false, false));
    trt_edgellm::DecoderXQARunner runner(
        DataType::kHALF, DataType::kINT8, batchSize, numQHeads, numKVHeads, headSize, smVersion);
    auto params = runner.initXQAParams();
    params.qInputPtr = thrust::raw_pointer_cast(qInputDevice.data());
    params.kvCache.data = thrust::raw_pointer_cast(kvInputInt8Device.data());
    params.kvCache.sequence_lengths = thrust::raw_pointer_cast(kvCacheLengthDevice.data());
    params.kvCache.capacity = kvCacheCapacity;
    params.output = thrust::raw_pointer_cast(outDevice.data());
    params.attentionScale = attentionScale;
    params.kScale = kScaleQuantOrig;
    params.vScale = vScaleQuantOrig;

    cudaStream_t stream{nullptr};
    runner.dispatchXQAKernel(params, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    thrust::host_vector<half> outHost(outDevice.size());
    thrust::copy(outDevice.begin(), outDevice.end(), outHost.begin());

    ASSERT_EQ(outHost.size(), outReferenceInt8.size());
    int32_t numErrorWithin1E_3 = 0;
    float maxAbsDiffFp16 = 0.0F;
    for (size_t i = 0; i < outHost.size(); ++i)
    {
        ASSERT_TRUE(std::isfinite(__half2float(outHost[i]))) << "non-finite output at " << i;
        EXPECT_TRUE(isclose(outHost[i], outReferenceInt8[i], 1e-2, 1e-2))
            << "INT8 reference mismatch at " << i << ": got " << __half2float(outHost[i]) << ", expected "
            << __half2float(outReferenceInt8[i]);
        if (isclose(outHost[i], outReferenceInt8[i], 1e-3, 1e-3))
        {
            numErrorWithin1E_3++;
        }
        float const absDiffFp16 = std::fabs(__half2float(outHost[i]) - __half2float(outReference[i]));
        maxAbsDiffFp16 = std::max(maxAbsDiffFp16, absDiffFp16);
        EXPECT_LE(absDiffFp16, vScaleQuantOrig)
            << "FP16 reference mismatch at " << i << ": got " << __half2float(outHost[i]) << ", expected "
            << __half2float(outReference[i]);
    }
    float const passRate1E_3 = static_cast<float>(numErrorWithin1E_3) / static_cast<float>(outHost.size());

    std::cout << "XQA Attention Decoding test. [INT8 KV cache] batch_size: " << batchSize
              << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
              << " kvcache lengths: " << kvCacheLengths << " k_scale: " << kScaleQuantOrig
              << " v_scale: " << vScaleQuantOrig << " pass_rate_1e-3: " << passRate1E_3
              << " max_abs_diff_vs_fp16_ref: " << maxAbsDiffFp16 << std::endl;
    EXPECT_GT(passRate1E_3, 0.9);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio3)
{
    TestXQAAttentionDecodingAccuracy(1, 24, 8, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 24, 8, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 24, 8, 128, 256);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio4)
{
    TestXQAAttentionDecodingAccuracy(1, 32, 8, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 32, 8, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 32, 8, 128, 256);
    TestXQAAttentionDecodingAccuracy(1, 32, 8, 64, 2048);
    TestXQAAttentionDecodingAccuracy(4, 16, 4, 64, 512);
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 256, 1024);
    TestXQAAttentionDecodingAccuracy(1, 16, 4, 256, 1024);
    TestXQAAttentionDecodingAccuracy(2, 16, 4, 256, 512);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio5)
{
    TestXQAAttentionDecodingAccuracy(1, 40, 8, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 40, 8, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 40, 8, 128, 512);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio7)
{
    TestXQAAttentionDecodingAccuracy(1, 28, 4, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 28, 4, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 28, 4, 128, 256);
    TestXQAAttentionDecodingAccuracy(1, 28, 4, 64, 1024);
    TestXQAAttentionDecodingAccuracy(4, 14, 2, 64, 512);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio8)
{
    TestXQAAttentionDecodingAccuracy(1, 32, 4, 128, 1024);
    TestXQAAttentionDecodingAccuracy(2, 32, 4, 128, 512);
    TestXQAAttentionDecodingAccuracy(4, 32, 4, 128, 256);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio8HeadDim256)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 256, 1024);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 256, 512);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio16HeadDim256)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 1, 256, 1024);
    TestXQAAttentionDecodingAccuracy(2, 16, 1, 256, 512);
    TestXQAAttentionDecodingAccuracy(1, 32, 2, 256, 512, false, 0, std::nullopt, 274);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio8HeadDim512)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 512, 256);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 512, 128);
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 512, 512, false, 0, std::nullopt, 274);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 512, 1024, false, 0, 1000);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio16HeadDim512)
{
    TestXQAAttentionDecodingAccuracy(2, 16, 1, 512, 128);
    TestXQAAttentionDecodingAccuracy(1, 16, 1, 512, 512, false, 0, std::nullopt, 274);
    TestXQAAttentionDecodingAccuracy(1, 16, 1, 512, 4096, false, 0, std::nullopt, 3200);
}

TEST(XQAAttentionDecodingTest, accuracyKVRatio6)
{
    TestXQAAttentionDecodingAccuracy(1, 24, 4, 256, 1024);
    TestXQAAttentionDecodingAccuracy(2, 24, 4, 256, 512);
    TestXQAAttentionDecodingAccuracy(4, 24, 4, 256, 256);
}

TEST(XQAAttentionDecodingTest, slidingWindowAccuracy)
{
    TestXQAAttentionDecodingAccuracy(3, 32, 4, 128, 512, false, 127);
    TestXQAAttentionDecodingAccuracy(2, 32, 8, 64, 384, false, 96);
    TestXQAAttentionDecodingAccuracy(2, 24, 4, 256, 384, false, 129);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 512, 192, false, 64);
    TestXQAAttentionDecodingAccuracy(2, 32, 4, 128, 96, false, 256);
}

TEST(XQAAttentionDecodingTest, multiBlockAccuracyGemma4Shapes)
{
    // Gemma 4 E4B: 8 Q / 2 KV heads, head dim 256 (sliding 512) and 512 (global).
    for (uint32_t splits : {2U, 8U})
    {
        gMultiBlockSplits = splits;
        TestXQAAttentionDecodingAccuracy(1, 8, 2, 256, 4096, false, 0, std::nullopt, 3300);
        TestXQAAttentionDecodingAccuracy(1, 8, 2, 256, 4096, false, 512, std::nullopt, 3300);
        TestXQAAttentionDecodingAccuracy(1, 8, 2, 256, 4096, false, 512, std::nullopt, 300);
    }
    gMultiBlockSplits = 1;
}

TEST(XQAAttentionDecodingTest, configurableAttentionScale)
{
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 128, 256, false, 0, 1.0F);
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 128, 256, false, 0, 0.37F);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 256, 1024, false, 512, 1.0F);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 256, 1024, false, 512, 0.37F);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 512, 256, false, 0, 1.0F);
    TestXQAAttentionDecodingAccuracy(1, 8, 1, 512, 256, false, 0, 0.37F);
}

TEST(XQAAttentionDecodingTest, attentionSinkAccuracy)
{
    constexpr int32_t kNUM_Q_HEADS = 32;
    std::vector<float> attentionSinks(kNUM_Q_HEADS);
    for (int32_t headIdx = 0; headIdx < kNUM_Q_HEADS; ++headIdx)
    {
        attentionSinks[headIdx] = static_cast<float>(headIdx - kNUM_Q_HEADS / 2) * 0.125F;
    }
    TestXQAAttentionDecodingAccuracy(1, kNUM_Q_HEADS, 2, 128, 256, false, 0, std::nullopt, 129, attentionSinks);
}

#if SUPPORTS_FP8
TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio3)
{
    TestXQAAttentionDecodingAccuracy(1, 24, 8, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 24, 8, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 24, 8, 128, 256, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio4)
{
    TestXQAAttentionDecodingAccuracy(1, 32, 8, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 32, 8, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 32, 8, 128, 256, true);
    TestXQAAttentionDecodingAccuracy(1, 32, 8, 64, 2048, true);
    TestXQAAttentionDecodingAccuracy(4, 16, 4, 64, 512, true);
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 256, 1024, true);
    TestXQAAttentionDecodingAccuracy(1, 16, 4, 256, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 16, 4, 256, 512, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio5)
{
    TestXQAAttentionDecodingAccuracy(1, 40, 8, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 40, 8, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 40, 8, 128, 512, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio7)
{
    TestXQAAttentionDecodingAccuracy(1, 28, 4, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 28, 4, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 28, 4, 128, 256, true);
    TestXQAAttentionDecodingAccuracy(1, 28, 4, 64, 1024, true);
    TestXQAAttentionDecodingAccuracy(4, 14, 2, 64, 512, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio8)
{
    TestXQAAttentionDecodingAccuracy(1, 32, 4, 128, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 32, 4, 128, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 32, 4, 128, 256, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio8HeadDim256)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 256, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 256, 512, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio8HeadDim512)
{
    TestXQAAttentionDecodingAccuracy(1, 16, 2, 512, 256, true);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 512, 128, true);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio16HeadDim512)
{
    TestXQAAttentionDecodingAccuracy(2, 16, 1, 512, 128, true);
    TestXQAAttentionDecodingAccuracy(1, 16, 1, 512, 512, true, 0, std::nullopt, 274);
}

TEST(XQAAttentionDecodingFP8Test, accuracyKVRatio6)
{
    TestXQAAttentionDecodingAccuracy(1, 24, 4, 256, 1024, true);
    TestXQAAttentionDecodingAccuracy(2, 24, 4, 256, 512, true);
    TestXQAAttentionDecodingAccuracy(4, 24, 4, 256, 256, true);
}

TEST(XQAAttentionDecodingFP8Test, slidingWindowAccuracy)
{
    TestXQAAttentionDecodingAccuracy(3, 32, 4, 128, 512, true, 127);
    TestXQAAttentionDecodingAccuracy(2, 16, 2, 256, 384, true, 96);
}

TEST(XQAAttentionDecodingFP8Test, configurableAttentionScale)
{
    TestXQAAttentionDecodingAccuracy(1, 8, 2, 128, 256, true, 0, 0.37F);
}
#endif

// INT8 converts are native on every XQA SM, so unlike FP8 these run on SM80+ including SM87.
TEST(XQAAttentionDecodingINT8Test, accuracyKVRatio4HeadDim256)
{
    // Gemma 4 sliding-window layers: 8 Q heads, 2 KV heads, head_dim 256.
    TestXQAAttentionDecodingInt8Accuracy(1, 8, 2, 256, 1024);
    TestXQAAttentionDecodingInt8Accuracy(2, 8, 2, 256, 512);
}

TEST(XQAAttentionDecodingINT8Test, accuracyKVRatio4HeadDim512)
{
    // Gemma 4 global layers: 8 Q heads, 2 KV heads, head_dim 512.
    TestXQAAttentionDecodingInt8Accuracy(1, 8, 2, 512, 256);
    TestXQAAttentionDecodingInt8Accuracy(2, 8, 2, 512, 128);
}
