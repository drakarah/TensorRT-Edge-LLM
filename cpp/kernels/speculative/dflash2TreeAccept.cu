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

#include "kernels/speculative/dflash2TreeAccept.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"

namespace trt_edgellm::kernel
{
namespace
{

__global__ void dflash2TreeSampleAcceptKernel(float const* __restrict__ topKProbs, int32_t const* __restrict__ topKIds,
    int32_t const* __restrict__ treeTokens, int32_t const* __restrict__ treeParents,
    int32_t const* __restrict__ validCounts, float const* __restrict__ uniforms,
    int32_t const* __restrict__ remaining, int32_t* __restrict__ acceptedIds, int32_t* __restrict__ acceptedIndices,
    int32_t* __restrict__ acceptLength, int32_t verifySize, int32_t topK, int32_t proposalLen, int32_t maxAccept,
    bool greedy)
{
    // One thread per request: the walk is a short serial chain (depth <= proposalLen) over <= 128 candidates.
    int32_t const b = static_cast<int32_t>(blockIdx.x);
    int32_t const valid = min(validCounts[b], verifySize);
    int32_t const cap = remaining != nullptr ? min(maxAccept, max(remaining[b], 0)) : maxAccept;
    int32_t cur = 0;
    int32_t n = 0;
    int32_t const uniformStride = 2 * proposalLen + 1;
    while (n < cap)
    {
        float const* probs = topKProbs + (static_cast<int64_t>(b) * verifySize + cur) * topK;
        int32_t const* ids = topKIds + (static_cast<int64_t>(b) * verifySize + cur) * topK;
        int32_t pick = 0;
        if (greedy)
        {
            float best = probs[0];
            for (int32_t k = 1; k < topK; ++k)
            {
                if (probs[k] > best)
                {
                    best = probs[k];
                    pick = k;
                }
            }
        }
        else
        {
            float total = 0.F;
            for (int32_t k = 0; k < topK; ++k)
            {
                total += probs[k];
            }
            int32_t const slot = n < proposalLen ? n : 2 * proposalLen;
            float const target = uniforms[b * uniformStride + slot] * total;
            float run = 0.F;
            pick = topK - 1;
            for (int32_t k = 0; k < topK; ++k)
            {
                run += probs[k];
                if (target < run)
                {
                    pick = k;
                    break;
                }
            }
        }
        int32_t const token = ids[pick];
        acceptedIds[b * maxAccept + n] = token;
        acceptedIndices[b * maxAccept + n] = cur;
        ++n;
        int32_t next = -1;
        for (int32_t c = 1; c < valid; ++c)
        {
            if (treeParents[b * verifySize + c] == cur && treeTokens[b * verifySize + c] == token)
            {
                next = c;
                break;
            }
        }
        if (next < 0)
        {
            break;
        }
        cur = next;
    }
    acceptLength[b] = n;
}

} // namespace

void dflash2TreeSampleAccept(rt::Tensor const& targetTopKProbs, rt::Tensor const& targetTopKIds,
    rt::Tensor const& treeTokenIds, rt::Tensor const& treeParentIds, rt::Tensor const& validCounts,
    rt::Tensor const& uniforms, int32_t proposalLen, bool greedy, rt::Tensor const* remainingGenerationLengths,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptedTokenIndices, rt::Tensor& acceptLength, cudaStream_t stream)
{
    auto const probsShape = targetTopKProbs.getShape();
    check::check(probsShape.getNumDims() == 3, "dflash2TreeSampleAccept: top-k probabilities must be [B, V, K]");
    int32_t const batch = static_cast<int32_t>(probsShape[0]);
    int32_t const verifySize = static_cast<int32_t>(probsShape[1]);
    int32_t const topK = static_cast<int32_t>(probsShape[2]);
    auto const acceptShape = acceptedTokenIds.getShape();
    check::check(acceptShape.getNumDims() == 2 && acceptShape[0] == batch, "dflash2TreeSampleAccept: bad accept shape");
    int32_t const maxAccept = static_cast<int32_t>(acceptShape[1]);
    check::check(maxAccept <= proposalLen + 1, "dflash2TreeSampleAccept: accept buffer exceeds the tree depth + 1");
    if (batch <= 0)
    {
        return;
    }
    dflash2TreeSampleAcceptKernel<<<batch, 1, 0, stream>>>(targetTopKProbs.dataPointer<float>(),
        targetTopKIds.dataPointer<int32_t>(), treeTokenIds.dataPointer<int32_t>(),
        treeParentIds.dataPointer<int32_t>(), validCounts.dataPointer<int32_t>(), uniforms.dataPointer<float>(),
        remainingGenerationLengths != nullptr ? remainingGenerationLengths->dataPointer<int32_t>() : nullptr,
        acceptedTokenIds.dataPointer<int32_t>(), acceptedTokenIndices.dataPointer<int32_t>(),
        acceptLength.dataPointer<int32_t>(), verifySize, topK, proposalLen, maxAccept, greedy);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace trt_edgellm::kernel
