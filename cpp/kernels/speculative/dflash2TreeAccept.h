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

#pragma once

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm::kernel
{

//! Lossless acceptance over a draft tree by sampling the target token at every visited node and following the child
//! that carries it ("sample and match"). The walk starts at the root (node 0), draws the target token from the node's
//! top-k distribution, emits it, and descends into the child whose token equals it; with no such child the emitted
//! token is the bonus token and the walk ends. Every draw comes from the target distribution at its own context, so the
//! output distribution is exactly the target's for any tree; greedy rows take the arg-max.
//!
//! The uniform for depth d is uniforms[b, d] for d < proposalLen and the bonus slot uniforms[b, 2 * proposalLen] for
//! d == proposalLen (the layout launchRequestStableSpecUniforms writes), so the draw at an absolute position does not
//! depend on the tree shape.
//!
//! targetTopKProbs/targetTopKIds: [batch, verifySize, K] (probabilities of the support, ids). treeTokenIds and
//! treeParentIds: [batch, verifySize] (root parent -1). validCounts: [batch]. remainingGenerationLengths (optional,
//! [batch]) caps the accepted length. Outputs: acceptedTokenIds and acceptedTokenIndices [batch, maxAcceptLength]
//! (the node whose logits produced each token), acceptLength [batch].
void dflash2TreeSampleAccept(rt::Tensor const& targetTopKProbs, rt::Tensor const& targetTopKIds,
    rt::Tensor const& treeTokenIds, rt::Tensor const& treeParentIds, rt::Tensor const& validCounts,
    rt::Tensor const& uniforms, int32_t proposalLen, bool greedy, rt::Tensor const* remainingGenerationLengths,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptedTokenIndices, rt::Tensor& acceptLength, cudaStream_t stream);

} // namespace trt_edgellm::kernel
