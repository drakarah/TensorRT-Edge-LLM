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

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kCpuMissMaxSlots = 32;
constexpr int32_t kCpuMissMaxTokens = 4;
constexpr int32_t kCpuMissMaxHidden = 4096;

//! Layer-to-host handshake of the CPU miss path, in mapped pinned memory. The GPU writes the fields below `seq` and then `seq`; the host answers by
//! writing `y` and then `done = seq`. Layers run one after another, so one mailbox serves all of them.
struct CpuMissMailbox
{
    alignas(64) volatile uint64_t seq;
    alignas(64) volatile uint64_t done;
    alignas(64) int32_t layer;
    int32_t numMiss;
    int32_t numTokens;
    int32_t hidden;
    int32_t expert[kCpuMissMaxSlots];
    int32_t token[kCpuMissMaxSlots];
    float weight[kCpuMissMaxSlots];
    alignas(64) uint16_t x[kCpuMissMaxTokens * kCpuMissMaxHidden]; // fp16 bits of the layer input
    alignas(64) float y[kCpuMissMaxTokens * kCpuMissMaxHidden];    // weighted sum of the missed experts' outputs
    alignas(64) uint64_t layerCalls;                               // counters written by the GPU, for the statistics line
    uint64_t doorbells;
    uint64_t waitCycles;
    uint64_t timeouts;
};

//! Device-side bookkeeping shared by the doorbell and wait kernels of consecutive layers.
struct CpuMissDeviceState
{
    uint64_t seq;
    uint64_t expect;
    int32_t pending;
    int32_t timeouts;
    uint64_t layerCalls;
    uint64_t doorbells;
    uint64_t waitCycles;
};

} // namespace kernel
} // namespace trt_edgellm
