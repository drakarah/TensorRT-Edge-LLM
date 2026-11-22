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

//! Inkling 2 short convolution kernel against a double-precision CPU reference that convolves each sequence's full
//! input history: kernel sizes, static and dynamic taps, padding slots, empty sequences, missing state rows, state
//! strides wider than the channel count, chunks shorter than the kernel and NaN isolation.

#include "common/cudaUtils.h"
#include "kernels/inkling2ShortConv/inkling2ShortConvKernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

namespace
{
constexpr float kSentinel = 55.0F;

struct Cfg
{
    int32_t channels = 40, kernelSize = 4, groups = 0, stateChannels = 40, stateRows = 4;
    float scale = 0.4472136F;
};

struct Stream
{
    std::vector<float> x;   // [total, C]
    std::vector<float> dyn; // [total, max(groups,1) * k]
};

float roundHalf(float v)
{
    return __half2float(__float2half(v));
}

Stream makeStream(Cfg const& c, int32_t total, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<float> nx(0.0F, 2.0F), nd(0.0F, 0.3F);
    Stream s;
    s.x.resize(size_t(total) * c.channels);
    s.dyn.resize(size_t(total) * std::max(c.groups, 1) * c.kernelSize);
    for (auto& v : s.x)
    {
        v = roundHalf(nx(gen));
    }
    for (auto& v : s.dyn)
    {
        v = roundHalf(nd(gen));
    }
    return s;
}

std::vector<float> makeTaps(Cfg const& c, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<float> n(0.0F, 0.5F);
    std::vector<float> t(size_t(c.channels) * c.kernelSize);
    for (auto& v : t)
    {
        v = n(gen);
    }
    return t;
}

template <typename T>
T* upload(std::vector<T> const& v)
{
    T* d{};
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
    if (!v.empty())
    {
        CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return d;
}

std::vector<half> toHalf(std::vector<float> const& v)
{
    std::vector<half> o(v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        o[i] = __float2half(v[i]);
    }
    return o;
}

double refOut(Cfg const& c, Stream const& s, std::vector<float> const& taps, int32_t t, int32_t ch)
{
    int32_t const perGroup = c.groups > 0 ? c.channels / c.groups : c.channels;
    double acc = 0;
    for (int32_t j = 0; j < c.kernelSize; ++j)
    {
        if (t - j < 0)
        {
            continue;
        }
        double tap = taps[size_t(ch) * c.kernelSize + j];
        if (c.groups > 0)
        {
            tap += s.dyn[(size_t(t) * c.groups + ch / perGroup) * c.kernelSize + j];
        }
        acc += tap * s.x[size_t(t - j) * c.channels + ch];
    }
    return (acc + s.x[size_t(t) * c.channels + ch]) * c.scale;
}

struct Step
{
    std::vector<int32_t> first, counts, rows;
    int32_t width = 0; // > 0: entry-padded slots
};

//! One launch for the given streams; verifies outputs, the state slots and everything that must stay untouched.
void runStep(Cfg const& c, std::vector<Stream> const& streams, std::vector<float> const& taps, std::vector<half>& state,
    Step const& st, bool checkOutputs = true)
{
    int32_t const batch = int32_t(streams.size());
    std::vector<int32_t> offsets{0};
    for (int32_t b = 0; b < batch; ++b)
    {
        offsets.push_back(offsets.back() + (st.width > 0 ? st.width : st.counts[b]));
    }
    int32_t const slots = offsets.back();
    int32_t const dynWidth = std::max(c.groups, 1) * c.kernelSize;
    std::vector<float> x(size_t(slots) * c.channels, 4.0F), dyn(size_t(slots) * dynWidth, 4.0F);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t i = 0; i < st.counts[b]; ++i)
        {
            for (int32_t ch = 0; ch < c.channels; ++ch)
            {
                x[size_t(offsets[b] + i) * c.channels + ch] = streams[b].x[size_t(st.first[b] + i) * c.channels + ch];
            }
            for (int32_t d = 0; d < dynWidth; ++d)
            {
                dyn[size_t(offsets[b] + i) * dynWidth + d] = streams[b].dyn[size_t(st.first[b] + i) * dynWidth + d];
            }
        }
    }
    std::vector<half> before = state;
    half* dx = upload(toHalf(x));
    float* dtaps = upload(taps);
    half* ddyn = c.groups > 0 ? upload(toHalf(dyn)) : nullptr;
    half* dstate = upload(state);
    int32_t* doff = upload(offsets);
    int32_t* dlen = upload(st.counts);
    int32_t* drows = upload(st.rows);
    half* dy{};
    CUDA_CHECK(cudaMalloc(&dy, std::max<size_t>(size_t(slots) * c.channels, 1) * sizeof(half)));
    std::vector<half> junk(size_t(slots) * c.channels, __float2half(9.0F));
    if (!junk.empty())
    {
        CUDA_CHECK(cudaMemcpy(dy, junk.data(), junk.size() * sizeof(half), cudaMemcpyHostToDevice));
    }
    Inkling2ShortConvParams p{};
    p.numTokens = slots;
    p.batchSize = batch;
    p.channels = c.channels;
    p.stateStride = c.stateChannels;
    p.stateRows = c.stateRows;
    p.kernelSize = c.kernelSize;
    p.groups = c.groups;
    p.scale = c.scale;
    launchInkling2ShortConv(p, dx, dtaps, ddyn, dstate, doff, dlen, drows, dy, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> y(size_t(slots) * c.channels);
    if (!y.empty())
    {
        CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * sizeof(half), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(state.data(), dstate, state.size() * sizeof(half), cudaMemcpyDeviceToHost));
    for (void* ptr : {(void*) dx, (void*) dtaps, (void*) ddyn, (void*) dstate, (void*) doff, (void*) dlen,
             (void*) drows, (void*) dy})
    {
        if (ptr != nullptr)
        {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
    int32_t const k = c.kernelSize;
    std::vector<bool> expectChanged(state.size(), false);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t slot = offsets[b]; slot < offsets[b + 1]; ++slot)
        {
            int32_t const local = slot - offsets[b];
            for (int32_t ch = 0; ch < c.channels; ++ch)
            {
                float const got = __half2float(y[size_t(slot) * c.channels + ch]);
                if (local >= st.counts[b])
                {
                    ASSERT_EQ(got, 0.0F) << "padding slot " << slot;
                    continue;
                }
                if (!checkOutputs)
                {
                    continue;
                }
                // History before the sequence start is zero unless a state row supplies it; the references below
                // were built with the state rows' content, so compare against the full-history reference.
                double want = refOut(c, streams[b], taps, st.first[b] + local, ch);
                if (st.rows[b] < 0)
                {
                    // no state row: only inputs inside this call are visible
                    Stream trimmed = streams[b];
                    int32_t const base = st.first[b];
                    for (int32_t i = 0; i < base; ++i)
                    {
                        std::fill(trimmed.x.begin() + size_t(i) * c.channels, trimmed.x.begin() + size_t(i + 1) * c.channels, 0.0F);
                    }
                    want = refOut(c, trimmed, taps, st.first[b] + local, ch);
                }
                EXPECT_NEAR(got, want, 2e-3 + 2e-3 * std::fabs(want)) << "slot " << slot << " ch " << ch;
            }
        }
        if (st.rows[b] < 0 || st.counts[b] <= 0)
        {
            continue;
        }
        for (int32_t ch = 0; ch < c.channels; ++ch)
        {
            for (int32_t m = 0; m < k - 1; ++m)
            {
                int32_t const idx = st.counts[b] - (k - 1) + m;
                size_t const at = (size_t(st.rows[b]) * c.stateChannels + ch) * k + m;
                float const want = idx >= 0 ? streams[b].x[size_t(st.first[b] + idx) * c.channels + ch]
                                            : __half2float(before[(size_t(st.rows[b]) * c.stateChannels + ch) * k + (st.counts[b] + m)]);
                ASSERT_EQ(__half2float(state[at]), want) << "row " << st.rows[b] << " ch " << ch << " slot " << m;
                expectChanged[at] = true;
            }
        }
    }
    for (size_t i = 0; i < state.size(); ++i)
    {
        if (!expectChanged[i])
        {
            ASSERT_EQ(__half2float(state[i]), __half2float(before[i])) << "state element " << i << " must be untouched";
        }
    }
}

std::vector<half> zeroState(Cfg const& c)
{
    return std::vector<half>(size_t(c.stateRows) * c.stateChannels * c.kernelSize, __float2half(0.0F));
}
} // namespace

TEST(Inkling2ShortConv, StaticAndDynamicDecodeSequencesMatchTheReference)
{
    for (auto [kernelSize, groups] : {std::pair<int32_t, int32_t>{4, 0}, {4, 4}, {2, 0}, {3, 8}, {8, 0}, {5, 10}})
    {
        Cfg c;
        c.kernelSize = kernelSize;
        c.groups = groups;
        c.scale = 1.0F / std::sqrt(float(groups ? 2 * kernelSize + 1 : kernelSize + 1));
        std::vector<Stream> streams{makeStream(c, 80, 1)};
        auto taps = makeTaps(c, 2);
        auto state = zeroState(c);
        int32_t first = 0;
        for (int32_t count : {13, 1, 1, 1, 2, 5, 1, 20, 1, 7})
        {
            runStep(c, streams, taps, state, Step{{first}, {count}, {1}, 0});
            first += count;
        }
    }
}

TEST(Inkling2ShortConv, ChunksShorterThanTheKernelMixOldStateAndNewInputs)
{
    Cfg c;
    c.kernelSize = 8;
    c.scale = 0.3F;
    std::vector<Stream> streams{makeStream(c, 60, 3)};
    auto taps = makeTaps(c, 4);
    auto state = zeroState(c);
    int32_t first = 0;
    for (int32_t count : {1, 2, 3, 1, 4, 2, 1, 7, 1, 1, 9, 6})
    {
        runStep(c, streams, taps, state, Step{{first}, {count}, {2}, 0});
        first += count;
    }
}

TEST(Inkling2ShortConv, BatchWithPermutedRowsPaddingAndEmptySequence)
{
    Cfg c;
    c.groups = 4;
    c.stateRows = 6;
    c.scale = 0.3F;
    std::vector<Stream> streams{makeStream(c, 40, 5), makeStream(c, 40, 6), makeStream(c, 40, 7)};
    auto taps = makeTaps(c, 8);
    auto state = zeroState(c);
    for (int32_t r : {3, 5})
    {
        for (int32_t ch = 0; ch < c.channels; ++ch)
        {
            for (int32_t m = 0; m < c.kernelSize; ++m)
            {
                state[(size_t(r) * c.stateChannels + ch) * c.kernelSize + m] = __float2half(kSentinel);
            }
        }
    }
    runStep(c, streams, taps, state, Step{{0, 0, 0}, {8, 3, 0}, {4, 0, 2}, 8});
    runStep(c, streams, taps, state, Step{{8, 3, 0}, {1, 5, 6}, {4, 0, 2}, 6});
}

TEST(Inkling2ShortConv, StateChannelStrideWiderThanTheConvolutionKeepsExtraChannelsUntouched)
{
    Cfg c;
    c.channels = 24;
    c.stateChannels = 80;
    std::vector<Stream> streams{makeStream(c, 40, 9)};
    auto taps = makeTaps(c, 10);
    auto state = zeroState(c);
    for (size_t i = 0; i < state.size(); ++i)
    {
        state[i] = __float2half(kSentinel);
    }
    // sentinel everywhere except the first k-1 slots of the first 24 channels of row 1, which a zero history needs
    for (int32_t ch = 0; ch < c.channels; ++ch)
    {
        for (int32_t m = 0; m < c.kernelSize - 1; ++m)
        {
            state[(size_t(1) * c.stateChannels + ch) * c.kernelSize + m] = __float2half(0.0F);
        }
    }
    int32_t first = 0;
    for (int32_t count : {9, 1, 1, 12})
    {
        runStep(c, streams, taps, state, Step{{first}, {count}, {1}, 0});
        first += count;
    }
}

TEST(Inkling2ShortConv, SequenceWithoutAStateRowSeesOnlyItsOwnInputsAndWritesNoState)
{
    Cfg c;
    c.scale = 0.5F;
    std::vector<Stream> streams{makeStream(c, 20, 11), makeStream(c, 20, 12)};
    auto taps = makeTaps(c, 13);
    auto state = zeroState(c);
    for (auto& v : state)
    {
        v = __float2half(kSentinel);
    }
    for (int32_t ch = 0; ch < c.channels; ++ch)
    {
        for (int32_t m = 0; m < c.kernelSize - 1; ++m)
        {
            state[(size_t(1) * c.stateChannels + ch) * c.kernelSize + m] = __float2half(0.0F);
        }
    }
    runStep(c, streams, taps, state, Step{{0, 0}, {6, 6}, {-1, 1}, 0});
}

TEST(Inkling2ShortConv, LargeAndZeroInputsStayFinite)
{
    Cfg c;
    c.scale = 0.5F;
    std::vector<Stream> streams{makeStream(c, 10, 14)};
    for (int32_t t : {2, 3, 4})
    {
        std::fill(streams[0].x.begin() + size_t(t) * c.channels, streams[0].x.begin() + size_t(t + 1) * c.channels, 0.0F);
    }
    std::fill(streams[0].x.begin() + size_t(6) * c.channels, streams[0].x.begin() + size_t(7) * c.channels, 3.0e3F);
    std::fill(streams[0].x.begin() + size_t(7) * c.channels, streams[0].x.begin() + size_t(8) * c.channels, -3.0e3F);
    auto taps = makeTaps(c, 15);
    auto state = zeroState(c);
    runStep(c, streams, taps, state, Step{{0}, {10}, {0}, 0});
}

TEST(Inkling2ShortConv, ZeroTokensLeavesEverythingUntouched)
{
    Cfg c;
    std::vector<Stream> streams{makeStream(c, 5, 16)};
    auto taps = makeTaps(c, 17);
    auto state = zeroState(c);
    for (auto& v : state)
    {
        v = __float2half(kSentinel);
    }
    std::vector<half> before = state;
    runStep(c, streams, taps, state, Step{{0}, {0}, {1}, 0});
    for (size_t i = 0; i < state.size(); ++i)
    {
        ASSERT_EQ(__half2float(state[i]), __half2float(before[i]));
    }
}

TEST(Inkling2ShortConv, NaNInOneInputOnlyAffectsTheNextKernelSizeOutputs)
{
    Cfg c;
    c.scale = 0.5F;
    std::vector<Stream> streams{makeStream(c, 20, 18)};
    auto taps = makeTaps(c, 19);
    int32_t const bad = 5;
    streams[0].x[size_t(bad) * c.channels + 3] = std::numeric_limits<float>::quiet_NaN();
    auto state = zeroState(c);
    // Run the kernel directly through runStep's machinery would assert on NaN; check the affected window here.
    int32_t const slots = 20;
    half* dx = upload(toHalf(streams[0].x));
    float* dtaps = upload(taps);
    half* dstate = upload(state);
    std::vector<int32_t> offsets{0, slots}, lens{slots}, rows{0};
    int32_t* doff = upload(offsets);
    int32_t* dlen = upload(lens);
    int32_t* drows = upload(rows);
    half* dy{};
    CUDA_CHECK(cudaMalloc(&dy, size_t(slots) * c.channels * sizeof(half)));
    Inkling2ShortConvParams p{};
    p.numTokens = slots;
    p.batchSize = 1;
    p.channels = c.channels;
    p.stateStride = c.stateChannels;
    p.stateRows = c.stateRows;
    p.kernelSize = c.kernelSize;
    p.groups = 0;
    p.scale = c.scale;
    launchInkling2ShortConv(p, dx, dtaps, nullptr, dstate, doff, dlen, drows, dy, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> y(size_t(slots) * c.channels);
    CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * sizeof(half), cudaMemcpyDeviceToHost));
    for (int32_t t = 0; t < slots; ++t)
    {
        for (int32_t ch = 0; ch < c.channels; ++ch)
        {
            bool const nan = std::isnan(__half2float(y[size_t(t) * c.channels + ch]));
            bool const expectNaN = ch == 3 && t >= bad && t < bad + c.kernelSize;
            // a zero tap times NaN is still NaN, so every tap position in the window is affected
            EXPECT_EQ(nan, expectNaN) << "t " << t << " ch " << ch;
        }
    }
    for (void* ptr : {(void*) dx, (void*) dtaps, (void*) dstate, (void*) doff, (void*) dlen, (void*) drows, (void*) dy})
    {
        CUDA_CHECK(cudaFree(ptr));
    }
}
