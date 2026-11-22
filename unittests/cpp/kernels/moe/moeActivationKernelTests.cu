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

//! Coverage for moeActivation(): SwiGLU, ReLU2, GeGLU and the softcapped SwiGLU
//! out = gateCap * tanh(gate / gateCap) * sigmoid(gate) * upCap * tanh(up / upCap)
//! against a double-precision reference, plus edge cases and guard canaries.

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/moe/moeActivationKernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using nvinfer1::DataType;

namespace
{
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

template <typename T>
float toF(T v);
template <>
float toF<half>(half v)
{
    return __half2float(v);
}
template <>
float toF<__nv_bfloat16>(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}
template <typename T>
T fromF(float v);
template <>
half fromF<half>(float v)
{
    return __float2half_rn(v);
}
template <>
__nv_bfloat16 fromF<__nv_bfloat16>(float v)
{
    return __float2bfloat16_rn(v);
}

template <typename T>
DataType dtypeOf();
template <>
DataType dtypeOf<half>()
{
    return DataType::kHALF;
}
template <>
DataType dtypeOf<__nv_bfloat16>()
{
    return DataType::kBF16;
}

//! Double-precision reference of one activation element. `gate` and `up` are the (already rounded) device inputs.
double refElement(int32_t type, double gate, double up, double gateCap, double upCap)
{
    switch (type)
    {
    case MoeActivationType::kMoeSwiGlu: return gate / (1.0 + std::exp(-gate)) * up;
    case MoeActivationType::kMoeGeGlu:
        return gate * 0.5 * (1.0 + std::tanh(0.7978845608028654 * (gate + 0.044715 * gate * gate * gate))) * up;
    case MoeActivationType::kMoeSoftcapSwiGlu:
        return gateCap * std::tanh(gate / gateCap) * (1.0 / (1.0 + std::exp(-gate))) * upCap * std::tanh(up / upCap);
    default: return 0.0;
    }
}

//! Runs the activation on host-side gate/up rows ([numTokens, 2 * inter], gate first) and returns the output as float.
//! `margin` extra canary elements (value `canary`) sit before and after the output and are verified untouched.
template <typename T>
std::vector<float> run(std::vector<float> const& input, int64_t numTokens, int64_t inter, int32_t type, float gateCap,
    float upCap, int32_t margin = 0, float canary = 0.0F, bool* canaryIntact = nullptr)
{
    bool const gated = type != MoeActivationType::kMoeRelu2;
    int64_t const inDim = gated ? 2 * inter : inter;
    std::vector<T> hostIn(static_cast<size_t>(numTokens * inDim));
    for (size_t i = 0; i < hostIn.size(); ++i)
    {
        hostIn[i] = fromF<T>(input[i]);
    }
    // 16-byte aligned canary margin: 8 elements of a 2-byte type.
    int64_t const pad = margin * 8;
    int64_t const outElems = numTokens * inter;
    T* dIn{};
    T* dOutBase{};
    CUDA_CHECK(cudaMalloc(&dIn, std::max<size_t>(hostIn.size(), 8) * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dOutBase, static_cast<size_t>(outElems + 2 * pad + 8) * sizeof(T)));
    if (!hostIn.empty())
    {
        CUDA_CHECK(cudaMemcpy(dIn, hostIn.data(), hostIn.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> init(static_cast<size_t>(outElems + 2 * pad + 8), fromF<T>(canary));
    CUDA_CHECK(cudaMemcpy(dOutBase, init.data(), init.size() * sizeof(T), cudaMemcpyHostToDevice));

    rt::Tensor in(dIn, {numTokens, inDim}, rt::DeviceType::kGPU, dtypeOf<T>());
    rt::Tensor out(dOutBase + pad, {numTokens, inter}, rt::DeviceType::kGPU, dtypeOf<T>());
    moeActivation(in, out, numTokens, inter, type, nullptr, gateCap, upCap);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> res(init.size());
    CUDA_CHECK(cudaMemcpy(res.data(), dOutBase, res.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOutBase));

    if (canaryIntact != nullptr)
    {
        bool ok = true;
        for (int64_t i = 0; i < pad; ++i)
        {
            ok &= toF(res[i]) == toF(fromF<T>(canary));
        }
        for (int64_t i = pad + outElems; i < static_cast<int64_t>(res.size()); ++i)
        {
            ok &= toF(res[i]) == toF(fromF<T>(canary));
        }
        *canaryIntact = ok;
    }
    std::vector<float> outF(static_cast<size_t>(outElems));
    for (int64_t i = 0; i < outElems; ++i)
    {
        outF[i] = toF(res[pad + i]);
    }
    return outF;
}

//! Compares against the reference computed from the rounded inputs. Tolerance: half/bf16 output rounding plus a
//! small absolute term (FP32 transcendental error).
template <typename T>
void expectMatchesReference(std::vector<float> const& input, std::vector<float> const& got, int64_t numTokens,
    int64_t inter, int32_t type, float gateCap, float upCap)
{
    bool const gated = type != MoeActivationType::kMoeRelu2;
    int64_t const inDim = gated ? 2 * inter : inter;
    double const relTol = std::is_same<T, half>::value ? 2e-3 : 1.6e-2;
    for (int64_t t = 0; t < numTokens; ++t)
    {
        for (int64_t j = 0; j < inter; ++j)
        {
            double const g = toF(fromF<T>(input[t * inDim + j]));
            double const u = gated ? toF(fromF<T>(input[t * inDim + inter + j])) : 0.0;
            double want = type == MoeActivationType::kMoeRelu2 ? std::pow(std::max(g, 0.0), 2.0)
                                                               : refElement(type, g, u, gateCap, upCap);
            double const have = got[t * inter + j];
            if (std::isnan(want))
            {
                EXPECT_TRUE(std::isnan(have)) << "token " << t << " col " << j;
                continue;
            }
            if (std::isinf(want) || std::fabs(want) > 60000.0)
            {
                continue; // overflow of the 16-bit output type is not part of the contract
            }
            // GeGLU evaluates 1 + tanh in FP32, which loses relative precision for strongly negative gates.
            double const absTol = 1e-5 + (type == MoeActivationType::kMoeGeGlu ? 5e-6 * std::fabs(g * u) : 0.0);
            EXPECT_NEAR(have, want, relTol * std::fabs(want) + absTol)
                << "token " << t << " col " << j << " gate " << g << " up " << u;
        }
    }
}

std::vector<float> randomInput(int64_t n, float scale, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0F, scale);
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v)
    {
        x = dist(gen);
    }
    return v;
}
} // namespace

template <typename T>
class MoeActivationTyped : public ::testing::Test
{
};
using ActivationTypes = ::testing::Types<half, __nv_bfloat16>;
TYPED_TEST_SUITE(MoeActivationTyped, ActivationTypes);

TYPED_TEST(MoeActivationTyped, ShippedActivationsMatchReference)
{
    for (int32_t type : {int32_t(MoeActivationType::kMoeSwiGlu), int32_t(MoeActivationType::kMoeRelu2),
             int32_t(MoeActivationType::kMoeGeGlu)})
    {
        for (auto [tokens, inter] : {std::pair<int64_t, int64_t>{1, 8}, {3, 64}, {8, 1152}, {33, 1024}})
        {
            int64_t const inDim = type == MoeActivationType::kMoeRelu2 ? inter : 2 * inter;
            auto const in = randomInput(tokens * inDim, 2.0F, 11);
            auto const got = run<TypeParam>(in, tokens, inter, type, 0.0F, 0.0F);
            expectMatchesReference<TypeParam>(in, got, tokens, inter, type, 0.0F, 0.0F);
        }
    }
}

TYPED_TEST(MoeActivationTyped, SoftcapSwiGluMatchesReferenceOnInklingShapes)
{
    // Inkling 2: intermediate 1152, caps 4 and 25. Shapes cover one token (decode), a Marlin-block sized batch and
    // a prefill sized batch with a non-power-of-two token count.
    for (auto [tokens, inter] : {std::pair<int64_t, int64_t>{1, 8}, {1, 1152}, {3, 64}, {8, 1152}, {257, 1152}})
    {
        auto const in = randomInput(tokens * 2 * inter, 3.0F, 7);
        auto const got = run<TypeParam>(in, tokens, inter, MoeActivationType::kMoeSoftcapSwiGlu, 4.0F, 25.0F);
        expectMatchesReference<TypeParam>(in, got, tokens, inter, MoeActivationType::kMoeSoftcapSwiGlu, 4.0F, 25.0F);
    }
}

TYPED_TEST(MoeActivationTyped, SoftcapSwiGluSaturatesInsteadOfOverflowing)
{
    // Gate and up at +-60000 (near the FP16 maximum): the soft caps bound the result by gateCap * upCap.
    std::vector<float> in;
    for (float g : {60000.0F, -60000.0F, 100.0F, -100.0F})
    {
        for (float u : {60000.0F, -60000.0F, 500.0F, -500.0F})
        {
            in.push_back(g);
        }
    }
    int64_t const inter = 16;
    std::vector<float> row(2 * inter);
    for (int i = 0; i < inter; ++i)
    {
        row[i] = in[i];
        row[inter + i] = (i % 4 == 0) ? 60000.0F : (i % 4 == 1) ? -60000.0F : (i % 4 == 2) ? 500.0F : -500.0F;
    }
    // bf16 cannot represent 60000 exactly; the reference uses the rounded value.
    auto const got = run<TypeParam>(row, 1, inter, MoeActivationType::kMoeSoftcapSwiGlu, 4.0F, 25.0F);
    for (float v : got)
    {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_LE(std::fabs(v), 4.0F * 25.0F * 1.01F);
    }
    expectMatchesReference<TypeParam>(row, got, 1, inter, MoeActivationType::kMoeSoftcapSwiGlu, 4.0F, 25.0F);
}

TYPED_TEST(MoeActivationTyped, SoftcapSwiGluZeroAndSignedZero)
{
    int64_t const inter = 16;
    std::vector<float> row(2 * inter, 0.0F);
    for (int i = 0; i < inter; ++i)
    {
        row[i] = (i % 2 == 0) ? 0.0F : -0.0F;
    }
    auto const got = run<TypeParam>(row, 1, inter, MoeActivationType::kMoeSoftcapSwiGlu, 4.0F, 25.0F);
    for (float v : got)
    {
        EXPECT_EQ(v, 0.0F);
    }
}

TYPED_TEST(MoeActivationTyped, SoftcapSwiGluInfinitiesAndNaN)
{
    int64_t const inter = 8;
    // gate / up pairs: (+inf, 1), (-inf, 1), (1, +inf), (1, -inf), (NaN, 1), (1, NaN), (+inf, +inf), (-inf, -inf)
    float const gate[8] = {kInf, -kInf, 1.0F, 1.0F, kNaN, 1.0F, kInf, -kInf};
    float const up[8] = {1.0F, 1.0F, kInf, -kInf, 1.0F, kNaN, kInf, -kInf};
    std::vector<float> row(2 * inter);
    for (int i = 0; i < inter; ++i)
    {
        row[i] = gate[i];
        row[inter + i] = up[i];
    }
    auto const got = run<TypeParam>(row, 1, inter, MoeActivationType::kMoeSoftcapSwiGlu, 4.0F, 25.0F);
    double const capUp1 = 25.0 * std::tanh(1.0 / 25.0), capGate1 = 4.0 * std::tanh(1.0 / 4.0);
    double const sig1 = 1.0 / (1.0 + std::exp(-1.0));
    EXPECT_NEAR(got[0], 4.0 * 1.0 * capUp1, 2e-2 * 4.0 * capUp1); // gate +inf: cap * 1 * sigmoid 1
    EXPECT_EQ(got[1], 0.0F);                                      // gate -inf: sigmoid 0 (the product is -0)
    EXPECT_NEAR(got[2], capGate1 * sig1 * 25.0, 2e-2 * capGate1 * sig1 * 25.0);
    EXPECT_NEAR(got[3], -capGate1 * sig1 * 25.0, 2e-2 * capGate1 * sig1 * 25.0);
    EXPECT_TRUE(std::isnan(got[4])) << "NaN gate must propagate";
    EXPECT_TRUE(std::isnan(got[5])) << "NaN up must propagate";
    EXPECT_NEAR(got[6], 4.0 * 25.0, 2e-2 * 100.0);
    EXPECT_EQ(got[7], 0.0F);
    (void) capUp1;
}

TYPED_TEST(MoeActivationTyped, SoftcapSwiGluApproachesPlainSwiGluForLargeCaps)
{
    int64_t const tokens = 4, inter = 64;
    auto const in = randomInput(tokens * 2 * inter, 1.0F, 3);
    auto const capped = run<TypeParam>(in, tokens, inter, MoeActivationType::kMoeSoftcapSwiGlu, 1e6F, 1e6F);
    auto const plain = run<TypeParam>(in, tokens, inter, MoeActivationType::kMoeSwiGlu, 0.0F, 0.0F);
    double const relTol = std::is_same<TypeParam, half>::value ? 4e-3 : 3e-2;
    for (size_t i = 0; i < capped.size(); ++i)
    {
        EXPECT_NEAR(capped[i], plain[i], relTol * std::fabs(plain[i]) + 1e-5);
    }
}

TYPED_TEST(MoeActivationTyped, SoftcapSwiGluTinyCapsClampToTheCap)
{
    // With a very small cap both factors saturate to +-cap: out = cg * tanh(+-inf) * sigmoid(g) * cu * tanh(+-inf).
    int64_t const inter = 8;
    std::vector<float> row(2 * inter);
    for (int i = 0; i < inter; ++i)
    {
        row[i] = 2.0F;
        row[inter + i] = (i % 2 == 0) ? 3.0F : -3.0F;
    }
    auto const got = run<TypeParam>(row, 1, inter, MoeActivationType::kMoeSoftcapSwiGlu, 1e-3F, 1e-3F);
    double const sig2 = 1.0 / (1.0 + std::exp(-2.0));
    for (int i = 0; i < inter; ++i)
    {
        double const want = 1e-3 * sig2 * 1e-3 * ((i % 2 == 0) ? 1.0 : -1.0);
        EXPECT_NEAR(got[i], want, 2e-2 * std::fabs(want) + 1e-9);
    }
}

TYPED_TEST(MoeActivationTyped, OutputWritesStayInsideTheOutputBuffer)
{
    for (int32_t type : {int32_t(MoeActivationType::kMoeSoftcapSwiGlu), int32_t(MoeActivationType::kMoeSwiGlu)})
    {
        bool intact = false;
        auto const in = randomInput(5 * 2 * 72, 1.0F, 5);
        auto const got = run<TypeParam>(in, 5, 72, type, 4.0F, 25.0F, 2, 123.0F, &intact);
        EXPECT_TRUE(intact) << "guard canaries around the output were overwritten";
        expectMatchesReference<TypeParam>(in, got, 5, 72, type, 4.0F, 25.0F);
    }
}

TYPED_TEST(MoeActivationTyped, ZeroTokensIsANoOp)
{
    bool intact = false;
    auto const got = run<TypeParam>({}, 0, 64, MoeActivationType::kMoeSoftcapSwiGlu, 4.0F, 25.0F, 1, 7.0F, &intact);
    EXPECT_TRUE(got.empty());
    EXPECT_TRUE(intact);
}

TYPED_TEST(MoeActivationTyped, InvalidSoftcapsAreRejected)
{
    int64_t const inter = 16;
    auto const in = randomInput(2 * 2 * inter, 1.0F, 9);
    struct Caps
    {
        float gate, up;
    };
    for (Caps c : {Caps{0.0F, 25.0F}, Caps{4.0F, 0.0F}, Caps{-4.0F, 25.0F}, Caps{4.0F, -25.0F}, Caps{kNaN, 25.0F},
             Caps{4.0F, kNaN}, Caps{kInf, 25.0F}, Caps{4.0F, kInf}, Caps{0.0F, 0.0F}})
    {
        EXPECT_THROW(
            (run<TypeParam>(in, 2, inter, MoeActivationType::kMoeSoftcapSwiGlu, c.gate, c.up)), std::runtime_error)
            << "gate cap " << c.gate << " up cap " << c.up;
    }
}

TYPED_TEST(MoeActivationTyped, SoftcapsAreIgnoredByOtherActivations)
{
    int64_t const inter = 16;
    auto const in = randomInput(2 * 2 * inter, 1.0F, 13);
    EXPECT_NO_THROW((run<TypeParam>(in, 2, inter, MoeActivationType::kMoeSwiGlu, -1.0F, kNaN)));
}

TYPED_TEST(MoeActivationTyped, ShapeErrorsAreRejected)
{
    using T = TypeParam;
    T* d{};
    CUDA_CHECK(cudaMalloc(&d, 1 << 16));
    {
        // intermediate dimension not a multiple of 8
        rt::Tensor in(d, {2, 2 * 12}, rt::DeviceType::kGPU, dtypeOf<T>());
        rt::Tensor out(d, {2, 12}, rt::DeviceType::kGPU, dtypeOf<T>());
        EXPECT_THROW(moeActivation(in, out, 2, 12, MoeActivationType::kMoeSoftcapSwiGlu, nullptr, 4.0F, 25.0F),
            std::runtime_error);
    }
    {
        // gated input must be 2 * intermediate wide
        rt::Tensor in(d, {2, 16}, rt::DeviceType::kGPU, dtypeOf<T>());
        rt::Tensor out(d, {2, 16}, rt::DeviceType::kGPU, dtypeOf<T>());
        EXPECT_THROW(moeActivation(in, out, 2, 16, MoeActivationType::kMoeSoftcapSwiGlu, nullptr, 4.0F, 25.0F),
            std::runtime_error);
    }
    {
        // token count mismatch
        rt::Tensor in(d, {2, 32}, rt::DeviceType::kGPU, dtypeOf<T>());
        rt::Tensor out(d, {2, 16}, rt::DeviceType::kGPU, dtypeOf<T>());
        EXPECT_THROW(moeActivation(in, out, 3, 16, MoeActivationType::kMoeSoftcapSwiGlu, nullptr, 4.0F, 25.0F),
            std::runtime_error);
    }
    {
        // unknown activation type
        rt::Tensor in(d, {2, 32}, rt::DeviceType::kGPU, dtypeOf<T>());
        rt::Tensor out(d, {2, 16}, rt::DeviceType::kGPU, dtypeOf<T>());
        EXPECT_THROW(moeActivation(in, out, 2, 16, 3, nullptr, 4.0F, 25.0F), std::runtime_error);
        EXPECT_THROW(moeActivation(in, out, 2, 16, 7, nullptr, 4.0F, 25.0F), std::runtime_error);
    }
    CUDA_CHECK(cudaFree(d));
}

TEST(MoeActivationErrors, DataTypeMismatchAndFp32AreRejected)
{
    half* d{};
    CUDA_CHECK(cudaMalloc(&d, 1 << 16));
    {
        rt::Tensor in(d, {2, 32}, rt::DeviceType::kGPU, DataType::kHALF);
        rt::Tensor out(d, {2, 16}, rt::DeviceType::kGPU, DataType::kBF16);
        EXPECT_THROW(moeActivation(in, out, 2, 16, MoeActivationType::kMoeSoftcapSwiGlu, nullptr, 4.0F, 25.0F),
            std::runtime_error);
    }
    {
        rt::Tensor in(d, {2, 32}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor out(d, {2, 16}, rt::DeviceType::kGPU, DataType::kFLOAT);
        EXPECT_THROW(moeActivation(in, out, 2, 16, MoeActivationType::kMoeSoftcapSwiGlu, nullptr, 4.0F, 25.0F),
            std::runtime_error);
    }
    CUDA_CHECK(cudaFree(d));
}
