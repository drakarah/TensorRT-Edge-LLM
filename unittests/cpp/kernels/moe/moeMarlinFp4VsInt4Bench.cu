// Decode-shape timing of the NVFP4 (E2M1, block 16) and INT4 (U4, group 32/128) Marlin MoE GEMMs on random data.
// Inkling 2 shapes: gate_up K=2304 N=2304 (2 x 1152 fused), down K=1152 N=2304; 1 token, top-8, expert pool 128 so no L2 reuse.
// Timing only; numerics are not checked here.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/moe/moe_marlin/moeMarlin.h"

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{
constexpr int32_t kPool = 128;
constexpr int32_t kTopK = 8;
constexpr int32_t kBlock = 8;
constexpr int32_t kRoutings = 64;
constexpr int32_t kIters = 400;

void fillRandom(Tensor& t, int64_t bytes)
{
    std::vector<uint8_t> h(static_cast<size_t>(bytes));
    std::mt19937 g(1);
    for (auto& b : h)
    {
        b = static_cast<uint8_t>(g());
    }
    CUDA_CHECK(cudaMemcpy(t.rawPointer(), h.data(), h.size(), cudaMemcpyHostToDevice));
}

struct Routing
{
    std::vector<int32_t> sorted, experts;
    int32_t post{};
};

// 1 input row routed to 8 distinct experts (rows = 8 slots, one block each) or 8 input rows with 1 expert each (down).
Routing makeRouting(std::mt19937& g, int32_t slots)
{
    std::vector<int32_t> pool(kPool);
    for (int i = 0; i < kPool; ++i) pool[i] = i;
    std::shuffle(pool.begin(), pool.end(), g);
    Routing r;
    r.sorted.assign(slots * kBlock, slots);
    r.experts.assign(slots, 0);
    for (int s = 0; s < slots; ++s)
    {
        r.sorted[s * kBlock] = s;
        r.experts[s] = pool[s];
    }
    r.post = slots * kBlock;
    return r;
}

struct Bench
{
    bool fp4;
    nvinfer1::DataType act;
    int32_t groupSize; // int4 only
    char const* name;
};

float runGemm(Bench const& b, int32_t K, int32_t N, int32_t rows, int32_t topK)
{
    int32_t numSms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, 0));
    int32_t const slots = rows * topK;
    int64_t const capacity = static_cast<int64_t>(slots) * kBlock;
    int64_t const ws = kernel::getMoeMarlinWorkspaceSize(capacity, N, kBlock, numSms);
    Tensor input({rows, K}, DeviceType::kGPU, b.act);
    Tensor output({slots, N}, DeviceType::kGPU, b.act);
    Tensor weights({kPool, K / 16, 2 * N}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    int64_t const wbytes = static_cast<int64_t>(kPool) * K / 16 * 2 * N * 4;
    fillRandom(weights, wbytes);
    Tensor sortedIds({capacity}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor expertIds({slots}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor post({1}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    Tensor topkW({capacity}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    Tensor workspace({ws}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    CUDA_CHECK(cudaMemset(workspace.rawPointer(), 0, ws * 4));
    CUDA_CHECK(cudaMemset(input.rawPointer(), 0, static_cast<size_t>(rows) * K * 2));
    std::vector<float> w(capacity, 0.5F);
    CUDA_CHECK(cudaMemcpy(topkW.rawPointer(), w.data(), w.size() * 4, cudaMemcpyHostToDevice));

    Tensor scales, globalScales;
    if (b.fp4)
    {
        scales = Tensor({kPool, K / 16, N}, DeviceType::kGPU, nvinfer1::DataType::kINT8);
        // realistic E4M3 block scales: positive, normal (exponent field 6..11). Fully random bytes put ~12% of the
        // scale pairs on the zero/subnormal path of the pre-sm89 software conversion (R25ga did not notice: sm89 converts in hardware).
        {
            std::vector<uint8_t> sb(static_cast<size_t>(kPool) * K / 16 * N);
            std::mt19937 rg(7);
            for (auto& v : sb) v = static_cast<uint8_t>((6 + rg() % 6) << 3 | (rg() & 7));
            CUDA_CHECK(cudaMemcpy(scales.rawPointer(), sb.data(), sb.size(), cudaMemcpyHostToDevice));
        }
        globalScales = Tensor({kPool}, DeviceType::kGPU, b.act);
        CUDA_CHECK(cudaMemset(globalScales.rawPointer(), 0x3c, kPool * 2));
    }
    else
    {
        scales = Tensor({kPool, K / b.groupSize, N}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
        std::vector<half> s(static_cast<size_t>(kPool) * K / b.groupSize * N, __float2half(0.01F));
        CUDA_CHECK(cudaMemcpy(scales.rawPointer(), s.data(), s.size() * 2, cudaMemcpyHostToDevice));
    }

    std::mt19937 g(7);
    std::vector<Routing> routings;
    for (int i = 0; i < kRoutings; ++i) routings.push_back(makeRouting(g, slots));
    std::vector<Tensor> sortedPer, expertPer;
    auto launch = [&](Routing const& r) {
        CUDA_CHECK(cudaMemcpy(sortedIds.rawPointer(), r.sorted.data(), r.sorted.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(expertIds.rawPointer(), r.experts.data(), r.experts.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(post.rawPointer(), &r.post, 4, cudaMemcpyHostToDevice));
        if (b.fp4)
            kernel::moeNvfp4A16MarlinGemm(input, output, weights, scales, globalScales, sortedIds, expertIds, post, topkW,
                workspace, kBlock, topK, false, nullptr);
        else
            kernel::moeAwqW4A16MarlinGemm(input, output, weights, scales, sortedIds, expertIds, post, topkW, workspace,
                kBlock, topK, false, nullptr);
    };
    // Indices are re-uploaded outside the timed region by capturing one graph per routing set.
    std::vector<cudaGraphExec_t> execs(kRoutings);
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));
    for (int i = 0; i < kRoutings; ++i)
    {
        // Each routing set gets its own index tensors so the graph replays without copies.
        sortedPer.emplace_back(Tensor({capacity}, DeviceType::kGPU, nvinfer1::DataType::kINT32));
        expertPer.emplace_back(Tensor({slots}, DeviceType::kGPU, nvinfer1::DataType::kINT32));
        CUDA_CHECK(cudaMemcpy(sortedPer[i].rawPointer(), routings[i].sorted.data(), routings[i].sorted.size() * 4,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(expertPer[i].rawPointer(), routings[i].experts.data(), routings[i].experts.size() * 4,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(post.rawPointer(), &routings[i].post, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        if (b.fp4)
            kernel::moeNvfp4A16MarlinGemm(input, output, weights, scales, globalScales, sortedPer[i], expertPer[i], post,
                topkW, workspace, kBlock, topK, false, st);
        else
            kernel::moeAwqW4A16MarlinGemm(input, output, weights, scales, sortedPer[i], expertPer[i], post, topkW,
                workspace, kBlock, topK, false, st);
        cudaGraph_t gr;
        CUDA_CHECK(cudaStreamEndCapture(st, &gr));
        CUDA_CHECK(cudaGraphInstantiate(&execs[i], gr, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(gr));
    }
    (void) launch;
    for (int i = 0; i < kRoutings; ++i) CUDA_CHECK(cudaGraphLaunch(execs[i], st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, st));
    for (int i = 0; i < kIters; ++i) CUDA_CHECK(cudaGraphLaunch(execs[i % kRoutings], st));
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    for (auto e : execs) cudaGraphExecDestroy(e);
    cudaStreamDestroy(st);
    return ms * 1000.F / kIters; // us per GEMM
}
} // namespace

TEST(MoeMarlinFp4VsInt4Bench, DecodeInkling2Shapes)
{
    Bench const cases[] = {
        {true, nvinfer1::DataType::kHALF, 16, "nvfp4 fp16-act"},
        {true, nvinfer1::DataType::kBF16, 16, "nvfp4 bf16-act"},
        {false, nvinfer1::DataType::kHALF, 32, "int4 g32 fp16-act"},
        {false, nvinfer1::DataType::kHALF, 128, "int4 g128 fp16-act"},
    };
    // bytes read per layer visit: 8 experts x (2304*2304 + 1152*2304) weights x 0.5 B (+ scales)
    double const weightBytes = 8.0 * (2304.0 * 2304 + 1152.0 * 2304) * 0.5;
    for (auto const& c : cases)
    {
        float const guUs = runGemm(c, 2304, 2304, 1, kTopK);
        float const dnUs = runGemm(c, 1152, 2304, kTopK, 1);
        double const tot = guUs + dnUs;
        printf("BENCH %-20s gate_up %7.2f us  down %7.2f us  visit %7.2f us  => %6.1f GB/s on weights only; 48 visits = %6.2f ms/token\n",
            c.name, guUs, dnUs, tot, weightBytes / (tot * 1e-6) / 1e9, tot * 48 / 1000.0);
    }
}

// Verify-step cost: the union of the routed experts of T tokens (U distinct experts, one 8-row block each) per layer visit, NVFP4 fp16 activations. For T uniform
// random tokens U is about 128 * (1 - (1 - 8/128)^T): 15.5, 22.6, 29.2, 35.4, 41.2, 51.4 for T = 2, 3, 4, 5, 6, 8.
TEST(MoeMarlinFp4VsInt4Bench, VerifyStepExpertUnion)
{
    Bench const c{true, nvinfer1::DataType::kHALF, 16, "nvfp4 fp16-act"};
    for (int32_t u : {8, 12, 16, 24, 32, 40, 48, 64})
    {
        float const guUs = runGemm(c, 2304, 2304, 1, u);
        float const dnUs = runGemm(c, 1152, 2304, u, 1);
        printf("BENCH union %2d experts: gate_up %7.2f us  down %7.2f us  visit %7.2f us  48 visits = %6.2f ms\n", u, guUs, dnUs, guUs + dnUs,
            (guUs + dnUs) * 48 / 1000.0);
    }
}
