// Standalone ctypes-loadable wrapper around the cloned affine W4A16 Marlin MoE kernel (no TensorRT / rt::Tensor).
// Build: nvcc -std=c++17 -arch=sm_87 --expt-relaxed-constexpr -shared -Xcompiler -fPIC -I<repo>/cpp affine_capi.cu -o libaffine_capi.so
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "kernels/moe/moe_marlin/marlin_moe_wna16_affine/ops.cuh"
#include "kernels/moe/moe_marlin/marlin_moe_wna16_affine/sm80_kernel_float16_u4_float16_affine.cuh"

namespace
{
int64_t divUp64(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}
} // namespace

// Same formula as getMoeMarlinWorkspaceSize() in moeMarlin.cu (int32 elements: locks + fp32 C_tmp)
extern "C" int64_t affine_workspace_elems(int64_t numTokensPadded, int64_t outDim, int64_t moeBlockSize)
{
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int64_t locks = std::min((outDim / 64) * divUp64(numTokensPadded, moeBlockSize), (int64_t) sms * 4);
    locks = divUp64(locks, 4) * 4;
    int64_t cTmp = std::min(outDim * numTokensPadded, (int64_t) sms * 4 * moeBlockSize * 256);
    if (moeBlockSize == 8)
        cTmp *= 2;
    return locks + cTmp;
}

// A [m, k] fp16, B marlin int32 [E, K/16, 2N], scales/bprime fp16 [E, K/64, N] (marlin permuted), C [m*top_k or m, N].
extern "C" int affine_moe_gemm(void const* A, void const* B, void* C, void* workspace, int64_t workspaceElems,
    void const* scales, void const* bprime, void const* sortedIds, void const* expertIds, void const* numPostPadded,
    void const* topkWeights, int moeBlock, int numExperts, int topK, int mulTopk, int m, int n, int k,
    int64_t numTokensPadded, void* stream)
{
    try
    {
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        int64_t locks = std::min((n / 64) * divUp64(numTokensPadded, moeBlock), (int64_t) sms * 4);
        locks = divUp64(locks, 4) * 4;
        if (workspaceElems < affine_workspace_elems(numTokensPadded, n, moeBlock))
            return -2;
        int32_t* ws = static_cast<int32_t*>(workspace);
        cudaMemsetAsync(ws, 0, locks * sizeof(int32_t), (cudaStream_t) stream);
        namespace md = trt_edgellm::marlin_dtypes;
        marlin_moe_wna16_affine::marlin_mm(const_cast<void*>(A), const_cast<void*>(B), C, ws + locks, nullptr, nullptr,
            const_cast<void*>(scales), nullptr, const_cast<void*>(bprime), nullptr, nullptr, nullptr,
            const_cast<void*>(sortedIds), const_cast<void*>(expertIds), const_cast<void*>(numPostPadded),
            const_cast<void*>(topkWeights), moeBlock, numExperts, topK, mulTopk != 0, m, n, k, ws, md::kFloat16,
            md::kU4, md::kFloat16, md::kFloat16, false, false, true, false, k / 64, 64, dev, (cudaStream_t) stream, -1,
            -1, sms, -1, false, true, false);
        return (int) cudaGetLastError();
    }
    catch (std::exception const& e)
    {
        fprintf(stderr, "affine_moe_gemm: %s\n", e.what());
        return -1;
    }
}
