// Flint affine W4A16 (w = (q-8)*s + b') explicit instantiations: fp16/kU4/fp16/fp16, group_blocks=4, 4 stages.
// Kept as .cuh (not .cu) so the kernels/*.cu glob in cpp/CMakeLists.txt does not compile it on its own;
// it is included by moeMarlinAffine.cu (same translation-unit style as moeMarlin.cu).
// clang-format off
#pragma once
#include <cuda_runtime.h>

#include "kernel.h"
#include "marlin_template_affine.h"

namespace MARLIN_NAMESPACE_NAME {

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 256, 1, 8, 8, true, 4, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 128, 1, 8, 4, true, 4, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 256, 1, 8, 8, false, 4, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 128, 1, 8, 4, false, 4, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 256, 2, 16, 4, false, 4, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 128, 2, 8, 4, false, 4, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 256, 3, 16, 4, false, 4, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 128, 3, 8, 4, false, 4, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 256, 4, 16, 4, false, 4, 4, false>( MARLIN_KERNEL_PARAMS );

template __global__ void Marlin<trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kU4.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), trt_edgellm::marlin_dtypes::kFloat16.id(), 128, 4, 8, 4, false, 4, 4, false>( MARLIN_KERNEL_PARAMS );

}
