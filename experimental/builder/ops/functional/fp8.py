# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FP8-weight, FP16-activation linear (decode GEMV / prefill GEMM plugin)."""

import numpy as np

from ..tensor import Tensor
from ._operation import operation, parameter

__all__ = ["fp8_linear", "int8_linear"]


def fp8_linear(hidden_states: Tensor,
               weight_bytes: np.ndarray,
               weight_scale: np.ndarray,
               *,
               name: str) -> Tensor:
    """y[T, N] = weight_scale[n] * sum_k hidden_states[T, k] * fp8(weight_bytes[n, k]).

    ``weight_bytes`` is the uint8 ``[N, K]`` array of E4M3 codes (K a multiple of 32, N of 8) and ``weight_scale`` one
    scale per output row. The activations stay FP16: unlike the TensorRT FP8 matmul no activation scale is involved.
    Both arrays are baked into the engine (the plugin takes the codes as int32 words).
    """
    if weight_bytes.dtype != np.uint8 or weight_bytes.ndim != 2:
        raise ValueError(f"{name}: fp8 weights must be a uint8 [N, K] array")
    n, k = weight_bytes.shape
    if n % 8 != 0 or k % 32 != 0:
        raise ValueError(f"{name}: fp8_linear needs N % 8 == 0 and K % 32 == 0, got [{n}, {k}]")
    scale = np.ascontiguousarray(np.asarray(weight_scale, np.float16).reshape(-1))
    if scale.shape[0] != n:
        raise ValueError(f"{name}: expected {n} weight scales, got {scale.shape[0]}")
    words = np.ascontiguousarray(weight_bytes).view(np.int32).reshape(n, k // 4)
    return operation("fp8_linear", [
        hidden_states,
        parameter(name + ".fp8_words", words, "fp8_linear"),
        parameter(name + ".fp8_scale", scale, "fp8_linear")
    ],
                     out_features=n,
                     in_features=k)


def int8_linear(hidden_states: Tensor,
                weight_int8: np.ndarray,
                weight_scale: np.ndarray,
                *,
                name: str) -> Tensor:
    """y[T, N] = weight_scale[n] * sum_k hidden_states[T, k] * weight_int8[n, k]: INT8 weight-only linear (FP16 activations, FP32 accumulation).

    ``weight_int8`` is the int8 ``[N, K]`` array (K a multiple of 32, N of 8), ``weight_scale`` one scale per output row. Runs on the
    Fp8LinearPlugin with ``weight_format=1``: a decode GEMV at memory bandwidth and a tensor-core GEMM for prefill. No FP8 hardware is needed.
    """
    if weight_int8.dtype != np.int8 or weight_int8.ndim != 2:
        raise ValueError(f"{name}: int8 weights must be an int8 [N, K] array")
    n, k = weight_int8.shape
    if n % 8 != 0 or k % 32 != 0:
        raise ValueError(f"{name}: int8_linear needs N % 8 == 0 and K % 32 == 0, got [{n}, {k}]")
    scale = np.ascontiguousarray(np.asarray(weight_scale, np.float32).reshape(-1))
    if scale.shape[0] != n:
        raise ValueError(f"{name}: expected {n} weight scales, got {scale.shape[0]}")
    words = np.ascontiguousarray(weight_int8).view(np.int32).reshape(n, k // 4)
    return operation("fp8_linear", [
        hidden_states,
        parameter(name + ".int8_words", words, "int8_linear"),
        parameter(name + ".int8_scale", scale, "int8_linear")
    ],
                     out_features=n,
                     in_features=k,
                     weight_format=1)
