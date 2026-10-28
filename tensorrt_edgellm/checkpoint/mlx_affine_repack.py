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
"""Repacking of MLX affine INT4 (group 64) linears for ``Int4AffineGroupwiseGemmPlugin``.

Checkpoint contract for a linear with out_features N and in_features K::

    weight : uint8 [N, K/2]   low nibble = even k, high nibble = odd k
    scales : bf16  [N, K/64]
    biases : bf16  [N, K/64]
    w[n, k] = scales[n, k // 64] * q[n, k] + biases[n, k // 64],  q in 0..15

Plugin inputs produced here:

* ``qweight`` int8 ``[rows, 512]`` with ``rows = ceil(N/128) * (K/64) * 8``: the cuteDSL
  fragment-order buffer (bN=128, bK=64, the same layout as the symmetric
  ``Int4GroupwiseGemmPluginV2``) holding the RAW nibbles ``q`` (the kernel subtracts 8).
  Out-of-range N rows are padded with 8.
* ``scales`` fp16 ``[K/64, N]`` and ``bias_prime`` fp16 ``[K/64, N]`` with
  ``bias_prime = fp16(float32(bias) + 8 * float32(scale))`` (the kernel evaluates
  ``fma(q - 8, scale, bias_prime)``).

The word layout mirrors ``kernelSrcs/int4_affine_cutedsl/int4_affine_reference.py``
(``repack_weights_for_tile``), which the standalone kernel tests validate; it is
reimplemented here so the package never imports from ``kernelSrcs`` at runtime.
"""

from typing import Tuple

import torch

MLX_AFFINE_GROUP_SIZE = 64
_BN = 128
_BK = 64
_THREADS = 128
# (bit position, N-offset selector, K offset) of the eight nibbles of a 32-bit word.
_NIBBLES = ((0, 0, 0), (4, 0, 8), (8, 32, 0), (12, 32, 8), (16, 0, 1),
            (20, 0, 9), (24, 32, 1), (28, 32, 9))

__all__ = [
    "MLX_AFFINE_GROUP_SIZE",
    "mlx_affine_fragment_rows",
    "unpack_mlx_nibbles",
    "repack_mlx_affine_weights",
    "repack_mlx_affine_scales_biases",
]


def mlx_affine_fragment_rows(n: int, k: int) -> int:
    """Row count of the fragment weight buffer for an ``[N, K]`` linear."""
    return ((n + _BN - 1) // _BN) * (k // _BK) * ((_BK // 16) * (_BN // 64))


def unpack_mlx_nibbles(weight: torch.Tensor) -> torch.Tensor:
    """uint8 ``[N, K/2]`` -> int16 ``q [N, K]`` in 0..15 (low nibble = even k)."""
    if weight.dtype != torch.uint8:
        raise TypeError(f"MLX affine weight must be uint8, got {weight.dtype}")
    w = weight.to(torch.int16)
    return torch.stack([w & 0xF, (w >> 4) & 0xF], dim=-1).flatten(-2)


def _to_i32(word: torch.Tensor) -> torch.Tensor:
    """Reinterpret the low 32 bits of an int64 tensor as int32."""
    return torch.where(word >= (1 << 31), word - (1 << 32),
                       word).to(torch.int32)


def repack_mlx_affine_weights(weight: torch.Tensor) -> torch.Tensor:
    """MLX ``weight [N, K/2]`` uint8 -> fragment buffer int8 ``[rows, 512]``.

    Runs on the device of *weight* (move it to CUDA first for the large lm_head).
    Requires ``K % 64 == 0`` and ``N % 8 == 0``.
    """
    n, k_half = weight.shape
    k = 2 * k_half
    if k % _BK != 0 or n % 8 != 0:
        raise ValueError(
            f"MLX affine INT4 repack needs K%64==0 and N%8==0, got N={n}, K={k}"
        )
    dev = weight.device
    k_blocks, n_pairs = _BK // 16, _BN // 64
    kn = k_blocks * n_pairs
    num_n_blocks = (n + _BN - 1) // _BN
    num_k_tiles = k // _BK
    n_pad = num_n_blocks * _BN

    q = unpack_mlx_nibbles(weight).to(torch.int32)
    if n_pad != n:
        q = torch.cat(
            [q, torch.full((n_pad - n, k), 8, dtype=torch.int32, device=dev)],
            0)

    # flat index (n_local * bK + k_local) of each of the 8 nibbles, [kbl, p, t, 8]
    t = torch.arange(_THREADS, device=dev)
    kbl = torch.arange(k_blocks, device=dev).view(-1, 1, 1)
    p = torch.arange(n_pairs, device=dev).view(1, -1, 1)
    n_base = (t // 4).view(1, 1, -1)
    kb = (2 * (t % 4)).view(1, 1, -1)
    idx = []
    for _, n_off, k_off in _NIBBLES:
        n_loc = n_base + 64 * p + n_off + 0 * kbl
        k_loc = 16 * kbl + kb + k_off + 0 * p
        idx.append(n_loc * _BK + k_loc)
    idx = torch.stack(idx, dim=-1)  # [kbl, p, t, 8]
    shifts = [s for s, _, _ in _NIBBLES]

    out = torch.empty((num_n_blocks, num_k_tiles, k_blocks, n_pairs, _THREADS),
                      dtype=torch.int32,
                      device=dev)
    chunk = 64
    for n0 in range(0, num_n_blocks, chunk):
        n1 = min(n0 + chunk, num_n_blocks)
        blk = q[n0 * _BN:n1 * _BN].view(n1 - n0, _BN, num_k_tiles, _BK)
        blk = blk.permute(0, 2, 1, 3).reshape(n1 - n0, num_k_tiles, _BN * _BK)
        nib = blk[:, :, idx.reshape(-1)].view(n1 - n0, num_k_tiles, k_blocks,
                                              n_pairs, _THREADS, 8)
        word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=dev)
        for j in range(8):
            word |= nib[..., j].to(torch.int64) << shifts[j]
        out[n0:n1] = _to_i32(word)
    frag = out.reshape(num_n_blocks * num_k_tiles * kn,
                       _THREADS).contiguous()  # [rows, 128] int32
    return frag.view(torch.int8).reshape(frag.shape[0], _THREADS * 4)


def repack_mlx_affine_scales_biases(
        scales: torch.Tensor,
        biases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint ``[N, K/64]`` scales/biases -> kernel ``[K/64, N]`` fp16 pair.

    ``bias_prime = fp16(float32(bias) + 8 * float32(scale))``; ``scales`` is the fp16 cast
    (exact for BF16 input unless the scale underflows fp16).
    """
    s32 = scales.to(torch.float32)
    b32 = biases.to(torch.float32)
    scales_t = s32.to(torch.float16).t().contiguous()
    bias_t = (b32 + 8.0 * s32).to(torch.float16).t().contiguous()
    return scales_t, bias_t
