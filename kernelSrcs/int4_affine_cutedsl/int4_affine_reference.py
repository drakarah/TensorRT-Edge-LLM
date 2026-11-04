# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Torch host-side helpers for the MLX affine INT4 (W4A16) kernels.

Used by the standalone test and by any offline repack step; the ``--export_only``
AOT path never imports it.

Checkpoint (MLX affine, group size 64) contract for a linear with out_features
N and in_features K:

  * weight : UINT8  [N, K/2]   low nibble = even k, high nibble = odd k
  * scales : BF16   [N, K/64]
  * biases : BF16   [N, K/64]
  * dequant: w[n, k] = scales[n, k // 64] * q[n, k] + biases[n, k // 64],  q in 0..15

    C[M, N] = A[M, K] @ w^T            (A, C FP16; FP32 accumulate)

Kernel-side tensors produced here:

  * ``repack_weights_for_tile``  -> int32 ``[num_n_blocks*num_k_tiles*kn, 128]``;
    the same fragment-order buffer as the symmetric kernel (bN=128/bK=64) but the
    nibbles hold the RAW ``q`` (the symmetric repack stores ``signed + 8``; the
    in-kernel ``- 8`` is identical).  Out-of-range N rows are padded with 8.
  * ``repack_scales_biases``     -> fp16 ``scales_t [K/64, N]`` and
    ``bias_t [K/64, N]`` with ``bias_t = round_fp16(bias + 8 * scale)`` computed in
    float32.  Both are the plain transpose of the checkpoint layout (this is the
    layout the symmetric kernel already expects for its scales); no further
    permutation is applied, and the bias is permuted exactly like the scales.
"""

from __future__ import annotations

import torch

GROUP_SIZE = 64
_THREADS = 128
# (bit position, N-offset selector, K offset) of the eight nibbles of a word.
_NIBBLES = [
    (0, 0, 0), (4, 0, 8), (8, 32, 0), (12, 32, 8),
    (16, 0, 1), (20, 0, 9), (24, 32, 1), (28, 32, 9),
]


def unpack_mlx_nibbles(weight: torch.Tensor) -> torch.Tensor:
    """uint8 ``[N, K/2]`` -> int16 ``q [N, K]`` in 0..15 (low nibble = even k)."""
    if weight.dtype != torch.uint8:
        raise TypeError("weight must be uint8")
    w = weight.to(torch.int16)
    return torch.stack([w & 0xF, (w >> 4) & 0xF], dim=-1).flatten(-2)


def dequant_reference(weight, scales, biases) -> torch.Tensor:
    """FP32 dequantisation ``s * q + b`` of the checkpoint tensors -> ``[N, K]``."""
    q = unpack_mlx_nibbles(weight).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=1)
    b = biases.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=1)
    return q * s + b


def affine_reference(act: torch.Tensor, weight, scales, biases) -> torch.Tensor:
    """FP32 reference ``act @ dequant(...)^T`` (returned FP32, un-rounded)."""
    w = dequant_reference(weight, scales, biases)
    return act.to(torch.float32) @ w.t()


def repack_scales_biases(scales: torch.Tensor, biases: torch.Tensor):
    """Checkpoint ``[N, K/64]`` BF16 scales/biases -> kernel ``[K/64, N]`` fp16 pair.

    ``bias_t = fp16(float32(bias) + 8 * float32(scale))``; ``scales_t`` is the
    fp16 cast of the BF16 scale (exact unless the scale underflows fp16).
    """
    s32 = scales.to(torch.float32)
    b32 = biases.to(torch.float32)
    scales_t = s32.to(torch.float16).t().contiguous()
    bias_t = (b32 + 8.0 * s32).to(torch.float16).t().contiguous()
    return scales_t, bias_t


def repack_weights_for_tile(weight: torch.Tensor, bN: int = 128, bK: int = 64) -> torch.Tensor:
    """MLX ``weight [N, K/2]`` uint8 -> fragment-order int32 ``[rows, 128]``.

    Word layout (see ``int4_fp16_gemm_cutedsl/int4_reference.py``): for N-block
    ``nb``, K-tile ``kt``, word ``idx = kbl * n_pairs + p`` and thread ``t``,
    ``n_base = t // 4``, ``kb = 2 * (t % 4)``, ``K0 = kt*bK + 16*kbl + kb``,
    ``Nlo = nb*bN + n_base + 64*p``, ``Nhi = Nlo + 32``:

        bits  0-3 (Nlo,K0)   4-7 (Nlo,K0+8)   8-11 (Nhi,K0)   12-15 (Nhi,K0+8)
        bits 16-19(Nlo,K0+1) 20-23(Nlo,K0+9) 24-27 (Nhi,K0+1) 28-31 (Nhi,K0+9)

    The nibble is the raw ``q``.  Requires ``K % bK == 0``.  Vectorised: runs on
    the device of ``weight`` (use CUDA for the 150144-row lm_head).
    """
    if bN % 64 != 0 or bK % 16 != 0:
        raise ValueError("repacked tile requires bN%64==0 and bK%16==0")
    n, k_half = weight.shape
    k = 2 * k_half
    if k % bK != 0:
        raise ValueError(f"K={k} must be a multiple of bK={bK}")
    dev = weight.device
    k_blocks, n_pairs = bK // 16, bN // 64
    kn = k_blocks * n_pairs
    num_n_blocks = (n + bN - 1) // bN
    num_k_tiles = k // bK
    n_pad = num_n_blocks * bN

    q = unpack_mlx_nibbles(weight).to(torch.int32)
    if n_pad != n:
        q = torch.cat([q, torch.full((n_pad - n, k), 8, dtype=torch.int32, device=dev)], 0)

    # flat index (n_local * bK + k_local) of each of the 8 nibbles, [kbl, p, t, 8]
    t = torch.arange(_THREADS, device=dev)
    kbl = torch.arange(k_blocks, device=dev).view(-1, 1, 1)
    p = torch.arange(n_pairs, device=dev).view(1, -1, 1)
    n_base = (t // 4).view(1, 1, -1)
    kb = (2 * (t % 4)).view(1, 1, -1)
    idx = []
    shifts = []
    for shift, n_off, k_off in _NIBBLES:
        n_loc = n_base + 64 * p + n_off + 0 * kbl
        k_loc = 16 * kbl + kb + k_off + 0 * p
        idx.append(n_loc * bK + k_loc)
        shifts.append(shift)
    idx = torch.stack(idx, dim=-1)                         # [kbl, p, t, 8]
    shifts = torch.tensor(shifts, dtype=torch.int64, device=dev)

    out = torch.empty((num_n_blocks, num_k_tiles, k_blocks, n_pairs, _THREADS),
                      dtype=torch.int32, device=dev)
    chunk = 64
    for n0 in range(0, num_n_blocks, chunk):
        n1 = min(n0 + chunk, num_n_blocks)
        blk = q[n0 * bN:n1 * bN].view(n1 - n0, bN, num_k_tiles, bK)
        blk = blk.permute(0, 2, 1, 3).reshape(n1 - n0, num_k_tiles, bN * bK)
        nib = blk[:, :, idx.reshape(-1)].view(n1 - n0, num_k_tiles, k_blocks, n_pairs, _THREADS, 8)
        word = torch.zeros(nib.shape[:-1], dtype=torch.int64, device=dev)
        for j in range(8):
            word |= nib[..., j].to(torch.int64) << int(shifts[j])
        out[n0:n1] = _to_i32(word)
    return out.reshape(num_n_blocks * num_k_tiles * kn, _THREADS).contiguous()


def _to_i32(word: torch.Tensor) -> torch.Tensor:
    """Reinterpret the low 32 bits of an int64 tensor as int32 (wraps >= 2**31)."""
    return torch.where(word >= (1 << 31), word - (1 << 32), word).to(torch.int32)


def repacked_rows(n: int, k: int, bN: int = 128, bK: int = 64) -> int:
    """Row count of the repacked weight buffer."""
    return ((n + bN - 1) // bN) * ((k + bK - 1) // bK) * (bK // 16) * (bN // 64)
