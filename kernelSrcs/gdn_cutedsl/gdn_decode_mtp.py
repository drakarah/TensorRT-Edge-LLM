# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# CuTe DSL GDN MTP (Multi-Token Processing) decode kernel.
#
# Used for speculative-decoding verification: the draft model produces T tokens
# per batch item; this kernel evolves the GDN recurrent state over all T steps
# and produces T output tokens per batch item.
#
# Key differences vs gdn_decode (seq_len=1):
#   1. seq_len = T >= 1; outer loop over T time steps.
#   2. Intermediate h-states are cached after each step for rollback support
#      (compile-time opt-in via cache_intermediate_states Constexpr).
#   3. Two-phase warp-specialized approach: Phase 1 precomputes sQ/sK/sG/sBeta
#      for all T steps into SMEM; Phase 2 processes V-tiles with state evolution,
#      reading per-step q/k/g/beta from SMEM (avoids T repeated global loads).
#
# Reference: FlashInfer gdn_decode_mtp.py (Apache-2.0) — adapted for edge-llm
#   AOT export pipeline (CuPy only, no PyTorch).
#   Source: https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_kernels/gdn_decode_mtp.py
#   SGLang integration: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py
#
# Test:  python gdn_decode_mtp.py --n 4 --h 8 --hv 8 --seq_len 4
# AOT:   python gdn_decode_mtp.py --export_only [--cache] [--gpu_arch sm_87]

import argparse
import os
import sys
import time
from typing import Dict, Tuple

_parsed_args = None
_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cutedsl_utils import aot_placeholders  # isort: skip

# ---------------------------------------------------------------------------
# Compile-time kernel constants (shared with gdn_decode.py for compatibility)
# ---------------------------------------------------------------------------
TILE_K = 128
TILE_V = 32
TILE_V_PADDED = 36
NUM_STAGES = 2
NUM_THREADS = 256       # 8 warps — same as gdn_decode large_batch
NUM_WARPS = 8
V_PER_WARP = 4          # TILE_V // NUM_WARPS
ROWS_PER_ITER = 8       # k-rows swept per inner k_iter step
NUM_K_ITERS = TILE_K // ROWS_PER_ITER   # = 16

# Max T steps baked into the Phase-1 SMEM layout. AOT export uses this value.
# At runtime, seq_len can be 1..T_MAX_AOT.
T_MAX_AOT = 16

# AOT placeholder shapes — only N/H/HV are dynamic shape markers.
AOT_PLACEHOLDER_N   = 1
AOT_PLACEHOLDER_H   = 14
AOT_PLACEHOLDER_HV  = 14
AOT_PLACEHOLDER_K   = 128
AOT_PLACEHOLDER_V   = 128
AOT_PLACEHOLDER_T   = 8     # representative T for compilation


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------

# Register-resident verify kernel (sm_80-class GPUs; Jetson Orin is sm_87), same layout as gdn_prefill.py: one block
# per (batch, value head, 32-column V tile), 4 warps of 8 V columns, each thread keeps 32 state rows of one column in
# registers. The state is read once from its resident pool row (h0_source[state_indices[n]]); each step writes only its
# output row. The caching (verify) variant never writes the pool and writes no per-step state snapshots: it stores the
# replay stash of the DDTree chunk verify (gdnTreeChunkKernels.h: per step L2-normalised k, gate g, beta, fp16 v) at
# the head of each batch row of intermediate_states, and the runtime commit replays the accepted steps from the pool
# (gdnTreeReplayCommitBatched with the path 0..L-1). That replaces T x 2 MB of snapshot writes per layer (Qwen3.5-4B)
# with ~16 KB of stash per step. (The previous kernel reloaded and stored every state tile from global memory
# for every step and V tile.) One warp per step stages the step's L2-normalised k, scaled q, gates and V tile.
REG_TILE_V = 32
REG_NUM_WARPS = 4
REG_NUM_THREADS = 32 * REG_NUM_WARPS
REG_K_PARTS = REG_NUM_WARPS                      # warp w owns state rows 32w..32w+31, lane = V column
REG_K_PER_THREAD = TILE_K // REG_K_PARTS         # 32 state rows per thread
REG_K_PER_LANE = TILE_K // 32                    # k / q elements per lane when staging a step
REG_MIN_BLOCKS_PER_SM = 5                        # register budget, see gdn_prefill.py


def _define_mtp_kernels():
    """Return (gdn_mtp_kernel_warpspec,) — one JIT launcher per Constexpr config."""

    @cute.kernel
    def gdn_mtp_kernel_warpspec(
        h0_source: cute.Tensor,           # [pool_rows, hv, K, V] FP32 resident state pool
        state_indices: cute.Tensor,       # [n] Int32: pool row of each sequence (out of range = zero state)
        num_v_tiles: Int32,
        q: cute.Tensor,                   # [n, seq_len, h, K] FP16
        k: cute.Tensor,                   # [n, seq_len, h, K] FP16
        v: cute.Tensor,                   # [n, seq_len, hv, V] FP16
        a: cute.Tensor,                   # [n, seq_len, hv] FP16
        b: cute.Tensor,                   # [n, seq_len, hv] FP16
        A_log: cute.Tensor,               # [hv] FP32
        dt_bias: cute.Tensor,             # [hv] FP16
        o: cute.Tensor,                   # [n, seq_len, hv, V] FP16 (output)
        intermediate_states: cute.Tensor, # [n, seq_len, hv, K, V] FP32 buffer; caching variant: replay stash at row heads
        seq_len: Int32,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],  # write the replay stash
        T_MAX: cutlass.Constexpr[int],    # SMEM prealloc upper bound (== T_MAX_AOT)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_id, _, _ = cute.arch.block_idx()
        v_tile = block_id % num_v_tiles
        head_block = block_id // num_v_tiles
        H = q.layout.shape[2]
        HV = v.layout.shape[2]
        i_n = head_block // HV
        i_hv = head_block % HV
        i_h = i_hv // (HV // H)

        # Warp = row group, lane = column: every state load is one contiguous 128-byte line per warp.
        k_part = warp_idx
        v_in_tile = lane
        v_global = v_tile * REG_TILE_V + v_in_tile
        k_base = k_part * REG_K_PER_THREAD
        r_A = cute.exp(cutlass.Float32(A_log[i_hv]))
        r_dt_bias = cutlass.Float32(dt_bias[i_hv])

        # Replay stash (float offsets): batch row = seq_len * HV * K * V floats; per step a cell of
        # [kNorm H*K f32 | g HV f32 | beta HV f32 | v HV*V f16] packed at cell_floats.
        cell_floats = H * TILE_K + 2 * HV + HV * (TILE_K // 2)
        row_floats = seq_len * HV * TILE_K * TILE_K
        stash_f32 = cute.make_tensor(intermediate_states.iterator, cute.make_layout(1 << 30))
        stash_f16 = cute.make_tensor(
            cute.recast_ptr(intermediate_states.iterator, dtype=cutlass.Float16), cute.make_layout(1 << 30))

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_MAX, TILE_K), stride=(TILE_K, 1)), 16)
        sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_MAX, TILE_K), stride=(TILE_K, 1)), 16)
        sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_MAX, REG_TILE_V), stride=(REG_TILE_V, 1)), 16)
        sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_MAX,), stride=(1,)), 16)
        sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_MAX,), stride=(1,)), 16)
        # Row-group partial sums, combined as (p0 + p1) + (p2 + p3) (the former xor-shuffle order).
        sPartHK = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((REG_K_PARTS, REG_TILE_V), stride=(REG_TILE_V, 1)), 16)
        sPartHQ = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((REG_K_PARTS, REG_TILE_V), stride=(REG_TILE_V, 1)), 16)

        # ---- Stage all steps: warp w prepares steps w, w + NUM_WARPS, ...
        for t in range(warp_idx, seq_len, REG_NUM_WARPS):
            rk = cute.make_rmem_tensor((REG_K_PER_LANE,), cutlass.Float32)
            rq = cute.make_rmem_tensor((REG_K_PER_LANE,), cutlass.Float32)
            sum_q = cutlass.Float32(0.0)
            sum_k = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(REG_K_PER_LANE):
                rk[j] = cutlass.Float32(k[i_n, t, i_h, lane * REG_K_PER_LANE + j])
                rq[j] = cutlass.Float32(q[i_n, t, i_h, lane * REG_K_PER_LANE + j])
                sum_k += rk[j] * rk[j]
                sum_q += rq[j] * rq[j]
            k_scale = cutlass.Float32(1.0)
            q_scale = cutlass.Float32(scale)
            if use_qk_l2norm:
                for offset in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                    sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                k_scale = cute.rsqrt(sum_k + 1e-6)
                q_scale = q_scale * cute.rsqrt(sum_q + 1e-6)
            for j in cutlass.range_constexpr(REG_K_PER_LANE):
                sK[(t, lane * REG_K_PER_LANE + j)] = rk[j] * k_scale
                sQ[(t, lane * REG_K_PER_LANE + j)] = rq[j] * q_scale
            v_raw = v[i_n, t, i_hv, v_tile * REG_TILE_V + lane]
            sV[(t, lane)] = cutlass.Float32(v_raw)
            cell = i_n * row_floats + t * cell_floats
            if cutlass.const_expr(cache_intermediate_states):
                if v_tile == 0:
                    if i_hv % (HV // H) == 0:  # one block per k head writes its normalised k
                        for j in cutlass.range_constexpr(REG_K_PER_LANE):
                            stash_f32[cell + i_h * TILE_K + lane * REG_K_PER_LANE + j] = rk[j] * k_scale
                stash_f16[2 * (cell + H * TILE_K + 2 * HV) + i_hv * TILE_K + v_tile * REG_TILE_V + lane] = v_raw
            if lane == 0:
                # g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b).
                x = cutlass.Float32(a[i_n, t, i_hv]) + r_dt_bias
                softplus_x = x
                if softplus_beta * x <= softplus_threshold:
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta)
                        * cutlass.Float32(cute.log(cutlass.Float32(1.0) + cute.exp(softplus_beta * x)))
                    )
                sG[t] = cute.exp(-r_A * softplus_x)
                sBeta[t] = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-cutlass.Float32(b[i_n, t, i_hv])))
                if cutlass.const_expr(cache_intermediate_states):
                    if v_tile == 0:
                        stash_f32[cell + H * TILE_K + i_hv] = sG[t]
                        stash_f32[cell + H * TILE_K + HV + i_hv] = sBeta[t]
        cute.arch.barrier()

        pool_row = state_indices[i_n]
        valid_row = Int32(0)
        if pool_row >= 0:
            if pool_row < h0_source.layout.shape[0]:
                valid_row = Int32(1)
        rH = cute.make_rmem_tensor((REG_K_PER_THREAD,), cutlass.Float32)
        rK = cute.make_rmem_tensor((REG_K_PER_THREAD,), cutlass.Float32)
        rQ = cute.make_rmem_tensor((REG_K_PER_THREAD,), cutlass.Float32)
        for i in cutlass.range_constexpr(REG_K_PER_THREAD):
            rH[i] = cutlass.Float32(0.0)
        if valid_row != 0:
            for i in cutlass.range_constexpr(REG_K_PER_THREAD):
                rH[i] = h0_source[(pool_row, i_hv, k_base + i, v_global)]

        for t in range(seq_len):
            r_g = sG[t]
            cute.autovec_copy(cute.local_tile(sK[(t, None)], (REG_K_PER_THREAD,), (k_part,)), rK)
            cute.autovec_copy(cute.local_tile(sQ[(t, None)], (REG_K_PER_THREAD,), (k_part,)), rQ)
            # correction = v - (g * H)^T k ; v_new = beta * correction.
            sum_hk = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(REG_K_PER_THREAD):
                rH[i] = rH[i] * r_g
                sum_hk += rH[i] * rK[i]
            sPartHK[(k_part, v_in_tile)] = sum_hk
            cute.arch.barrier()
            sum_hk = (sPartHK[(0, v_in_tile)] + sPartHK[(1, v_in_tile)]) + (
                sPartHK[(2, v_in_tile)] + sPartHK[(3, v_in_tile)])
            v_new = (sV[(t, v_in_tile)] - sum_hk) * sBeta[t]
            # H = g * H + k v_new^T ; o = H^T q.
            sum_hq = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(REG_K_PER_THREAD):
                rH[i] = rH[i] + rK[i] * v_new
                sum_hq += rH[i] * rQ[i]
            sPartHQ[(k_part, v_in_tile)] = sum_hq
            cute.arch.barrier()
            if k_part == 0:
                sum_hq = (sPartHQ[(0, v_in_tile)] + sPartHQ[(1, v_in_tile)]) + (
                    sPartHQ[(2, v_in_tile)] + sPartHQ[(3, v_in_tile)])
                o[i_n, t, i_hv, v_global] = cutlass.Float16(sum_hq)

        if cutlass.const_expr(not cache_intermediate_states):
            if valid_row != 0:
                for i in cutlass.range_constexpr(REG_K_PER_THREAD):
                    h0_source[(pool_row, i_hv, k_base + i, v_global)] = rH[i]

    return (gdn_mtp_kernel_warpspec,)


_jit_mtp = None


def _get_jit_mtp():
    global _jit_mtp
    if _jit_mtp is None:
        _jit_mtp = _define_mtp_kernels()
    return _jit_mtp


# ---------------------------------------------------------------------------
# JIT wrapper functions  (copy_atom / layouts created inside @cute.jit context)
# ---------------------------------------------------------------------------

_jit_mtp_wrappers = None


def _create_jit_mtp_wrappers():
    (gdn_mtp_kernel_warpspec,) = _get_jit_mtp()

    @cute.jit
    def run_mtp(
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        intermediate_states: cute.Tensor,
        seq_len: Int32,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        T_MAX: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        n_batch = q.layout.shape[0]
        hv_dim = v.layout.shape[2]
        num_v_tiles = cute.ceil_div(v.layout.shape[3], REG_TILE_V)
        smem_bytes = 4 * (2 * T_MAX * TILE_K + T_MAX * REG_TILE_V + 2 * T_MAX + 2 * REG_K_PARTS * REG_TILE_V) + 7 * 16

        gdn_mtp_kernel_warpspec(
            h0_source,
            state_indices,
            num_v_tiles,
            q, k, v, a, b,
            A_log, dt_bias,
            o,
            intermediate_states,
            seq_len,
            softplus_beta,
            softplus_threshold,
            scale,
            use_qk_l2norm,
            cache_intermediate_states,
            T_MAX,
        ).launch(
            grid=(n_batch * hv_dim * num_v_tiles, 1, 1),
            block=[REG_NUM_THREADS, 1, 1],
            smem=smem_bytes,
            min_blocks_per_mp=REG_MIN_BLOCKS_PER_SM,
            stream=stream,
        )

    return run_mtp


def _get_jit_mtp_wrapper():
    global _jit_mtp_wrappers
    if _jit_mtp_wrappers is None:
        _jit_mtp_wrappers = _create_jit_mtp_wrappers()
    return _jit_mtp_wrappers


_compiled_mtp: Dict[Tuple, object] = {}


# ---------------------------------------------------------------------------
# Placeholder / tensor helpers  (mirrors gdn_decode.py patterns)
# ---------------------------------------------------------------------------

def _cp_dtype_fp16():
    return cp.float16


def _make_mtp_placeholder_tensors(n, h, hv, k, v, seq_len, with_cache):
    dt = _cp_dtype_fp16()
    ph = {
        "q":             cp.zeros((n, seq_len, h, k),    dtype=dt),
        "k":             cp.zeros((n, seq_len, h, k),    dtype=dt),
        "v":             cp.zeros((n, seq_len, hv, v),   dtype=dt),
        "a":             cp.zeros((n, seq_len, hv),      dtype=dt),
        "b":             cp.zeros((n, seq_len, hv),      dtype=dt),
        "A_log":         cp.zeros(hv,                    dtype=cp.float32),
        "dt_bias":       cp.zeros(hv,                    dtype=dt),
        "h0_source":     cp.zeros((n, hv, k, v),         dtype=cp.float32),
        "state_indices": cp.arange(n,                    dtype=cp.int32),
        "o":             cp.zeros((n, seq_len, hv, v),   dtype=dt),
    }
    # Always allocate a full 5D tensor so MLIR type matches 5D store coordinates
    # even in the no-cache path (the store is guarded by Constexpr but the
    # MLIR verifier checks type congruence before dead-code elimination).
    ph["intermediate_states"] = cp.zeros((n, seq_len, hv, k, v), dtype=cp.float32)
    return ph


def _mark_dynamic_4d(tensor):
    so = (0, 1, 2, 3)
    return (tensor.mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so)
              .mark_compact_shape_dynamic(mode=3, stride_order=so))


def _mark_dynamic_3d(tensor):
    so = (0, 1, 2)
    return (tensor.mark_layout_dynamic(leading_dim=2)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so))


def _mark_h0_source_dynamic(tensor):
    so = (0, 1, 2, 3)
    return (tensor.mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so))


def _mark_intermediate_states_dynamic(tensor, with_cache):
    # Both variants use a full 5D tensor now; mark identically.
    so = (0, 1, 2, 3, 4)
    ct = from_dlpack(tensor, assumed_align=32)
    return (ct.mark_layout_dynamic(leading_dim=4)
              .mark_compact_shape_dynamic(mode=0, stride_order=so)
              .mark_compact_shape_dynamic(mode=1, stride_order=so)
              .mark_compact_shape_dynamic(mode=2, stride_order=so)
              .mark_compact_shape_dynamic(mode=3, stride_order=so))


def _to_mtp_cute_tensors(ph, with_cache):
    def wrap(arr, leading_dim=None):
        ct = from_dlpack(arr, assumed_align=16)
        ld = (len(arr.shape) - 1) if leading_dim is None else leading_dim
        return ct.mark_layout_dynamic(leading_dim=ld)

    q_ct  = wrap(ph["q"])
    v_ct  = wrap(ph["v"])
    return {
        "q":          _mark_dynamic_4d(q_ct),
        "k":          wrap(ph["k"]),
        "v":          _mark_dynamic_4d(v_ct),
        "a":          _mark_dynamic_3d(wrap(ph["a"])),
        "b":          _mark_dynamic_3d(wrap(ph["b"])),
        "A_log":      wrap(ph["A_log"], leading_dim=0),
        "dt_bias":    wrap(ph["dt_bias"], leading_dim=0),
        "h0_source":  _mark_h0_source_dynamic(from_dlpack(ph["h0_source"], assumed_align=32)),
        "state_indices": wrap(ph["state_indices"], leading_dim=0).mark_compact_shape_dynamic(mode=0),
        "o":          wrap(ph["o"]),
        "intermediate_states": _mark_intermediate_states_dynamic(ph["intermediate_states"], with_cache),
    }


def _make_aot_mtp_cute_tensors(n, h, hv, k, v, seq_len):
    def compact(dtype, shape, assumed_align=16):
        return aot_placeholders.make_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    q = compact(cutlass.Float16, (n, seq_len, h, k))
    v_tensor = compact(cutlass.Float16, (n, seq_len, hv, v))
    a = compact(cutlass.Float16, (n, seq_len, hv))
    b = compact(cutlass.Float16, (n, seq_len, hv))
    h0_source = compact(cutlass.Float32, (n, hv, k, v), assumed_align=32)
    intermediate_states = compact(
        cutlass.Float32, (n, seq_len, hv, k, v), assumed_align=32
    )
    return {
        "q": _mark_dynamic_4d(q.mark_layout_dynamic(leading_dim=3)),
        "k": compact(cutlass.Float16, (n, seq_len, h, k)).mark_layout_dynamic(leading_dim=3),
        "v": _mark_dynamic_4d(v_tensor.mark_layout_dynamic(leading_dim=3)),
        "a": _mark_dynamic_3d(a.mark_layout_dynamic(leading_dim=2)),
        "b": _mark_dynamic_3d(b.mark_layout_dynamic(leading_dim=2)),
        "A_log": compact(cutlass.Float32, (hv,)).mark_layout_dynamic(leading_dim=0),
        "dt_bias": compact(cutlass.Float16, (hv,)).mark_layout_dynamic(leading_dim=0),
        "h0_source": _mark_h0_source_dynamic(h0_source),
        "state_indices": compact(cutlass.Int32, (n,)).mark_layout_dynamic(leading_dim=0),
        "o": compact(cutlass.Float16, (n, seq_len, hv, v)).mark_layout_dynamic(leading_dim=3),
        "intermediate_states": (
            intermediate_states.mark_layout_dynamic(leading_dim=4)
            .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
            .mark_compact_shape_dynamic(mode=1, stride_order=(0, 1, 2, 3, 4))
            .mark_compact_shape_dynamic(mode=2, stride_order=(0, 1, 2, 3, 4))
            .mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3, 4))
        ),
    }


def _compile_mtp(
    n, h, hv, k, v, seq_len, with_cache, stream, gpu_arch="", export_only=False
):
    key = (with_cache,)
    if key in _compiled_mtp:
        return _compiled_mtp[key]

    run_mtp = _get_jit_mtp_wrapper()

    if export_only:
        t = _make_aot_mtp_cute_tensors(n, h, hv, k, v, seq_len)
    else:
        ph = _make_mtp_placeholder_tensors(n, h, hv, k, v, seq_len, with_cache)
        t = _to_mtp_cute_tensors(ph, with_cache)

    # Only the export path may pin a foreign target arch in the compile
    # options; a native JIT run must compile for the local GPU (see the
    # native-vs-cross note in cutedsl_utils/cutedsl_compile_wrapper.py).
    compile_opts = aot_placeholders.compile_options(
        f"--gpu-arch={gpu_arch}" if gpu_arch else ""
    ) if export_only else None
    compiled = cute.compile(
        run_mtp,
        t["h0_source"],
        t["state_indices"],
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"],
        t["o"],
        t["intermediate_states"],
        seq_len,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=k ** -0.5,
        use_qk_l2norm=True,
        cache_intermediate_states=with_cache,
        T_MAX=T_MAX_AOT,
        stream=stream,
        **(dict(options=compile_opts) if compile_opts else {}),
    )
    _compiled_mtp[key] = compiled
    return compiled


# ---------------------------------------------------------------------------
# AOT export
# ---------------------------------------------------------------------------

def export_gdn_decode_mtp(
    n, h, hv, k, v,
    output_dir, file_name, function_prefix,
    gpu_arch="",
    cache_only=False,
):
    """AOT-compile and export gdn_decode_mtp kernel (cache variant only).

    Exports into output_dir:
      <file_name>_cache.h/.o   — cache variant (writes intermediate_states per step)

    The cache variant is used for speculative-decoding rollback.
    """
    stream = aot_placeholders.make_stream()
    os.makedirs(output_dir, exist_ok=True)

    fn = file_name + "_cache"
    fp = function_prefix + "_cache"

    print("[gdn_decode_mtp] AOT compile cache variant gpu_arch=%r" % (gpu_arch or "default"))
    t0 = time.time()
    compiled = _compile_mtp(
        n, h, hv, k, v,
        seq_len=AOT_PLACEHOLDER_T,
        with_cache=True,
        stream=stream,
        gpu_arch=gpu_arch,
        export_only=True,
    )
    print("[gdn_decode_mtp] Compilation time: %.4fs" % (time.time() - t0))

    compiled.export_to_c(
        file_path=output_dir,
        file_name=fn,
        function_prefix=fp,
    )
    print("[gdn_decode_mtp] Exported %s/%s.h and %s.o" % (output_dir, fn, fn))


# ---------------------------------------------------------------------------
# NumPy reference implementation
# ---------------------------------------------------------------------------

def _softplus_np(x, beta=1.0, threshold=20.0):
    bx = beta * x
    if bx <= threshold:
        return (1.0 / beta) * np.log(1.0 + np.exp(bx))
    return float(x)


def _run_numpy_mtp_reference(
    q_f32,         # [n, T, h, k]
    k_f32,         # [n, T, h, k]
    v_f32,         # [n, T, hv, v]
    a_f32,         # [n, T, hv]
    b_f32,         # [n, T, hv]
    A_log_f32,     # [hv]
    dt_bias_f32,   # [hv]
    h0_f32,        # [n, hv, k, v]  — updated in-place
    n, h, hv, k_dim, v_dim, seq_len,
    scale,
    use_qk_l2norm=True,
):
    """CPU reference: evolves GDN state over seq_len (T) time steps per batch item.

    Returns:
        o_ref   : [n, T, hv, v] float32
        h0_out  : [n, hv, k, v] float32  (equals h0_f32 after in-place update)
        interm  : [n, T, hv, k, v] float32  (per-step intermediate states)
    """
    h0 = h0_f32.copy()   # don't modify caller's array
    o_ref  = np.zeros((n, seq_len, hv, v_dim), dtype=np.float32)
    interm = np.zeros((n, seq_len, hv, k_dim, v_dim), dtype=np.float32)

    for i_n in range(n):
        for i_hv in range(hv):
            i_h = i_hv // (hv // h) if h else 0
            H = h0[i_n, i_hv].copy()   # [k, v]

            for t in range(seq_len):
                q_vec = q_f32[i_n, t, i_h].astype(np.float64)
                k_vec = k_f32[i_n, t, i_h].astype(np.float64)
                v_vec = v_f32[i_n, t, i_hv].astype(np.float64)
                a_val = float(a_f32[i_n, t, i_hv])
                b_val = float(b_f32[i_n, t, i_hv])

                if use_qk_l2norm:
                    nq = np.sqrt(np.sum(q_vec ** 2) + 1e-6)
                    nk = np.sqrt(np.sum(k_vec ** 2) + 1e-6)
                    q_eff = q_vec / nq * scale
                    k_eff = k_vec / nk
                else:
                    q_eff = q_vec * scale
                    k_eff = k_vec

                A_val  = float(A_log_f32[i_hv])
                dt_val = float(dt_bias_f32[i_hv])
                sp     = _softplus_np(a_val + dt_val, 1.0, 20.0)
                g      = np.exp(-np.exp(A_val) * sp)
                beta   = 1.0 / (1.0 + np.exp(-b_val))

                H_gated    = H * g
                correction = (v_vec - H_gated.T @ k_eff) * beta
                H_new      = H_gated + np.outer(k_eff, correction)
                o_ref[i_n, t, i_hv, :] = (H_new.T @ q_eff).astype(np.float32)

                H = H_new
                interm[i_n, t, i_hv] = H_new.astype(np.float32)

            # Padded steps: o = 0 (already zeroed), state unchanged
            h0[i_n, i_hv] = H.astype(np.float32)

    return o_ref, h0, interm


# ---------------------------------------------------------------------------
# JIT launcher helper
# ---------------------------------------------------------------------------

def _run_mtp_decode_jit(
    q_cp, k_cp, v_cp, a_cp, b_cp,
    A_log_cp, dt_bias_cp,
    h0_cp,
    o_cp, intermediate_states_cp,
    seq_len, with_cache, stream, gpu_arch="",
):
    n   = h0_cp.shape[0]
    hv  = h0_cp.shape[1]
    k   = h0_cp.shape[2]
    v   = h0_cp.shape[3]
    h   = q_cp.shape[2]

    compiled = _compile_mtp(n, h, hv, k, v, seq_len, with_cache, stream, gpu_arch=gpu_arch)

    def wrap(arr, assumed_align=16):
        return from_dlpack(arr, assumed_align=assumed_align)

    compiled(
        wrap(h0_cp, 32),
        wrap(cp.arange(n, dtype=cp.int32)),
        wrap(q_cp), wrap(k_cp), wrap(v_cp),
        wrap(a_cp), wrap(b_cp),
        wrap(A_log_cp), wrap(dt_bias_cp),
        wrap(o_cp),
        wrap(intermediate_states_cp, 32),
        seq_len,
        stream,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def run_test_mtp(
    n=4, h=8, hv=8, k=128, v=128, seq_len=4,
    with_cache=False,
    skip_ref_check=False,
    tolerance=0.1,
    warmup=3,
    iterations=100,
    gpu_arch="",
):
    dt     = _cp_dtype_fp16()
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    scale  = k ** -0.5

    # Generate random FP32 data; convert to FP16 for GPU inputs.
    q_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    k_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    v_f32  = np.random.randn(n, seq_len, hv, v).astype(np.float32) * 0.1
    a_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    b_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    A_log_f32   = np.random.randn(hv).astype(np.float32) * 0.1
    dt_bias_f32 = np.random.randn(hv).astype(np.float32) * 0.1
    h0_f32      = np.random.randn(n, hv, k, v).astype(np.float32) * 0.01

    # GPU arrays
    q_cp  = cp.asarray(q_f32.astype(np.float16))
    k_cp  = cp.asarray(k_f32.astype(np.float16))
    v_cp  = cp.asarray(v_f32.astype(np.float16))
    a_cp  = cp.asarray(a_f32.astype(np.float16))
    b_cp  = cp.asarray(b_f32.astype(np.float16))
    A_log_cp    = cp.asarray(A_log_f32)
    dt_bias_cp  = cp.asarray(dt_bias_f32.astype(np.float16))
    h0_cp       = cp.asarray(h0_f32)
    o_cp = cp.zeros((n, seq_len, hv, v), dtype=dt)

    if with_cache:
        interm_cp = cp.zeros((n, seq_len, hv, k, v), dtype=cp.float32)
    else:
        interm_cp = cp.zeros((1,), dtype=cp.float32)  # dummy

    # Warmup
    for _ in range(warmup):
        h0_warm = cp.asarray(h0_f32)
        _run_mtp_decode_jit(
            q_cp, k_cp, v_cp, a_cp, b_cp,
            A_log_cp, dt_bias_cp,
            h0_warm,
            o_cp, interm_cp, seq_len, with_cache, stream, gpu_arch,
        )
    cp.cuda.get_current_stream().synchronize()

    if not skip_ref_check:
        h0_kernel_cp = cp.asarray(h0_f32)
        o_cp_check   = cp.zeros((n, seq_len, hv, v), dtype=dt)
        if with_cache:
            interm_check = cp.zeros((n, seq_len, hv, k, v), dtype=cp.float32)
        else:
            interm_check = cp.zeros((1,), dtype=cp.float32)

        _run_mtp_decode_jit(
            q_cp, k_cp, v_cp, a_cp, b_cp,
            A_log_cp, dt_bias_cp,
            h0_kernel_cp,
            o_cp_check, interm_check, seq_len, with_cache, stream, gpu_arch,
        )
        cp.cuda.get_current_stream().synchronize()

        o_ref, h0_ref, interm_ref = _run_numpy_mtp_reference(
            q_f32, k_f32, v_f32, a_f32, b_f32,
            A_log_f32, dt_bias_f32,
            h0_f32, n, h, hv, k, v, seq_len, scale,
        )

        o_kernel  = cp.asnumpy(o_cp_check).astype(np.float32)
        h0_kernel = cp.asnumpy(h0_kernel_cp)

        np.testing.assert_allclose(o_kernel, o_ref, atol=tolerance, rtol=1e-2)
        print("[gdn_decode_mtp] Output reference check PASSED.")
        np.testing.assert_allclose(h0_kernel, h0_ref, atol=tolerance, rtol=1e-2)
        print("[gdn_decode_mtp] h0 reference check PASSED.")

        if with_cache:
            interm_kernel = cp.asnumpy(interm_check)
            np.testing.assert_allclose(interm_kernel, interm_ref, atol=tolerance, rtol=1e-2)
            print("[gdn_decode_mtp] Intermediate states reference check PASSED.")

        label = "uniform" + ("+cache" if with_cache else "")
        print("[gdn_decode_mtp] Reference check PASSED (%s)." % label)

    # Benchmark
    h0_bench = cp.asarray(h0_f32)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _run_mtp_decode_jit(
            q_cp, k_cp, v_cp, a_cp, b_cp,
            A_log_cp, dt_bias_cp,
            h0_bench,
            o_cp, interm_cp, seq_len, with_cache, stream, gpu_arch,
        )
    cp.cuda.get_current_stream().synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print("[gdn_decode_mtp] Latency: %.4f us (n=%d h=%d hv=%d k=%d v=%d T=%d)" % (
        us, n, h, hv, k, v, seq_len))
    return us


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parsed_args
    np.random.seed(42)

    if args.export_only:
        export_gdn_decode_mtp(
            n=AOT_PLACEHOLDER_N,
            h=AOT_PLACEHOLDER_H,
            hv=AOT_PLACEHOLDER_HV,
            k=AOT_PLACEHOLDER_K,
            v=AOT_PLACEHOLDER_V,
            output_dir=args.output_dir,
            file_name=args.file_name,
            function_prefix=args.function_prefix,
            gpu_arch=args.gpu_arch,
            cache_only=args.cache_only,
        )
        return

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU required.")
    cp.random.seed(42)

    run_test_mtp(
        n=args.n,
        h=args.h,
        hv=args.hv,
        k=args.k,
        v=args.v,
        seq_len=args.seq_len,
        with_cache=args.cache,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup=args.warmup,
        iterations=args.iterations,
        gpu_arch=args.gpu_arch,
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL GDN MTP decode: AOT export and test (Ampere+)."
    )
    p.add_argument("--export_only",   action="store_true",
                   help="AOT export only (no test run)")
    p.add_argument("--output_dir",    type=str,  default="./gdn_mtp_artifacts")
    p.add_argument("--file_name",     type=str,  default="gdn_decode_mtp")
    p.add_argument("--function_prefix", type=str, default="gdn_decode_mtp")
    p.add_argument("--n",             type=int,  default=4)
    p.add_argument("--h",             type=int,  default=8)
    p.add_argument("--hv",            type=int,  default=8)
    p.add_argument("--k",             type=int,  default=128)
    p.add_argument("--v",             type=int,  default=128)
    p.add_argument("--seq_len",       type=int,  default=4,
                   help="Number of MTP draft tokens T (1 .. T_MAX=%d)" % T_MAX_AOT)
    p.add_argument("--cache",         action="store_true",
                   help="Enable intermediate-state caching in test mode")
    p.add_argument("--cache_only",    action="store_true",
                   help="AOT export: only export the cache variant (skip base)")
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--tolerance",     type=float, default=0.1)
    p.add_argument("--warmup",        type=int,   default=3)
    p.add_argument("--iterations",    type=int,   default=100)
    p.add_argument("--gpu_arch",      type=str,   default="",
                   help="Target GPU arch for export (e.g. sm_87). Empty = current GPU.")
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
