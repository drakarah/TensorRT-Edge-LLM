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

# Origin: Adapted from SGLang CuTe DSL GDN decode kernel (Apache-2.0):
#   https://github.com/sgl-project/sglang/blob/v0.5.9/python/sglang/jit_kernel/cutedsl_gdn.py
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang Team. All Rights Reserved.

# This version add support for prefill(seq_len T > 1) stage. Uses CuPy only (no PyTorch) and
# is built for edge-llm AOT export and C++ plugin integration.
# CuTe DSL GDN prefill (seq_len T > 1). Test: python gdn_prefill.py --n 16 ... ; AOT: --export_only ...

import argparse
import os
import sys
import time

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

# Register-resident recurrence (sm_80-class GPUs; Jetson Orin is sm_87). One block per (batch, value head,
# 32-column V tile): the V columns of the state recur independently. 4 warps, each owning 8 V columns; lane =
# k_part * 8 + v_local, and each thread keeps K rows [32 * k_part, 32 * k_part + 32) of its column in registers.
# Tokens go in chunks of CHUNK: the block stages the chunk's L2-normalised k, scaled q, gates and its V tile in
# shared memory (one warp per token), then every thread runs the recurrence over the chunk from shared memory
# (8 lanes read the same address) with no block barrier inside the chunk; the h·k and h·q dot products reduce over
# the 4 k_part lanes with two shuffles.
TILE_K = 128        # state head dim K
TILE_V = 32         # V columns per block
NUM_WARPS = 4
NUM_THREADS = 32 * NUM_WARPS
V_PER_WARP = TILE_V // NUM_WARPS    # 8 V columns per warp
K_PARTS = 32 // V_PER_WARP          # 4 lanes share a column
K_PER_THREAD = TILE_K // K_PARTS    # 32 state rows per thread
K_PER_LANE = TILE_K // 32           # k / q elements per lane when staging a token
CHUNK = 16          # tokens staged per block barrier
# Register budget (65536 / (MIN_BLOCKS_PER_SM * NUM_THREADS)): fewer passes of the token loop over the grid of
# n * hv * v / TILE_V blocks (128 for Qwen3.5-4B on Orin's 14 SMs).
MIN_BLOCKS_PER_SM = int(os.environ.get("GDN_PREFILL_MIN_BLOCKS", "5"))

# AOT placeholders; h0_source batch-dense [n, hv, k, v].
AOT_PLACEHOLDER_N = 1
AOT_PLACEHOLDER_H = 8
AOT_PLACEHOLDER_HV = 8
AOT_PLACEHOLDER_K = 128
AOT_PLACEHOLDER_V = 128
AOT_PLACEHOLDER_SEQLEN = 2


def _define_prefill_kernel():
    @cute.kernel
    def gdn_kernel_prefill(
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        num_v_tiles: Int32,
        seq_len: Int32,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_id, _, _ = cute.arch.block_idx()
        v_tile = block_id % num_v_tiles
        batch_idx = block_id // num_v_tiles
        HV = v.layout.shape[2]
        H = q.layout.shape[2]
        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)
        state_slot = cutlass.Int32(i_n)
        if use_state_indices != 0:
            state_slot = cutlass.Int32(state_indices[i_n])

        k_part = lane // V_PER_WARP
        v_in_tile = warp_idx * V_PER_WARP + lane % V_PER_WARP
        v_global = v_tile * TILE_V + v_in_tile
        k_base = k_part * K_PER_THREAD
        ctx_len = cutlass.Int32(context_lengths[i_n])
        r_A = cute.exp(cutlass.Float32(A_log[i_hv]))
        r_dt_bias = cutlass.Float32(dt_bias[i_hv])

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((CHUNK, TILE_K), stride=(TILE_K, 1)), 16)
        sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((CHUNK, TILE_K), stride=(TILE_K, 1)), 16)
        sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((CHUNK, TILE_V), stride=(TILE_V, 1)), 16)
        sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((CHUNK,), stride=(1,)), 16)
        sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((CHUNK,), stride=(1,)), 16)

        rH = cute.make_rmem_tensor((K_PER_THREAD,), cutlass.Float32)
        rK = cute.make_rmem_tensor((K_PER_THREAD,), cutlass.Float32)
        rQ = cute.make_rmem_tensor((K_PER_THREAD,), cutlass.Float32)
        for i in cutlass.range_constexpr(K_PER_THREAD):
            rH[i] = h0_source[(state_slot, i_hv, k_base + i, v_global)]

        num_chunks = (seq_len + CHUNK - 1) // CHUNK
        for chunk in range(num_chunks):
            t0 = chunk * CHUNK
            # ---- Stage the chunk: warp w prepares tokens t0 + w, t0 + w + NUM_WARPS, ...
            for c in range(warp_idx, CHUNK, NUM_WARPS):
                t = t0 + c
                if cutlass.Int32(t) < ctx_len:
                    rk = cute.make_rmem_tensor((K_PER_LANE,), cutlass.Float32)
                    rq = cute.make_rmem_tensor((K_PER_LANE,), cutlass.Float32)
                    sum_q = cutlass.Float32(0.0)
                    sum_k = cutlass.Float32(0.0)
                    for j in cutlass.range_constexpr(K_PER_LANE):
                        rk[j] = cutlass.Float32(k[i_n, t, i_h, lane * K_PER_LANE + j])
                        rq[j] = cutlass.Float32(q[i_n, t, i_h, lane * K_PER_LANE + j])
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
                    for j in cutlass.range_constexpr(K_PER_LANE):
                        sK[(c, lane * K_PER_LANE + j)] = rk[j] * k_scale
                        sQ[(c, lane * K_PER_LANE + j)] = rq[j] * q_scale
                    sV[(c, lane)] = cutlass.Float32(v[i_n, t, i_hv, v_tile * TILE_V + lane])
                    if lane == 0:
                        # g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b).
                        x = cutlass.Float32(a[i_n, t, i_hv]) + r_dt_bias
                        softplus_x = x
                        if softplus_beta * x <= softplus_threshold:
                            softplus_x = cutlass.Float32(
                                (cutlass.Float32(1.0) / softplus_beta)
                                * cutlass.Float32(cute.log(cutlass.Float32(1.0) + cute.exp(softplus_beta * x)))
                            )
                        sG[c] = cute.exp(-r_A * softplus_x)
                        sBeta[c] = cutlass.Float32(1.0) / (
                            cutlass.Float32(1.0) + cute.exp(-cutlass.Float32(b[i_n, t, i_hv])))
            cute.arch.barrier()

            # ---- Recurrence over the chunk.
            for c in range(CHUNK):
                t = t0 + c
                if cutlass.Int32(t) < seq_len:
                    if cutlass.Int32(t) < ctx_len:
                        r_g = sG[c]
                        cute.autovec_copy(cute.local_tile(sK[(c, None)], (K_PER_THREAD,), (k_part,)), rK)
                        cute.autovec_copy(cute.local_tile(sQ[(c, None)], (K_PER_THREAD,), (k_part,)), rQ)
                        # correction = v - (g * H)^T k ; v_new = beta * correction.
                        sum_hk = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(K_PER_THREAD):
                            rH[i] = rH[i] * r_g
                            sum_hk += rH[i] * rK[i]
                        for offset in [V_PER_WARP, 2 * V_PER_WARP]:
                            sum_hk += cute.arch.shuffle_sync_bfly(sum_hk, offset=offset, mask=-1, mask_and_clamp=31)
                        v_new = (sV[(c, v_in_tile)] - sum_hk) * sBeta[c]
                        # H = g * H + k v_new^T ; o = H^T q.
                        sum_hq = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(K_PER_THREAD):
                            rH[i] = rH[i] + rK[i] * v_new
                            sum_hq += rH[i] * rQ[i]
                        for offset in [V_PER_WARP, 2 * V_PER_WARP]:
                            sum_hq += cute.arch.shuffle_sync_bfly(sum_hq, offset=offset, mask=-1, mask_and_clamp=31)
                        if k_part == 0:
                            o[(i_n, t, i_hv, v_global)] = cutlass.Float16(sum_hq)
                    else:
                        if k_part == 0:
                            o[(i_n, t, i_hv, v_global)] = cutlass.Float16(0.0)
            cute.arch.barrier()

        for i in cutlass.range_constexpr(K_PER_THREAD):
            h0_source[(state_slot, i_hv, k_base + i, v_global)] = rH[i]

    return gdn_kernel_prefill


_jit_function_prefill = None


def _create_jit_function_prefill():
    gdn_prefill_kernel = _define_prefill_kernel()

    @cute.jit
    def run_prefill(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        state_indices: cute.Tensor,
        use_state_indices: Int32,
        context_lengths: cute.Tensor,
        o: cute.Tensor,
        seq_len: Int32,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_initial_state: cutlass.Constexpr[bool],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        n_batch = q.layout.shape[0]
        hv_dim = v.layout.shape[2]
        num_v_tiles = cute.ceil_div(v.layout.shape[3], TILE_V)
        smem_bytes = 4 * (2 * CHUNK * TILE_K + CHUNK * TILE_V + 2 * CHUNK) + 5 * 16
        gdn_prefill_kernel(
            h0_source,
            state_indices,
            use_state_indices,
            num_v_tiles,
            seq_len,
            q, k, v, a, b,
            A_log, dt_bias, context_lengths, o,
            softplus_beta, softplus_threshold, scale,
            use_qk_l2norm,
        ).launch(
            grid=(n_batch * hv_dim * num_v_tiles, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes,
            min_blocks_per_mp=MIN_BLOCKS_PER_SM,
            stream=stream,
        )

    return run_prefill


def _get_jit_function_prefill():
    global _jit_function_prefill
    if _jit_function_prefill is None:
        _jit_function_prefill = _create_jit_function_prefill()
    return _jit_function_prefill


_compiled_kernels_prefill = {}


def _cp_dtype_fp16():
    return cp.float16


def _get_leading_dim(arr):
    for i, s in enumerate(arr.strides):
        if s == arr.dtype.itemsize:
            return i
    return len(arr.shape) - 1


def _make_placeholder_tensors(n, h, hv, k, v, seq_len):
    dt = _cp_dtype_fp16()
    return {
        "q":          cp.zeros((n, seq_len, h, k), dtype=dt),
        "k":          cp.zeros((n, seq_len, h, k), dtype=dt),
        "v":          cp.zeros((n, seq_len, hv, v), dtype=dt),
        "a":          cp.zeros((n, seq_len, hv), dtype=dt),
        "b":          cp.zeros((n, seq_len, hv), dtype=dt),
        "A_log":      cp.zeros(hv, dtype=cp.float32),
        "dt_bias":    cp.zeros(hv, dtype=dt),
        "h0_source":  cp.zeros((n, hv, k, v), dtype=cp.float32),
        "state_indices": cp.arange(n, dtype=cp.int32),
        "context_lengths": cp.full((n,), seq_len, dtype=cp.int32),
        "o":          cp.zeros((n, seq_len, hv, v), dtype=dt),
    }


def _mark_gdn_prefill_qv_dynamic(tensor):
    so = (0, 1, 2, 3)
    return (tensor.mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so))

def _mark_gdn_1d_dynamic(tensor):
    return (tensor.mark_layout_dynamic(leading_dim=0)
            .mark_compact_shape_dynamic(mode=0, stride_order=(0,)))

def _mark_h0_source_dynamic(tensor):
    """Batch (n) and hv dims dynamic; stride stays static."""
    so = (0, 1, 2, 3)
    return tensor.mark_compact_shape_dynamic(mode=0, stride_order=so).mark_compact_shape_dynamic(
        mode=1, stride_order=so
    )

def _to_cute_tensors(ph):
    def wrap(arr, leading_dim=None, skip_dynamic=False, assumed_align=16):
        ct = from_dlpack(arr, assumed_align=assumed_align)
        if skip_dynamic:
            return ct
        ld = _get_leading_dim(arr) if leading_dim is None else leading_dim
        return ct.mark_layout_dynamic(leading_dim=ld)

    q = wrap(ph["q"])
    v = wrap(ph["v"])
    h0_src = _mark_h0_source_dynamic(from_dlpack(ph["h0_source"], assumed_align=32))
    state_indices = from_dlpack(ph["state_indices"], assumed_align=16)
    state_indices = _mark_gdn_1d_dynamic(state_indices)
    ctx = from_dlpack(ph["context_lengths"], assumed_align=16)
    ctx = ctx.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
    return {
        "q":          _mark_gdn_prefill_qv_dynamic(q),
        "k":          wrap(ph["k"]),
        "v":          _mark_gdn_prefill_qv_dynamic(v),
        "a":          wrap(ph["a"]),
        "b":          wrap(ph["b"]),
        "A_log":      wrap(ph["A_log"], leading_dim=0),
        "dt_bias":    wrap(ph["dt_bias"], leading_dim=0),
        "h0_source":  h0_src,
        "state_indices": state_indices,
        "context_lengths": ctx,
        "o":          wrap(ph["o"]),
    }


def _make_aot_cute_tensors(n, h, hv, k, v, seq_len):
    def compact(dtype, shape, assumed_align=16):
        return aot_placeholders.make_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    q = compact(cutlass.Float16, (n, seq_len, h, k))
    v_tensor = compact(cutlass.Float16, (n, seq_len, hv, v))
    h0_source = compact(cutlass.Float32, (n, hv, k, v), assumed_align=32)
    state_indices = compact(cutlass.Int32, (n,))
    context_lengths = compact(cutlass.Int32, (n,))
    return {
        "q": _mark_gdn_prefill_qv_dynamic(q.mark_layout_dynamic(leading_dim=3)),
        "k": compact(cutlass.Float16, (n, seq_len, h, k)).mark_layout_dynamic(leading_dim=3),
        "v": _mark_gdn_prefill_qv_dynamic(v_tensor.mark_layout_dynamic(leading_dim=3)),
        "a": compact(cutlass.Float16, (n, seq_len, hv)).mark_layout_dynamic(leading_dim=2),
        "b": compact(cutlass.Float16, (n, seq_len, hv)).mark_layout_dynamic(leading_dim=2),
        "A_log": compact(cutlass.Float32, (hv,)).mark_layout_dynamic(leading_dim=0),
        "dt_bias": compact(cutlass.Float16, (hv,)).mark_layout_dynamic(leading_dim=0),
        "h0_source": _mark_h0_source_dynamic(h0_source),
        "state_indices": _mark_gdn_1d_dynamic(state_indices),
        "context_lengths": _mark_gdn_1d_dynamic(context_lengths),
        "o": compact(cutlass.Float16, (n, seq_len, hv, v)).mark_layout_dynamic(leading_dim=3),
    }


def _compile_prefill(n, h, hv, k, v, seq_len, stream, gpu_arch="", export_only=False):
    if "_" in _compiled_kernels_prefill:
        return _compiled_kernels_prefill["_"]

    if export_only:
        t = _make_aot_cute_tensors(n, h, hv, k, v, seq_len)
    else:
        ph = _make_placeholder_tensors(n, h, hv, k, v, seq_len)
        t = _to_cute_tensors(ph)
    run_prefill = _get_jit_function_prefill()
    # Only the export path may pin a foreign target arch in the compile
    # options; a native JIT run must compile for the local GPU (see the
    # native-vs-cross note in cutedsl_utils/cutedsl_compile_wrapper.py).
    compile_opts = aot_placeholders.compile_options(
        f"--gpu-arch={gpu_arch}" if gpu_arch else ""
    ) if export_only else None
    compiled = cute.compile(
        run_prefill,
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
        seq_len,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=k ** -0.5,
        use_initial_state=True,
        use_qk_l2norm=True,
        stream=stream,
        **(dict(options=compile_opts) if compile_opts else {}),
    )
    _compiled_kernels_prefill["_"] = compiled
    return compiled


def export_gdn_prefill(n, h, hv, k, v, seq_len,
                       output_dir, file_name, function_prefix, gpu_arch=""):
    if seq_len < 2:
        raise ValueError("Prefill requires seq_len >= 2.")
    stream = aot_placeholders.make_stream()
    print("[gdn_prefill] AOT compile gpu_arch=%r" % (gpu_arch or "default"))
    t0 = time.time()
    compiled = _compile_prefill(
        n, h, hv, k, v, seq_len, stream, gpu_arch=gpu_arch, export_only=True
    )
    print("[gdn_prefill] Compilation time: %.4fs" % (time.time() - t0))

    os.makedirs(output_dir, exist_ok=True)
    compiled.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    print("[gdn_prefill] Exported to %s/%s.h and %s.o" % (output_dir, file_name, file_name))
    return compiled


def _softplus_np(x, beta=1.0, threshold=20.0):
    bx = beta * x
    if bx <= threshold:
        return (1.0 / beta) * np.log(1.0 + np.exp(bx))
    return float(x)


def _run_numpy_prefill_reference(
    q_f32, k_f32, v_f32, a_f32, b_f32,
    A_log_f32, dt_bias_f32, h0_source_f32,
    n, h, hv, k_dim, v_dim, seq_len, scale,
    context_lengths_np,
    use_qk_l2norm=True,
):
    h0 = h0_source_f32.copy()  # (n, HV, K, V) batch-dense
    o_ref = np.zeros((n, seq_len, hv, v_dim), dtype=np.float32)

    for i_n in range(n):
        for i_hv in range(hv):
            i_h = i_hv // (hv // h) if h else 0
            H = h0[i_n, i_hv].copy()  # (K, V)
            max_t = int(context_lengths_np[i_n])
            for t in range(seq_len):
                if t >= max_t:
                    o_ref[i_n, t, i_hv, :] = 0.0
                    continue
                q_vec = q_f32[i_n, t, i_h].astype(np.float64)
                k_vec = k_f32[i_n, t, i_h].astype(np.float64)
                v_vec = v_f32[i_n, t, i_hv].astype(np.float64)
                a_val = float(a_f32[i_n, t, i_hv])
                b_val = float(b_f32[i_n, t, i_hv])
                A_val = float(A_log_f32[i_hv])
                dt_val = float(dt_bias_f32[i_hv])

                if use_qk_l2norm:
                    nq = np.sqrt(np.sum(q_vec ** 2) + 1e-6)
                    nk = np.sqrt(np.sum(k_vec ** 2) + 1e-6)
                    q_eff = q_vec / nq * scale
                    k_eff = k_vec / nk
                else:
                    q_eff = q_vec * scale
                    k_eff = k_vec

                sp = _softplus_np(a_val + dt_val, 1.0, 20.0)
                g = np.exp(-np.exp(A_val) * sp)
                beta = 1.0 / (1.0 + np.exp(-b_val))

                H_gated = H * g                           # (K, V)
                correction = v_vec - H_gated.T @ k_eff    # (V,)
                v_new = correction * beta                  # (V,)
                H = H_gated + np.outer(k_eff, v_new)      # (K, V)
                o_ref[i_n, t, i_hv] = H.T @ q_eff
            h0[i_n, i_hv] = H
    return o_ref, h0


def _prefill_context_lengths_array(n, seq_len, preset):
    """INT32 [n]: valid token count per row; must be in [0, seq_len]."""
    if preset == "full" or preset is None:
        return np.full((n,), seq_len, dtype=np.int32)
    if preset == "half":
        t = max(1, seq_len // 2)
        return np.full((n,), t, dtype=np.int32)
    if preset == "staggered":
        out = np.empty((n,), dtype=np.int32)
        span = max(1, seq_len - 1)
        for i in range(n):
            out[i] = max(1, seq_len - (i % span))
        return out
    raise ValueError("Unknown context_lengths preset: %r" % (preset,))


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test_prefill(n, h, hv, k, v, seq_len,
                     skip_ref_check=False, tolerance=0.2,
                     warmup=3, iterations=100, gpu_arch="",
                     context_lengths_preset="full"):
    if seq_len < 2:
        raise ValueError("Prefill requires seq_len >= 2.")

    dt = _cp_dtype_fp16()
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    # Generate float32 data, cast to fp16 for kernel input. h0 batch-dense [n, hv, k, v].
    q_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    k_f32  = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    v_f32  = np.random.randn(n, seq_len, hv, v).astype(np.float32) * 0.1
    a_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    b_f32  = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    A_log_f32  = np.random.randn(hv).astype(np.float32) * 0.1
    dt_bias_f32 = np.random.randn(hv).astype(np.float32) * 0.1
    h0_f32 = np.random.randn(n, hv, k, v).astype(np.float32) * 0.01

    ctx_np = _prefill_context_lengths_array(n, seq_len, context_lengths_preset)
    ph = {
        "q":          cp.asarray(q_f32).astype(dt),
        "k":          cp.asarray(k_f32).astype(dt),
        "v":          cp.asarray(v_f32).astype(dt),
        "a":          cp.asarray(a_f32).astype(dt),
        "b":          cp.asarray(b_f32).astype(dt),
        "A_log":      cp.asarray(A_log_f32),
        "dt_bias":    cp.asarray(dt_bias_f32).astype(dt),
        "h0_source":  cp.asarray(h0_f32),
        "state_indices": cp.arange(n, dtype=cp.int32),
        "context_lengths": cp.asarray(ctx_np),
        "o":          cp.zeros((n, seq_len, hv, v), dtype=dt),
    }

    compiled = _compile_prefill(n, h, hv, k, v, seq_len, stream, gpu_arch=gpu_arch)
    t = _to_cute_tensors(ph)
    args = (
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
        seq_len,
        stream,
    )

    for _ in range(warmup):
        compiled(*args)
    cp.cuda.get_current_stream().synchronize()

    if not skip_ref_check:
        # Reset h0_source to original (kernel mutates it).
        ph["h0_source"] = cp.asarray(h0_f32)
        t = _to_cute_tensors(ph)
        args = (
            t["q"], t["k"], t["v"], t["a"], t["b"],
            t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
            seq_len,
            stream,
        )
        compiled(*args)
        cp.cuda.get_current_stream().synchronize()

        o_ref, h0_ref = _run_numpy_prefill_reference(
            q_f32, k_f32, v_f32, a_f32, b_f32,
            A_log_f32, dt_bias_f32, h0_f32,
            n, h, hv, k, v, seq_len, scale=k ** -0.5,
            context_lengths_np=ctx_np,
            use_qk_l2norm=True,
        )
        o_kernel = cp.asnumpy(ph["o"]).astype(np.float32)
        max_err = np.max(np.abs(o_kernel - o_ref))
        print("[gdn_prefill] Max abs error vs NumPy ref: %.6f (tol=%.4f)" % (max_err, tolerance))
        np.testing.assert_allclose(o_kernel, o_ref, atol=tolerance, rtol=1e-2)
        h0_kernel = cp.asnumpy(ph["h0_source"]).astype(np.float32)
        max_err_h0 = np.max(np.abs(h0_kernel - h0_ref))
        print("[gdn_prefill] h0 max abs error vs NumPy ref: %.6f (tol=%.4f)" % (max_err_h0, tolerance))
        np.testing.assert_allclose(h0_kernel, h0_ref, atol=tolerance, rtol=1e-2)
        print("[gdn_prefill] Reference check PASSED (o + h0). context_lengths preset=%r" % (context_lengths_preset,))

    # Benchmark
    ph["h0_source"] = cp.asarray(h0_f32)
    t = _to_cute_tensors(ph)
    args = (
        t["q"], t["k"], t["v"], t["a"], t["b"],
        t["A_log"], t["dt_bias"], t["h0_source"], t["state_indices"], 1, t["context_lengths"], t["o"],
        seq_len,
        stream,
    )
    t0 = time.perf_counter()
    for _ in range(iterations):
        compiled(*args)
    cp.cuda.get_current_stream().synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iterations
    print(
        "[gdn_prefill] Latency: %.4f us "
        "(n=%d T=%d h=%d hv=%d k=%d v=%d)" % (us, n, seq_len, h, hv, k, v)
    )
    return us


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parsed_args
    np.random.seed(42)

    if args.export_only:
        export_gdn_prefill(
            n=AOT_PLACEHOLDER_N,
            h=AOT_PLACEHOLDER_H,
            hv=AOT_PLACEHOLDER_HV,
            k=AOT_PLACEHOLDER_K,
            v=AOT_PLACEHOLDER_V,
            seq_len=AOT_PLACEHOLDER_SEQLEN,
            output_dir=args.output_dir,
            file_name=args.file_name,
            function_prefix=args.function_prefix,
            gpu_arch=args.gpu_arch,
        )
        return

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU required.")
    cp.random.seed(42)

    run_test_prefill(
        n=args.n, h=args.h, hv=args.hv, k=args.k, v=args.v,
        seq_len=args.seq_len,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup=args.warmup,
        iterations=args.iterations,
        gpu_arch=args.gpu_arch,
        context_lengths_preset=args.context_lengths_preset,
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL GDN prefill: AOT export and test (CuPy only, Ampere+)."
    )
    p.add_argument("--export_only", action="store_true")
    p.add_argument("--output_dir", type=str, default="./gdn_aot_artifacts")
    p.add_argument("--file_name", type=str, default="gdn_prefill")
    p.add_argument("--function_prefix", type=str, default="gdn_prefill")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--hv", type=int, default=8)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--v", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument(
        "--context_lengths_preset",
        type=str,
        default="full",
        choices=("full", "half", "staggered"),
        help="Prefill mask test: full=T for all rows; half=all rows use max(1,T//2) valid tokens; "
        "staggered=each row i uses max(1, T - i%%(T-1)) valid tokens (variable-length within batch).",
    )
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--gpu_arch", type=str, default="",
                   help="Target GPU arch for export (e.g. sm_87 for Orin). Empty = current GPU.")
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
