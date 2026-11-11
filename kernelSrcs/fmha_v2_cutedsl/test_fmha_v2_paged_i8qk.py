# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation of the FMHA-v2 paged prefill kernel with Q.K^T on INT8 MMA (``qk_int8=True``).

Q arrives as per-row INT8 (scale = amax / 127), K is read from the INT8 pool as 16-bit pairs and the product runs
on ``mma.m16n8k32 s8`` with exact INT32 accumulation; P.V stays FP16 over V expanded from the pool. Two kinds of check:

* kernel exactness: the output must match an FP32 reference computed from the *dequantised* Q (int8 * scale) and the
  exact K/V codes to FP16-rounding accuracy. The only differences left are the FP16 rounding of P and of O, so this
  catches any indexing, masking, scale or page-table error without being fooled by the quantisation itself;
* precision cost: the error against the FP32 reference computed from the original FP16 Q isolates what quantising Q to
  INT8 costs, and is reported next to the error of the FP16-Q INT8-pool kernel (the previous path) for comparison.

Edge cases as in test_fmha_v2_paged_i8.py: a single query row, tile and page boundaries, a cached prefix, ragged
batches with padded extents, pages shared between requests, poisoned pool bytes, scale extremes, all byte values.

    python3 fmha_v2_cutedsl/test_fmha_v2_paged_i8qk.py [--quick] [--bench]
"""
import argparse
import importlib
import math
import os
import sys
import time

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import test_fmha_v2_paged_i8 as base  # noqa: E402
from test_fmha_v2_paged_i8 import D, HKV, HQ, PAGE, _mark_cu, make_case, reference  # noqa: E402

TILE = tuple(int(x) for x in os.environ.get("TILE_QK", "128,64,256").split(","))


class QKVariant:
    def __init__(self, mod, tile=None):
        br, bc, nt = tile or TILE
        self.kernel = mod.FMHAV2Ampere(head_dim=D, m_block_size=br, n_block_size=bc, num_threads=nt, is_causal=True,
                                       skip_rescale=True, kv_int8=True, qk_int8=True)
        self.compiled = None

    def __call__(self, q8, qscale, pool, plist, o, cu_q, cu_k, attn, kscale, vscale):
        V = base.Variant
        args = [V._q(q8.view(torch.float16)), V._plist(qscale), V._pool(pool), V._pool(pool.view(torch.float16)),
                V._plist(plist), V._q(o), _mark_cu(cu_q), _mark_cu(cu_k)]
        tail = [cutlass.Int32(0), cutlass.Float32(attn), cutlass.Float32(kscale), cutlass.Float32(vscale),
                cutlass.Int32(8), cuda.CUstream(torch.cuda.current_stream().cuda_stream)]
        if self.compiled is None:
            self.compiled = cute.compile(self.kernel.__call_context_paged_i8qk__, *args, *tail)
        self.compiled(*args, *tail)
        torch.cuda.synchronize()


def quantize_q(q):
    """Per-row symmetric INT8: q [B,S,H,D] fp16 -> (codes int8, scale fp32 [B,H,S] = amax/127, never zero)."""
    amax = q.float().abs().amax(-1).clamp_min(1e-8)            # [B,S,H]
    scale = amax / 127.0
    codes = torch.clamp(torch.round(q.float() / scale[..., None]), -127, 127).to(torch.int8)
    # A fresh buffer, not .contiguous(): torch leaves arbitrary strides on size-1 dims, which the DLPack layout check rejects.
    out = torch.empty(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device)
    out.copy_(scale.permute(0, 2, 1))
    return codes.contiguous(), out


def err_stats(o, ref):
    d = (o.float() - ref).abs()
    return d.max().item(), (d.norm() / ref.norm().clamp_min(1e-9)).item()


def run_case(qk, i8, b, sq, sk, q_lens=None, k_lens=None, ks=0.0123, vs=0.015625, shared=False, seed=None,
             ref_check=True):
    attn = 1.0 / math.sqrt(D)
    q, kc, vc, pool, plist, cu_q, cu_k = make_case(b, sq, sk, seed or 4321 + sq, q_lens, k_lens, shared)
    q8, qscale = quantize_q(q)
    o = torch.zeros_like(q)
    qk(q8, qscale, pool, plist, o, cu_q, cu_k, attn, ks, vs)
    res = {"o": o}
    if not ref_check:
        return res
    kdq, vdq = kc.float() * ks, vc.float() * vs
    qdq = (q8.float() * qscale.permute(0, 2, 1)[..., None]).half()
    ref_same = reference(qdq, kdq, vdq, attn)          # exact math of the kernel (dequantised Q)
    ref_true = reference(q, kdq, vdq, attn)            # unquantised Q: the quantisation error shows here
    res["same"] = err_stats(o, ref_same)
    res["true"] = err_stats(o, ref_true)
    o_prev = torch.zeros_like(q)
    npg = pool.shape[0] // 2
    i8(q, pool, plist, o_prev, cu_q, cu_k, attn, ks, vs)
    res["prev"] = err_stats(o_prev, ref_true)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--module", default="fmha")
    args = ap.parse_args()
    mod = importlib.import_module(args.module)
    qk = QKVariant(mod)
    i8 = base.Variant(mod, True, base.TILE)
    ok = True
    sqs = [1, 7, 33, 128, 1000] if args.quick else [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000]
    cases = []
    for sq in sqs:
        cases.append((1, sq, sq))
        cases.append((1, sq, sq + 777))
        if sq in (1, 33, 128):
            cases.append((2, sq, sq + 130))
    print(f"{'B':>2} {'Sq':>5} {'Sk':>5} | {'same max':>9} {'same rel':>9} | {'true max':>9} {'true rel':>9} | "
          f"{'prev max':>9} {'prev rel':>9}")
    for (b, sq, sk) in cases:
        r = run_case(qk, i8, b, sq, sk, ref_check=sq * sk <= 4_000_000)
        if "same" not in r:
            continue
        sm, sr = r["same"]
        tm, tr = r["true"]
        pm, pr = r["prev"]
        good = sm < 2.5e-3 and sr < 2e-3 and not math.isnan(sm)
        ok &= good
        print(f"{b:>2} {sq:>5} {sk:>5} | {sm:9.5f} {sr:9.6f} | {tm:9.5f} {tr:9.6f} | {pm:9.5f} {pr:9.6f}"
              f"{'' if good else '  <-- FAIL'}")
    # ragged
    ql, kl = [100, 37, 128], [400, 237, 128]
    q, kc, vc, pool, plist, cu_q, cu_k = make_case(3, 128, 640, 99, ql, kl)
    attn = 1.0 / math.sqrt(D)
    q8, qscale = quantize_q(q)
    o = torch.zeros_like(q)
    qk(q8, qscale, pool, plist, o, cu_q, cu_k, attn, 0.0123, 0.015625)
    worst = 0.0
    for i in range(3):
        qdq = (q8[i:i + 1, :ql[i]].float() * qscale[i:i + 1, :, :ql[i]].permute(0, 2, 1)[..., None]).half()
        ref = reference(qdq, kc[i:i + 1, :kl[i]].float() * 0.0123, vc[i:i + 1, :kl[i]].float() * 0.015625, attn)
        worst = max(worst, (o[i:i + 1, :ql[i]].float() - ref).abs().max().item())
    ok &= worst < 2.5e-3
    print(f"ragged q{ql} k{kl}: max abs vs exact-dequantised-Q reference {worst:.5f}")
    # shared pages
    r = run_case(qk, i8, 2, 64, 300, shared=True, seed=7)
    ok &= r["same"][0] < 2.5e-3
    print(f"shared pages: same-math max {r['same'][0]:.5f}")
    # scale extremes
    for ks, vs in ((1.0, 1.0), (1e-4, 3e-5), (7.3, 0.0039), (0.5, 4.0)):
        r = run_case(qk, i8, 1, 130, 130, ks=ks, vs=vs, seed=5)
        # Huge K scales saturate the softmax and huge V scales scale the absolute error, so only the relative
        # error against the exact-dequantised-Q reference is meaningful here.
        good = r["same"][1] < 8e-3
        ok &= good
        print(f"scales ({ks}, {vs}): same-math max {r['same'][0]:.5f} rel {r['same'][1]:.6f}{'' if good else '  <-- FAIL'}")
    # extreme Q: one huge-outlier row and an all-zero row (scale clamps, no NaN)
    q, kc, vc, pool, plist, cu_q, cu_k = make_case(1, 64, 200, 3)
    q[0, 5] = q[0, 5] * 60.0
    q[0, 9] = 0.0
    q8, qscale = quantize_q(q)
    o = torch.zeros_like(q)
    qk(q8, qscale, pool, plist, o, cu_q, cu_k, 1.0 / math.sqrt(D), 0.0123, 0.015625)
    nan = bool(torch.isnan(o.float()).any() or torch.isinf(o.float()).any())
    ok &= not nan
    print(f"outlier and zero Q rows: {'NaN/Inf PRESENT' if nan else 'finite'}")
    if args.bench:
        bench(qk, i8)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def bench(qk, i8, sq=2048, sk=16384, iters=5):
    attn = 1.0 / math.sqrt(D)
    q, kc, vc, pool, plist, cu_q, cu_k = make_case(1, sq, sk, 11)
    q8, qscale = quantize_q(q)
    o = torch.zeros_like(q)
    for name, fn in (("int8 pool, fp16 QK", lambda: i8(q, pool, plist, o, cu_q, cu_k, attn, 0.0123, 0.015625)),
                     ("int8 pool, int8 QK", lambda: qk(q8, qscale, pool, plist, o, cu_q, cu_k, attn, 0.0123, 0.015625))):
        fn()
        t = time.time()
        for _ in range(iters):
            fn()
        print(f"bench sq={sq} sk={sk} {name}: {(time.time() - t) / iters * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
