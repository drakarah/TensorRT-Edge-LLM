# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical validation of the FMHA-v2 paged prefill kernel with learned attention sinks.

Compiles the paged context kernel (``FMHAV2Ampere.__call_context_paged__`` for the plain variants and
``__call_context_paged_sink__`` for the sink variants) against live torch tensors, runs it over a
(batch, seqlen_q, seqlen_k, window, sink-vector) matrix, and compares with an FP32 torch reference that
adds the sink as an extra softmax column without a value contribution:

    out = sum_j exp(l_j - m) v_j / (sum_j exp(l_j - m) + exp(s_h - m)),   l_j = scale * q . k_j

It also runs the has_sink=False kernel next to a pristine copy of the module (``--base-module``) and checks
the outputs are bitwise equal, so the sink work cannot have changed the existing variants.

Run from the kernelSrcs directory (needs a CUDA GPU, torch and nvidia-cutlass-dsl):

    python3 fmha_v2_cutedsl/test_fmha_v2_sink.py [--ckpt DIR] [--quick] [--base-module NAME]
"""
import argparse
import glob
import importlib
import json
import math
import os
import struct
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

HQ, HKV, D, PAGE = 32, 4, 128, 128


def load_softmax_offsets(ckpt_dir):
    """Return {layer_index: float32 [32]} of ``self_attn.softmax_offset`` read from safetensors shards."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors"))):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            for name, meta in hdr.items():
                if name.endswith("self_attn.softmax_offset"):
                    lo, hi = meta["data_offsets"]
                    fh.seek(8 + n + lo)
                    raw = torch.frombuffer(bytearray(fh.read(hi - lo)), dtype=torch.bfloat16)
                    out[int(name.split(".")[2])] = raw.float().clone()
    return out


def _mark_cu(t):
    t = from_dlpack(t, assumed_align=16)
    return t.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, stride_order=(0,))


class Variant:
    """One compiled paged kernel: (has_sink, sliding_window) for head_dim 128, Br=Bc=64, 128 threads."""

    def __init__(self, mod, has_sink, sliding):
        kw = dict(head_dim=D, m_block_size=64, n_block_size=64, num_threads=128, is_causal=True,
                  use_sliding_window=sliding, skip_rescale=True)
        if has_sink:
            kw["has_sink"] = True
        self.kernel = mod.FMHAV2Ampere(**kw)
        self.has_sink = has_sink
        self.compiled = None

    def __call__(self, q, pool, plist, o, cu_q, cu_k, window, scale, sink=None):
        args = [
            self._q(q), self._pool(pool), self._plist(plist), self._q(o), _mark_cu(cu_q), _mark_cu(cu_k),
        ]
        tail = [cutlass.Int32(window), cutlass.Float32(scale), cutlass.Int32(8),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream)]
        if self.has_sink:
            args.append(_mark_cu(sink))
            fn = self.kernel.__call_context_paged_sink__
        else:
            fn = self.kernel.__call_context_paged__
        if self.compiled is None:
            self.compiled = cute.compile(fn, *args, *tail)
        self.compiled(*args, *tail)
        torch.cuda.synchronize()

    @staticmethod
    def _q(t):
        t = from_dlpack(t, assumed_align=16)
        so = (0, 1, 2, 3)
        return (t.mark_compact_shape_dynamic(mode=0, stride_order=so)
                .mark_compact_shape_dynamic(mode=1, stride_order=so)
                .mark_compact_shape_dynamic(mode=2, stride_order=so))

    @staticmethod
    def _pool(phys):
        t = from_dlpack(phys.permute(0, 2, 1, 3), assumed_align=16)
        so = (0, 2, 1, 3)
        return (t.mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(mode=0, stride_order=so)
                .mark_compact_shape_dynamic(mode=1, stride_order=so))

    @staticmethod
    def _plist(t):
        t = from_dlpack(t, assumed_align=16)
        so = (0, 1, 2)
        return (t.mark_layout_dynamic(leading_dim=2).mark_compact_shape_dynamic(mode=0, stride_order=so)
                .mark_compact_shape_dynamic(mode=2, stride_order=so))


def make_case(b, sq, sk, seed, q_lens=None, k_lens=None):
    """Random Q/K/V plus a scrambled, poisoned paged pool holding K/V.

    ``q_lens`` / ``k_lens`` (optional, length ``b``) give per-batch logical lengths; ``sq`` / ``sk`` are then the
    padded extents and the cumulative-length tensors carry the real lengths.
    """
    q_lens = q_lens or [sq] * b
    k_lens = k_lens or [sk] * b
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(b, sq, HQ, D, device="cuda", generator=g) * 1.5).half()
    k = (torch.randn(b, sk, HKV, D, device="cuda", generator=g) * 1.5).half()
    v = torch.randn(b, sk, HKV, D, device="cuda", generator=g).half()
    maxp = (sk + PAGE - 1) // PAGE
    per = maxp + 1
    npg = b * per
    pool = torch.empty(2 * npg, PAGE, HKV, D, device="cuda", dtype=torch.half)
    pool[:npg] = -127.0
    pool[npg:] = 109.0
    plist = torch.empty(b, 2, maxp, dtype=torch.int32)
    rng = torch.Generator().manual_seed(seed)
    for bi in range(b):
        base = bi * per
        kp = base + 1 + torch.randperm(maxp, generator=rng)
        vp = base + 1 + torch.randperm(maxp, generator=rng)
        plist[bi, 0] = base
        plist[bi, 1] = npg + base
        for lp in range((k_lens[bi] + PAGE - 1) // PAGE):
            lo, hi = lp * PAGE, min(lp * PAGE + PAGE, k_lens[bi])
            plist[bi, 0, lp] = int(kp[lp])
            plist[bi, 1, lp] = npg + int(vp[lp])
            pool[int(kp[lp]), : hi - lo] = k[bi, lo:hi]
            pool[npg + int(vp[lp]), : hi - lo] = v[bi, lo:hi]
    cu_q = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32).cuda()
    cu_k = torch.tensor([0] + torch.tensor(k_lens).cumsum(0).tolist(), dtype=torch.int32).cuda()
    return q, k, v, pool, plist.cuda(), cu_q, cu_k


def reference(q, k, v, scale, window, sink_sets):
    """FP32 reference with sinks; returns one output per entry of ``sink_sets`` (None = no sink)."""
    b, sq, _, _ = q.shape
    sk = k.shape[1]
    qi = torch.arange(sq, device="cuda")[:, None]
    ki = torch.arange(sk, device="cuda")[None, :]
    off = sk - sq
    mask = ki <= qi + off
    if window >= 0:
        mask &= ki >= qi + off - window
    outs = [torch.zeros(q.shape, device="cuda", dtype=torch.float32) for _ in sink_sets]
    grp = HQ // HKV
    for bi in range(b):
        for h in range(HQ):
            kh, vh = k[bi, :, h // grp].float(), v[bi, :, h // grp].float()
            s = (q[bi, :, h].float() @ kh.T) * scale
            s = s.masked_fill(~mask, float("-inf"))
            for out, sinks in zip(outs, sink_sets):
                if sinks is None:
                    m = s.amax(1, keepdim=True)
                    p = torch.exp(s - m)
                    p = p / p.sum(1, keepdim=True)
                else:
                    sv = sinks[h].to(torch.float32)
                    m = torch.maximum(s.amax(1, keepdim=True), sv)
                    p = torch.exp(s - m)
                    p = p / (p.sum(1, keepdim=True) + torch.exp(sv - m))
                out[bi, :, h] = p @ vh
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/work/flint-int4")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--base-module", default=None, help="pristine copy of fmha.py for the bitwise check")
    ap.add_argument("--module", default="fmha")
    args = ap.parse_args()
    mod = importlib.import_module(args.module)
    base = importlib.import_module(args.base_module) if args.base_module else None

    sink_sets = {"none": None}
    if os.path.isdir(args.ckpt):
        offs = load_softmax_offsets(args.ckpt)
        for layer in sorted(offs)[:: max(1, len(offs) // 4)]:
            sink_sets[f"layer{layer}"] = offs[layer].cuda()
        allv = torch.cat(list(offs.values()))
        print(f"softmax_offset over {len(offs)} layers: min {allv.min():.3f} max {allv.max():.3f} "
              f"mean {allv.mean():.3f}")
    sink_sets["+30"] = torch.full((HQ,), 30.0, device="cuda")
    sink_sets["-30"] = torch.full((HQ,), -30.0, device="cuda")
    sink_sets["0"] = torch.zeros(HQ, device="cuda")
    sink_sets["+-30mix"] = torch.where(torch.arange(HQ, device="cuda") % 2 == 0, 30.0, -30.0)
    sink_sets["+1000"] = torch.full((HQ,), 1000.0, device="cuda")
    names = list(sink_sets)
    scale = 1.0 / math.sqrt(D)

    variants = {}
    for sliding in (False, True):
        variants[(True, sliding)] = Variant(mod, True, sliding)
        variants[(False, sliding)] = Variant(mod, False, sliding)
    base_variants = {s: Variant(base, False, s) for s in (False, True)} if base else {}

    sqs = [1, 7, 128, 1000] if args.quick else [1, 7, 128, 1000, 4096]
    cases = []
    for b in (1, 2):
        for sq in sqs:
            if b == 2 and sq == 4096 and args.quick:
                continue
            cases.append((b, sq, sq))
            if sq in (7, 1000, 4096) or sq == 1:
                cases.append((b, sq, sq + 777))
    print(f"{'B':>2} {'Sq':>5} {'Sk':>5} {'win':>4} {'sink':>9} {'maxabs':>9} {'rel_l2':>9} {'plain_maxabs':>12}")
    worst = 0.0
    bit_ok = True
    for (b, sq, sk) in cases:
        q, k, v, pool, plist, cu_q, cu_k = make_case(b, sq, sk, 1234 + sq)
        for sliding in (False, True):
            window = 512 if sliding else -1
            refs = reference(q, k, v, scale, window, [sink_sets[n] for n in names])
            for n, ref in zip(names, refs):
                o = torch.zeros_like(q)
                if n == "none":
                    variants[(False, sliding)](q, pool, plist, o, cu_q, cu_k, max(window, 0), scale)
                    kind = "plain"
                    if base is not None:
                        ob = torch.zeros_like(q)
                        base_variants[sliding](q, pool, plist, ob, cu_q, cu_k, max(window, 0), scale)
                        same = torch.equal(o, ob)
                        bit_ok &= same
                        if not same:
                            print(f"   BITWISE MISMATCH vs base: B{b} Sq{sq} Sk{sk} win{window}")
                else:
                    variants[(True, sliding)](q, pool, plist, o, cu_q, cu_k, max(window, 0), scale,
                                              sink_sets[n])
                    kind = n
                err = (o.float() - ref).abs()
                mabs = err.max().item()
                rnorm = ref.norm().item()
                mrel = (err.norm().item() / rnorm) if rnorm > 1e-3 else float("nan")
                worst = max(worst, mabs)
                extra = ""
                if kind != "plain" and n != "none":
                    # How far the sink-free kernel is from this reference: shows the sink is exercised.
                    o0 = torch.zeros_like(q)
                    variants[(False, sliding)](q, pool, plist, o0, cu_q, cu_k, max(window, 0), scale)
                    extra = f"{(o0.float() - ref).abs().max().item():12.4f}"
                print(f"{b:>2} {sq:>5} {sk:>5} {window:>4} {n:>9} {mabs:9.5f} {mrel:9.5f} {extra:>12}")
    # Ragged batch: per-batch logical lengths below the padded extents (chunked prefill with different
    # prefix lengths per request). Only the valid rows are compared.
    q_lens, k_lens = [100, 37, 128], [400, 237, 128]
    q, k, v, pool, plist, cu_q, cu_k = make_case(3, 128, 640, 99, q_lens, k_lens)
    for sliding in (False, True):
        window = 512 if sliding else -1
        for n in ("layer0", "+30", "-30", "+-30mix"):
            if n not in sink_sets:
                continue
            o = torch.zeros_like(q)
            variants[(True, sliding)](q, pool, plist, o, cu_q, cu_k, max(window, 0), scale, sink_sets[n])
            mabs = 0.0
            for bi in range(3):
                ql, kl = q_lens[bi], k_lens[bi]
                ref = reference(q[bi:bi + 1, :ql], k[bi:bi + 1, :kl], v[bi:bi + 1, :kl], scale, window,
                                [sink_sets[n]])[0]
                mabs = max(mabs, (o[bi:bi + 1, :ql].float() - ref).abs().max().item())
            worst = max(worst, mabs)
            print(f"ragged q{q_lens} k{k_lens} win{window:>3} {n:>8} maxabs {mabs:.5f}")
    print(f"WORST max abs error over matrix: {worst:.5f}")
    if base is not None:
        print("BITWISE has_sink=False vs base module:", "IDENTICAL" if bit_ok else "DIFFERENT")
    print("PASS" if worst < 0.05 and bit_ok else "FAIL")


if __name__ == "__main__":
    main()
