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

"""Standalone validation of the affine INT4 GEMM/GEMV kernels on real checkpoint tensors.

Run inside the dev container (torch + CUDA + nvidia-cutlass-dsl)::

    PYTHONPATH=<repo>/kernelSrcs:<repo>/kernelSrcs/int4_affine_cutedsl \
        python3 test_int4_affine.py --ckpt /work/flint-int4 [--mode all|analysis|kernels]

``analysis``  scans every ``*.scales`` / ``*.biases`` pair of the checkpoint and
              reports the fp16 range/precision of ``s`` and ``b' = b + 8 s`` and
              the dequantised-weight error of ``fma_f16(q - 8, s16, b'16)`` vs the
              fp32 dequantisation ``s * q + b``.
``kernels``   runs the GEMV (M <= 8) and every baked GEMM variant on the Flint
              shapes with real weights, comparing against an FP32 reference.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time

_saved_argv = list(sys.argv)
sys.argv = [sys.argv[0]]  # cutlass inspects argv at import

import cuda.bindings.driver as cuda  # noqa: E402
import cupy as cp  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from common import ceil_div, mark_lock_1d, mark_row_major_2d  # noqa: E402
from int4_affine_gemm_ampere import Int4AffineGemmAmpere  # noqa: E402
from int4_affine_gemv_ampere import Int4AffineGemvAmpere  # noqa: E402
from int4_affine_reference import (  # noqa: E402
    GROUP_SIZE,
    repack_scales_biases,
    repack_weights_for_tile,
    unpack_mlx_nibbles,
)

# (label, tensor prefix, K, N) -- K/N as stated by the task; checked against the file.
SHAPES = [
    ("q_proj 2048->4096", "model.layers.3.self_attn.q_proj", 2048, 4096),
    ("k_proj 2048->512", "model.layers.3.self_attn.k_proj", 2048, 512),
    ("o_proj 4096->2048", "model.layers.0.self_attn.o_proj", 4096, 2048),
    ("dense gate 2048->6144", "model.layers.0.mlp.gate_proj", 2048, 6144),
    ("dense down 6144->2048", "model.layers.0.mlp.down_proj", 6144, 2048),
    ("shared gate 2048->1024", "model.layers.3.mlp.shared_expert.gate_proj", 2048, 1024),
    ("shared down 1024->2048", "model.layers.3.mlp.shared_expert.down_proj", 1024, 2048),
    ("lm_head 2048->150144", "lm_head", 2048, 150144),
]
M_LIST = [1, 2, 8, 16, 64, 512, 2048]
# (bM, stages, split_k) -- the symmetric kernel's 16 baked GEMM variants.
GEMM_VARIANTS = [
    (16, 3, 1), (16, 4, 1), (16, 4, 2), (16, 4, 4),
    (32, 2, 1), (32, 3, 2), (32, 3, 4), (32, 4, 1),
    (64, 2, 1), (64, 3, 1), (64, 3, 4), (64, 4, 2), (64, 4, 4),
    (128, 2, 1), (128, 2, 4), (128, 4, 1),
]
# Default tactic per M (used for the headline table).
DEFAULT_GEMM = {16: (16, 4, 1), 64: (64, 3, 1), 512: (128, 2, 1), 2048: (128, 2, 1)}


class Checkpoint:
    """Lazy safetensors reader over all shards."""

    def __init__(self, path: str):
        self.files = sorted(glob.glob(f"{path}/model-*.safetensors"))
        self.where = {}
        for f in self.files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    self.where[k] = f

    def get(self, name: str, device="cuda") -> torch.Tensor:
        with safe_open(self.where[name], framework="pt", device="cpu") as sf:
            return sf.get_tensor(name).to(device)

    def quant_prefixes(self):
        return sorted(k[: -len(".scales")] for k in self.where if k.endswith(".scales"))


# ---------------------------------------------------------------------------
# fp16 range / precision of s and b' on the real checkpoint
# ---------------------------------------------------------------------------
def analysis(ck: Checkpoint) -> None:
    q = torch.arange(16, device="cuda", dtype=torch.float32).view(16, 1)
    n_groups = 0
    agg = dict(smax=0.0, smin=float("inf"), bpmax=0.0, bpmin=float("inf"), nonfin=0, zeros=0, subn=0,
               s_inexact=0, abs_err=0.0, rel_w_sig=0.0, rel_s_sig=0.0, floor_abs=0.0, excess_abs=0.0,
               excess_rel_sig=0.0, n_small=0, small_abs=0.0,
               ratio_min=float("inf"), ratio_max=-float("inf"), ratio_sum=0.0, bp_over_s_max=0.0)
    t0 = time.time()
    for pre in ck.quant_prefixes():
        s32 = ck.get(pre + ".scales").to(torch.float32).flatten()
        b32 = ck.get(pre + ".biases").to(torch.float32).flatten()
        for c0 in range(0, s32.numel(), 1 << 22):
            s = s32[c0:c0 + (1 << 22)]
            b = b32[c0:c0 + (1 << 22)]
            s16 = s.to(torch.float16)
            bp32 = b + 8.0 * s
            bp16 = bp32.to(torch.float16)
            n_groups += s.numel()
            agg["nonfin"] += int((~torch.isfinite(s16)).sum() + (~torch.isfinite(bp16)).sum())
            agg["zeros"] += int((s16 == 0).sum())
            agg["subn"] += int(((s16.abs() < 6.103515625e-05) & (s16 != 0)).sum()
                               + ((bp16.abs() < 6.103515625e-05) & (bp16 != 0)).sum())
            agg["s_inexact"] += int((s16.to(torch.float32) != s).sum())
            nz = s16 != 0
            agg["smax"] = max(agg["smax"], float(s.abs().max()))
            agg["smin"] = min(agg["smin"], float(s.abs()[nz].min()) if nz.any() else agg["smin"])
            agg["bpmax"] = max(agg["bpmax"], float(bp32.abs().max()))
            nzb = bp16 != 0
            agg["bpmin"] = min(agg["bpmin"], float(bp32.abs()[nzb].min()) if nzb.any() else agg["bpmin"])
            ratio = (b / s)[nz]
            agg["ratio_min"] = min(agg["ratio_min"], float(ratio.min()))
            agg["ratio_max"] = max(agg["ratio_max"], float(ratio.max()))
            agg["ratio_sum"] += float(ratio.sum())
            agg["bp_over_s_max"] = max(agg["bp_over_s_max"], float((bp32 / s)[nz].abs().max()))
            # weight error for all 16 nibble values of every group
            w32 = q * s.view(1, -1) + b.view(1, -1)                   # fp32 dequant of the BF16 values
            wk = ((q - 8.0) * s16.to(torch.float32).view(1, -1) + bp16.to(torch.float32).view(1, -1)).to(torch.float16)
            err = (wk.to(torch.float32) - w32).abs()
            floor = (w32.to(torch.float16).to(torch.float32) - w32).abs()  # fp16 storage of w itself
            sabs = s.abs().clamp_min(1e-30).view(1, -1)
            gmax = w32.abs().amax(dim=0, keepdim=True).clamp_min(1e-30)
            agg["abs_err"] = max(agg["abs_err"], float(err.max()))
            agg["floor_abs"] = max(agg["floor_abs"], float(floor.max()))
            sig = gmax.view(-1) >= 1e-3                                # groups whose weights are not ~0
            agg["n_small"] += int((~sig).sum())
            if sig.any():
                agg["rel_w_sig"] = max(agg["rel_w_sig"], float((err[:, sig] / gmax[:, sig]).max()))
                agg["excess_rel_sig"] = max(agg["excess_rel_sig"], float(((err - floor)[:, sig] / gmax[:, sig]).max()))
                agg["rel_s_sig"] = max(agg["rel_s_sig"], float((err[:, sig] / sabs[:, sig]).max()))
            agg["abs_err"] = max(agg["abs_err"], float(err.max()))
            agg["excess_abs"] = max(agg["excess_abs"], float((err - floor).max()))
            agg["floor_abs"] = max(agg["floor_abs"], float(floor.max()))
            agg["small_abs"] = max(agg["small_abs"], float(err[:, ~sig].max()) if (~sig).any() else 0.0)
    print(f"scanned {n_groups} (scale, bias) groups from {len(ck.quant_prefixes())} tensors in {time.time() - t0:.0f}s")
    print(f"  scale |s|      : min {agg['smin']:.3e}  max {agg['smax']:.3e}   (fp16 normal range 6.1e-5 .. 65504)")
    print(f"  b' = b + 8 s   : min|b'| {agg['bpmin']:.3e}  max|b'| {agg['bpmax']:.3e}   max |b'/s| {agg['bp_over_s_max']:.2f}")
    print(f"  b/s            : min {agg['ratio_min']:.3f}  max {agg['ratio_max']:.3f}  mean {agg['ratio_sum'] / n_groups:.3f}")
    print(f"  fp16 problems  : non-finite {agg['nonfin']}, zero scales {agg['zeros']}, "
          f"subnormal s/b' {agg['subn']}, bf16->fp16 inexact scales {agg['s_inexact']}")
    print("  dequant error of fma_f16(q-8, s16, b'16) vs fp32 s*q+b, all 16 q of every group:")
    print(f"      max abs err                         {agg['abs_err']:.3e}  (storing w itself in fp16 gives {agg['floor_abs']:.3e}; "
          f"max excess over that {agg['excess_abs']:.3e})")
    print(f"      groups with max|w| < 1e-3 (ignored below): {agg['n_small']} ({100.0 * agg['n_small'] / n_groups:.2f}%), "
          f"their max abs err {agg['small_abs']:.2e}")
    print(f"      other groups: max err / max|w|      {agg['rel_w_sig']:.3e}  (excess over fp16 rounding of w: {agg['excess_rel_sig']:.3e})")
    print(f"      other groups: max err / |scale|     {agg['rel_s_sig']:.3e}")


# ---------------------------------------------------------------------------
# kernel validation
# ---------------------------------------------------------------------------
class KernelSet:
    """Compile each variant once (dynamic M/N/K) on throw-away tensors."""

    def __init__(self):
        self.stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
        self.gemm = {}
        self.gemv = {}
        self._seed = None

    def _dummy(self):
        if self._seed is None:
            d = lambda r, c, dt: mark_row_major_2d(cp.zeros((r, c), dtype=dt))
            self._seed = (
                d(16, 64, cp.float16), d(16 * 4, 128, cp.uint32), d(1, 128, cp.float16),
                d(1, 128, cp.float16), d(16, 128, cp.float16),
            )
        return self._seed

    def get_gemm(self, v):
        if v not in self.gemm:
            bM, stages, sk = v
            kern = Int4AffineGemmAmpere(cta_tiler_mnk=(bM, 128, 64), num_stages=stages, split_k=sk)
            a, qw, s, b, c = self._dummy()
            locks = mark_lock_1d(cp.zeros(1, dtype=cp.int32))
            self.gemm[v] = cute.compile(kern, a, qw, s, b, c, c, locks, cutlass.Int32(1), self.stream)
        return self.gemm[v]

    def get_gemv(self, m):
        if m not in self.gemv:
            from int4_affine_gemv_ampere import _gemv_defaults  # noqa: PLC0415

            u2, eager, minb = _gemv_defaults(m)
            kern = Int4AffineGemvAmpere(m=m, group_size=64, unroll2=u2, u2_eager=eager, min_blocks=minb)
            a, qw, s, b, c = self._dummy()
            self.gemv[m] = cute.compile(kern, a, qw, s, b, c, self.stream)
        return self.gemv[m]


def make_activations(ck_dump, m, k, seed):
    """Layer-norm-like activations; the first rows are real (RMS-normalised) Flint hidden states when K == 2048."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((m, k), device="cuda", generator=g)
    if ck_dump is not None and k == ck_dump.shape[1]:
        n = min(m, ck_dump.shape[0])
        h = ck_dump[:n].to("cuda")
        a[:n] = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    return a.to(torch.float16)


def run_tactic(ks, tactic, act, qw_cp, s_cp, b_cp, n):
    m = act.shape[0]
    out = torch.zeros((m, n), device="cuda", dtype=torch.float16)
    a_cp = mark_row_major_2d(cp.from_dlpack(act))
    o_cp = mark_row_major_2d(cp.from_dlpack(out))
    if tactic[0] == "gemv":
        ks.get_gemv(m)(a_cp, qw_cp, s_cp, b_cp, o_cp, ks.stream)
    else:
        bM, stages, sk = tactic[1]
        locks = torch.zeros(max(ceil_div(m, bM) * ceil_div(n, 128), 1) if sk > 1 else 1,
                            device="cuda", dtype=torch.int32)
        ks.get_gemm(tactic[1])(a_cp, qw_cp, s_cp, b_cp, o_cp, o_cp, mark_lock_1d(cp.from_dlpack(locks)),
                               cutlass.Int32(1), ks.stream)
    torch.cuda.synchronize()
    return out


def reference_errors(act, out, weight, scales, biases):
    """Max abs / rel error of ``out`` vs the FP32 reference, chunked over N."""
    n = weight.shape[0]
    a32 = act.to(torch.float32)
    max_abs = 0.0
    ref_max = 0.0
    floor = 0.0
    num = 0.0
    den = 0.0
    for n0 in range(0, n, 8192):
        n1 = min(n0 + 8192, n)
        q = unpack_mlx_nibbles(weight[n0:n1]).to(torch.float32)
        w = q * scales[n0:n1].to(torch.float32).repeat_interleave(GROUP_SIZE, 1) \
            + biases[n0:n1].to(torch.float32).repeat_interleave(GROUP_SIZE, 1)
        ref = a32 @ w.t()
        d = (out[:, n0:n1].to(torch.float32) - ref)
        max_abs = max(max_abs, float(d.abs().max()))
        ref_max = max(ref_max, float(ref.abs().max()))
        floor = max(floor, float((ref.to(torch.float16).to(torch.float32) - ref).abs().max()))
        num += float(d.pow(2).sum())
        den += float(ref.pow(2).sum())
    return max_abs, ref_max, floor, (num / max(den, 1e-30)) ** 0.5


def kernels(ck: Checkpoint, all_variants: bool, shapes) -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    ks = KernelSet()
    dump = None
    try:
        d = torch.load("/proj/logs/flint_ref_chat.pt", map_location="cpu")
        dump = d["dump"]["l3.out"].float()
    except Exception as e:  # noqa: BLE001
        print(f"(no real hidden states: {e}; using random activations)")
    fails = 0
    hdr = f"{'shape':24s} {'M':>5s} {'tactic':>16s} {'max|ref|':>9s} {'max abs err':>11s} {'rel(max)':>9s} {'rel(rms)':>9s} {'fp16 floor':>10s}"
    print(hdr)
    for label, pre, k, n in shapes:
        weight = ck.get(pre + ".weight")
        scales = ck.get(pre + ".scales")
        biases = ck.get(pre + ".biases")
        assert weight.shape == (n, k // 2) and scales.shape == (n, k // 64), (label, weight.shape, scales.shape)
        rep = repack_weights_for_tile(weight)
        s_t, b_t = repack_scales_biases(scales, biases)
        qw_cp = mark_row_major_2d(cp.from_dlpack(rep).view(cp.uint32))
        s_cp = mark_row_major_2d(cp.from_dlpack(s_t))
        b_cp = mark_row_major_2d(cp.from_dlpack(b_t))
        for m in M_LIST:
            act = make_activations(dump, m, k, seed=m * 7 + n)
            tactics = []
            if m <= 8:
                tactics.append(("gemv", None))
            tactics.append(("gemm", DEFAULT_GEMM.get(m, (16, 4, 1)) if m > 8 else (16, 4, 1)))
            if all_variants:
                tactics += [("gemm", v) for v in GEMM_VARIANTS if ("gemm", v) not in tactics]
            worst = None
            for t in tactics:
                if t[0] == "gemm" and (k // 64) % t[1][2] != 0:
                    continue
                out = run_tactic(ks, t, act, qw_cp, s_cp, b_cp, n)
                max_abs, ref_max, floor, rms = reference_errors(act, out, weight, scales, biases)
                rel = max_abs / max(ref_max, 1e-30)
                name = "gemv" if t[0] == "gemv" else f"gemm{t[1][0]}_s{t[1][1]}_sk{t[1][2]}"
                if t is tactics[0] or t is tactics[1] and m <= 8:
                    print(f"{label:24s} {m:5d} {name:>16s} {ref_max:9.2f} {max_abs:11.2e} {rel:9.2e} {rms:9.2e} {floor:10.2e}")
                if rel > 4e-3 or rms > 2e-3 or not torch.isfinite(out.float()).all():
                    fails += 1
                    print(f"  FAIL {label} M={m} {name}: rel={rel:.2e} rms={rms:.2e}")
                worst = max(worst or 0.0, rel)
            if all_variants:
                print(f"{'':24s} {m:5d} {'worst of ' + str(len(tactics)):>16s} {'':9s} {'':11s} {worst:9.2e}")
        del weight, scales, biases, rep
        torch.cuda.empty_cache()
    print("KERNELS FAILED" if fails else "KERNELS OK", f"({fails} failures)")
    return fails


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/work/flint-int4")
    p.add_argument("--mode", default="all", choices=["all", "analysis", "kernels"])
    p.add_argument("--all_variants", action="store_true", help="run every baked GEMM variant at every M")
    p.add_argument("--only", default="", help="substring filter on the shape label")
    args = p.parse_args(_saved_argv[1:])
    ck = Checkpoint(args.ckpt)
    if args.mode in ("all", "analysis"):
        analysis(ck)
    rc = 0
    if args.mode in ("all", "kernels"):
        shapes = [s for s in SHAPES if args.only in s[0]]
        rc = kernels(ck, args.all_variants, shapes)
    sys.exit(1 if rc else 0)


if __name__ == "__main__":
    main()
