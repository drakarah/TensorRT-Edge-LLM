# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation of the FMHA-v2 paged prefill kernel over an INT8 KV pool.

The INT8 variant (``FMHAV2Ampere(kv_int8=True).__call_context_paged_i8__``) stages INT8 tiles and expands them to
exact FP16 codes in shared memory. The K scale joins the softmax scale and the V scale multiplies the normalised
output, so there is no per-element dequantisation. Three kinds of check:

* bit-exact: the FP16 paged kernel run on the pool's codes (exact in FP16) with the same folded softmax scale, its
  output then multiplied by a power-of-two V scale (exact), must equal the INT8 kernel's output bit for bit;
* bound: with arbitrary (non power-of-two) scales the output stays within the FP32 reference's tolerance and
  closer to it than the old gather + FP16 path (which rounds ``code * scale`` to FP16);
* edges: a single query row, tile-boundary lengths (31/32/33, 63/64/65, 127/128/129), a chunked-prefill prefix
  (Sk > Sq), ragged per-batch lengths with padded extents, scrambled page order, pages shared between requests,
  and poisoned (extreme) pool bytes outside the live region and in the unused tail of the last page, which must
  never influence the output.

Run inside the container from the kernelSrcs directory (needs a CUDA GPU, torch and nvidia-cutlass-dsl):

    python3 fmha_v2_cutedsl/test_fmha_v2_paged_i8.py [--quick] [--bench]
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
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

HQ, HKV, D, PAGE = 24, 4, 256, 128  # Qwen3.8-27B attention layers
# (Br, Bc, threads) of the INT8 kernel as built by build_cutedsl.py (fmha_v2_d256_paged_i8). The FP16 kernel it is
# compared with always runs the production tile (64, 32, 128). TILE_I8=64,32,128 selects the smaller INT8 tile.
TILE = tuple(int(x) for x in os.environ.get("TILE_I8", "128,64,256").split(","))
FP16_TILE = (64, 32, 128)


def _mark_cu(t):
    t = from_dlpack(t, assumed_align=16)
    return t.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(mode=0, stride_order=(0,))


class Variant:
    """One compiled D256 paged kernel (Br=64, Bc=32, 128 threads), FP16 pool or INT8 pool."""

    def __init__(self, mod, int8, tile=None):
        br, bc, nt = tile or TILE
        kw = dict(head_dim=D, m_block_size=br, n_block_size=bc, num_threads=nt, is_causal=True,
                  skip_rescale=True)
        if int8:
            kw["kv_int8"] = True
        self.kernel = mod.FMHAV2Ampere(**kw)
        self.int8 = int8
        self.compiled = None

    def __call__(self, q, pool, plist, o, cu_q, cu_k, scale, kscale=1.0, vscale=1.0):
        args = [self._q(q), self._pool(pool), self._plist(plist), self._q(o), _mark_cu(cu_q), _mark_cu(cu_k)]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if self.int8:
            tail = [cutlass.Int32(0), cutlass.Float32(scale), cutlass.Float32(kscale), cutlass.Float32(vscale),
                    cutlass.Int32(8), stream]
            fn = self.kernel.__call_context_paged_i8__
        else:
            tail = [cutlass.Int32(0), cutlass.Float32(scale), cutlass.Int32(8), stream]
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


def make_case(b, sq, sk, seed, q_lens=None, k_lens=None, shared_pages=False):
    """Random Q, INT8 K/V codes, and a scrambled, poisoned INT8 pool holding them."""
    q_lens = q_lens or [sq] * b
    k_lens = k_lens or [sk] * b
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(b, sq, HQ, D, device="cuda", generator=g) * 1.5).half()
    kc = torch.randint(-127, 128, (b, sk, HKV, D), device="cuda", generator=g, dtype=torch.int8)
    vc = torch.randint(-127, 128, (b, sk, HKV, D), device="cuda", generator=g, dtype=torch.int8)
    if shared_pages and b > 1:
        kc[1:] = kc[0:1]
        vc[1:] = vc[0:1]
    maxp = (sk + PAGE - 1) // PAGE
    per = maxp + 1
    npg = b * per
    pool = torch.empty(2 * npg, PAGE, HKV, D, device="cuda", dtype=torch.int8)
    # Poison: extreme codes everywhere, so any read outside the live region shows up.
    pool[:npg] = -128
    pool[npg:] = 127
    plist = torch.empty(b, 2, maxp, dtype=torch.int32)
    rng = torch.Generator().manual_seed(seed)
    for bi in range(b):
        base = bi * per
        kp = base + 1 + torch.randperm(maxp, generator=rng)
        vp = base + 1 + torch.randperm(maxp, generator=rng)
        plist[bi, 0] = base  # unused logical slots point at a poisoned page
        plist[bi, 1] = npg + base
        for lp in range((k_lens[bi] + PAGE - 1) // PAGE):
            lo, hi = lp * PAGE, min(lp * PAGE + PAGE, k_lens[bi])
            plist[bi, 0, lp] = int(kp[lp])
            plist[bi, 1, lp] = npg + int(vp[lp])
            pool[int(kp[lp]), : hi - lo] = kc[bi, lo:hi]
            pool[npg + int(vp[lp]), : hi - lo] = vc[bi, lo:hi]
    cu_q = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32).cuda()
    cu_k = torch.tensor([0] + torch.tensor(k_lens).cumsum(0).tolist(), dtype=torch.int32).cuda()
    return q, kc, vc, pool, plist.cuda(), cu_q, cu_k


def dequant(codes, scale):
    """The gather kernel's arithmetic: half(float(code) * scale), round to nearest."""
    return (codes.float() * torch.tensor(scale, dtype=torch.float32, device=codes.device)).half()


def folded_scale(attn_scale, kscale):
    """float32(attn * kscale): the softmax scale the INT8 kernel forms from its two inputs."""
    return float(torch.tensor(attn_scale, dtype=torch.float32) * torch.tensor(kscale, dtype=torch.float32))


def same_bits(a, b, vscale=1.0):
    """Bitwise equality, except in the FP16 subnormal range of the double rounding.

    The reference scales an already rounded FP16 output by the V scale. Where the unscaled value or the scaled
    result is subnormal that rounds twice, whereas the kernel rounds once; such entries may differ by one
    subnormal step of the coarser grid (2^-24 * max(1, vscale)). Every other entry must match bit for bit.
    """
    span = max(1.0, vscale)
    a32, b32 = a.float(), b.float()
    tiny = (a32.abs() < 2.0 ** -14 * span) | (b32.abs() < 2.0 ** -14 * span)
    if not torch.equal(a[~tiny], b[~tiny]):
        return False
    return bool(((a32 - b32).abs()[tiny] <= 2.0 ** -24 * span).all())


def exact_reference(fp16, q, pool, plist, cu_q, cu_k, attn_scale, kscale, vscale_pow2):
    """FP16 paged kernel on the exact codes with the folded scale; V scale applied afterwards (power of two)."""
    o = torch.zeros_like(q)
    fp16(q, pool.half(), plist, o, cu_q, cu_k, folded_scale(attn_scale, kscale))
    return (o.float() * vscale_pow2).half()


def reference(q, k, v, scale):
    b, sq, _, _ = q.shape
    sk = k.shape[1]
    mask = torch.arange(sk, device="cuda")[None, :] <= torch.arange(sq, device="cuda")[:, None] + (sk - sq)
    out = torch.zeros(q.shape, device="cuda", dtype=torch.float32)
    grp = HQ // HKV
    for bi in range(b):
        for h in range(HQ):
            s = (q[bi, :, h].float() @ k[bi, :, h // grp].float().T) * scale
            p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=1)
            out[bi, :, h] = p @ v[bi, :, h // grp].float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench", action="store_true", help="time the INT8 kernel against gather + FP16 kernel")
    ap.add_argument("--module", default="fmha")
    args = ap.parse_args()
    mod = importlib.import_module(args.module)
    # Bit-exactness compares kernels of the same tile (a different Bc reorders the online softmax); the gather
    # baseline of the error column is the production FP16 tile.
    fp16, int8 = Variant(mod, False, TILE), Variant(mod, True, TILE)
    fp16_prod = Variant(mod, False, FP16_TILE)
    attn_scale = 1.0 / math.sqrt(D)
    kscale, vscale = 0.0123, 0.015625  # K arbitrary (not a power of two); V a power of two for bit-exactness

    cases = []  # (b, sq, sk)
    sqs = [1, 7, 33, 128, 1000] if args.quick else [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 2048]
    for sq in sqs:
        cases.append((1, sq, sq))
        cases.append((1, sq, sq + 777))       # chunked prefill: a prefix already in the cache
        if sq in (1, 33, 128):
            cases.append((2, sq, sq + 130))
    ok = True
    print(f"{'B':>2} {'Sq':>5} {'Sk':>5} {'bitwise':>8} {'err_int8':>9} {'err_gather':>11}")
    for (b, sq, sk) in cases:
        q, kc, vc, pool, plist, cu_q, cu_k = make_case(b, sq, sk, 4321 + sq)
        o8 = torch.zeros_like(q)
        int8(q, pool, plist, o8, cu_q, cu_k, attn_scale, kscale, vscale)
        same = same_bits(o8, exact_reference(fp16, q, pool, plist, cu_q, cu_k, attn_scale, kscale, vscale), vscale)
        ok &= same
        e8 = eg = float("nan")
        if sq * sk <= 4_000_000:
            # exact (unrounded) dequantised K/V reference: what the codes really mean
            ref = reference(q, kc.float() * kscale, vc.float() * vscale, attn_scale)
            e8 = (o8.float() - ref).abs().max().item()
            npg = pool.shape[0] // 2
            pool_h = torch.cat([dequant(pool[:npg], kscale), dequant(pool[npg:], vscale)])
            og = torch.zeros_like(q)
            fp16_prod(q, pool_h, plist, og, cu_q, cu_k, attn_scale)
            eg = (og.float() - ref).abs().max().item()
            ok &= e8 <= 1.5 * eg + 2e-4  # never meaningfully worse than gather + FP16
        print(f"{b:>2} {sq:>5} {sk:>5} {'EQUAL' if same else 'DIFF':>8} {e8:9.6f} {eg:11.6f}")
    # Ragged batch: per-batch logical lengths below the padded extents. Compare the valid rows only.
    q_lens, k_lens = [100, 37, 128], [400, 237, 128]
    q, kc, vc, pool, plist, cu_q, cu_k = make_case(3, 128, 640, 99, q_lens, k_lens)
    o8 = torch.zeros_like(q)
    int8(q, pool, plist, o8, cu_q, cu_k, attn_scale, kscale, vscale)
    oe = exact_reference(fp16, q, pool, plist, cu_q, cu_k, attn_scale, kscale, vscale)
    same = all(same_bits(o8[i, :q_lens[i]], oe[i, :q_lens[i]], vscale) for i in range(3))
    ok &= same
    print(f"ragged q{q_lens} k{k_lens}: {'EQUAL' if same else 'DIFF'}")
    # Pages shared between two requests (a prefix-cache hit maps the same pool pages twice).
    q, kc, vc, pool, plist, cu_q, cu_k = make_case(2, 64, 300, 7, shared_pages=True)
    plist[1] = plist[0]
    o8 = torch.zeros_like(q)
    int8(q, pool, plist, o8, cu_q, cu_k, attn_scale, kscale, vscale)
    same = same_bits(o8, exact_reference(fp16, q, pool, plist, cu_q, cu_k, attn_scale, kscale, vscale), vscale)
    ok &= same
    print(f"shared pages: {'EQUAL' if same else 'DIFF'}")
    # Scale edges: unit scales, tiny and large K scale, several V powers of two.
    for ks, vs in ((1.0, 1.0), (1e-4, 2.0 ** -15), (0.5, 0.25), (7.3, 2.0 ** -8), (3.0, 4.0)):
        q, kc, vc, pool, plist, cu_q, cu_k = make_case(1, 130, 130, 5)
        o8 = torch.zeros_like(q)
        int8(q, pool, plist, o8, cu_q, cu_k, attn_scale, ks, vs)
        same = same_bits(o8, exact_reference(fp16, q, pool, plist, cu_q, cu_k, attn_scale, ks, vs), vs)
        ok &= same
        print(f"scales ({ks}, {vs}): {'EQUAL' if same else 'DIFF'}")
    # All 256 byte values through the converter: codes -128..127 laid out across K and V, compared exactly.
    q, kc, vc, pool, plist, cu_q, cu_k = make_case(1, 64, 256, 3)
    codes = (torch.arange(256, device="cuda", dtype=torch.int32) - 128).to(torch.int8)
    pool[plist[0, 0, 0]] = codes.repeat(PAGE * HKV * D // 256).view(PAGE, HKV, D)
    pool[plist[0, 1, 0]] = codes.flip(0).repeat(PAGE * HKV * D // 256).view(PAGE, HKV, D)
    o8 = torch.zeros_like(q)
    int8(q, pool, plist, o8, cu_q, cu_k, attn_scale, 1.0, 1.0)
    same = same_bits(o8, exact_reference(fp16, q, pool, plist, cu_q, cu_k, attn_scale, 1.0, 1.0))
    ok &= same
    print(f"every byte value -128..127: {'EQUAL' if same else 'DIFF'}")
    if args.bench or os.environ.get('BENCH_ONLY'):
        bench(int8, fp16_prod, attn_scale, kscale, vscale)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def bench(int8, fp16, attn_scale, kscale, vscale, sq=2048, sk=16384, iters=5):
    """INT8 kernel vs the FP16 kernel on the same shape (the gather it removes is timed separately)."""
    q, kc, vc, pool, plist, cu_q, cu_k = make_case(1, sq, sk, 11)
    npg = pool.shape[0] // 2
    pool_h = torch.cat([dequant(pool[:npg], kscale), dequant(pool[npg:], vscale)])
    o = torch.zeros_like(q)
    for name, fn, p in (("fp16 pool", lambda: fp16(q, pool_h, plist, o, cu_q, cu_k, attn_scale), pool_h),
                        ("int8 pool", lambda: int8(q, pool, plist, o, cu_q, cu_k, attn_scale, kscale, vscale), pool)):
        fn()
        t = time.time()
        for _ in range(iters):
            fn()
        ms = (time.time() - t) / iters * 1e3
        print(f"bench sq={sq} sk={sk} {name}: {ms:.1f} ms")


if __name__ == "__main__":
    main()
