"""Standalone validation of the cloned affine W4A16 Marlin MoE kernel on REAL Flint checkpoint tensors.
Usage (in edgellm-dev): python3 -I test_affine_moe.py LAYER [LAYER...]   (lib built from affine_capi.cu)
"""
import ctypes, sys, time
import torch
sys.path.insert(0, "/work/wt-flint-marlin/tests/flint_affine_moe")
from flint_affine_lib import *

lib = ctypes.CDLL("/tmp/aff/libaffine_capi.so")
lib.affine_workspace_elems.restype = ctypes.c_int64
lib.affine_workspace_elems.argtypes = [ctypes.c_int64] * 3
P = ctypes.c_void_p
lib.affine_moe_gemm.argtypes = [P] * 3 + [P, ctypes.c_int64] + [P] * 6 + [ctypes.c_int] * 7 + [ctypes.c_int64, P]


def gemm(A, qw, s, bp, sorted_ids, exp_ids, npp, topk_w, block, E, top_k, mul, m, n, k, out_rows):
    C = torch.zeros((out_rows, n), dtype=torch.float16, device=DEV)
    padded = sorted_ids.numel()
    ws = torch.zeros(lib.affine_workspace_elems(padded, n, block), dtype=torch.int32, device=DEV)
    torch.cuda.synchronize()
    rc = lib.affine_moe_gemm(A.data_ptr(), qw.data_ptr(), C.data_ptr(), ws.data_ptr(), ws.numel(), s.data_ptr(), bp.data_ptr(),
                             sorted_ids.data_ptr(), exp_ids.data_ptr(), npp.data_ptr(), topk_w.data_ptr(), block, E,
                             top_k, int(mul), m, n, k, padded, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert rc == 0, rc
    return C


def errs(x, ref):
    x = x.float(); e = (x - ref).abs()
    return e.max().item(), (e.norm() / ref.norm()).item(), (e.max() / ref.abs().max()).item()


def activations(ck, l, T, cfg_eps=1e-6):
    d = torch.load("/proj/logs/flint_ref_chat.pt", map_location="cpu")["dump"][f"l{l}.attn_out"].to(DEV)   # real residual
    lnw = ck.get(f"model.layers.{l}.post_attention_layernorm.weight").float()
    def rms(x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg_eps) * lnw
    real = rms(d)
    if T <= real.shape[0]:
        h = real[:T]
    else:  # pad with synthetic layernorm-like rows (unit-rms gaussian x ln weight)
        g = torch.Generator(device=DEV).manual_seed(1234 + T)
        syn = rms(torch.randn(T - real.shape[0], 2048, device=DEV, generator=g) * real.pow(2).mean().sqrt())
        h = torch.cat([real, syn], 0)
    return h.half()


import os
BLOCKS = [8, 16, 32, 48, 64]
# Pre-scale the SwiGLU output by 2^-K and fold 2^K into the routing weights of the down GEMM (lossless) so that deep
# layers with massive activations (layer 46: |silu(g)*u| up to 7.6e4 > fp16 max) do not overflow fp16.
ACT_K = int(os.environ.get('ACT_SCALE_LOG2', '0'))


def run(l, Ts, E=256, top_k=6):
    ck = Ckpt("/work/flint-int4")
    pre = f"model.layers.{l}.mlp"
    print(f"=== layer {l}: repacking (torch reference repack) ...", flush=True)
    t0 = time.time()
    gu_q, gu_s, gu_b = repack_proj_pair(ck, pre + ".switch_mlp.gate_proj", pre + ".switch_mlp.up_proj")
    dn_q, dn_s, dn_b = repack_proj_pair(ck, pre + ".switch_mlp.down_proj")
    print(f"    repack {time.time()-t0:.1f}s  gate_up qw{tuple(gu_q.shape)} s{tuple(gu_s.shape)}  down qw{tuple(dn_q.shape)} s{tuple(dn_s.shape)}")
    # fp32 reference weights
    Wg = dequant_fp32(ck.get(pre + ".switch_mlp.gate_proj.weight"), ck.get(pre + ".switch_mlp.gate_proj.scales"), ck.get(pre + ".switch_mlp.gate_proj.biases"))
    Wu = dequant_fp32(ck.get(pre + ".switch_mlp.up_proj.weight"), ck.get(pre + ".switch_mlp.up_proj.scales"), ck.get(pre + ".switch_mlp.up_proj.biases"))
    Wd = dequant_fp32(ck.get(pre + ".switch_mlp.down_proj.weight"), ck.get(pre + ".switch_mlp.down_proj.scales"), ck.get(pre + ".switch_mlp.down_proj.biases"))
    router = ck.get(pre + ".gate.weight").float()
    rows = []
    for T in Ts:
        h = activations(ck, l, T)
        hf = h.float()
        # ---------- (1) single expert GEMM: all T tokens -> expert e, top_k=1, gate_proj-style (fused gate|up) ----------
        e = 37
        for block in BLOCKS:
            ids1 = torch.full((T, 1), e, dtype=torch.int64, device=DEV)
            sid, eid, npp = build_marlin_indices(ids1, E, block)
            sid, eid, npp = sid.to(DEV), eid.to(DEV), npp.to(DEV)
            C = gemm(h, gu_q, gu_s, gu_b, sid, eid, npp, torch.ones(T + 1, device=DEV), block, E, 1, 0, T, 1024, 2048, T)
            ref = hf @ torch.cat([Wg[e], Wu[e]], 0).T                  # [T, 1024] fp32
            fl = (hf @ torch.cat([Wg[e], Wu[e]], 0).half().float().T).half()    # ideal fp16-rounded weights, fp32 accum, fp16 out
            rows.append((l, T, f"1 expert gate|up block={block}", *errs(C, ref), *errs(fl, ref)))
        # down for single expert
        a = torch.nn.functional.silu(hf @ Wg[e].T) * (hf @ Wu[e].T)
        a16 = a.half()
        block = 8 if T <= 8 else 32
        ids1 = torch.full((T, 1), e, dtype=torch.int64, device=DEV)
        sid, eid, npp = build_marlin_indices(ids1, E, block)
        C = gemm(a16, dn_q, dn_s, dn_b, sid.to(DEV), eid.to(DEV), npp.to(DEV), torch.ones(T + 1, device=DEV), block, E, 1, 0, T, 2048, 512, T)
        ref = a16.float() @ Wd[e].T
        fl = (a16.float() @ Wd[e].half().float().T).half()
        rows.append((l, T, f"1 expert down block={block}", *errs(C, ref), *errs(fl, ref)))
        # ---------- (2) full MoE: real router, sigmoid top-6 renorm, gate|up -> SwiGLU(fp16 as plugin) -> down*w -> sum ----------
        logits = hf @ router.T
        wts, ids = routing(logits, top_k)
        yref = ref_moe(hf, Wg, Wu, Wd, wts, ids)
        Wg16, Wu16, Wd16 = Wg.half().float(), Wu.half().float(), Wd.half().float()
        yfl = torch.zeros_like(hf)                                    # same fp16 pipeline, ideal fp16 weights, fp32 accum
        for e_ in ids.unique().tolist():
            tok, slot = (ids == e_).nonzero(as_tuple=True)
            he = hf[tok]
            g_ = (he @ Wg16[e_].T).half(); u_ = (he @ Wu16[e_].T).half()
            a_ = (torch.nn.functional.silu(g_.float()) * u_.float() * 2.0 ** -ACT_K).half() if ACT_K else torch.nn.functional.silu(g_.float()).half() * u_
            o_ = ((a_.float() @ Wd16[e_].T) * (wts[tok, slot][:, None] * 2.0 ** ACT_K)).half().float()
            yfl.index_add_(0, tok, o_)
        for block in BLOCKS:
            sid, eid, npp = build_marlin_indices(ids, E, block)
            sid, eid, npp = sid.to(DEV), eid.to(DEV), npp.to(DEV)
            total = T * top_k
            dummy = torch.ones(total + 1, device=DEV)
            gu = gemm(h, gu_q, gu_s, gu_b, sid, eid, npp, dummy, block, E, top_k, 0, T, 1024, 2048, total)
            g, u = gu[:, :512], gu[:, 512:]
            act = ((torch.nn.functional.silu(g.float()) * u.float() * 2.0 ** -ACT_K).half() if ACT_K else (torch.nn.functional.silu(g.float()).half() * u))  # plugin: half(silu(g_f32)) * up in half
            twp = torch.zeros(sid.numel() + 8, device=DEV); twp[:total] = wts.flatten() * 2.0 ** ACT_K
            dn = gemm(act.contiguous(), dn_q, dn_s, dn_b, sid, eid, npp, twp, block, E, 1, 1, total, 2048, 512, total)
            y = dn.float().view(T, top_k, 2048).sum(1)
            rows.append((l, T, f"full MoE block={block}", *errs(y, yref), *errs(yfl, yref)))
        del h
    return rows


if __name__ == "__main__":
    layers = [int(x) for x in sys.argv[1:]] or [20]
    allrows = []
    for l in layers:
        allrows += run(l, [1, 8, 64, 512])
    print("\nlayer  T   case                          max_abs_err   rel_L2(err/ref)  max_err/max|ref|  | ideal-fp16-weights: max_abs  rel_L2  max/max")
    for r in allrows:
        print(f"{r[0]:4d} {r[1]:4d}  {r[2]:30s} {r[3]:.3e}   {r[4]:.3e}      {r[5]:.3e}     | {r[6]:.3e} {r[7]:.3e} {r[8]:.3e}")
