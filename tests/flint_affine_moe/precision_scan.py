"""All MoE layers: fp16 overflow check and weight-error ratio (kernel formulation vs fp16 floor), summarised."""
import sys, torch
sys.path.insert(0, "/work/wt-flint-marlin/tests/flint_affine_moe")
from flint_affine_lib import *
ck = Ckpt("/work/flint-int4")
worst = {}
print("layer proj  max|s| max|b'| inf  rmse_kernel/rms(w)  rmse_floor/rms(w)  max_abs_err_kernel  max_abs_err_floor")
for l in range(2, 49):
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = f"model.layers.{l}.mlp.switch_mlp.{proj}"
        w = ck.get(p + ".weight"); s = ck.get(p + ".scales"); b = ck.get(p + ".biases")
        s16, bp16 = make_sb(s, b)
        ref = dequant_fp32(w, s, b)
        q = unpack_q(w).float().unflatten(-1, (-1, GROUP))
        hw = ((q - 8) * s16.float().unsqueeze(-1) + bp16.float().unsqueeze(-1)).flatten(-2).half().float()
        fl = ref.half().float()
        rms = ref.pow(2).mean().sqrt()
        a = ((hw - ref).pow(2).mean().sqrt() / rms).item(); f = ((fl - ref).pow(2).mean().sqrt() / rms).item()
        print(f"{l:3d} {proj:9s} {s.float().abs().max().item():.2e} {bp16.float().abs().max().item():.2e} {int(torch.isinf(bp16).sum())} {a:.2e} {f:.2e} {(hw-ref).abs().max().item():.2e} {(fl-ref).abs().max().item():.2e}", flush=True)
        del w, s, b, ref, q, hw, fl
