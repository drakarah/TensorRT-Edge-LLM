"""Step 2: range / rounding of fp16 s and b' = bias + 8*scale on real checkpoint tensors vs fp32 dequant."""
import sys, torch
sys.path.insert(0, "/work/wt-flint-marlin/tests/flint_affine_moe")
from flint_affine_lib import *

ck = Ckpt("/work/flint-int4")
layers = [int(x) for x in sys.argv[1:]] or [2, 20, 48]
for l in layers:
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = f"model.layers.{l}.mlp.switch_mlp.{proj}"
        w = ck.get(p + ".weight"); s = ck.get(p + ".scales"); b = ck.get(p + ".biases")
        s32, b32 = s.float(), b.float()
        s16, bp16 = make_sb(s, b)
        bp32 = b32 + 8 * s32
        ref = dequant_fp32(w, s, b)                                   # fp32 truth
        q = unpack_q(w).float().unflatten(-1, (-1, GROUP))
        hw = ((q - 8) * s16.float().unsqueeze(-1) + bp16.float().unsqueeze(-1)).flatten(-2).half().float()  # kernel math
        ref16 = ref.half().float()                                    # best possible fp16 weight
        # naive alt: fp16 of bias used directly with q*s (zero point not folded): w = q*s16 + fp16(bias)
        alt = (q * s16.float().unsqueeze(-1) + b32.half().float().unsqueeze(-1)).flatten(-2).half().float()
        grp_max = ref.abs().unflatten(-1, (-1, GROUP)).amax(-1, keepdim=True).expand_as(ref.unflatten(-1, (-1, GROUP))).flatten(-2)
        def st(x):
            e = (x - ref).abs()
            return f"max_abs={e.max().item():.3e} rel_to_grp_max(max)={(e / grp_max.clamp_min(1e-30)).max().item():.3e} rmse/rms(w)={(e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item():.3e}"
        nz = s32 != 0
        print(f"== layer {l} {proj}  w[{w.shape}]")
        print(f"  scale: min|s|={s32.abs()[nz].min().item():.3e} max|s|={s32.abs().max().item():.3e} zero={int((~nz).sum())}"
              f"  fp16 subnormal(<6.1e-5)={int(((s32.abs() < 6.1035e-5) & nz).sum())} ({100*((s32.abs()<6.1035e-5)&nz).float().mean():.3f}%)  s16 rel.err max={((s16.float()-s32).abs()/s32.abs().clamp_min(1e-30)).max().item():.3e}")
        print(f"  bias : range [{b32.min().item():.3e},{b32.max().item():.3e}]  bias/scale mean={((b32/s32)[nz]).mean().item():.3f} min={((b32/s32)[nz]).min().item():.3f} max={((b32/s32)[nz]).max().item():.3f}")
        print(f"  b'   : range [{bp32.min().item():.3e},{bp32.max().item():.3e}]  max|b'|/|s| = {(bp32.abs()/s32.abs().clamp_min(1e-30)).max().item():.3f}  fp16 overflow={int(torch.isinf(bp16).sum())}  b' abs err max={(bp16.float()-bp32).abs().max().item():.3e}  b' subnormal={int(((bp16.float().abs()<6.1035e-5)&(bp16!=0)).sum())} zero-in-fp16(nonzero in fp32)={int(((bp16==0)&(bp32!=0)).sum())}")
        print(f"  weight err  kernel (q-8)*s16+b'16 -> fp16 : {st(hw)}")
        print(f"  weight err  fp16(round(w_fp32)) floor      : {st(ref16)}")
        print(f"  weight err  naive q*s16+fp16(bias)         : {st(alt)}")
