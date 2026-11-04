# CuTe DSL affine INT4 (MLX-style) W4A16 GEMM / GEMV (Ampere)

Dense W4A16 kernels for checkpoints quantised with MLX affine INT4, group size 64
(the Flint model).  Derived from `../int4_fp16_gemm_cutedsl/` (same cp.async
multistage pipeline, fragment-order weight buffer, serial split-K, narrow-CTA
decode GEMV); the symmetric kernels are untouched.

    C[M, N] = A[M, K] @ W^T,    W[n, k] = scales[n, k/64] * q[n, k] + biases[n, k/64],   q in 0..15

## Dequant

The symmetric kernel computes `(q - 8) * s` (magic-number nibble conversion, then
`mul.f16x2`).  The affine kernel keeps the conversion that yields `q - 8` and
replaces the multiply by one `fma.rn.f16x2`:

    w = fma(q - 8, s, b'),     b' = b + 8 * s   (float32, rounded once to fp16 at repack time)

The inner loop has the same instruction count as the symmetric kernel; the only
addition is one `f16x2` splat of `b'` per word (the scale splat was already there).
See `int4_dequant.py`.  Error of this formulation over all 621 M (scale, bias)
groups of the Flint checkpoint is printed by `test_int4_affine.py
--mode analysis` (max abs 9.8e-4, i.e. fp16 rounding of w itself; no overflow).

## Tensors (ABI)

| Tensor | Dtype | Shape | Notes |
|---|---|---|---|
| `mA` | fp16 | `[M, K]` row-major | `K % 64 == 0` |
| `mQW` | uint32 | `[ceil(N/128) * (K/64) * 8, 128]` | `int4_affine_reference.repack_weights_for_tile(weight)`; raw nibbles `q` |
| `mScales` | fp16 | `[K/64, N]` row-major | transpose of checkpoint scales; `N % 8 == 0` |
| `mBias` | fp16 | `[K/64, N]` row-major | `fp16(float32(bias) + 8 * float32(scale))`, same layout as scales |
| `mC` / `mOut` | fp16 | `[M, N]` row-major | |

GEMM export signature: `(mA, mQW, mScales, mBias, mC, mWorkspace, mLocks, swizzle, stream)`
(`mWorkspace` placeholder: pass `mC`; `mLocks` zero int32, `ceil(M/bM)*ceil(N/128)`
entries for `split_k > 1`, 1 otherwise; `swizzle` runtime Int32, 1 = none).
GEMV export signature: `(mA, mQW, mScales, mBias, mOut, stream)`; M is baked
(`int4_affine_gemv_m1..m8`).

Variants (registered as group `int4_affine_gemm` in `build_cutedsl.py`):
`int4_affine_gemm_{bM}x128x64_s{stages}_sk{split_k}` for the same 16
`(bM, stages, split_k)` tuples as the symmetric group, and `int4_affine_gemv_m{1..8}`.
`split_k` must divide `K / 64`.

## Standalone testing

```bash
export PYTHONPATH=$PWD/kernelSrcs:$PWD/kernelSrcs/int4_affine_cutedsl
cd kernelSrcs/int4_affine_cutedsl
python3 int4_affine_gemm_ampere.py --mnk 256,512,1024 --cta_tiler_mnk 64,128,64 --num_stages 3
python3 int4_affine_gemv_ampere.py --mnk 4,2048,2048
# Real Flint tensors vs an FP32 reference (needs the checkpoint):
python3 test_int4_affine.py --ckpt /path/to/flint-int4 [--all_variants]
```
