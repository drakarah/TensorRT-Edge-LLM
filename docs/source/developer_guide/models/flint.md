# Flint

Flint (`RecursiveSI/flint`) is a 39.8B-parameter sparse mixture-of-experts decoder for coding agents
(about 2.5B active parameters per token). Edge-LLM runs its **quantisation-aware-trained INT4 release**
(`RecursiveSI/flint-int4`, MLX affine format, 22.6 GB) on Jetson Orin (sm_87). The BF16 release does
not fit the device and the NVFP4 release needs FP4/FP8 hardware that Orin does not have.

This page describes the architecture, the checkpoint format, what Edge-LLM adds to run it, how to
export, build and run it, the numerics notes that matter, and what has and has not been validated.

## Architecture

Flint is a Qwen3-MoE-style decoder with four differences. The semantics below were taken from the
vendor's SGLang model (`flint_sglang`) and llama.cpp's `src/models/flint.cpp`, which agree with each
other, and were confirmed against a plain-PyTorch reference that reproduces the model on the real
checkpoint (see [Validation](#validation)).

| Property | Value |
| :-- | :-- |
| Layers | 49: layers 0-1 dense SwiGLU MLP (intermediate 6144), layers 2-48 sparse MoE |
| Hidden size / heads | 2048; 32 query heads, 4 KV heads, head dim 128, attention scale 1/sqrt(128) |
| Attention layers | 35 sliding-window layers (window 513 = the current token plus the 512 before it, RoPE theta 10000) and 14 full-attention layers (NoPE: no positional encoding; layers 0, 1, 5, 9, ..., 45, 48) |
| QK norm | per-head RMSNorm on Q and K (weight of size head_dim), applied before RoPE |
| Attention sinks | one learned logit per query head and layer (`self_attn.softmax_offset`), added to the softmax denominator with no value contribution |
| Experts | 256 routed experts per MoE layer, top-6, intermediate 512, SwiGLU; plus an always-on **ungated** shared expert (intermediate 1024) |
| Routing | sigmoid scores, top-6, weights renormalised to sum to one; no expert groups, no score-correction bias |
| Vocabulary / context | 150,144 tokens; 262,144 tokens configured |

The sink adds `exp(s_h)` to the softmax denominator: `out = sum_j exp(l_j - m) v_j / (sum_j exp(l_j - m) + exp(s_h - m))`
with `l_j = attention_scale * q.k_j` and `m` the row maximum. The sink is used as given, it is not multiplied by
the attention scale, and it applies on sliding-window layers too.

## Checkpoint format (`flint-int4`, MLX affine INT4)

- Projections, experts and `lm_head` are INT4 with **group size 64** along the input dimension. Each tensor
  comes as `weight` (`uint8`, two values per byte, **low nibble = even k**), `scales` and `biases` (BF16,
  `[out, in/64]`). The dequantised weight is `w = scale * q + bias` with `q` in 0..15.
- The format is **asymmetric**: `bias/scale` is a continuous per-group value (mean about -7.5, range about
  -230 to 0 over the whole checkpoint), so the zero point cannot be folded into the weights or rounded to
  an integer. Edge-LLM's existing GPTQ path folds zero points into the weights with a clamp, which is lossy
  here, and the vendored Marlin kernels have their zero-point path removed. This is why Flint needs its own
  kernels.
- `model.embed_tokens` is INT8 affine (`q` is the byte, same group size). It is dequantised to the FP16
  embedding table at export.
- Routed experts are stored stacked per layer (`mlp.switch_mlp.{gate,up,down}_proj.*`, leading dimension 256).
- The router (`mlp.gate.weight`, `[256, 2048]`), all norms and the sinks are BF16.
- The shards (`model-0000N.safetensors`) ship **without** a `model.safetensors.index.json`.

## What Edge-LLM adds

New files unless noted. Existing kernels and plugins are unchanged: Flint has its own kernels and plugins,
and the existing files only received additive wiring (listed at the end).

**Kernels**

| Component | Location | Notes |
| :-- | :-- | :-- |
| Dense affine W4A16 GEMM and GEMV (CuTe DSL) | `kernelSrcs/int4_affine_cutedsl/` | 16 GEMM variants plus GEMV for M = 1..8; ABI in the README there |
| Affine Marlin MoE GEMM | `cpp/kernels/moe/moe_marlin/marlin_moe_wna16_affine/`, `moeMarlinAffine.{h,cu}` | clone of the Marlin template in its own namespace with a bias term |
| Scaled SwiGLU | `cpp/kernels/moe/moeScaledSwiGluKernels.{h,cu}` | FP32 SwiGLU scaled by 2^-k, one rounding to FP16 |
| Attention sinks in paged prefill | `kernelSrcs/fmha_v2_cutedsl/fmha.py` (`has_sink`), variants `fmha_v2_d128_sink_paged` and `fmha_v2_d128_sw_sink_paged` | decode already supported sinks through the XQA kernel |

The affine dequantisation keeps the existing cheap nibble conversion, which yields `q - 8`, and replaces the
scale multiply by one fused multiply-add: `w = fma(q - 8, s, b')` with `b' = fp16(float32(bias) + 8 * float32(scale))`.
The instruction count is unchanged; the bias is read next to the scale, so the scale-side bytes double.

**Plugins**

- `Int4AffineGroupwiseGemmPlugin` (`cpp/plugins/int4AffineGemmPlugin/`): dense linears (q/k/v/o, dense MLP,
  shared expert, `lm_head`). Inputs: FP16 activation, int8 weight fragments, FP16 scales, FP16 `b'`.
  Attributes: `gemm_n`, `gemm_k`, `group_size` (must be 64). Tactic ids 1-16 are the GEMM variants, 17 is the
  decode GEMV (M <= 8, otherwise a GEMM fallback). Constraints: K % 64 == 0, N % 8 == 0, split-K must divide K/64.
  `EDGELLM_INT4_AFFINE_FORCE_TACTIC=<id>` forces a tactic for tests.
- `Int4MoeAffinePlugin` (`cpp/plugins/int4MoeAffinePlugin/`): the routed experts of one layer. Eight inputs:
  router logits (FP32), hidden (FP16), gate|up qweights/scales/`b'`, down qweights/scales/`b'`. Routing uses the
  existing `moeSigmoidGroupTopk` kernel with one group. Attributes as `Int4MoePlugin` plus `act_scale_log2`
  (4 for Flint), `norm_topk_prob` and `routed_scaling_factor`. The gate and up projections are fused along N
  with the gate columns first.

**Python and export**

- `tensorrt_edgellm/models/flint/`: `FlintForCausalLM` (reuses the dual-RoPE token-major interface of the
  Muse-Glimmer model: sliding layers get the RoPE table, full layers the NoPE table), `FlintAttention`,
  `FlintSparseMoeBlock`, `FlintRuntimeEmbedding`.
- Quant type `QUANT_INT4_MLX_AFFINE`, recognised for `model_type: flint` (or a `flint-int4.json` sidecar with
  `weight_format: mlx-affine-int4`) and `quantization.bits == 4`; group sizes other than 64 are rejected.
- `MlxAffineInt4Linear` (`models/linear.py`) and the export-time repack in
  `tensorrt_edgellm/checkpoint/mlx_affine_repack.py`.
- The loader now accepts sharded safetensors checkpoints that have no index file (`checkpoint/loader.py`).
- Config: per-layer sliding/full labels, dual RoPE with NoPE on the full-attention layers (validated against
  `no_rope_layer_indices`), QK norm as an architectural property (`config.py`).

## Export, build and run

```bash
# Export. Routed experts are streamed from the checkpoint layer by layer (see Memory), so the export does
# not need the 22 GB of experts in RAM. --num-decoder-layer N truncates the model for quick checks.
python3 -m tensorrt_edgellm.scripts.export <flint-int4 dir> <onnx dir> [--num-decoder-layer N]

# Build the engine (the build copies the 21 GB expert sidecar into the engine directory).
./build/examples/llm/llm_build --onnxDir <onnx dir> --engineDir <engine dir> \
  --maxBatchSize <B> --maxInputLen <L> --maxKVCacheCapacity <C>

# Run.
./build/examples/llm/llm_inference --inputFile <requests.json> --engineDir <engine dir> \
  --outputFile <out.json> --dumpOutput --dumpProfile
```

The engine that was validated has `builder_config`: `max_batch_size 4`, `max_input_len 1024`,
`max_kv_cache_capacity 4096`, FP16 KV cache. The exact `llm_build` flags that produced it were not recorded, so
treat the values above as the resulting limits, not as a recipe. **Contexts longer than 4096 tokens have not
been built or tested.** The 49-layer export took about 2 minutes and the build about 1.5 minutes (implementing session).

Build prerequisites that bit during bring-up:

- The CuTe DSL artifacts for the new groups must be built for sm_87 (`kernelSrcs/build_cutedsl.py --kernels
  int4_affine_gemm` and `--kernels fmha`) and the CMake build configured with
  `-DENABLE_CUTE_DSL="fmha;int4_fp16_gemm;int4_affine_gemm"`. The checked-in prebuilt sm_87 tarball does not
  contain the new variants.
- New plugin source files are found by `GLOB_RECURSE`: re-run `cmake .` once after adding them.
- The build script pins `nvidia-cutlass-dsl` 4.7.0 and `cupy` 13.6.0. The development container had 4.8.0 and
  14.2.0, so the local artifact mixes objects built with both runtimes.

## Memory

The device has 30 GB shared between CPU and GPU, and the INT4 experts alone are 21.3 GB.

- **Export**: the experts never enter the PyTorch model or the ONNX graph. The model traces the expert op with
  tiny placeholder inputs (each encodes layer and slot); after the ONNX export, `checkpoint/flint_moe.py`
  turns them into static graph inputs and streams the repack one layer at a time (about 0.5 GB resident) into
  `external_flint_moe_weights.safetensors`, which the existing `external_weight_files` mechanism binds at
  runtime. The loader skips the `mlp.switch_mlp.` keys. The shards are closed after each layer, since open
  memory-mapped shards were the dominant RSS (24.4 GB before that fix, 7.4 GB after, reported by the
  implementing session).
- **Runtime**: `llm_inference` on the 49-layer engine reported `Peak CPU Memory 27,052 MB` and
  `Peak GPU Memory 21,930 MB` (these overlap on a shared-memory device). Run nothing else alongside it.
- Before the streaming hooks existed, exporting without `--num-decoder-layer` loaded all 49 layers, reached
  23-26 GB resident and was killed by the out-of-memory daemon (rc 143). The few-layer exports are still the
  quickest way to iterate.

## Numerics

- **FP16 range**: on massive-activation tokens the routed-expert SwiGLU product `silu(gate) * up` reaches about
  7.6e4 at layer 46 (limit 65504) and 3.4e4 at layer 48; an FP16 pipeline gives NaN there even with ideal
  weights. The plugin evaluates SwiGLU in FP32, scales it by 2^-4 before the single rounding to FP16 and
  multiplies the routing weights of the down GEMM by 2^4 (exact, both are powers of two).
  The residual stream stays far below the limit (maximum about 9.9e3 over the two reference prompts), and the
  same check on the shared expert and dense MLPs (FP16 emulation, both reference prompts) peaks at 6.2e3, so
  those need no scaling. That check covers only those two prompts.
- **GEMV accumulation**: the decode GEMV accumulates in FP16x2 and is looser than the GEMM (relative error
  up to 2.1e-3 of the maximum on the 6144 -> 2048 shape against 2e-4 for the GEMM). An FP32-accumulating GEMV
  has not been tried; the plan is to compare the two once the engine is tuned.
- **fp16 scale constants**: `b'` and the scales in FP16 add no error beyond FP16 rounding of the weight itself
  (scan of all 486 scale/bias tensors, 621 million groups; about 0.3% of groups have subnormal scales and
  contribute only negligible absolute error).

## Validation

Oracle: `tools/flint_ref.py` in the notes repository (plain PyTorch, fp32, streams the real INT4 shards,
sinks on). It answers "The capital of France is" with ` Paris` and, with the chat template, opens with
`<think>We need to write a Python function that reverses a string. ...`.

Numbers marked (implementing session) were reported by the sessions that built each piece and were not
re-measured by the author of this page; the unit tests listed below can be re-run to reproduce the
plugin-level ones.

| Check | Result |
| :-- | :-- |
| Dense plugin vs fp32 reference, q/k/dense-down/lm_head, M 1..512, builder-chosen tactics | relative L2 2.1e-4 to 8.9e-4 (implementing session) |
| MoE plugin vs fp32 reference, layers 2/20/46/48, T 1..512 | relative L2 3.9e-4 to 6.2e-4, 1.05e-3 at layer 48 with one token; no NaN or Inf at layer 46 (implementing session) |
| Attention sinks in prefill, 396 kernel comparisons | max abs 2.9e-3 against fp32 (the sink-free kernel has 2.7e-3 against its own reference); old variants bit-identical (implementing session) |
| 2-layer engine (attention and dense MLP) vs reference, three prompts, prefill and 5 decode steps | logits relative L2 at most 3.0e-3, top-1 always equal, K/V at most 1.3e-3 (implementing session) |
| 3-layer engine (adds one MoE layer) | logits relative L2 at most 2.5e-3, top-1 6/6 (implementing session) |
| **49-layer engine, greedy chat prompt, 40 tokens** | **identical to the reference for the first 194 characters (about 33 tokens), then diverges where the reference is nearly tied**: at that step the reference's top two logits are ` a` (-0.83) and `:\n\n` (-0.91), a margin of 0.08, and the FP16 engine picks the second (measured in the main session) |
| 49-layer engine logits drift vs reference | relative L2 3.7e-4 to 2.0e-3 on the chat prompt, 1.6e-2 to 4.8e-2 on the raw and long prompts, top-1 equal in all rounds, cosine at least 0.9996 (implementing session; the larger drift on the non-chat prompts was not investigated) |
| 49-layer engine speed, chat prompt, 40 tokens, nothing tuned | prefill of 11 tokens 83.8 ms; decode 51.1 tokens/s (19.6 ms/token, higher is better); wall-clock 45.2 tokens/s (measured in the main session; power mode and clocks not checked) |

Unit tests (run inside the development container with `LLM_SDK_DIR` set):

- `tests/defs/test_flint_model.py`: config and module-tree structure on a synthetic tiny config.
- `tests/python-unittests/test_int4_affine_gemm_plugin.py`, `test_mlx_affine_int4.py`: dense plugin and wiring
  on real checkpoint tensors.
- `tests/python-unittests/test_int4_moe_affine_plugin.py`: MoE plugin on real tensors of layers 2, 20, 46, 48.
- `kernelSrcs/int4_affine_cutedsl/test_int4_affine.py`, `kernelSrcs/fmha_v2_cutedsl/test_fmha_v2_sink.py`,
  `tests/flint_affine_moe/`: standalone kernel validation.

## Limitations and not yet validated

- Context above 4096 tokens, and the 262,144 configured context, are untested.
- Sliding-window layers are covered by kernel-level tests only; the engine checks used the 2-layer and
  3-layer models, whose layers are all full attention, and the 49-layer comparison covers the
  sliding-window variant only through the greedy text above.
- Chunked prefill with sinks, batch larger than 1, tensor parallelism and `--externalize-weights` for the
  affine initialisers are not exercised. Prefill sinks support only head dim 128 with FP16 and reject the
  other paths (padding mask, quantised KV, bounded SWA cache, vision-block attention).
- Bounded sliding-window KV storage is not wired for Flint: every layer keeps a full KV cache.
- K/V dumps for all 49 layers kill the inference process (memory), so layer-by-layer cache comparison was
  done for layers 0-1 only.
- Tuning: no Flint-specific tile tuning of the dense kernels, no timing of the MoE plugin, no profile of the
  49-layer engine.

## Edits to existing files

Additive only (new branches in the code, new entries, one defaulted parameter): `tensorrt_edgellm/config.py`,
`models/linear.py`, `models/ops.py`, `onnx/{onnx_custom_schemas,dynamo_translations,export}.py`,
`checkpoint/{loader,repacking}.py`, `_export_api.py`, `kernelSrcs/build_cutedsl.py`,
`kernelSrcs/fmha_v2_cutedsl/fmha.py` (a `has_sink` flag; the `has_sink=False` kernels were checked bit-identical),
`cpp/kernels/contextAttentionKernels/{cuteDslFMHAV2Runner.{h,cpp},cuteDslFmhaParams.h}` and
`cpp/plugins/attentionPlugin/attentionPlugin.cpp` (defaulted sink parameter and a narrowed rejection check).
