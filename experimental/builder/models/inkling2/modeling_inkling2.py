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
"""Inkling 2 (looped MoE) checkpoint-direct graph.

The residual stream, block attention residuals and message gates run in FP32; projections, the attention visit, the short
convolutions and the experts run in FP16. Each visit of a layer owns paged-KV entry v and conv-state entry v.
"""

import logging
import math
import os
from typing import Dict, List, Optional

import numpy as np
import tensorrt as trt
import torch

from ...ops import (Linear, Module, NetworkModule, RaggedDecoderInputs,
                    add_ragged_decoder_inputs)
from ...ops import functional as F
from ...core.ragged import bounded_swa_capacity
from ...ops.moe import prepare_gated_int4_weights
from . import runtime_config as rc
from . import weights as weight_conversion

LOGGER = logging.getLogger("builder.inkling2")
_EPS = 1e-6
_FUSED_RESIDUAL = os.environ.get("EDGELLM_INKLING_FUSED_RESIDUAL") == "1"
# EDGELLM_INKLING_FUSED_GATE=1 at build: the message gates of the looped passes run as one plugin (BF16 weights, FP32 activations) instead of three TF32 matmuls
_FUSED_GATE = os.environ.get("EDGELLM_INKLING_FUSED_GATE") == "1"
# EDGELLM_INKLING_TAPS=1 at build: the block-residual plugins that see the stream entering visits 5, 23, 41 and the final mix carry a debug capture slot (drafter study)
_TAPS = os.environ.get("EDGELLM_INKLING_TAPS") == "1"
_TAP_VISITS = {5: 0, 23: 1, 41: 2}
# EDGELLM_INKLING_VERIFY=N at build: the generation profile takes 1..N tokens per step (chain verification of N - 1 drafted tokens). Every short convolution
# then also emits its per-token states (the rollback source after a partly accepted window) and the engine outputs the drafter's four taps.
_VERIFY = int(os.environ.get("EDGELLM_INKLING_VERIFY", "0"))


def _f32(weights, name: str) -> np.ndarray:
    return np.ascontiguousarray(weights.f32(name), dtype=np.float32)


_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _round_nvfp4(x: torch.Tensor):
    """FP32 [R, C] -> NVFP4 (packed bytes [R, C/2], E4M3 block scales [R, C/16] as uint8, per-matrix multiplier)."""
    rows, cols = x.shape
    amax = x.abs().max().clamp_min(1e-30)
    multiplier = amax / (6.0 * 448.0)
    blocks = x.view(rows, cols // 16, 16)
    scale = (blocks.abs().amax(-1, keepdim=True) / 6.0 /
             multiplier).clamp(max=448.0).to(torch.float8_e4m3fn)
    scaled = (blocks /
              (scale.float() * multiplier).clamp_min(1e-30)).abs().clamp(
                  max=6.0)
    index = (scaled.unsqueeze(-1) - _E2M1_LEVELS).abs().argmin(-1)
    code = (index + 8 * ((blocks < 0) & (index > 0)).long()).to(
        torch.uint8).view(rows, cols)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), scale.view(
        torch.uint8).view(rows, cols // 16), multiplier


def _expert_nvfp4(weights, name: str):
    """Per-expert NVFP4 tensors (uint8 [E, R, C/2], uint8 [E, R, C/16], float32 [E]) of one fused expert matrix.

    NVFP4 layers pass through unchanged. MXFP8 layers (no amax tensor) are dequantized and rounded to NVFP4.
    """
    raw = torch.from_numpy(
        np.ascontiguousarray(weights.array(name)).view(np.uint8))
    scale = torch.from_numpy(
        np.ascontiguousarray(weights.array(name + "_scale")).view(np.uint8))
    if weights.has(name + "_amax"):
        multiplier = _f32(weights, name + "_amax") / _f32(
            weights, name + "_outer_scale_denominator")
        return raw, scale, torch.from_numpy(multiplier.astype(np.float32))
    experts, rows, cols = raw.shape
    packed = torch.empty(experts, rows, cols // 2, dtype=torch.uint8)
    block_scale = torch.empty(experts, rows, cols // 16, dtype=torch.uint8)
    multiplier = torch.empty(experts, dtype=torch.float32)
    for i in range(experts):
        dense = (raw[i].view(torch.float8_e4m3fn).float().view(
            rows, cols // 32, 32) *
                 torch.exp2(scale[i].float() - 127.0)[..., None]).view(
                     rows, cols)
        packed[i], block_scale[i], multiplier[i] = _round_nvfp4(dense)
    return packed, block_scale, multiplier


def encode_routing(logits, bias: np.ndarray, top_k: int):
    """FP32 router logits [T, E] -> logits for a softmax top-k router that selects by sigmoid + bias and weights by sigmoid.

    Selected experts get +30 so the softmax renormalizes the plain scores over exactly the chosen set.
    """
    score = logits.sigmoid().maximum(1e-30)
    biased = score + F.constant(
        bias.reshape(1, -1).astype(np.float32), "router_bias")
    values, _ = F.topk(biased, top_k, 1)
    kth = values.reduce(trt.ReduceOperation.MIN, values._reduce_dims(1), True)
    selected = ((biased - kth + 1e-6) * 1e6).minimum(1.0).maximum(0.0)
    return score.log() + selected * 30.0


def rms_unweighted(x, eps: float = _EPS):
    return x / ((x * x).mean(-1, keepdim=True) + eps).sqrt()


def rms_weighted(x, weight: np.ndarray, eps: float = _EPS):
    return rms_unweighted(x, eps) * F.constant(
        weight.astype(np.float32).reshape(1, -1), "norm_w")


def l2_clip_rows(x, threshold: float):
    """x [..., d] FP32: scale each row to at most the given L2 norm."""
    norm = (x * x).sum(-1, keepdim=True).sqrt()
    factor = (threshold / (norm + 1e-6)).minimum(1.0)
    return x * factor


def _bf16_words(w, key: str) -> np.ndarray:
    """A BF16 checkpoint matrix [rows, cols] as int32 words [rows, cols / 2] (two BF16 per word, element 2i in the low half): exact, the plugin widens BF16 to FP32."""
    f32 = np.ascontiguousarray(_f32(w, key))
    bits = f32.view(np.uint32)
    if np.any(bits & 0xFFFF):
        raise ValueError(f"{key}: not exactly representable in BF16")
    half = (bits >> 16).astype(np.uint16)
    return np.ascontiguousarray(half).view(np.int32).reshape(f32.shape[0], f32.shape[1] // 2)


def _matmul_const(x, weight_kn: np.ndarray, name: str):
    return x.matmul(
        F.constant(np.ascontiguousarray(weight_kn, np.float32), name))


def block_attention_residual(states: List, queries: np.ndarray, hidden: int, tap: int = -1):
    """Softmax-weighted mix of the (normalized) block states, one output row per query; returns a list of [T, H] tensors."""
    if _FUSED_RESIDUAL and len(states) <= 8:
        q_rows = np.ascontiguousarray(queries.reshape(-1, hidden),
                                      dtype=np.float32)
        return F.inkling2_block_residual(states,
                                         F.constant(q_rows, "attn_res_q"),
                                         q_rows.shape[0], hidden, _EPS, tap)
    stacked = F.concatenate([s.reshape([-1, 1, hidden]) for s in states],
                            dim=1)
    normed = rms_unweighted(stacked)
    q = queries.reshape(-1, hidden).astype(np.float32)
    rows = []
    for i in range(q.shape[0]):
        column = np.ascontiguousarray(q[i].reshape(1, hidden, 1))
        scores = _matmul_const(normed, column, "attn_res_q") * (hidden**-0.5)
        probs = scores.softmax(1)
        rows.append(
            probs.matmul(normed, lhs_op=trt.MatrixOperation.TRANSPOSE).reshape(
                [-1, hidden]))
    return rows


class MessageGate(Module):

    def __init__(self, ctx, prefix: str):
        super().__init__(ctx, prefix)
        w = ctx.weights
        self.down = _f32(w, self.key("down_weight")).T  # [2H, rank]
        self.forget = _f32(w, self.key("forget_weight")).T  # [rank, H]
        self.gain = _f32(w, self.key("gain_weight")).T
        self.rank = int(self.down.shape[1])
        self.hidden = int(self.down.shape[0]) // 2
        if _FUSED_GATE:
            # the checkpoint weights are BF16: keep their bits (exact) as packed int32 words for the fused plugin; layouts [rank, 2H], [H, rank], [H, rank]
            self.packed = tuple(_bf16_words(w, self.key(name)) for name in ("down_weight", "forget_weight", "gain_weight"))

    def forward(self, hidden, message):
        if _FUSED_GATE:
            down, forget, gain = self.packed
            return F.inkling2_message_gate(hidden, message, F.constant(down, "gate_down"), F.constant(forget, "gate_forget"), F.constant(gain, "gate_gain"),
                                           self.hidden, self.rank, _EPS)
        gate_in = F.concatenate(
            [rms_unweighted(hidden),
             rms_unweighted(message)], dim=1)
        low = _matmul_const(gate_in, self.down, "msg_down")
        forget = _matmul_const(low, self.forget, "msg_forget").tanh()
        gain = _matmul_const(low, self.gain, "msg_gain").tanh()
        return (1.0 - forget) * hidden + gain * message


def static_taps(raw: np.ndarray) -> np.ndarray:
    """Depthwise conv1d weight [C, 1, k] (last tap = current token) -> taps [C, k] indexed by lag."""
    return np.ascontiguousarray(raw[:, 0, ::-1], dtype=np.float32)


class Inkling2Layer(Module):

    def __init__(self, ctx, index: int):
        prefix = f"model.llm.layers.{index}"
        super().__init__(ctx, prefix)
        cfg, w = ctx.cfg, ctx.weights
        text = cfg.raw_component
        self.index = index
        self.hidden = cfg.hidden_size
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.k_dyn_groups = self.nkv
        self.rel_dim = int(text["d_rel"])
        self.window = 0 if index not in set(text["local_layer_ids"]) else int(
            text["sliding_window_size"])
        self.full = self.window == 0
        self.qk_clip = float(text["qk_l2_clip_threshold"])
        self.rel_clip = float(text["rel_l2_clip_threshold"])
        self.log_alpha = float(text["log_scaling_alpha"])
        self.log_floor = float(text["log_scaling_n_floor"])
        n_block = int(text["n_layers_per_attn_res_block"])
        self.boundary = index % n_block == 0
        loop = text["loop_config"]
        self.in_core = int(loop["core_start"]) <= index < int(loop["core_end"])
        self.eps = float(cfg.rms_norm_eps)

        a = self.key("attn")
        self.wq = Linear(ctx, a + ".wq_du")
        self.wk = Linear(ctx, a + ".wk_dv")
        self.wv = Linear(ctx, a + ".wv_dv")
        self.wo = Linear(ctx, a + ".wo_ud")
        self.gate_attn = Linear(ctx, a + ".gate_attn")
        self.sink = Linear(ctx, a + ".attn_sink_proj")
        self.wr = Linear(ctx, a + ".wr_du")
        self.q_scale = _f32(w, a + ".q_l2_clip.scale").reshape(-1)
        self.rel_scale = _f32(w, a + ".rel_l2_clip.scale").reshape(-1)
        self.rel_proj_t = np.ascontiguousarray(
            _f32(w, a + ".rel_logits_proj.proj").T)
        self.rel_extent = int(self.rel_proj_t.shape[0])
        if self.full:
            self.k_dyn_w = np.ascontiguousarray(
                w.f16(a + ".k_sconv.weight_proj").T)
            self.v_dyn_w = np.ascontiguousarray(
                w.f16(a + ".v_sconv.weight_proj").T)
            self.k_taps = np.ascontiguousarray(
                _f32(w, a + ".k_sconv.static_w").T)
            self.v_taps = np.ascontiguousarray(
                _f32(w, a + ".v_sconv.static_w").T)
        else:
            self.k_taps = static_taps(_f32(w, a + ".k_sconv.weight"))
            self.v_taps = static_taps(_f32(w, a + ".v_sconv.weight"))
        self.attn_taps = static_taps(_f32(w, self.key("attn_sconv.weight")))
        self.mlp_taps = static_taps(_f32(w, self.key("mlp_sconv.weight")))
        if self.full:
            self.q_norm = _f32(w, self.key("q_input_norm.weight"))
            self.k_norm = _f32(w, self.key("k_input_norm.weight"))
            self.v_norm = _f32(w, self.key("v_input_norm.weight"))
        else:
            self.in_norm = _f32(w, self.key("attn_norm.weight"))
        self.mlp_norm = _f32(w, self.key("mlp_norm.weight"))
        self.attn_input_query = (_f32(
            w, f"model.llm.attn_res.attn_queries.{index - 1}")
                                 if index > 0 else None)
        self.mlp_input_query = _f32(w,
                                    f"model.llm.attn_res.mlp_queries.{index}")
        if self.in_core:
            self.gates = {
                name: MessageGate(ctx, self.key(f"controllers.{name}"))
                for name in ("attn", "k", "v", "mlp")
            }
        self.router_w = np.ascontiguousarray(
            _f32(w, self.key("mlp.gate.weight")).T)
        self.router_bias = _f32(w, self.key("mlp.gate.bias"))
        self.int4 = w.has(self.key("mlp.experts.0.gate_proj.qweight"))
        self.nvfp4_off = w.has(self.key("mlp.experts.fc1_qweights"))
        if not self.int4 and not self.nvfp4_off:
            self.moe_weights, self.moe_inter = self._pack_experts(ctx)

    def _pack_experts(self, ctx):
        from tensorrt_edgellm.checkpoint.repacking import \
            repack_nvfp4_a16_marlin_gated_moe_experts
        w = ctx.weights
        text = ctx.cfg.raw_component
        e, inter = int(text["n_routed_experts"]), int(
            text["intermediate_size"])
        padded = ((inter + 127) // 128) * 128
        base = self.key("mlp.experts")
        w13, s13, g13 = _expert_nvfp4(w, base + ".w13_weight")
        w2, s2, g2 = _expert_nvfp4(w, base + ".w2_weight")
        gp, gs, gg, up, us, ug, dp, ds, dg = ([] for _ in range(9))
        for i in range(e):
            gp.append(w13[i, 0::2])
            up.append(w13[i, 1::2])
            gs.append(s13[i, 0::2])
            us.append(s13[i, 1::2])
            gg.append(g13[i])
            ug.append(g13[i])
            dp.append(w2[i])
            ds.append(s2[i])
            dg.append(g2[i])
        fc1q, fc1s, fc1g, fc2q, fc2s, fc2g = repack_nvfp4_a16_marlin_gated_moe_experts(
            gp, gs, gg, up, us, ug, dp, ds, dg, padded)
        packed = {
            "fc1_qweights": fc1q.numpy().astype(np.int8),
            "fc1_block_scales": fc1s.numpy().astype(np.int8),
            "fc1_global_scales": fc1g.numpy().astype(np.float16),
            "fc2_qweights": fc2q.numpy().astype(np.int8),
            "fc2_block_scales": fc2s.numpy().astype(np.int8),
            "fc2_global_scales": fc2g.numpy().astype(np.float16),
            "e_score_correction_bias": self.router_bias,
        }
        return packed, padded

    # -- one visit ------------------------------------------------------------------------------------------------
    def _fused_qrsg(self, hq):
        """EDGELLM_INKLING_FUSED_QRSG=1 at build (with INT8 projections): wq, gate_attn, wr and the sink projection share the input, so they run as one INT8 GEMV of
        nh * 273 rows laid out [q | gate | rel | sink] (a launch of 6,552 rows instead of four launches that each sit near the launch-latency floor). The attention
        plugin reads the four parts as column ranges of that tensor (``fused_qrsg``). Returns the fused [T, nh * 273] tensor or None."""
        if os.environ.get("EDGELLM_INKLING_FUSED_QRSG") != "1":
            return None
        linears = (self.wq, self.gate_attn, self.wr, self.sink)
        if not all(l._int8_gemv_enabled() and not l._int4_gemv_enabled() for l in linears):
            return None
        parts = [l.int8_rows() for l in linears]
        if any(p is None for p in parts) or self.nh * 273 != sum(p[0].shape[0] for p in parts):
            return None
        # EDGELLM_INT8_MMA=1: the fused projection (6,552 rows) takes the tensor-core path like the other INT8 projections (METRICS 210)
        mma = os.environ.get("EDGELLM_INT8_MMA") == "1" and parts[0][0].shape[1] % 64 == 0
        return F.int8_linear(hq, np.ascontiguousarray(np.concatenate([p[0] for p in parts])),
                             np.ascontiguousarray(np.concatenate([p[1] for p in parts])), name=self.wq.prefix.rsplit(".", 1)[0] + ".qrsg", mma=mma)

    def attention(self, hq, hk, hv, past_kv, conv_state,
                  ragged: RaggedDecoderInputs):
        state_in = conv_state
        nh, nkv, hd = self.nh, self.nkv, self.hd
        if os.environ.get("EDGELLM_INKLING_HESSIAN") == "1":  # calibration engine: GPTQ Hessians of the dense attention inputs (builder-time switch)
            hq = F.inkling2_hessian_tap(hq, int(hq.shape[-1]), self.index * 4)
            hk = F.inkling2_hessian_tap(hk, int(hk.shape[-1]), self.index * 4 + 1)
            hv = F.inkling2_hessian_tap(hv, int(hv.shape[-1]), self.index * 4 + 2)
        fused_qrsg = self._fused_qrsg(hq)
        if fused_qrsg is not None:
            q_lin = rel_lin = sink_lin = gate_lin = fused_qrsg
        else:
            q_lin = self.wq(hq)
            rel_lin = self.wr(hq)
            sink_lin = self.sink(hq)
            gate_lin = self.gate_attn(hq)
        k_lin = self.wk(hk)
        v_lin = self.wv(hv)
        kv_ch = nkv * hd
        if self.full:
            k_conv, _, *ik = F.inkling2_short_conv(
                k_lin, F.constant(self.k_taps, "k_taps"), state_in, ragged,
                kv_ch, 4, 1.0 / math.sqrt(9.0),
                hk.matmul(F.constant(self.k_dyn_w,
                                     "k_dyn_w")), self.k_dyn_groups, 0, _VERIFY)
            v_conv, _, *iv = F.inkling2_short_conv(
                v_lin, F.constant(self.v_taps, "v_taps"), state_in, ragged,
                kv_ch, 4, 1.0 / math.sqrt(9.0),
                hv.matmul(F.constant(self.v_dyn_w,
                                     "v_dyn_w")), self.k_dyn_groups, kv_ch, _VERIFY)
        else:
            k_conv, _, *ik = F.inkling2_short_conv(
                k_lin, F.constant(self.k_taps, "k_taps"), state_in, ragged,
                kv_ch, 4, 1.0 / math.sqrt(5.0), None, 0, 0, _VERIFY)
            v_conv, _, *iv = F.inkling2_short_conv(
                v_lin, F.constant(self.v_taps, "v_taps"), state_in, ragged,
                kv_ch, 4, 1.0 / math.sqrt(5.0), None, 0, kv_ch, _VERIFY)
        k32 = k_conv.cast(trt.float32).reshape([-1, nkv, hd])
        k_new = l2_clip_rows(k32,
                             self.qk_clip).reshape([-1,
                                                    kv_ch]).cast(trt.float16)
        v32 = v_conv.cast(trt.float32).reshape([-1, nkv, hd])
        v_new = rms_unweighted(v32,
                               self.eps).reshape([-1, kv_ch]).cast(trt.float16)
        attn, present_kv = F.inkling2_attention(
            q_lin, rel_lin, sink_lin, gate_lin, k_new, v_new,
            F.constant(self.q_scale, "q_scale"),
            F.constant(self.rel_scale, "rel_scale"),
            F.constant(self.rel_proj_t, "rel_proj_t"), past_kv, ragged, nh,
            nkv, self.rel_extent, self.window, self.log_alpha, self.log_floor,
            self.qk_clip, self.rel_clip, hd**-0.5,
            bounded=self.window > 0 and ragged.swa_kv_page_table is not None,
            kv_int8=self.cfg.kv_cache_dtype == "int8",
            fused_qrsg=fused_qrsg is not None)
        if os.environ.get("EDGELLM_INKLING_HESSIAN") == "1":
            attn = F.inkling2_hessian_tap(attn, int(attn.shape[-1]), self.index * 4 + 3)
        out = self.wo(attn)
        out, _, *io = F.inkling2_short_conv(out,
                                            F.constant(self.attn_taps, "attn_taps"),
                                            state_in, ragged, self.hidden, 4,
                                            1.0 / math.sqrt(5.0), None, 0,
                                            2 * kv_ch, _VERIFY)
        self.inter = ik + iv + io  # per-token conv states of K, V and the attention output (channel order of the conv state)
        return out.cast(trt.float32), present_kv, state_in

    def _int4_moe(self, logits, hidden16):
        cfg, w = self.cfg, self.ctx.weights
        text = cfg.raw_component
        base = self.key("mlp.experts")
        experts = int(text["n_routed_experts"])
        # the group size comes from the tensors: with mixed-precision checkpoints cfg.group_size is the dense linears' one
        group_size = self.hidden // int(
            w.parameter_spec(f"{base}.0.gate_proj.scales", np.float16).shape[0])

        def materialize():

            def load(expert, projection):
                name = f"{base}.{expert}.{projection}"
                return (w.array(name + ".qweight"), w.array(name + ".qzeros"),
                        w.array(name + ".scales"))

            return prepare_gated_int4_weights(self.ctx, load, group_size)

        moe_weights = w.parameter_value(
            "int4_moe", base,
            lambda: weight_conversion.int4_expert_specs(w, base, experts),
            materialize)
        bindings = weight_conversion.int4_expert_bindings(
            w, base, experts, group_size, cfg.quant.gptq_zero_point_offset)
        encoded, route_bias = self._route(logits)
        routed = F.int4_moe(encoded,
                            hidden16,
                            moe_weights,
                            experts,
                            8,
                            self.hidden,
                            int(text["intermediate_size"]),
                            group_size,
                            route_bias=route_bias,
                            weight_prefix=base,
                            weight_bindings=bindings,
                            zero_point_offset=cfg.quant.gptq_zero_point_offset,
                            gate_softcap=float(text["mlp_gate_softcap"]),
                            up_softcap=float(text["mlp_up_softcap"]),
                            packed_dtype=np.int32)
        return routed * float(text["route_scale"])

    def _route(self, logits):
        """(router input of the plugin, selection bias or None): the plugin does Inkling 2's routing itself unless
        EDGELLM_INKLING_GRAPH_ROUTING=1 asks for the encoded-logits graph."""
        if os.environ.get("EDGELLM_INKLING_GRAPH_ROUTING") == "1":
            return encode_routing(logits, self.router_bias, 8), None
        return logits, F.constant(
            self.router_bias.astype(np.float32).reshape(-1), "router_bias")

    def _nvfp4_offload_moe(self, logits, hidden16, pred_self=None, pred_next=None):
        """Marlin NVFP4 experts from the sidecar, host-resident like the INT4 ones (Int4MoePlugin weight_format 1)."""
        cfg, w = self.cfg, self.ctx.weights
        text = cfg.raw_component
        base = self.key("mlp.experts")
        experts = int(text["n_routed_experts"])
        specs = lambda: weight_conversion.nvfp4_expert_specs(w, base)

        def materialize():
            return {
                engine: w.array(f"{base}.{stored}")
                for engine, stored, _ in weight_conversion._NVFP4_TENSORS
            }

        moe_weights = w.parameter_value("int4_moe", base, specs, materialize)
        inter = int(moe_weights["fc_down_scales"].shape[1]) * 16
        encoded, route_bias = self._route(logits)
        routed = F.int4_moe(encoded,
                            hidden16,
                            moe_weights,
                            experts,
                            8,
                            self.hidden,
                            inter,
                            16,
                            route_bias=route_bias,
                            pred_self=pred_self,
                            pred_next=pred_next,
                            weight_prefix=base,
                            weight_bindings=weight_conversion.nvfp4_expert_bindings(w, base),
                            gate_softcap=float(text["mlp_gate_softcap"]),
                            up_softcap=float(text["mlp_up_softcap"]),
                            packed_dtype=np.int32,
                            weight_format="nvfp4")
        return routed * float(text["route_scale"])

    def moe(self, x32, conv_state, ragged):
        logits = _matmul_const(x32, self.router_w, "router_w")
        hidden16 = x32.cast(trt.float16)
        if self.nvfp4_off:
            # predicted prefetch (EDGELLM_INKLING_PREFETCH=1 at build): this visit's router input run through the NEXT visit's router
            next_w = getattr(self, "prefetch_next_w", None)
            pred_next = _matmul_const(x32, next_w, "router_w") if next_w is not None else None
            pred_self = getattr(self, "prefetch_self_pred", None)
            self.prefetch_pred_out = pred_next
            routed = self._nvfp4_offload_moe(logits, hidden16, pred_self, pred_next)
            return self._moe_conv(routed, conv_state, ragged)
        if self.int4:
            routed = self._int4_moe(logits, hidden16)
            return self._moe_conv(routed, conv_state, ragged)
        routed = F.nvfp4_a16_moe(
            logits,
            hidden16,
            self.moe_weights,
            int(self.moe_weights["fc1_global_scales"].shape[0]),
            8,
            self.hidden,
            self.moe_inter,
            F.MoeActivation.SOFTCAP_SWIGLU,
            1,
            1,
            1,
            float(self.cfg.raw_component["route_scale"]),
            F.MoeRouting.SIGMOID_GROUP_TOPK,
            weight_prefix=self.key("mlp.experts"),
            gate_softcap=float(self.cfg.raw_component["mlp_gate_softcap"]),
            up_softcap=float(self.cfg.raw_component["mlp_up_softcap"]))
        return self._moe_conv(routed, conv_state, ragged)

    def _moe_conv(self, routed, conv_state, ragged):
        out, conv_state, *im = F.inkling2_short_conv(
            routed, F.constant(self.mlp_taps, "mlp_taps"), conv_state, ragged,
            self.hidden, 4, 1.0 / math.sqrt(5.0), None, 0,
            2 * self.nkv * self.hd + self.hidden, _VERIFY)
        self.inter = self.inter + im
        return out.cast(trt.float32), conv_state

    def _tap(self) -> int:
        return _TAP_VISITS.get(getattr(self, "visit", -1), -1) if _TAPS else -1

    def forward(self, h, bs, messages, past_kv, conv_state, ragged):
        hidden = self.hidden
        residual_in = h
        gates = getattr(self, "gates", None)
        if self.full:
            if self.attn_input_query is None:
                qi = ki = vi = h
            else:
                qi, ki, vi = block_attention_residual(bs + [h],
                                                      self.attn_input_query,
                                                      hidden, self._tap())
            if messages is not None:
                qi, ki, vi = (gates["attn"](qi, messages[0]),
                              gates["k"](ki,
                                         messages[1]), gates["v"](vi,
                                                                  messages[2]))
            hq = rms_weighted(qi, self.q_norm, self.eps).cast(trt.float16)
            hk = rms_weighted(ki, self.k_norm, self.eps).cast(trt.float16)
            hv = rms_weighted(vi, self.v_norm, self.eps).cast(trt.float16)
        else:
            x = h if self.attn_input_query is None else block_attention_residual(
                bs + [h], self.attn_input_query, hidden, self._tap())[0]
            if messages is None:
                hq = hk = hv = rms_weighted(x, self.in_norm,
                                            self.eps).cast(trt.float16)
            else:
                hq = rms_weighted(gates["attn"](x, messages[0]), self.in_norm,
                                  self.eps).cast(trt.float16)
                hk = rms_weighted(gates["k"](x, messages[1]), self.in_norm,
                                  self.eps).cast(trt.float16)
                hv = rms_weighted(gates["v"](x, messages[2]), self.in_norm,
                                  self.eps).cast(trt.float16)
        a, present_kv, conv_state = self.attention(hq, hk, hv, past_kv,
                                                   conv_state, ragged)
        if self.boundary:
            bs_local = bs + [residual_in]
            res = a
        else:
            bs_local = bs
            res = residual_in + a
        x = block_attention_residual(bs_local + [res], self.mlp_input_query,
                                     hidden)[0]
        if messages is not None:
            x = gates["mlp"](x, messages[3])
        mo, conv_state = self.moe(rms_weighted(x, self.mlp_norm, self.eps),
                                  conv_state, ragged)
        return res + mo, present_kv, conv_state


class Inkling2ForCausalLM(NetworkModule):

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        cfg = ctx.cfg
        text = cfg.raw_component
        loop = text["loop_config"]
        self.core_start, self.core_end, self.passes = int(
            loop["core_start"]), int(loop["core_end"]), int(loop["num_passes"])
        self.num_layers = int(text["num_hidden_layers"])
        self.layers = [Inkling2Layer(ctx, i) for i in range(self.num_layers)]
        w = ctx.weights
        self.embed_norm = _f32(w, "model.llm.embed_norm.weight")
        self.final_query = _f32(w, "model.llm.attn_res.output_query")
        self.norm = _f32(w, "model.llm.norm.weight")
        q = _f32(w, "model.llm.middle_cycle_cross_pass_queries")
        kv = _f32(w, "model.llm.middle_cycle_cross_pass_kv_queries")
        n_core = self.core_end - self.core_start
        self.message_queries = np.stack([
            np.stack([q[2 * j], kv[2 * j], kv[2 * j + 1], q[2 * j + 1]])
            for j in range(n_core)
        ])
        self.lm_head = Linear(ctx, "model.llm.unembed")
        self.visits = rc.visit_count(cfg)
        self.softcap = float(text["final_logit_softcapping"])

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        conv_channels = rc.conv_channels(cfg)
        kv_dtype = (trt.DataType.INT8 if cfg.kv_cache_dtype == "int8" else
                    trt.DataType.FP8 if cfg.kv_cache_dtype == "fp8" else trt.float16)
        kv_head_dim = cfg.layer_head_dim(0)  # 128, or 132 for INT8 rows (EDGELLM_INKLING_KV_INT8=1)
        result = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{i}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                kv_head_dim)) for i in range(self.visits)
            ],
            "conv_states": [
                self.add_input(f"conv_state_{i}", trt.float16,
                               (-1, conv_channels, rc.CONV_KERNEL))
                for i in range(self.visits)
            ],
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
            "recurrent_states": [
                self.add_input(f"recurrent_state_{i}", trt.float32,
                               (-1, ) + rc.RECURRENT_SHAPE)
                for i in range(self.visits)
            ],
        }
        bounded = bool(cfg.kv_bounded_entries) and bounded_swa_capacity(
            cfg.kv_bounded_window, self.ctx.args.max_input_len,
            self.ctx.args.max_kv_cache_capacity) > 0
        result.update(
            add_ragged_decoder_inputs(self.add_input,
                                      include_swa=bounded,
                                      swa_mode_dtype=trt.int32).as_dict())
        return result

    def forward(self, **io):
        ragged = RaggedDecoderInputs.from_dict(io)
        hidden = self.cfg.hidden_size
        visit = [0]
        present_kv: List = [None] * self.visits
        present_conv: List = [None] * self.visits
        present_rec: List = [None] * self.visits
        inter_conv: List = [None] * self.visits
        tap_states: List = []

        # The runtime always binds a rope table; the model has no rotary embedding, so tie it to a dummy output.
        keep_rope = io["rope"].reshape([-1]).sum(0, keepdim=True).reshape(
            [1, 1, 1, 1]) * 0.0

        order = (list(range(self.core_start)) +
                 list(range(self.core_start, self.core_end)) * self.passes +
                 list(range(self.core_end, self.num_layers)))
        prefetch = os.environ.get("EDGELLM_INKLING_PREFETCH") == "1"
        carry = [None]

        def run_layer(l, h, bs, messages):
            v = visit[0]
            visit[0] += 1
            layer = self.layers[l]
            layer.visit = v
            if prefetch:
                layer.prefetch_next_w = (self.layers[order[v + 1]].router_w
                                         if v + 1 < len(order) else None)
                layer.prefetch_self_pred = carry[0]
            if _VERIFY and v in _TAP_VISITS:
                tap_states.append(h)  # the stream entering the visit (drafter tap 0..2)
            new, kv, conv = layer(h, bs, messages, io["past_key_values"][v],
                                  io["conv_states"][v], ragged)
            if _VERIFY:
                inter_conv[v] = F.concatenate(layer.inter, 1)
            if prefetch:
                carry[0] = layer.prefetch_pred_out
            present_kv[v], present_conv[v] = kv, conv
            present_rec[v] = io["recurrent_states"][v] + keep_rope
            if layer.boundary:
                bs = bs + [h]
            return new, bs

        h = rms_weighted(io["inputs_embeds"].cast(trt.float32),
                         self.embed_norm)
        bs: List = []
        for l in range(self.core_start):
            h, bs = run_layer(l, h, bs, None)
        pre_h, pre_bs = h, list(bs)
        n_pre = len(pre_bs)
        for l in range(self.core_start, self.core_end):
            h, bs = run_layer(l, h, bs, None)
        for p in range(1, self.passes):
            prev_states, prev_h = bs[n_pre + 1:], h
            h, bs = pre_h, list(pre_bs)
            for l in range(self.core_start, self.core_end):
                messages = block_attention_residual(
                    prev_states + [prev_h],
                    self.message_queries[l - self.core_start], hidden)
                h, bs = run_layer(l, h, bs, messages)
        for l in range(self.core_end, self.num_layers):
            h, bs = run_layer(l, h, bs, None)
        if _VERIFY:
            tap_states.append(h)  # the stream after the last visit (drafter tap 3, before the final mix and norm)
        final = block_attention_residual(bs + [h], self.final_query, hidden, 3 if _TAPS else -1)[0]
        normed = rms_weighted(final, self.norm).cast(trt.float16)
        selected = F.gather_token_rows(normed, ragged.logits_indices)
        logits = self.lm_head(selected).cast(trt.float32)
        logits = self.softcap * (logits / self.softcap).tanh()
        padded = int(
            self.ctx.weights.parameter_spec("model.llm.unembed.weight").shape[0])
        unpadded = int(self.cfg.raw_component.get("unpadded_vocab_size", padded))
        if unpadded < padded:
            # the unembedding is padded beyond the tokenizer's ids; those rows are untrained and must never be sampled
            mask = np.zeros((1, padded), np.float32)
            mask[:, unpadded:] = -30000.0
            logits = logits + F.constant(mask, "vocab_pad_mask")
        outputs = {"logits": logits}
        for i in range(self.visits):
            outputs[f"present_key_values_{i}"] = present_kv[i]
            outputs[f"present_conv_state_{i}"] = present_conv[i]
            outputs[f"present_recurrent_state_{i}"] = present_rec[i]
        if _VERIFY:
            for i in range(self.visits):
                outputs[f"intermediate_conv_state_{i}"] = inter_conv[i]
                # placeholder (the recurrent state is unused): one row, whatever the token count, so a long prefill never overruns the runtime's [verify, ...] buffer; distinct per visit
                outputs[f"intermediate_recurrent_state_{i}"] = io["recurrent_states"][i] + F.constant(np.array([[[[float(i), 0.0]]]], np.float32), f"inter_rec_{i}")
            outputs["inkling_taps"] = F.concatenate(tap_states, 1)
        return outputs
