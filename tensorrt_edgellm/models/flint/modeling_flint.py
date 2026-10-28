# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Flint (Recursive) token-major text decoder for checkpoint export.

Flint is a Qwen3-MoE-style decoder with four differences:

* Learned per-head attention sinks (``self_attn.softmax_offset``): one logit per
  query head added to the softmax denominator, no value contribution.
* Hybrid attention: ``layer_types`` marks each layer sliding (window 513, the
  current token included) or full; full-attention layers use no positional
  encoding (NoPE, ``no_rope_layer_indices``), sliding layers use plain RoPE.
* Sigmoid routing: top-k of the sigmoid router scores, weights renormalised to
  sum to one (no group selection, no score-correction bias).
* An always-on, ungated shared expert next to the 256 routed experts; layers in
  ``mlp_only_layers`` use a dense SwiGLU MLP instead.

Reference semantics: Recursive's ``flint_sglang`` model and llama.cpp
``src/models/flint.cpp`` (both agree on all of the above).

Checkpoint keys (``flint-int4``, MLX affine INT4, group 64)::

    model.embed_tokens.{weight,scales,biases}                (INT8)
    model.layers.N.input_layernorm.weight
    model.layers.N.post_attention_layernorm.weight
    model.layers.N.self_attn.{q,k,v,o}_proj.{weight,scales,biases}
    model.layers.N.self_attn.{q,k}_norm.weight               [head_dim]
    model.layers.N.self_attn.softmax_offset                  [num_heads]
    model.layers.N.mlp.{gate,up,down}_proj.*                 (layers 0-1, dense)
    model.layers.N.mlp.gate.weight                           [E, hidden], BF16
    model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.*      stacked [E, ...]
    model.layers.N.mlp.shared_expert.{gate,up,down}_proj.*
    model.norm.weight
    lm_head.{weight,scales,biases}
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..default.modeling_default import (MLP, Attention, RMSNorm, qkv_concat)
from ..linear import make_linear
from ..muse_glimmer.modeling_muse_glimmer_text import (
    MuseGlimmerForCausalLM, _select_rope_for_layer)
from ..ops import attention_plugin
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoERouter

logger = logging.getLogger(__name__)

# ``config.QUANT_INT4_MLX_AFFINE`` (branch feat/flint-int4-affine-gemm); spelled out here so
# this module imports without that branch merged (the INT8 embedding then simply never matches).
_MLX_AFFINE_QUANT = "int4_mlx_affine"

__all__ = [
    "FlintRuntimeEmbedding",
    "FlintAttention",
    "FlintSparseMoeBlock",
    "FlintDecoderLayer",
    "FlintTransformer",
    "FlintForCausalLM",
]


def _is_moe_layer(config: ModelConfig, layer_idx: int) -> bool:
    """Flint: every layer outside ``mlp_only_layers`` is a sparse MoE layer."""
    return (layer_idx not in config.mlp_only_layers
            and config.num_experts > 0)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
class FlintRuntimeEmbedding(nn.Module):
    """INT8 affine embedding table, materialised as FP16 for the C++ runtime.

    The ``flint-int4`` checkpoint stores ``model.embed_tokens`` as MLX affine INT8
    (``weight`` uint8 ``[V, H]``, BF16 ``scales`` / ``biases`` ``[V, H/group]``,
    ``w = scale * q + bias`` with ``q`` the byte). The runtime looks tokens up in an
    FP16 ``embedding.safetensors``, so the export asks :meth:`runtime_weight` for the
    dequantised table (the lookup is not part of the engine graph).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        group = max(int(config.quant.group_size), 1)
        self.register_buffer(
            "weight",
            torch.zeros(config.vocab_size,
                        config.hidden_size,
                        dtype=torch.uint8))
        for name in ("scales", "biases"):
            self.register_buffer(
                name,
                torch.zeros(config.vocab_size,
                            config.hidden_size // group,
                            dtype=torch.bfloat16))

    def runtime_weight(self) -> torch.Tensor:
        """FP16 table ``fp16(scale * q + bias)`` evaluated in float32."""
        vocab, hidden = self.weight.shape
        groups = self.scales.shape[1]
        group = hidden // groups
        out = torch.empty(vocab, hidden, dtype=torch.float16)
        chunk = 8192
        for r0 in range(0, vocab, chunk):
            r1 = min(r0 + chunk, vocab)
            q = self.weight[r0:r1].to(torch.float32).view(
                r1 - r0, groups, group)
            scale = self.scales[r0:r1].to(torch.float32).unsqueeze(-1)
            bias = self.biases[r0:r1].to(torch.float32).unsqueeze(-1)
            out[r0:r1] = (q * scale + bias).view(r1 - r0,
                                                 hidden).to(torch.float16)
        return out


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class FlintAttention(Attention):
    """GQA attention with QK-norm, learned sinks, per-layer sliding window.

    QK-norm is fused in the attention plugin (gamma captured by the base
    class). The base ``sliding_window_size`` is overridden per layer: sliding
    layers get ``config.sliding_window_size`` (513), full layers ``-1``.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.attention_type = config.attention_layer_types[layer_idx]
        self.sliding_window_size = (config.sliding_window_size
                                    if self.attention_type
                                    == "sliding_attention" else -1)
        # One logit per query head; stored FP32, captured as a plain list for
        # the plugin attribute (see ``Attention._q_norm_gamma_list``).
        self.register_buffer("softmax_offset",
                             torch.zeros(self.num_heads, dtype=torch.float32))
        self._attention_sinks_list: List[float] = []
        self.register_load_state_dict_post_hook(
            lambda *_args, **_kwargs: self._capture_attention_sinks())

    def _capture_attention_sinks(self) -> None:
        self._attention_sinks_list = self.softmax_offset.detach().to(
            torch.float32).cpu().flatten().tolist()

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        positions: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        skip_softmax_scale: "torch.Tensor | None" = None,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del positions, state_indices, tree_parent_ids, tree_depths
        del valid_tree_counts, skip_softmax_scale
        if hasattr(self, "qkv_proj_fused"):
            qkv = self.qkv_proj_fused(hidden_states)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            qkv = (qkv_concat(q, k, v)
                   if self._uses_int4_qkv else torch.cat([q, k, v], dim=-1))
        qkv = qkv.reshape(-1, qkv.shape[-1])
        kwargs: dict = {
            "num_q_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "sliding_window_size": self.sliding_window_size,
            "enable_tree_attention": packed_attention_mask is not None,
            "enable_fp8_kv_cache": self.enable_fp8_kv_cache,
            "attention_scale": self.attention_scale,
            "enable_context_mask_selector": False,
            "enable_vision_block_attention": False,
            "skip_softmax_scale_factor": self.skip_softmax_scale_factor,
            "qkv_scales": getattr(self, "_qkv_scales_float", [1.0, 1.0, 1.0]),
            "query_start_offsets": query_start_offsets,
            "attention_sequence_lengths": attention_sequence_lengths,
            "execution_phase_marker": execution_phase_marker,
            "context_sequence_count_carrier": context_sequence_count_carrier,
            "q_norm_gamma": self._q_norm_gamma_list,
            "k_norm_gamma": self._k_norm_gamma_list,
            "rms_norm_eps": float(self._rms_norm_eps),
            "enable_qk_norm": 1,
            "attention_sinks": self._attention_sinks_list,
            "enable_attention_sink": 1,
        }
        if packed_attention_mask is not None:
            kwargs["attention_mask"] = packed_attention_mask
            kwargs["attention_pos_id"] = attention_position_ids
        attn_output, present_key_value = attention_plugin(
            qkv, past_key_value, query_lengths, rope_rotary_cos_sin,
            past_lengths, kv_page_table, **kwargs)
        attn_output = attn_output.reshape(*hidden_states.shape[:-1],
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value


# ---------------------------------------------------------------------------
# Sparse MoE block
# ---------------------------------------------------------------------------
class _ExpertProj(nn.Module):
    """Stacked routed-expert projection (MLX affine INT4, group 64).

    ``weight`` is uint8 ``[E, N, K/2]`` (two nibbles per byte, low nibble
    first along K), ``scales`` / ``biases`` are BF16 ``[E, N, K/group]`` with
    ``w = scale * q + bias``.
    """

    def __init__(self, num_experts: int, out_features: int, in_features: int,
                 group_size: int) -> None:
        super().__init__()
        self.register_buffer(
            "weight",
            torch.zeros(num_experts,
                        out_features,
                        in_features // 2,
                        dtype=torch.uint8))
        for name in ("scales", "biases"):
            self.register_buffer(
                name,
                torch.zeros(num_experts,
                            out_features,
                            in_features // group_size,
                            dtype=torch.bfloat16))


class FlintSparseMoeBlock(nn.Module):
    """Sigmoid top-k routed experts + always-on ungated shared expert.

    ``y = routed(x) + shared_expert(x)`` where ``routed`` is one fused plugin
    op (routing, expert GEMMs, SwiGLU, weighted combine). The routed op comes
    from the affine-INT4 MoE plugin (``feat/flint-marlin-zp-moe``);
    ``_routed_experts`` is the single hook that branch fills in.
    """

    def __init__(self, config: ModelConfig, module_prefix: str) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.norm_topk_prob = config.norm_topk_prob
        self.group_size = config.quant.group_size
        self.gate = Qwen3MoERouter(config)
        self.switch_mlp = nn.Module()
        group = max(self.group_size, 1)
        self.switch_mlp.gate_proj = _ExpertProj(self.num_experts,
                                                self.moe_intermediate_size,
                                                self.hidden_size, group)
        self.switch_mlp.up_proj = _ExpertProj(self.num_experts,
                                              self.moe_intermediate_size,
                                              self.hidden_size, group)
        self.switch_mlp.down_proj = _ExpertProj(self.num_experts,
                                                self.hidden_size,
                                                self.moe_intermediate_size,
                                                group)
        shared = config.moe_shared_expert_intermediate_size
        self.shared_expert = nn.Module()
        self.shared_expert.gate_proj = make_linear(
            config,
            self.hidden_size,
            shared,
            module_name=f"{module_prefix}.shared_expert.gate_proj")
        self.shared_expert.up_proj = make_linear(
            config,
            self.hidden_size,
            shared,
            module_name=f"{module_prefix}.shared_expert.up_proj")
        self.shared_expert.down_proj = make_linear(
            config,
            shared,
            self.hidden_size,
            module_name=f"{module_prefix}.shared_expert.down_proj")

    def _shared_expert_forward(self,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.shared_expert.gate_proj(hidden_states)
        up = self.shared_expert.up_proj(hidden_states)
        return self.shared_expert.down_proj(F.silu(gate) * up)

    def _routed_experts(self, router_logits: torch.Tensor,
                        hidden_flat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Flint routed experts need the affine-INT4 MoE plugin "
            "(branch feat/flint-marlin-zp-moe).")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_flat, self.gate.weight).float()
        routed = self._routed_experts(router_logits, hidden_flat)
        routed = routed.reshape(hidden_states.shape)
        return routed + self._shared_expert_forward(hidden_states)


# ---------------------------------------------------------------------------
# Decoder layer / transformer / causal LM
# ---------------------------------------------------------------------------
class FlintDecoderLayer(nn.Module):
    """Pre-norm layer: Flint attention, then dense MLP or sparse MoE block."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = FlintAttention(config, layer_idx=layer_idx)
        if _is_moe_layer(config, layer_idx):
            self.mlp = FlintSparseMoeBlock(
                config, module_prefix=f"layers.{layer_idx}.mlp")
        else:
            self.mlp = MLP(config, layer_idx=layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

    def forward_ragged(self, hidden_states: torch.Tensor,
                       past_key_value: torch.Tensor,
                       rope_rotary_cos_sin: torch.Tensor,
                       **metadata) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_output, present_key_value = self.self_attn.forward_ragged(
            self.input_layernorm(hidden_states), past_key_value,
            rope_rotary_cos_sin, **metadata)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class FlintTransformer(nn.Module):
    """Flint decoder stack with dual RoPE tables (sliding RoPE, full NoPE)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.quant.quant_type == _MLX_AFFINE_QUANT:
            self.embed_tokens = FlintRuntimeEmbedding(config)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size,
                                             config.hidden_size)
        self.layers = nn.ModuleList([
            FlintDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.target_hidden_concat: "torch.Tensor | None" = None

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin_sliding: torch.Tensor,
        rope_rotary_cos_sin_full: torch.Tensor,
        positions: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        attention_sequence_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_position_ids: "torch.Tensor | None" = None,
        packed_attention_mask: "torch.Tensor | None" = None,
        tree_parent_ids: "torch.Tensor | None" = None,
        tree_depths: "torch.Tensor | None" = None,
        valid_tree_counts: "torch.Tensor | None" = None,
        dflash_target_layer_ids: "List[int] | None" = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        del dflash_target_layer_ids
        hidden_states = inputs_embeds
        present: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            rope = _select_rope_for_layer(layer, rope_rotary_cos_sin_sliding,
                                          rope_rotary_cos_sin_full)
            hidden_states, kv = layer.forward_ragged(
                hidden_states,
                past_key_values[i],
                rope,
                positions=positions,
                query_start_offsets=query_start_offsets,
                query_lengths=query_lengths,
                past_lengths=past_lengths,
                attention_sequence_lengths=attention_sequence_lengths,
                state_indices=state_indices,
                execution_phase_marker=execution_phase_marker,
                context_sequence_count_carrier=context_sequence_count_carrier,
                kv_page_table=kv_page_table,
                attention_position_ids=attention_position_ids,
                packed_attention_mask=packed_attention_mask,
                tree_parent_ids=tree_parent_ids,
                tree_depths=tree_depths,
                valid_tree_counts=valid_tree_counts)
            present.append(kv)
        self.target_hidden_concat = None
        return self.norm(hidden_states), tuple(present)


class FlintForCausalLM(MuseGlimmerForCausalLM):
    """Flint causal LM.

    Reuses the dual-RoPE token-major ABI and ONNX export spec of
    :class:`MuseGlimmerForCausalLM` (same input/output contract: sliding and
    full RoPE tables, per-layer paged KV, ragged metadata); only the
    transformer stack differs. ``final_logit_softcapping`` and
    ``output_multiplier`` stay at their identity defaults for Flint.
    """

    # Muse keeps FP32 norm gains; Flint's RMSNorm gains follow the model dtype.
    match_fp32_elementwise_initializers = False

    def __init__(self, config: ModelConfig) -> None:
        # Skip MuseGlimmerForCausalLM.__init__: it would build a Muse stack.
        nn.Module.__init__(self)
        self.config = config
        self.model = FlintTransformer(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")
        self.output_multiplier = 1.0
        self.final_logit_softcapping = None
