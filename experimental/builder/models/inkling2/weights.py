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
"""Inkling 2 checkpoint weight mapping (original checkpoint layout, model.llm.* names)."""

import numpy as np

from ...core.weights import ParameterSpec

_ALIASES = {
    "model.embed_tokens.weight": "model.llm.embed.weight",
    "lm_head.weight": "model.llm.unembed.weight",
}


def writes_runtime_embedding(args) -> bool:
    return True


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    alias = _ALIASES.get(name)
    return (alias, ) if alias else ()


def normalize_checkpoint_name(name: str) -> str:
    return name[len("model.llm."):] if name.startswith("model.llm.") else name


def int4_expert_bindings(weights, experts_prefix: str, num_experts: int,
                         group_size: int, zero_point_offset: int) -> dict:
    """Map per-expert GPTQ tensors to the inputs of Int4MoePlugin."""

    def names(projections, leaves):
        return [
            f"{experts_prefix}.{expert}.{projection}.{leaf}"
            for expert in range(num_experts) for projection in projections
            for leaf in leaves
        ]

    common = {
        "num_experts": num_experts,
        "group_size": group_size,
        "zero_point_offset": zero_point_offset,
    }
    return {
        "fc_gate_up_qweights":
        weights.checkpoint_binding(
            names(("gate_proj", "up_proj"), ("qweight", "qzeros", "g_idx")),
            "plugin", "int4_moe_gate_up", **common),
        "fc_gate_up_scales":
        weights.checkpoint_binding(
            names(("gate_proj", "up_proj"), ("scales", )), "plugin",
            "int4_moe_gate_up_scales", **common),
        "fc_down_qweights":
        weights.checkpoint_binding(
            names(("down_proj", ), ("qweight", "qzeros", "g_idx")), "plugin",
            "int4_moe_down", **common),
        "fc_down_scales":
        weights.checkpoint_binding(names(("down_proj", ), ("scales", )),
                                   "plugin", "int4_moe_down_scales", **common),
    }


def int4_expert_specs(weights, experts_prefix: str, num_experts: int) -> dict:
    """Describe the final Marlin MoE buffers without loading provider tensors.

    The packed weights are int32 words (the plugin also takes them as int8 bytes): TensorRT refuses INT8 tensors
    in a network with FP8 Q/DQ, and UInt8 is not a plugin format."""
    gate_qweight = weights.parameter_spec(
        f"{experts_prefix}.0.gate_proj.qweight", np.int32)
    gate_scales = weights.parameter_spec(
        f"{experts_prefix}.0.gate_proj.scales", np.float16)
    down_qweight = weights.parameter_spec(
        f"{experts_prefix}.0.down_proj.qweight", np.int32)
    down_scales = weights.parameter_spec(
        f"{experts_prefix}.0.down_proj.scales", np.float16)
    hidden_size = gate_qweight.shape[0] * 8
    intermediate_size = gate_qweight.shape[1]
    down_input_size = down_qweight.shape[0] * 8
    down_output_size = down_qweight.shape[1]
    return {
        "fc_gate_up_qweights":
        ParameterSpec((num_experts, hidden_size // 16, 4 * intermediate_size),
                      np.int32),
        "fc_gate_up_scales":
        ParameterSpec(
            (num_experts, gate_scales.shape[0], 2 * intermediate_size),
            np.float16),
        "fc_down_qweights":
        ParameterSpec(
            (num_experts, down_input_size // 16, 2 * down_output_size),
            np.int32),
        "fc_down_scales":
        ParameterSpec((num_experts, down_scales.shape[0], down_output_size),
                      np.float16),
    }


_NVFP4_TENSORS = (("fc_gate_up_qweights", "fc1_qweights", np.int32),
                  ("fc_gate_up_scales", "fc1_scales", np.int32),
                  ("fc_down_qweights", "fc2_qweights", np.int32),
                  ("fc_down_scales", "fc2_scales", np.int32),
                  ("fc1_global_scales", "fc1_global_scales", np.float16),
                  ("fc2_global_scales", "fc2_global_scales", np.float16))


def nvfp4_expert_specs(weights, experts_prefix: str) -> dict:
    """Describe the Marlin-packed NVFP4 sidecar tensors of one layer (int32 words, fp16 global scales)."""
    return {
        engine: weights.parameter_spec(f"{experts_prefix}.{stored}", dtype)
        for engine, stored, dtype in _NVFP4_TENSORS
    }


def nvfp4_expert_bindings(weights, experts_prefix: str) -> dict:
    """Identity bindings: the runtime copies each sidecar tensor as it is."""
    return {
        engine:
        weights.checkpoint_binding([f"{experts_prefix}.{stored}"], "plugin",
                                   "identity")
        for engine, stored, _ in _NVFP4_TENSORS
    }
