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
    """Describe the final Marlin MoE buffers without loading provider tensors."""
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
        ParameterSpec((num_experts, hidden_size // 16, 16 * intermediate_size),
                      np.int8),
        "fc_gate_up_scales":
        ParameterSpec(
            (num_experts, gate_scales.shape[0], 2 * intermediate_size),
            np.float16),
        "fc_down_qweights":
        ParameterSpec(
            (num_experts, down_input_size // 16, 8 * down_output_size),
            np.int8),
        "fc_down_scales":
        ParameterSpec((num_experts, down_scales.shape[0], down_output_size),
                      np.float16),
    }
