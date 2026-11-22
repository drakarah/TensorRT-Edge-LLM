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
"""Inkling 2 checkpoint configuration."""

from ...core import contracts


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        return root.get("text_config") or root
    raise ValueError(f"Inkling 2 has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    config.setdefault("model_type", root.get("model_type", "inkling2_loop"))
    loop = config["loop_config"]
    layers = int(config["num_hidden_layers"])
    visits = layers + (int(loop["core_end"]) -
                       int(loop["core_start"])) * (int(loop["num_passes"]) - 1)
    # Every visit of a layer owns one paged-KV entry (attention) and one conv-state entry that carries the four short
    # convolutions of the visit (the recurrent-state slot is a dummy of two floats).
    config["layer_types"] = ["full_attention", "linear_attention"] * visits
    conv_channels = (
        2 * int(config["num_key_value_heads"]) * int(config["head_dim"]) +
        2 * int(config["hidden_size"]))
    config["linear_num_key_heads"] = (conv_channels - 2) // 2
    config["linear_key_head_dim"] = 1
    config["linear_num_value_heads"] = 1
    config["linear_value_head_dim"] = 2
    config["linear_conv_kernel_dim"] = 4
    config.setdefault("moe_intermediate_size", config["intermediate_size"])
    # The model has no rotary embedding; the runtime still requires a positive rotary width.
    config["partial_rotary_factor"] = 8.0 / float(config["head_dim"])
    config.setdefault("tie_word_embeddings", False)
    return config
