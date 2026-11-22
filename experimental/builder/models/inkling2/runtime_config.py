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
"""Inkling 2 runtime configuration.

Every layer visit owns one paged-KV entry and one conv-state entry that carries the four short convolutions of the visit
(K, V, post-attention, post-MLP) as consecutive channel ranges. A looped layer therefore appears once per pass.
"""

from typing import Any, Dict

from ...core import contracts

CONV_KERNEL = 4
RECURRENT_SHAPE = (1, 1, 2)


def visit_count(cfg) -> int:
    text = cfg.raw_component
    loop = text["loop_config"]
    return int(text["num_hidden_layers"]) + (int(loop["core_end"]) - int(
        loop["core_start"])) * (int(loop["num_passes"]) - 1)


def conv_channels(cfg) -> int:
    return 2 * cfg.num_key_value_heads * cfg.head_dim + 2 * cfg.hidden_size


def update_llm_config(config: Dict[str, Any], root: Dict[str, Any], cfg,
                      args) -> None:
    config["use_rope"] = False


def component_runtime_config(bundle, component: contracts.Component, args):
    return None
