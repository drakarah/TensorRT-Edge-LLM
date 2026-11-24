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

import os
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
    verify = int(os.environ.get("EDGELLM_INKLING_VERIFY", "0"))
    if verify > 0:
        # chain verification engine: generation profile of up to `verify` tokens, per-token conv states and drafter taps as extra outputs
        config["inkling_verify_tokens"] = verify
    if cfg.inkling_int8_kv():
        # INT8 KV rows of 132 bytes (128 values, FP16 scale, 2 pad): the runtime sizes the pools from head_dim and kv_cache_dtype
        config["kv_cache_dtype"] = "int8"
        config["head_dim"] = cfg.layer_head_dim(0)
        for entry in config.get("kv_layer_configs", []):
            if isinstance(entry, dict):  # None marks entries that own no pool of that kind
                entry["head_dim"] = cfg.layer_head_dim(0)


def component_runtime_config(bundle, component: contracts.Component, args):
    return None
