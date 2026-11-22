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

from ...core.weights import \
    ParameterSpec  # noqa: F401  (kept for hook symmetry with other families)

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
