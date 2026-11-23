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
"""Bounded sliding-window KV bookkeeping of the Inkling 2 family: capacity and pool arithmetic (mirrors the C++
``computeMinimumSwaPoolPages``), which visits are marked as window entries, and the runtime config they produce."""

from __future__ import annotations

import pytest

ragged = pytest.importorskip("experimental.builder.core.ragged")
configuration = pytest.importorskip("experimental.builder.models.inkling2.configuration")
contracts = pytest.importorskip("experimental.builder.core.contracts")


def cpp_min_swa_pool_pages(batch: int, capacity: int, page: int = 128, replacement: int = 1, headroom: int = 1) -> int:
    retained = (capacity + page - 1) // page + headroom
    return batch * (2 * retained + replacement)


@pytest.mark.parametrize("window,chunk,max_kv,expected", [
    (1024, 1024, 262144, 2048),  # window + chunk, page aligned
    (1024, 1000, 262144, 2048),  # rounds up to the page
    (16, 64, 2048, 128),
    (16, 64, 128, 0),  # bounding saves nothing when the capacity reaches the full one
    (16, 64, 100, 0),
    (1024, 1024, 2048, 0),  # equal capacity is not a reduction
    (1024, 1024, 2049, 2048),
    (1, 1, 4096, 128),  # minimal window still occupies one page
])
def test_bounded_capacity(window, chunk, max_kv, expected):
    assert ragged.bounded_swa_capacity(window, chunk, max_kv) == expected


@pytest.mark.parametrize("batch", [1, 2, 7])
@pytest.mark.parametrize("capacity", [128, 129, 256, 2048, 4096])
def test_swa_pool_pages_matches_cpp(batch, capacity):
    assert ragged.swa_pool_pages(batch, capacity) == cpp_min_swa_pool_pages(batch, capacity)


def test_swa_pool_pages_scales_linearly_with_batch():
    assert ragged.swa_pool_pages(4, 2048) == 4 * ragged.swa_pool_pages(1, 2048)


def real_config() -> dict:
    return {
        "num_hidden_layers": 24,
        "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3},
        "local_layer_ids": [i for i in range(24) if i not in (5, 11, 17, 23)],
        "sliding_window_size": 1024,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 2304,
        "intermediate_size": 1152,
    }


def prepared(config: dict) -> dict:
    return configuration.prepare_text_config(dict(config), config, contracts.Component.LLM, "")


def test_visit_marking_of_the_real_layout():
    text = prepared(real_config())
    entries = text["kv_bounded_entries"]
    assert text["kv_bounded_window"] == 1024
    # 48 visits: 8 full-attention visits (layer 5 once, 11 and 17 three times, 23 once), 40 window visits
    assert len(entries) == 40 and len(set(entries)) == 40 and max(entries) == 46  # the last visit (layer 23) is full attention
    full_visits = sorted(set(range(48)) - set(entries))
    assert len(full_visits) == 8
    order = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    assert sorted(order[v] for v in full_visits) == [5, 11, 11, 11, 17, 17, 17, 23]


def test_visit_marking_without_window_layers():
    config = real_config()
    config["local_layer_ids"] = []
    assert prepared(config)["kv_bounded_entries"] == []


def test_visit_marking_all_window_layers():
    config = real_config()
    config["local_layer_ids"] = list(range(24))
    assert prepared(config)["kv_bounded_entries"] == list(range(48))


def test_visit_marking_without_loop_passes():
    config = real_config()
    config["loop_config"]["num_passes"] = 1
    assert len(prepared(config)["kv_bounded_entries"]) == 20
