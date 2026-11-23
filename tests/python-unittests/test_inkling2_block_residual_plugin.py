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
"""Tests for Inkling2BlockResidualPlugin: the softmax-weighted mix of RMS-normalised block states.

The reference normalises every state row to unit RMS, scores it against each query row with a dot product scaled by
hidden ** -0.5, takes the softmax over the states and mixes the normalised states with those probabilities (float64).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, PluginUnsupportedError,
                              assert_close, pf_float32, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

PLUGIN_NAME = "Inkling2BlockResidualPlugin"
EPS = 1e-6


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    if DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


@dataclass(frozen=True)
class Cfg:
    hidden: int = 96
    num_states: int = 3
    num_queries: int = 2
    eps: float = EPS

    def fields(self):
        return [
            pf_int32("hidden", self.hidden),
            pf_int32("num_states", self.num_states),
            pf_int32("num_queries", self.num_queries),
            pf_float32("eps", self.eps),
        ]


def _specs(cfg: Cfg):
    specs = [(f"s{j}", trt.float32, (-1, cfg.hidden))
             for j in range(cfg.num_states)]
    specs.append(("queries", trt.float32, (cfg.num_queries, cfg.hidden)))
    return specs


def _build(cfg: Cfg, expect_unsupported=False, fields=None, specs=None,
           max_tokens=300) -> PluginRunner:
    specs = _specs(cfg) if specs is None else specs
    profiles = {}
    for name, _, shape in specs:
        if shape[0] == -1:
            profiles[name] = ((1, ) + shape[1:], (8, ) + shape[1:],
                              (max_tokens, ) + shape[1:])
        else:
            profiles[name] = (shape, shape, shape)
    runner = PluginRunner()
    runner.build(input_specs=specs,
                 output_names=[f"out{i}" for i in range(cfg.num_queries)],
                 plugin_name=PLUGIN_NAME,
                 plugin_version="1",
                 plugin_fields=cfg.fields() if fields is None else fields,
                 profiles=profiles,
                 expect_unsupported=expect_unsupported)
    return runner


def _reference(cfg: Cfg, states, queries):
    stacked = torch.stack([s.double() for s in states], dim=1)  # [T, k, H]
    normed = stacked / torch.sqrt((stacked**2).mean(-1, keepdim=True) + cfg.eps)
    outs = []
    for i in range(cfg.num_queries):
        scores = (normed @ queries[i].double()) * cfg.hidden**-0.5  # [T, k]
        probs = torch.softmax(scores, dim=1)
        outs.append((probs.unsqueeze(-1) * normed).sum(1))
    return outs


def _run(runner, cfg, states, queries):
    tokens = states[0].shape[0]
    tensors = {f"s{j}": s.cuda() for j, s in enumerate(states)}
    tensors["queries"] = queries.cuda()
    outs = [
        torch.full((tokens, cfg.hidden), 77.0, dtype=torch.float32).cuda()
        for _ in range(cfg.num_queries)
    ]
    for i, o in enumerate(outs):
        tensors[f"out{i}"] = o
    runner.execute(tensors)
    return outs


def _inputs(cfg, tokens, seed, magnitude=1.0):
    gen = torch.Generator().manual_seed(seed)
    states = [
        torch.randn(tokens, cfg.hidden, generator=gen) * magnitude
        for _ in range(cfg.num_states)
    ]
    queries = torch.randn(cfg.num_queries, cfg.hidden, generator=gen)
    return states, queries


def _check(name, cfg, states, queries, outs):
    for i, ref in enumerate(_reference(cfg, states, queries)):
        assert_close(f"{name}[q{i}]",
                     ref.float().cuda(),
                     outs[i],
                     atol=2e-5,
                     rtol=1e-4,
                     cos_threshold=0.999999)


@pytest.mark.parametrize("num_states", [1, 2, 5, 8])
@pytest.mark.parametrize("num_queries", [1, 2, 4])
@pytest.mark.parametrize("tokens", [1, 7, 129])
def test_matches_the_reference(num_states, num_queries, tokens):
    cfg = Cfg(num_states=num_states, num_queries=num_queries)
    runner = _build(cfg)
    states, queries = _inputs(cfg, tokens, 3 + tokens)
    _check("fused", cfg, states, queries, _run(runner, cfg, states, queries))


@pytest.mark.parametrize("hidden", [1, 33, 1023, 1025, 4096])
def test_hidden_sizes_off_the_thread_grid(hidden):
    cfg = Cfg(hidden=hidden, num_states=4, num_queries=3)
    runner = _build(cfg, max_tokens=16)
    states, queries = _inputs(cfg, 5, hidden)
    _check("hidden", cfg, states, queries, _run(runner, cfg, states, queries))


def test_one_engine_serves_changing_token_counts():
    cfg = Cfg(num_states=5, num_queries=4)
    runner = _build(cfg)
    for tokens in (1, 64, 3, 300, 1, 17):
        states, queries = _inputs(cfg, tokens, tokens)
        _check(f"T{tokens}", cfg, states, queries,
               _run(runner, cfg, states, queries))


def test_rows_are_independent_and_row_order_is_preserved():
    cfg = Cfg(num_states=4, num_queries=2)
    runner = _build(cfg)
    states, queries = _inputs(cfg, 20, 5)
    full = _run(runner, cfg, states, queries)
    perm = torch.randperm(20, generator=torch.Generator().manual_seed(1))
    shuffled = _run(runner, cfg, [s[perm] for s in states], queries)
    for a, b in zip(full, shuffled):
        assert torch.equal(a[perm].cpu(), b.cpu())


@pytest.mark.parametrize("magnitude", [3e4, 1e-4, 0.0])
def test_extreme_magnitudes_and_all_zero_rows_stay_finite(magnitude):
    cfg = Cfg(num_states=5, num_queries=4)
    runner = _build(cfg)
    states, queries = _inputs(cfg, 9, 2, magnitude)
    outs = _run(runner, cfg, states, queries)
    for o in outs:
        assert torch.isfinite(o).all()
    if magnitude > 0:
        _check("extreme", cfg, states, queries, outs)
    else:
        for o in outs:
            assert torch.count_nonzero(o) == 0


def test_a_zero_state_among_normal_states_is_ignored_by_the_mix_scores():
    cfg = Cfg(num_states=3, num_queries=1)
    runner = _build(cfg)
    states, queries = _inputs(cfg, 6, 8)
    states[1] = torch.zeros_like(states[1])
    _check("zero-state", cfg, states, queries,
           _run(runner, cfg, states, queries))


def test_engine_serialization_round_trip_keeps_attributes():
    cfg = Cfg(num_states=4, num_queries=3, eps=1e-3)
    runner = _build(cfg)
    serialized = runner.engine.serialize()
    assert serialized is not None
    runtime = trt.Runtime(runner.logger)
    runner.engine = runtime.deserialize_cuda_engine(serialized)
    runner.context = runner.engine.create_execution_context()
    runner._runtime = runtime
    states, queries = _inputs(cfg, 11, 4)
    _check("round-trip", cfg, states, queries,
           _run(runner, cfg, states, queries))


@pytest.mark.parametrize("change", [
    {"hidden": 0},
    {"hidden": 4097},
    {"num_states": 0},
    {"num_states": 9},
    {"num_queries": 0},
    {"num_queries": 5},
    {"eps": 0.0},
    {"eps": -1e-6},
    {"eps": float("nan")},
    {"eps": float("inf")},
],
                         ids=[
                             "zero_hidden", "hidden_too_large", "no_states",
                             "too_many_states", "no_queries",
                             "too_many_queries", "zero_eps", "negative_eps",
                             "nan_eps", "inf_eps"
                         ])
def test_invalid_attributes_are_rejected(change):
    with pytest.raises(PluginUnsupportedError):
        _build(Cfg(),
               expect_unsupported=True,
               fields=replace(Cfg(), **change).fields())


def test_missing_and_duplicate_attributes_are_rejected():
    fields = Cfg().fields()
    with pytest.raises(PluginUnsupportedError):
        _build(Cfg(), expect_unsupported=True, fields=fields[:-1])
    with pytest.raises(PluginUnsupportedError):
        _build(Cfg(), expect_unsupported=True, fields=fields + [fields[0]])


def test_shape_and_input_count_mismatches_are_rejected():
    cfg = Cfg()
    base = _specs(cfg)

    def with_shape(name, shape):
        return [(n, d, shape if n == name else s) for n, d, s in base]

    cases = [
        with_shape("queries", (cfg.num_queries + 1, cfg.hidden)),
        with_shape("queries", (cfg.num_queries, cfg.hidden + 1)),
        with_shape("s1", (-1, cfg.hidden + 1)),
        base[:-1],  # no queries input
        base + [("extra", trt.float32, (-1, cfg.hidden))],
    ]
    for specs in cases:
        with pytest.raises(PluginUnsupportedError):
            _build(cfg, expect_unsupported=True, specs=specs)


def test_wrong_data_types_are_rejected():
    cfg = Cfg()
    for name in ("s0", "queries"):
        for dtype in (trt.float16, trt.int32):
            specs = [(n, dtype if n == name else d, s)
                     for n, d, s in _specs(cfg)]
            with pytest.raises(PluginUnsupportedError):
                _build(cfg, expect_unsupported=True, specs=specs)
