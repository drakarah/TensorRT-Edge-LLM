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
"""Accuracy, rolling-state and rejection tests for ``Inkling2ShortConvPlugin``.

The reference convolves each sequence's full input history with zero history before its start:
y[t, c] = scale * (sum_j (taps[c, j] + dyn[t, c // (C / groups), j]) * x[t - j, c] + x[t, c]).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, PluginUnsupportedError,
                              assert_close, pf_float32, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

PLUGIN_NAME = "Inkling2ShortConvPlugin"
SENTINEL = 55.0


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
    channels: int = 64
    kernel_size: int = 4
    groups: int = 0
    scale: float = 0.4472135954999579  # 1 / sqrt(5)
    batch: int = 1
    state_rows: int = 4
    state_channels: int = 64
    state_offset: int = 0

    def fields(self):
        return [
            pf_int32("channels", self.channels),
            pf_int32("kernel_size", self.kernel_size),
            pf_int32("groups", self.groups),
            pf_float32("scale", self.scale),
            pf_int32("state_offset", self.state_offset),
        ]


def _specs(cfg: Cfg):
    specs = [
        ("x", trt.float16, (-1, cfg.channels)),
        ("taps", trt.float32, (cfg.channels, cfg.kernel_size)),
        ("state", trt.float16,
         (cfg.state_rows, cfg.state_channels, cfg.kernel_size)),
        ("query_start_offsets", trt.int32, (cfg.batch + 1, )),
        ("query_lengths", trt.int32, (cfg.batch, )),
        ("state_indices", trt.int32, (cfg.batch, )),
    ]
    if cfg.groups > 0:
        specs.append(("dyn", trt.float16, (-1, cfg.groups * cfg.kernel_size)))
    return specs


def _build(cfg: Cfg,
           expect_unsupported=False,
           fields=None,
           specs=None,
           max_tokens=256) -> PluginRunner:
    specs = _specs(cfg) if specs is None else specs
    profiles = {}
    for name, _, shape in specs:
        if shape[0] == -1:
            profiles[name] = ((1, ) + shape[1:], (1, ) + shape[1:],
                              (max_tokens, ) + shape[1:])
        else:
            profiles[name] = (shape, shape, shape)
    runner = PluginRunner()
    runner.build(input_specs=specs,
                 output_names=["y", "present_state"],
                 plugin_name=PLUGIN_NAME,
                 plugin_version="1",
                 plugin_fields=cfg.fields() if fields is None else fields,
                 profiles=profiles,
                 expect_unsupported=expect_unsupported)
    return runner


class Stream:
    """Inputs of one sequence over its whole life, on the FP16 grid."""

    def __init__(self, cfg: Cfg, total: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.x = (torch.randn(total, cfg.channels, generator=g) *
                  2.0).half().float()
        self.dyn = (torch.randn(total, max(cfg.groups, 1) * cfg.kernel_size,
                                generator=g) * 0.3).half().float()


def _taps(cfg: Cfg, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(cfg.channels, cfg.kernel_size, generator=g) * 0.5


def _reference(cfg: Cfg, stream: Stream, taps, first: int, count: int):
    x = stream.x
    out = torch.zeros(count, cfg.channels)
    per_group = cfg.channels // cfg.groups if cfg.groups else cfg.channels
    for i in range(count):
        t = first + i
        acc = torch.zeros(cfg.channels)
        for j in range(cfg.kernel_size):
            if t - j < 0:
                continue
            tap = taps[:, j].clone()
            if cfg.groups:
                dyn = stream.dyn[t].view(cfg.groups, cfg.kernel_size)[:, j]
                tap += dyn.repeat_interleave(per_group)
            acc += tap * x[t - j]
        out[i] = cfg.scale * (acc + x[t])
    return out


def _new_state(cfg: Cfg):
    return torch.full((cfg.state_rows, cfg.state_channels, cfg.kernel_size),
                      0.0,
                      dtype=torch.float16).cuda()


def _run(runner,
         cfg: Cfg,
         streams: Sequence[Stream],
         state,
         taps,
         first: Sequence[int],
         counts: Sequence[int],
         state_indices: Sequence[int],
         width: Optional[int] = None):
    batch = len(streams)
    slots = batch * width if width else sum(counts)
    offsets, cursor = [0], 0
    for b in range(batch):
        cursor += width if width else counts[b]
        offsets.append(cursor)
    x = torch.full((slots, cfg.channels), 4.0)
    dyn = torch.full((slots, max(cfg.groups, 1) * cfg.kernel_size), 4.0)
    for b in range(batch):
        n = counts[b]
        x[offsets[b]:offsets[b] + n] = streams[b].x[first[b]:first[b] + n]
        dyn[offsets[b]:offsets[b] + n] = streams[b].dyn[first[b]:first[b] + n]
    tensors = {
        "x": x.half().cuda(),
        "taps": taps.float().cuda(),
        "state": state,
        "present_state": state,
        "query_start_offsets": torch.tensor(offsets, dtype=torch.int32).cuda(),
        "query_lengths": torch.tensor(counts, dtype=torch.int32).cuda(),
        "state_indices": torch.tensor(state_indices,
                                      dtype=torch.int32).cuda(),
    }
    if cfg.groups:
        tensors["dyn"] = dyn.half().cuda()
    y = torch.full((slots, cfg.channels), 9.0, dtype=torch.float16).cuda()
    tensors["y"] = y
    runner.execute(tensors)
    return y, offsets


def _check(name, cfg, streams, taps, first, counts, y, offsets):
    for b, n in enumerate(counts):
        if n == 0:
            continue
        ref = _reference(cfg, streams[b], taps, first[b], n)
        assert_close(f"{name}[seq{b}]",
                     ref.half().cuda(),
                     y[offsets[b]:offsets[b] + n],
                     atol=3e-3,
                     rtol=1e-2,
                     cos_threshold=0.99999)


def _check_state(cfg, streams, state, lengths, rows, untouched_slot=True):
    """State slots 0..k-2 hold the last k-1 inputs oldest first; the unused slot and extra channels keep their value."""
    k = cfg.kernel_size
    for b, row in enumerate(rows):
        if row < 0:
            continue
        for m in range(k - 1):
            idx = lengths[b] - (k - 1) + m
            want = streams[b].x[idx] if idx >= 0 else torch.zeros(
                cfg.channels)
            got = state[row, cfg.state_offset:cfg.state_offset +
                        cfg.channels, m].float().cpu()
            assert torch.equal(got, want), f"row {row} slot {m}"


@pytest.mark.parametrize("kernel,groups", [(4, 0), (4, 4), (2, 0), (3, 8),
                                           (8, 0)],
                         ids=["k4", "k4_dyn4", "k2", "k3_dyn8", "k8"])
def test_prefill_then_decode_steps_match_the_reference(kernel, groups):
    scale = 1.0 / (2 * kernel + 1)**0.5 if groups else 1.0 / (kernel + 1)**0.5
    cfg = Cfg(kernel_size=kernel, groups=groups, scale=scale)
    runner = _build(cfg)
    stream = Stream(cfg, 60, 1)
    taps = _taps(cfg)
    state = _new_state(cfg)
    first = 0
    for step, count in enumerate([13, 1, 1, 1, 2, 5, 1, 20, 1]):
        y, offsets = _run(runner, cfg, [stream], state, taps, [first], [count],
                          [2])
        _check(f"step{step}", cfg, [stream], taps, [first], [count], y,
               offsets)
        first += count
        _check_state(cfg, [stream], state, [first], [2])


def test_chunks_shorter_than_the_kernel_mix_old_state_and_new_inputs():
    cfg = Cfg(kernel_size=8, scale=1.0 / 3.0)
    runner = _build(cfg)
    stream = Stream(cfg, 40, 2)
    taps = _taps(cfg)
    state = _new_state(cfg)
    first = 0
    for count in [1, 2, 3, 1, 4, 2, 1, 7, 1, 1, 9]:
        y, offsets = _run(runner, cfg, [stream], state, taps, [first], [count],
                          [0])
        _check(f"chunk@{first}", cfg, [stream], taps, [first], [count], y,
               offsets)
        first += count
        _check_state(cfg, [stream], state, [first], [0])


def test_batch_with_permuted_state_rows_padding_and_empty_sequence():
    cfg = Cfg(batch=3, state_rows=6, groups=4, scale=1.0 / 3.0)
    runner = _build(cfg)
    streams = [Stream(cfg, 40, 4), Stream(cfg, 40, 5), Stream(cfg, 40, 6)]
    taps = _taps(cfg)
    state = _new_state(cfg)
    state[3] = SENTINEL  # a row no sequence uses
    state[5] = SENTINEL
    rows = [4, 0, 2]
    first, counts = [0, 0, 0], [8, 3, 0]
    y, offsets = _run(runner,
                      cfg,
                      streams,
                      state,
                      taps,
                      first,
                      counts,
                      rows,
                      width=8)
    _check("batch1", cfg, streams, taps, first, counts, y, offsets)
    pad = y[offsets[1] + counts[1]:offsets[2]].float()
    assert bool((pad == 0).all()), "padding slots must be zero"
    assert bool((y[offsets[2]:offsets[3]].float() == 0).all()
                ), "an empty sequence produces only padding"
    _check_state(cfg, streams, state, counts, rows)
    assert bool((state[3] == SENTINEL).all()) and bool(
        (state[5] == SENTINEL).all())
    first = [8, 3, 0]
    counts = [1, 5, 6]
    y, offsets = _run(runner,
                      cfg,
                      streams,
                      state,
                      taps,
                      first,
                      counts,
                      rows,
                      width=6)
    _check("batch2", cfg, streams, taps, first, counts, y, offsets)
    _check_state(cfg, streams, state, [9, 8, 6], rows)


def test_state_channels_beyond_the_conv_channels_and_unused_slot_are_untouched():
    cfg = Cfg(channels=32, state_channels=96)
    runner = _build(cfg)
    stream = Stream(cfg, 30, 7)
    taps = _taps(cfg)
    state = _new_state(cfg)
    state[:, cfg.channels:, :] = SENTINEL
    state[:, :, cfg.kernel_size - 1] = SENTINEL
    for first, count in [(0, 11), (11, 1), (12, 9)]:
        y, offsets = _run(runner, cfg, [stream], state, taps, [first],
                          [count], [1])
        _check(f"step@{first}", cfg, [stream], taps, [first], [count], y,
               offsets)
    assert bool((state[:, cfg.channels:, :] == SENTINEL).all())
    assert bool((state[:, :, cfg.kernel_size - 1] == SENTINEL).all())
    _check_state(cfg, [stream], state, [21], [1])


def test_negative_state_index_means_empty_history_and_no_state_write():
    cfg = Cfg(batch=2, scale=0.5)
    runner = _build(cfg)
    streams = [Stream(cfg, 20, 8), Stream(cfg, 20, 9)]
    taps = _taps(cfg)
    state = _new_state(cfg)
    state[:] = SENTINEL  # a stale state must not leak into a sequence without a state row
    y, offsets = _run(runner, cfg, streams, state, taps, [0, 0], [6, 6],
                      [-1, 1])
    # sequence 0 has no state row: its first outputs only see zero history
    ref = _reference(cfg, streams[0], taps, 0, 6)
    assert_close("no-row",
                 ref.half().cuda(),
                 y[offsets[0]:offsets[0] + 6],
                 atol=3e-3,
                 rtol=1e-2,
                 cos_threshold=0.99999)
    assert bool((state[0] == SENTINEL).all()) and bool(
        (state[2:] == SENTINEL).all())


def test_extreme_and_signed_zero_inputs_stay_finite_and_exact():
    cfg = Cfg(kernel_size=4, scale=0.5)
    runner = _build(cfg)
    stream = Stream(cfg, 10, 10)
    stream.x[2:5] = 0.0
    stream.x[5] = -0.0
    stream.x[6] = 6.0e4
    stream.x[7] = -6.0e4
    taps = torch.zeros(cfg.channels, cfg.kernel_size)
    taps[:, 0] = 1.0
    state = _new_state(cfg)
    y, offsets = _run(runner, cfg, [stream], state, taps, [0], [10], [0])
    ref = _reference(cfg, stream, taps, 0, 10)
    finite = torch.isfinite(ref.half().float())
    assert bool(torch.isfinite(y.float()).cpu()[finite].all())
    assert_close("extreme",
                 torch.where(finite, ref, torch.zeros(())).half().cuda(),
                 torch.where(finite.cuda(), y, torch.zeros((),
                                                           dtype=y.dtype).cuda()),
                 atol=1.0,
                 rtol=2e-3,
                 cos_threshold=0.9999)


def test_engine_serialization_round_trip_keeps_attributes():
    cfg = Cfg(groups=4, scale=0.3)
    runner = _build(cfg)
    serialized = runner.engine.serialize()
    assert serialized is not None
    runtime = trt.Runtime(runner.logger)
    runner.engine = runtime.deserialize_cuda_engine(serialized)
    runner.context = runner.engine.create_execution_context()
    runner._runtime = runtime
    stream = Stream(cfg, 20, 11)
    taps = _taps(cfg)
    state = _new_state(cfg)
    y, offsets = _run(runner, cfg, [stream], state, taps, [0], [12], [3])
    _check("round-trip", cfg, [stream], taps, [0], [12], y, offsets)


def test_convolutions_sharing_one_state_row_use_disjoint_channel_ranges():
    base = Cfg(state_channels=96, state_rows=3, scale=0.4)
    parts = [(24, 0, 0), (24, 24, 4), (48, 48, 0)]
    cfgs = [
        replace(base, channels=ch, state_offset=off, groups=g)
        for ch, off, g in parts
    ]
    runners = [_build(c) for c in cfgs]
    streams = [Stream(c, 50, 30 + i) for i, c in enumerate(cfgs)]
    taps = [_taps(c, 40 + i) for i, c in enumerate(cfgs)]
    state = _new_state(base)
    first = 0
    for count in [5, 1, 1, 9, 2, 12]:
        for i, c in enumerate(cfgs):
            y, offsets = _run(runners[i], c, [streams[i]], state, taps[i],
                              [first], [count], [1])
            _check(f"part{i}@{first}", c, [streams[i]], taps[i], [first],
                   [count], y, offsets)
        first += count
    for i, c in enumerate(cfgs):
        _check_state(c, [streams[i]], state, [first], [1])


@pytest.mark.parametrize("change", [
    {
        "state_offset": -1
    },
    {
        "state_offset": 1
    },
    {
        "channels": 0
    },
    {
        "kernel_size": 1
    },
    {
        "kernel_size": 9
    },
    {
        "groups": -1
    },
    {
        "groups": 5
    },
    {
        "scale": 0.0
    },
    {
        "scale": float("nan")
    },
    {
        "scale": float("inf")
    },
],
                         ids=[
                             "negative_offset", "offset_beyond_state",
                             "zero_channels", "kernel1", "kernel9",
                             "negative_groups", "groups_not_dividing",
                             "zero_scale", "nan_scale", "inf_scale"
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
    cfg = Cfg(groups=4)
    base = _specs(cfg)

    def with_shape(name, shape):
        return [(n, d, shape if n == name else s) for n, d, s in base]

    cases = [
        with_shape("taps", (cfg.channels, cfg.kernel_size + 1)),
        with_shape("state", (cfg.state_rows, cfg.state_channels,
                             cfg.kernel_size + 1)),
        with_shape("state", (cfg.state_rows, cfg.channels - 1,
                             cfg.kernel_size)),
        with_shape("dyn", (-1, cfg.groups * cfg.kernel_size + 1)),
        base[:-1],  # groups > 0 but no dynamic taps input
    ]
    for specs in cases:
        with pytest.raises(PluginUnsupportedError):
            _build(cfg, expect_unsupported=True, specs=specs)
    with pytest.raises(PluginUnsupportedError):
        # dynamic taps supplied to a static plugin
        _build(Cfg(),
               expect_unsupported=True,
               specs=_specs(Cfg(groups=4)))


def test_wrong_data_types_are_rejected():
    cfg = Cfg()
    for name, dtype in (("x", trt.float32), ("taps", trt.float16),
                        ("state", trt.float32), ("query_lengths", trt.int64)):
        specs = [(n, dtype if n == name else d, s)
                 for n, d, s in _specs(cfg)]
        with pytest.raises(PluginUnsupportedError):
            _build(cfg, expect_unsupported=True, specs=specs)
