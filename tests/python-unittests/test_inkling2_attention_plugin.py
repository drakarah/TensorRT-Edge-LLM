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
"""Accuracy, paging, padding and rejection tests for ``Inkling2AttentionPlugin``.

The reference is an FP32 torch implementation of the Inkling 2 attention visit over the cached keys and values of a
sequence: L2-clipped scaled query, relative-position bias from a clipped relative state, a per-token sink logit,
an optional sliding window, log-length query scaling on full layers and a sigmoid output gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, PluginUnsupportedError,
                              assert_close, pf_float32, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

HEAD_DIM = 128
REL_DIM = 16
PAGE = 128
PLUGIN_NAME = "Inkling2AttentionPlugin"
SENTINEL = 77.0


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
    num_q_heads: int = 6
    num_kv_heads: int = 2
    rel_extent: int = 64
    window: int = 0
    log_alpha: float = 0.1
    log_n_floor: float = 16.0
    q_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    scaling: float = HEAD_DIM**-0.5
    batch: int = 1
    max_pages: int = 4
    num_pages: int = 12  # pool pages per plane

    def fields(self):
        return [
            pf_int32("num_q_heads", self.num_q_heads),
            pf_int32("num_kv_heads", self.num_kv_heads),
            pf_int32("rel_extent", self.rel_extent),
            pf_int32("window", self.window),
            pf_float32("log_alpha", self.log_alpha),
            pf_float32("log_n_floor", self.log_n_floor),
            pf_float32("q_clip", self.q_clip),
            pf_float32("rel_clip", self.rel_clip),
            pf_float32("scaling", self.scaling),
        ]


def _specs(cfg: Cfg, max_tokens: int):
    nh, nkv = cfg.num_q_heads, cfg.num_kv_heads
    return [
        ("q_lin", trt.float16, (-1, nh * HEAD_DIM)),
        ("rel_lin", trt.float16, (-1, nh * REL_DIM)),
        ("sink_lin", trt.float16, (-1, nh)),
        ("gate_lin", trt.float16, (-1, nh * HEAD_DIM)),
        ("k_new", trt.float16, (-1, nkv * HEAD_DIM)),
        ("v_new", trt.float16, (-1, nkv * HEAD_DIM)),
        ("q_scale", trt.float32, (nh * HEAD_DIM, )),
        ("rel_scale", trt.float32, (nh * REL_DIM, )),
        ("rel_proj_t", trt.float32, (cfg.rel_extent, REL_DIM)),
        ("past_kv", trt.float16, (2, cfg.num_pages, PAGE, nkv, HEAD_DIM)),
        ("query_start_offsets", trt.int32, (cfg.batch + 1, )),
        ("query_lengths", trt.int32, (cfg.batch, )),
        ("past_lengths", trt.int32, (cfg.batch, )),
        ("kv_page_table", trt.int32, (cfg.batch, 2, cfg.max_pages)),
    ]


def _build(cfg: Cfg,
           max_tokens: int = 512,
           expect_unsupported=False,
           fields=None,
           specs=None) -> PluginRunner:
    specs = _specs(cfg, max_tokens) if specs is None else specs
    profiles = {}
    for name, _, shape in specs:
        if shape[0] == -1:
            profiles[name] = ((1, ) + shape[1:], (1, ) + shape[1:],
                              (max_tokens, ) + shape[1:])
        else:
            profiles[name] = (shape, shape, shape)
    runner = PluginRunner()
    runner.build(input_specs=specs,
                 output_names=["attention", "present_kv"],
                 plugin_name=PLUGIN_NAME,
                 plugin_version="1",
                 plugin_fields=cfg.fields() if fields is None else fields,
                 profiles=profiles,
                 expect_unsupported=expect_unsupported)
    return runner


def _l2clip(x, threshold):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def _reference(cfg: Cfg, q_lin, rel_lin, sink, gate, q_scale, rel_scale,
               proj_t, keys, values, first_pos: int):
    """Attention of ``n`` new tokens at positions first_pos.. over keys/values [first_pos + n, nkv, 128] (FP32)."""
    n = q_lin.shape[0]
    nh, nkv = cfg.num_q_heads, cfg.num_kv_heads
    total = keys.shape[0]
    pos = torch.arange(first_pos, first_pos + n, dtype=torch.float32)
    log_scale = (torch.ones(n) if cfg.window > 0 else 1 +
                 cfg.log_alpha * torch.log(
                     ((pos + 1) / cfg.log_n_floor).clamp(min=1.0)))
    q = _l2clip(
        q_lin.view(n, nh, HEAD_DIM) * q_scale.view(nh, HEAD_DIM),
        cfg.q_clip) * log_scale[:, None, None]
    rs = _l2clip(
        rel_lin.view(n, nh, REL_DIM) * rel_scale.view(nh, REL_DIM),
        cfg.rel_clip) * log_scale[:, None, None]
    rel = rs @ proj_t.t()  # [n, nh, extent]
    rep = nh // nkv
    kh = keys.repeat_interleave(rep, 1)
    vh = values.repeat_interleave(rep, 1)
    scores = torch.einsum("qhd,shd->hqs", q, kh) * cfg.scaling
    qi = pos.long()[:, None]
    si = torch.arange(total)[None, :]
    dist = qi - si
    in_extent = (dist >= 0) & (dist < cfg.rel_extent)
    gathered = rel.permute(1, 0, 2).gather(
        -1,
        dist.clamp(0, cfg.rel_extent - 1)[None].expand(nh, -1, -1))
    scores = scores + torch.where(in_extent[None], gathered, torch.zeros(()))
    visible = dist >= 0
    if cfg.window > 0:
        visible = visible & (dist < cfg.window)
    scores = scores.masked_fill(~visible[None], float("-inf"))
    full = torch.cat([scores, sink.t().unsqueeze(-1)], -1)
    probs = torch.softmax(full, -1)[..., :-1]
    out = torch.einsum("hqs,shd->qhd", probs, vh)
    return (out * torch.sigmoid(gate.view(n, nh, HEAD_DIM))).reshape(
        n, nh * HEAD_DIM)


class Sequence_:
    """Random per-sequence data on the FP16 grid plus its page assignment."""

    def __init__(self, cfg: Cfg, total: int, seed: int, scale: float = 0.8):
        g = torch.Generator().manual_seed(seed)
        nh, nkv = cfg.num_q_heads, cfg.num_kv_heads

        def h(*shape, s=scale):
            return (torch.randn(*shape, generator=g) * s).half().float()

        self.total = total
        self.q = h(total, nh * HEAD_DIM)
        self.rel = h(total, nh * REL_DIM)
        self.sink = h(total, nh)
        self.gate = h(total, nh * HEAD_DIM)
        self.k = h(total, nkv, HEAD_DIM, s=1.5)
        self.v = h(total, nkv, HEAD_DIM, s=1.0)


def _params(cfg: Cfg, seed: int = 5):
    g = torch.Generator().manual_seed(seed)
    nh = cfg.num_q_heads
    q_scale = 1 + 0.1 * torch.randn(nh * HEAD_DIM, generator=g)
    rel_scale = 1 + 0.1 * torch.randn(nh * REL_DIM, generator=g)
    proj_t = torch.randn(cfg.rel_extent, REL_DIM, generator=g) * 0.5
    return q_scale, rel_scale, proj_t


def _page_table(cfg: Cfg, page_counts: Sequence[int], seed: int):
    """Scrambled K pages per sequence; V entries are K + num_pages (flattened plane axis)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(cfg.num_pages, generator=g).tolist()
    table = torch.zeros((cfg.batch, 2, cfg.max_pages), dtype=torch.int32)
    used: List[int] = []
    cursor = 0
    for b, count in enumerate(page_counts):
        for p in range(count):
            page = perm[cursor]
            cursor += 1
            table[b, 0, p] = page
            table[b, 1, p] = page + cfg.num_pages
            used.append(page)
    return table, used


def _run_step(runner: PluginRunner,
              cfg: Cfg,
              seqs: Sequence[Sequence_],
              pool,
              table,
              first: Sequence[int],
              counts: Sequence[int],
              width: Optional[int] = None):
    """One engine call: sequence b contributes counts[b] new tokens starting at position first[b].

    ``width`` selects entry-padded slots (every sequence owns ``width`` slots), otherwise the slots are packed.
    Returns the plugin output [slots, nh*128] and the reference rows for the real tokens.
    """
    q_scale, rel_scale, proj_t = _params(cfg)
    batch = len(seqs)
    slots = batch * width if width else sum(counts)
    offsets, cursor = [0], 0
    for b in range(batch):
        cursor += width if width else counts[b]
        offsets.append(cursor)
    nh, nkv = cfg.num_q_heads, cfg.num_kv_heads

    def pack(name, cols, dtype=torch.float16):
        out = torch.full((slots, cols), 3.0)  # padding slots carry junk
        for b in range(batch):
            n = counts[b]
            src = getattr(seqs[b], name)
            out[offsets[b]:offsets[b] + n] = src[first[b]:first[b] +
                                                 n].reshape(n, cols)
        return out.to(dtype).cuda().contiguous()

    tensors = {
        "q_lin": pack("q", nh * HEAD_DIM),
        "rel_lin": pack("rel", nh * REL_DIM),
        "sink_lin": pack("sink", nh),
        "gate_lin": pack("gate", nh * HEAD_DIM),
        "k_new": pack("k", nkv * HEAD_DIM),
        "v_new": pack("v", nkv * HEAD_DIM),
        "q_scale": q_scale.cuda(),
        "rel_scale": rel_scale.cuda(),
        "rel_proj_t": proj_t.cuda(),
        "past_kv": pool,
        "present_kv": pool,
        "query_start_offsets": torch.tensor(offsets, dtype=torch.int32).cuda(),
        "query_lengths": torch.tensor(counts, dtype=torch.int32).cuda(),
        "past_lengths": torch.tensor(first, dtype=torch.int32).cuda(),
        "kv_page_table": table.cuda(),
    }
    out = torch.full((slots, nh * HEAD_DIM), 9.0, dtype=torch.float16).cuda()
    tensors["attention"] = out
    runner.execute(tensors)
    refs = []
    for b in range(batch):
        n = counts[b]
        if n == 0:
            refs.append(torch.zeros((0, nh * HEAD_DIM)))
            continue
        end = first[b] + n
        refs.append(
            _reference(cfg, seqs[b].q[first[b]:end], seqs[b].rel[first[b]:end],
                       seqs[b].sink[first[b]:end], seqs[b].gate[first[b]:end],
                       q_scale, rel_scale, proj_t, seqs[b].k[:end],
                       seqs[b].v[:end], first[b]))
    return out, offsets, refs


def _new_pool(cfg: Cfg):
    return torch.full((2, cfg.num_pages, PAGE, cfg.num_kv_heads, HEAD_DIM),
                      SENTINEL,
                      dtype=torch.float16).cuda()


def _check_rows(name, out, offsets, refs, counts, atol=4e-3, rtol=2e-2):
    for b, n in enumerate(counts):
        if n == 0:
            continue
        assert_close(f"{name}[seq{b}]",
                     refs[b].half().cuda(),
                     out[offsets[b]:offsets[b] + n],
                     atol=atol,
                     rtol=rtol,
                     cos_threshold=0.9999)


def _pool_row(pool, table, b, pos, nkv, plane):
    page = int(table[b, plane, pos // PAGE])
    return pool.view(2 * pool.shape[1], PAGE, nkv, HEAD_DIM)[page, pos % PAGE]


def _check_pool(cfg: Cfg, pool, table, seqs, upto, used_pages):
    """Cached rows equal the inputs; every pool page outside the page table is untouched."""
    for b, seq in enumerate(seqs):
        for pos in range(upto[b]):
            assert torch.equal(
                _pool_row(pool, table, b, pos, cfg.num_kv_heads,
                          0).float().cpu(), seq.k[pos])
            assert torch.equal(
                _pool_row(pool, table, b, pos, cfg.num_kv_heads,
                          1).float().cpu(), seq.v[pos])
    untouched = [p for p in range(cfg.num_pages) if p not in set(used_pages)]
    for plane in (0, 1):
        for page in untouched:
            assert bool((pool[plane, page] == SENTINEL
                         ).all()), f"page {page} plane {plane} was written"


@pytest.mark.parametrize("window,extent", [(0, 64), (0, 300), (20, 20),
                                           (200, 64)],
                         ids=["full", "full_wide_extent", "swa20", "swa200"])
def test_prefill_matches_reference_and_fills_the_cache(window, extent):
    cfg = Cfg(window=window, rel_extent=extent)
    runner = _build(cfg)
    seq = Sequence_(cfg, 300, 1)
    table, used = _page_table(cfg, [3], 7)
    pool = _new_pool(cfg)
    out, offsets, refs = _run_step(runner, cfg, [seq], pool, table, [0], [300])
    _check_rows("prefill", out, offsets, refs, [300])
    _check_pool(cfg, pool, table, [seq], [300], used)


def test_decode_steps_after_prefill_across_page_boundaries():
    cfg = Cfg()
    runner = _build(cfg)
    seq = Sequence_(cfg, 262, 2)
    table, used = _page_table(cfg, [3], 8)
    pool = _new_pool(cfg)
    out, offsets, refs = _run_step(runner, cfg, [seq], pool, table, [0], [126])
    _check_rows("prefill126", out, offsets, refs, [126])
    # positions 126, 127 end the first page, 128 opens the second, then a multi-row verify-style step
    for pos in (126, 127, 128):
        out, offsets, refs = _run_step(runner, cfg, [seq], pool, table, [pos],
                                       [1])
        _check_rows(f"decode@{pos}", out, offsets, refs, [1])
    out, offsets, refs = _run_step(runner, cfg, [seq], pool, table, [129],
                                   [133])
    _check_rows("rows129+133", out, offsets, refs, [133])
    _check_pool(cfg, pool, table, [seq], [262], used)


def test_log_length_scaling_applies_on_full_layers_only():
    # log_n_floor 4 makes the factor differ from 1 for every position after the first few tokens
    cfg = Cfg(log_n_floor=4.0, log_alpha=0.5)
    seq = Sequence_(cfg, 200, 3)
    table, _ = _page_table(cfg, [2], 9)
    for window in (0, 64):
        c = replace(cfg, window=window)
        runner = _build(c)
        out, offsets, refs = _run_step(runner, c, [seq], _new_pool(c), table,
                                       [0], [200])
        _check_rows(f"window{window}", out, offsets, refs, [200])


def test_clipping_paths_are_exercised():
    # large raw q and relative state force both L2 clips to trigger
    cfg = Cfg(q_clip=3.0, rel_clip=0.5)
    runner = _build(cfg)
    seq = Sequence_(cfg, 150, 4, scale=3.0)
    table, _ = _page_table(cfg, [2], 10)
    out, offsets, refs = _run_step(runner, cfg, [seq], _new_pool(cfg), table,
                                   [0], [150])
    _check_rows("clips", out, offsets, refs, [150])


def test_batch_with_entry_padding_skips_padding_slots():
    cfg = Cfg(batch=2)
    runner = _build(cfg)
    seqs = [Sequence_(cfg, 40, 11), Sequence_(cfg, 25, 12)]
    table, used = _page_table(cfg, [1, 1], 13)
    pool = _new_pool(cfg)
    counts = [8, 3]  # second sequence is padded to the common width of 8
    out, offsets, refs = _run_step(runner,
                                   cfg,
                                   seqs,
                                   pool,
                                   table, [0, 0],
                                   counts,
                                   width=8)
    _check_rows("padded", out, offsets, refs, counts)
    pad = out[offsets[1] + counts[1]:offsets[2]].float()
    assert bool((pad == 0).all()), "padding slots must produce zero rows"
    _check_pool(cfg, pool, table, seqs, counts, used)
    # the padded sequence continues; only its real rows are cached
    out, offsets, refs = _run_step(runner,
                                   cfg,
                                   seqs,
                                   pool,
                                   table, [8, 3], [4, 20],
                                   width=20)
    _check_rows("padded-step2", out, offsets, refs, [4, 20])
    _check_pool(cfg, pool, table, seqs, [12, 23], used)


def test_sequence_with_no_new_tokens_in_a_batch():
    cfg = Cfg(batch=2)
    runner = _build(cfg)
    seqs = [Sequence_(cfg, 30, 14), Sequence_(cfg, 30, 15)]
    table, used = _page_table(cfg, [1, 1], 16)
    pool = _new_pool(cfg)
    out, offsets, refs = _run_step(runner, cfg, seqs, pool, table, [0, 0],
                                   [5, 0])
    _check_rows("one-empty", out, offsets, refs, [5, 0])
    _check_pool(cfg, pool, table, seqs, [5, 0], used)


def test_single_token_and_maximum_context_boundary():
    cfg = Cfg(max_pages=2, num_pages=4)
    runner = _build(cfg)
    seq = Sequence_(cfg, 256, 17)
    table, used = _page_table(cfg, [2], 18)
    pool = _new_pool(cfg)
    out, offsets, refs = _run_step(runner, cfg, [seq], pool, table, [0], [255])
    _check_rows("fill255", out, offsets, refs, [255])
    out, offsets, refs = _run_step(runner, cfg, [seq], pool, table, [255], [1])
    _check_rows("last slot", out, offsets, refs, [1])
    _check_pool(cfg, pool, table, [seq], [256], used)


def test_engine_serialization_round_trip_keeps_attributes():
    cfg = Cfg(window=20, rel_extent=20)
    runner = _build(cfg)
    serialized = runner.engine.serialize()
    assert serialized is not None
    runtime = trt.Runtime(runner.logger)
    runner.engine = runtime.deserialize_cuda_engine(serialized)
    runner.context = runner.engine.create_execution_context()
    runner._runtime = runtime
    seq = Sequence_(cfg, 100, 19)
    table, _ = _page_table(cfg, [1], 20)
    out, offsets, refs = _run_step(runner, cfg, [seq], _new_pool(cfg), table,
                                   [0], [100])
    _check_rows("round-trip", out, offsets, refs, [100])


@pytest.mark.parametrize("change", [
    {
        "num_q_heads": 5
    },
    {
        "num_kv_heads": 0
    },
    {
        "rel_extent": 0
    },
    {
        "window": -1
    },
    {
        "scaling": 0.0
    },
    {
        "scaling": float("nan")
    },
    {
        "q_clip": -1.0
    },
    {
        "rel_clip": float("inf")
    },
    {
        "log_n_floor": 0.0
    },
],
                         ids=[
                             "heads_not_multiple", "zero_kv_heads",
                             "zero_extent", "negative_window", "zero_scaling",
                             "nan_scaling", "negative_q_clip", "inf_rel_clip",
                             "zero_log_floor"
                         ])
def test_invalid_attributes_are_rejected(change):
    cfg = replace(Cfg(), **change)
    with pytest.raises(PluginUnsupportedError):
        _build(Cfg(), expect_unsupported=True, fields=cfg.fields())


def test_missing_and_duplicate_attributes_are_rejected():
    fields = Cfg().fields()
    with pytest.raises(PluginUnsupportedError):
        _build(Cfg(), expect_unsupported=True, fields=fields[:-1])
    with pytest.raises(PluginUnsupportedError):
        _build(Cfg(), expect_unsupported=True, fields=fields + [fields[0]])


def test_shape_mismatches_are_rejected():
    cfg = Cfg()
    base = _specs(cfg, 512)

    def with_shape(name, shape):
        return [(n, d, shape if n == name else s) for n, d, s in base]

    wrong_head_dim = with_shape("past_kv",
                                (2, cfg.num_pages, PAGE, cfg.num_kv_heads, 64))
    wrong_page = with_shape("past_kv",
                            (2, cfg.num_pages, 64, cfg.num_kv_heads, HEAD_DIM))
    wrong_kv_heads = with_shape(
        "past_kv", (2, cfg.num_pages, PAGE, cfg.num_kv_heads + 1, HEAD_DIM))
    wrong_rel = with_shape("rel_proj_t", (cfg.rel_extent + 1, REL_DIM))
    wrong_table = with_shape("kv_page_table", (cfg.batch, 3, cfg.max_pages))
    for specs in (wrong_head_dim, wrong_page, wrong_kv_heads, wrong_rel,
                  wrong_table):
        with pytest.raises(PluginUnsupportedError):
            _build(cfg, expect_unsupported=True, specs=specs)


def test_wrong_input_data_types_are_rejected():
    cfg = Cfg()
    specs = [(n, trt.float32 if n == "q_lin" else d, s)
             for n, d, s in _specs(cfg, 512)]
    with pytest.raises(PluginUnsupportedError):
        _build(cfg, expect_unsupported=True, specs=specs)
