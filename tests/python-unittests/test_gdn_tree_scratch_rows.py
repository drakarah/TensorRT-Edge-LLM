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
"""The gated-delta-net op's tree_scratch_rows attribute (build-time layout of the DDTree verify scratch)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt")

from experimental.builder.ops.functional import recurrent


def _capture(monkeypatch):
    captured = {}

    def fake_operation(name, inputs, output_count=1, **attributes):
        captured.update(name=name, inputs=inputs, count=output_count, attrs=attributes)
        return list(range(output_count))

    monkeypatch.setattr(recurrent, "operation", fake_operation)
    return captured


def _call(**kwargs):
    ragged = SimpleNamespace(query_lengths="ql",
                             query_start_offsets="qs",
                             state_indices="si",
                             execution_phase_marker="ep",
                             context_sequence_count_carrier="cc")
    args = dict(q="q", k="k", v="v", a="a", b="b", a_log="al", dt_bias="dt", state="st",
                ragged=ragged, key_head_dim=128, value_head_dim=128)
    args.update(kwargs)
    return recurrent.gated_delta_net(**args)


def test_scratch_rows_become_a_plugin_attribute_for_ddtree(monkeypatch):
    captured = _capture(monkeypatch)
    _call(tree_parent_ids="p", tree_depths="d", use_intermediate=True, tree_scratch_rows=2)
    assert captured["attrs"]["tree_scratch_rows"] == 2
    assert captured["attrs"]["use_ddtree"] == 1
    assert captured["count"] == 3


@pytest.mark.parametrize("rows", [0, None])
def test_no_scratch_rows_leaves_the_attribute_out(monkeypatch, rows):
    captured = _capture(monkeypatch)
    kwargs = {} if rows is None else {"tree_scratch_rows": rows}
    _call(tree_parent_ids="p", tree_depths="d", use_intermediate=True, **kwargs)
    assert "tree_scratch_rows" not in captured["attrs"]


def test_scratch_rows_need_a_ddtree(monkeypatch):
    _capture(monkeypatch)
    with pytest.raises(ValueError, match="DDTree"):
        _call(use_intermediate=True, tree_scratch_rows=2)
    with pytest.raises(ValueError, match="DDTree"):
        _call(tree_scratch_rows=2)


def test_negative_scratch_rows_are_rejected(monkeypatch):
    _capture(monkeypatch)
    with pytest.raises(ValueError, match="non-negative"):
        _call(tree_parent_ids="p", tree_depths="d", use_intermediate=True, tree_scratch_rows=-1)


def _parse(*argv):
    import argparse

    from experimental.builder import cli
    parser = argparse.ArgumentParser()
    cli._add_common_engine_args(parser)
    cli._add_build_args(parser)
    return cli, parser.parse_args(["--model-dir", "m", "--engine-dir", "e", *argv])


def test_cli_option_defaults_to_the_token_major_layout_and_reaches_the_build_args():
    cli, args = _parse()
    assert args.gdn_tree_scratch_rows == 0
    assert cli._build_args(args, "llm").gdn_tree_scratch_rows == 0
    cli, args = _parse("--gdn-tree-scratch-rows", "2")
    assert args.gdn_tree_scratch_rows == 2
    assert cli._build_args(args, "llm").gdn_tree_scratch_rows == 2
