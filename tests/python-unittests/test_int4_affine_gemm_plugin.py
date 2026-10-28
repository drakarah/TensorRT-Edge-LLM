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
"""Accuracy tests for ``Int4AffineGroupwiseGemmPlugin`` (MLX affine INT4, group 64).

Each case builds one dynamic-profile engine around the plugin (weights as engine
constants, exactly how the exported ONNX binds them), lets the TensorRT builder time every
tactic, serialises / deserialises it and compares the output with an FP32 reference
``act @ (scale * q + bias)^T`` on the unquantised checkpoint tensors. A second sweep
forces every tactic (16 GEMM variants + the decode GEMV) through the
``EDGELLM_INT4_AFFINE_FORCE_TACTIC`` override and checks each against the same reference.
Shapes mirror the Flint projections; real checkpoint tensors are used when ``FLINT_CKPT``
(default ``/work/flint-int4``) exists, otherwise random MLX-format tensors.
"""

from __future__ import annotations

import os

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

    from tensorrt_edgellm.checkpoint.mlx_affine_repack import (
        repack_mlx_affine_scales_biases, repack_mlx_affine_weights)

_PLUGIN_NAME = "Int4AffineGroupwiseGemmPlugin"
_PLUGIN_VERSION = "1"
_FORCE_ENV = "EDGELLM_INT4_AFFINE_FORCE_TACTIC"
_GEMV_TACTIC = 17
_NUM_TACTICS = 17
_MAX_M = 512
_CKPT = os.environ.get("FLINT_CKPT", "/work/flint-int4")

# (label, checkpoint module name, N, K)
_SHAPES = [
    ("q_proj", "model.layers.0.self_attn.q_proj", 4096, 2048),
    ("k_proj", "model.layers.0.self_attn.k_proj", 512, 2048),
    ("dense_down", "model.layers.0.mlp.down_proj", 2048, 6144),
    ("lm_head", "lm_head", 150144, 2048),
]
_MS = [1, 8, 64, 512]


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    if DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


_SHARD_MAP = {}


def _load_checkpoint_tensors(module: str):
    """(weight, scales, biases) of ``module`` from the Flint checkpoint, or None.

    The MLX checkpoint ships ``model-0000N.safetensors`` shards without an index file, so
    the key -> shard map is built by scanning the shard headers once.
    """
    if not os.path.isdir(_CKPT):
        return None
    from safetensors import safe_open
    if not _SHARD_MAP:
        for name in sorted(os.listdir(_CKPT)):
            if name.endswith(".safetensors"):
                with safe_open(os.path.join(_CKPT, name), "pt") as f:
                    _SHARD_MAP.update({k: name for k in f.keys()})
    out = []
    for suffix in ("weight", "scales", "biases"):
        key = f"{module}.{suffix}"
        if key not in _SHARD_MAP:
            return None
        with safe_open(os.path.join(_CKPT, _SHARD_MAP[key]), "pt") as f:
            out.append(f.get_tensor(key))
    return tuple(out)


def _random_tensors(n: int, k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
    scales = (torch.rand(n, k // 64, generator=g) * 0.02 +
              0.002).to(torch.bfloat16)
    # Flint-like: the bias is about -8 * scale plus a per-group offset.
    biases = (-8.0 * scales.float() +
              (torch.rand(n, k // 64, generator=g) - 0.5) * 0.01).to(
                  torch.bfloat16)
    return weight, scales, biases


def _dequant_reference(weight, scales, biases) -> "torch.Tensor":
    """FP32 ``[N, K]`` dequantisation ``scale * q + bias`` of the checkpoint tensors."""
    n, k_half = weight.shape
    w = weight.to(torch.int32)
    q = torch.stack([w & 0xF, (w >> 4) & 0xF], dim=-1).reshape(n, 2 * k_half)
    s = scales.float().repeat_interleave(64, dim=1)
    b = biases.float().repeat_interleave(64, dim=1)
    return q.float() * s + b


def _build(runner, n: int, k: int, m_opt: int, tensors):
    weight, scales, biases = tensors
    qweight = repack_mlx_affine_weights(weight.cuda()).cpu()
    scales_t, bias_t = repack_mlx_affine_scales_biases(scales, biases)
    runner.build(
        input_specs=[("x", trt.float16, (-1, k))],
        output_names=["y"],
        plugin_name=_PLUGIN_NAME,
        plugin_version=_PLUGIN_VERSION,
        plugin_fields=[
            pf_int32("gemm_n", n),
            pf_int32("gemm_k", k),
            pf_int32("group_size", 64)
        ],
        profiles={"x": ((1, k), (m_opt, k), (_MAX_M, k))},
        constant_specs=[
            ("qweight", trt.int8, tuple(qweight.shape), qweight),
            ("scales", trt.float16, tuple(scales_t.shape), scales_t),
            ("bias_prime", trt.float16, tuple(bias_t.shape), bias_t),
        ],
        plugin_input_order=["x", "qweight", "scales", "bias_prime"],
    )


def _rel_l2(ref: "torch.Tensor", out: "torch.Tensor") -> float:
    diff = (ref.double() - out.double()).norm()
    return float(diff / ref.double().norm().clamp_min(1e-30))


def _run(runner, x, n: int) -> "torch.Tensor":
    y = torch.empty(x.shape[0], n, dtype=torch.float16, device="cuda")
    runner.execute({"x": x, "y": y})
    return y


# Relative L2 error of an fp16 W4A16 GEMM against the fp32 dequant reference. The kernel rounds
# the dequantised weight to fp16 (|err| ~ 5e-4 relative per element, random sign), so the
# accumulated error is ~1e-3 relative; 3e-3 catches layout / scale / bias mistakes (O(1)).
_REL_L2_TOL = 3e-3

_REPORT = []


@pytest.mark.parametrize("m", _MS)
@pytest.mark.parametrize("label,module,n,k", _SHAPES)
def test_builder_tactics_match_reference(label, module, n, k, m, monkeypatch):
    """Builder-selected tactic (opt M = m) over the profile, incl. engine round trip."""
    monkeypatch.delenv(_FORCE_ENV, raising=False)
    tensors = _load_checkpoint_tensors(module) or _random_tensors(n, k, 0)
    w_ref = _dequant_reference(*tensors).cuda()
    runner = PluginRunner()
    _build(runner, n, k, m, tensors)
    g = torch.Generator(device="cuda").manual_seed(1)
    for m_run in sorted({m, 1, min(_MAX_M, 3 * m)}):
        x = torch.randn(m_run, k, device="cuda", dtype=torch.float16,
                        generator=g)
        ref = x.float() @ w_ref.t()
        err = _rel_l2(ref, _run(runner, x, n))
        _REPORT.append((label, n, k, m, m_run, "builder", err))
        assert err < _REL_L2_TOL, (
            f"{label} N={n} K={k} opt M={m} run M={m_run}: rel L2 {err:.2e}")


@pytest.mark.parametrize("label,module,n,k", _SHAPES)
def test_every_tactic_matches_reference(label, module, n, k, monkeypatch):
    """Force each tactic id (1..16 GEMM variants, 17 GEMV) at M in {1, 8, 64, 512}."""
    tensors = _load_checkpoint_tensors(module) or _random_tensors(n, k, 0)
    w_ref = _dequant_reference(*tensors).cuda()
    runner = PluginRunner()
    monkeypatch.delenv(_FORCE_ENV, raising=False)
    _build(runner, n, k, 64, tensors)
    g = torch.Generator(device="cuda").manual_seed(2)
    xs = {
        m: torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g)
        for m in _MS
    }
    for tactic in range(1, _NUM_TACTICS + 1):
        monkeypatch.setenv(_FORCE_ENV, str(tactic))
        for m, x in xs.items():
            if tactic == _GEMV_TACTIC and m > 8:
                continue  # GEMV tactic falls back to the default GEMM variant for M > 8
            ref = x.float() @ w_ref.t()
            err = _rel_l2(ref, _run(runner, x, n))
            _REPORT.append((label, n, k, m, m, f"tactic {tactic}", err))
            assert err < _REL_L2_TOL, (
                f"{label} tactic {tactic} M={m}: rel L2 {err:.2e}")


def test_rejects_unsupported_geometry():
    """N % 8 != 0, K % 64 != 0 and group_size != 64 are refused at creation."""
    runner = PluginRunner()  # loads the plugin library
    creator = trt.get_plugin_registry().get_creator(_PLUGIN_NAME,
                                                    _PLUGIN_VERSION, "")
    assert creator is not None
    for n, k, g in ((4100, 2048, 64), (4096, 2080, 64), (4096, 2048, 128)):
        fc = trt.PluginFieldCollection([
            pf_int32("gemm_n", n),
            pf_int32("gemm_k", k),
            pf_int32("group_size", g)
        ])
        assert creator.create_plugin(_PLUGIN_NAME, fc,
                                     trt.TensorRTPhase.BUILD) is None
    del runner


def teardown_module(module):
    if _REPORT:
        print("\nlabel N K M_opt M_run mode rel_L2")
        for row in _REPORT:
            print("%s %d %d %d %d %s %.2e" % row)
