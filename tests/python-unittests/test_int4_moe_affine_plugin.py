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
"""Accuracy tests for ``Int4MoeAffinePlugin`` (Flint routed experts) on real checkpoint tensors.

One dynamic-profile engine per layer (weights as engine constants), run at several token
counts against the FP32 reference (sigmoid top-6 renormalised, SwiGLU, weighted combine on
the dequantised ``scale * q + bias`` weights). Layer 46 is the FP16-overflow layer
(``|silu(gate) * up|`` reaches 7.6e4 on massive-activation tokens): the plugin must stay
finite there. Needs the Flint checkpoint (``FLINT_CKPT``, default ``/work/flint-int4``) and
the golden activations ``FLINT_GOLDEN`` (default ``/proj/logs/flint_ref_chat.pt``).
"""

from __future__ import annotations

import os
import sys

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, pf_float32, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

    from tensorrt_edgellm.checkpoint.flint_moe import repack_layer

_CKPT = os.environ.get("FLINT_CKPT", "/work/flint-int4")
_GOLDEN = os.environ.get("FLINT_GOLDEN", "/proj/logs/flint_ref_chat.pt")
_E, _H, _I, _TOPK = 256, 2048, 512, 6
_ACT_LOG2 = 4
_LAYERS = [2, 20, 46, 48]
_TS = [1, 8, 64, 512]
_REPORT = []


@pytest.fixture(autouse=True)
def require_deps(request):
    if DEPENDENCIES_AVAILABLE and os.path.isdir(_CKPT):
        return
    pytest.skip(f"TensorRT/torch CUDA or {_CKPT} not available: {IMPORT_ERROR}")


def _ckpt():
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(__file__), "..", "flint_affine_moe"))
    from flint_affine_lib import Ckpt
    return Ckpt(_CKPT)


def _activations(ck, layer, n_tokens):
    """Post-attention-norm activations: real chat tokens, padded with synthetic rows."""
    d = torch.load(_GOLDEN, map_location="cpu")["dump"][f"l{layer}.attn_out"]
    d = d.to("cuda").float()
    lnw = ck.get(f"model.layers.{layer}.post_attention_layernorm.weight").float()

    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * lnw

    real = rms(d)
    if n_tokens <= real.shape[0]:
        return real[:n_tokens].half()
    g = torch.Generator(device="cuda").manual_seed(1234 + n_tokens)
    syn = rms(
        torch.randn(n_tokens - real.shape[0], _H, device="cuda", generator=g)
        * real.pow(2).mean().sqrt())
    return torch.cat([real, syn], 0).half()


def _build(layer, ck):
    def get(key):
        return ck.get(key, "cpu")

    tensors = repack_layer(get, layer, torch.device("cuda"))
    runner = PluginRunner()
    shapes = [tuple(t.shape) for t in tensors]
    dts = [trt.int8, trt.float16, trt.float16] * 2
    cnames = ["gu_q", "gu_s", "gu_bp", "dn_q", "dn_s", "dn_bp"]
    runner.build(
        input_specs=[("router_logits", trt.float32, (-1, _E)),
                     ("hidden", trt.float16, (-1, _H))],
        output_names=["y"],
        plugin_name="Int4MoeAffinePlugin",
        plugin_version="1",
        plugin_fields=[
            pf_int32("num_experts", _E),
            pf_int32("top_k", _TOPK),
            pf_int32("hidden_size", _H),
            pf_int32("moe_inter_size", _I),
            pf_int32("activation_type", 0),
            pf_int32("quantization_group_size", 64),
            pf_int32("act_scale_log2", _ACT_LOG2),
            pf_int32("norm_topk_prob", 1),
            pf_float32("routed_scaling_factor", 1.0),
        ],
        profiles={
            "router_logits": ((1, _E), (64, _E), (512, _E)),
            "hidden": ((1, _H), (64, _H), (512, _H))
        },
        constant_specs=[(n, d, s, t.cpu())
                        for n, d, s, t in zip(cnames, dts, shapes, tensors)],
        plugin_input_order=["router_logits", "hidden"] + cnames,
    )
    return runner


def _reference(ck, layer, h):
    from flint_affine_lib import dequant_fp32
    pre = f"model.layers.{layer}.mlp"
    W = [
        dequant_fp32(ck.get(f"{pre}.switch_mlp.{p}.weight"),
                     ck.get(f"{pre}.switch_mlp.{p}.scales"),
                     ck.get(f"{pre}.switch_mlp.{p}.biases"))
        for p in ("gate_proj", "up_proj", "down_proj")
    ]
    router = ck.get(f"{pre}.gate.weight").float()
    return W, router


@pytest.mark.parametrize("layer", _LAYERS)
def test_plugin_matches_reference(layer):
    ck = _ckpt()
    from flint_affine_lib import ref_moe, routing
    runner = _build(layer, ck)
    W, router = _reference(ck, layer, None)
    for T in _TS:
        h = _activations(ck, layer, T)
        logits = h.float() @ router.T
        wts, ids = routing(logits, _TOPK)
        ref = ref_moe(h.float(), W[0], W[1], W[2], wts, ids)
        y = torch.empty(T, _H, dtype=torch.float16, device="cuda")
        runner.execute({"router_logits": logits.contiguous(), "hidden": h.contiguous(), "y": y})
        yf = y.float()
        finite = bool(torch.isfinite(yf).all())
        err = (yf - ref).abs()
        rel = float(err.norm() / ref.norm())
        _REPORT.append((layer, T, finite, rel, float(err.max()),
                        float(ref.abs().max())))
        assert finite, f"layer {layer} T={T}: non-finite output"
        assert rel < 3e-3, f"layer {layer} T={T}: rel L2 {rel:.2e}"


def teardown_module(module):
    if _REPORT:
        print("\nlayer T finite relL2 maxabs max|ref|")
        for r in _REPORT:
            print("%d %d %s %.2e %.3e %.3e" % r)
