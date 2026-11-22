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
"""``Int4MoePlugin`` with ``weight_format = 1`` (Marlin NVFP4 experts, E4M3 block scales, per-expert global scales).

The expert fixture is the one of the NVFP4 A16 plugin tests (softcapped SwiGLU, sigmoid routing of Inkling 2); the router is
handed over in the encoded form the Inkling 2 graph uses for the softmax top-k router of this plugin
(log(score) + 30 for the selected experts), and the result is compared with the same fp32 reference as that plugin's tests.
"""

from __future__ import annotations

import pytest
from test_nvfp4_a16_moe_plugin import (_INKLING_CASE, _MAX_SEQUENCE_LENGTH,
                                       _make_qwen_fixture, _moe_reference,
                                       _reference_dense_weights)
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, PluginUnsupportedError,
                              assert_close, pf_float32, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    if DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def _fields(case, weight_format=1, group=16, route_mode=0):
    return [
        pf_int32("route_mode", route_mode),
        pf_int32("num_experts", case.num_experts),
        pf_int32("top_k", case.top_k),
        pf_int32("hidden_size", case.hidden_size),
        pf_int32("moe_inter_size", case.moe_inter_size),
        pf_int32("activation_type", 0),
        pf_int32("quantization_group_size", group),
        pf_int32("weight_format", weight_format),
        pf_float32("gate_softcap", case.gate_softcap),
        pf_float32("up_softcap", case.up_softcap),
    ]


def _specs(case, words: bool, route: bool = False):
    e, h, i = case.num_experts, case.hidden_size, case.moe_inter_size
    qtype = trt.int32 if words else trt.int8
    div = 4 if words else 1
    return [
        ("router_logits", trt.float32, (-1, e)),
        ("hidden_states", trt.float16, (-1, h)),
        ("fc_gate_up_qweights", qtype, (e, h // 16, 16 * i // div)),
        ("fc_gate_up_scales", qtype, (e, h // 16, 2 * i // div)),
        ("fc_down_qweights", qtype, (e, i // 16, 8 * h // div)),
        ("fc_down_scales", qtype, (e, i // 16, h // div)),
        ("fc1_global_scales", trt.float16, (e, )),
        ("fc2_global_scales", trt.float16, (e, )),
    ] + ([("route_bias", trt.float32, (e, ))] if route else [])


def _profiles(specs):
    profiles = {}
    for name, _, shape in specs:
        if shape[0] == -1:
            profiles[name] = ((1, ) + shape[1:], (1, ) + shape[1:],
                              (_MAX_SEQUENCE_LENGTH, ) + shape[1:])
        else:
            profiles[name] = (shape, shape, shape)
    return profiles


def _build(case, words=False, route=False, **kw):
    runner = PluginRunner()
    specs = _specs(case, words, route)
    runner.build(input_specs=specs,
                 output_names=["output"],
                 plugin_name="Int4MoePlugin",
                 plugin_version="1",
                 plugin_fields=_fields(case, **kw),
                 profiles=_profiles(specs),
                 expect_unsupported=kw.get("expect_unsupported", False)
                 if False else False)
    return runner


def _encode(router_logits, bias, case):
    """The graph-side encoding of Inkling 2's routing for the softmax top-k router of Int4MoePlugin."""
    score = router_logits.sigmoid().clamp_min(1e-30)
    biased = score + bias
    kth = biased.topk(case.top_k, dim=-1).values[:, -1:]
    selected = ((biased - kth + 1e-6) * 1e6).clamp(0.0, 1.0)
    return score.log() + 30.0 * selected


@pytest.mark.parametrize("words", [False, True], ids=["int8_view", "int32_words"])
@pytest.mark.parametrize("num_tokens", [1, 3, 9, _MAX_SEQUENCE_LENGTH],
                         ids=lambda n: f"T{n}")
def test_nvfp4_experts_match_the_reference(words, num_tokens):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("The Marlin kernels need SM80 or newer")
    case = _INKLING_CASE
    fixture = _make_qwen_fixture(torch.float16, case)
    runner = _build(case, words)
    dense = _reference_dense_weights(fixture)
    packed = {k: v.to("cuda").contiguous() for k, v in fixture.packed_inputs.items()}
    if words:
        for key in ("fc1_qweights", "fc1_block_scales", "fc2_qweights", "fc2_block_scales"):
            packed[key] = packed[key].view(torch.int32)
    generator = torch.Generator().manual_seed(500 + num_tokens)
    hidden = (torch.randn((num_tokens, case.hidden_size), generator=generator) * 0.25).half().cuda()
    router = torch.randn((num_tokens, case.num_experts), generator=generator).cuda()
    bias = (torch.randn((case.num_experts, ), generator=generator) * 0.01).cuda()
    expected = _moe_reference(fixture, hidden, router, bias, dense)
    actual = torch.empty_like(hidden)
    runner.execute({
        "router_logits": _encode(router, bias, case),
        "hidden_states": hidden,
        "fc_gate_up_qweights": packed["fc1_qweights"],
        "fc_gate_up_scales": packed["fc1_block_scales"],
        "fc_down_qweights": packed["fc2_qweights"],
        "fc_down_scales": packed["fc2_block_scales"],
        "fc1_global_scales": packed["fc1_global_scales"],
        "fc2_global_scales": packed["fc2_global_scales"],
        "output": actual,
    })
    assert bool(torch.isfinite(actual.float()).all())
    # the plugin has no route scale: the graph multiplies it afterwards
    assert_close(f"nvfp4 experts words={words} T={num_tokens}", expected, actual * case.routed_scaling_factor,
                 atol=0.05, rtol=0.02, cos_threshold=0.99)


def test_nvfp4_rejects_unknown_weight_format_and_wrong_group():
    case = _INKLING_CASE
    for kw in ({"weight_format": 2}, {"weight_format": 1, "group": 32}):
        with pytest.raises((PluginUnsupportedError, RuntimeError)):
            runner = PluginRunner()
            specs = _specs(case, False)
            runner.build(input_specs=specs, output_names=["output"], plugin_name="Int4MoePlugin", plugin_version="1",
                         plugin_fields=_fields(case, **kw), profiles=_profiles(specs), expect_unsupported=True)


def test_nvfp4_rejects_missing_global_scale_inputs():
    case = _INKLING_CASE
    specs = _specs(case, False)[:6]  # six inputs with weight_format 1: the plugin needs eight
    runner = PluginRunner()
    with pytest.raises((PluginUnsupportedError, RuntimeError)):
        runner.build(input_specs=specs, output_names=["output"], plugin_name="Int4MoePlugin", plugin_version="1",
                     plugin_fields=_fields(case), profiles=_profiles(specs), expect_unsupported=True)


@pytest.mark.parametrize("num_tokens", [1, 3, 9, _MAX_SEQUENCE_LENGTH], ids=lambda n: f"T{n}")
def test_plugin_side_sigmoid_routing_matches_the_reference(num_tokens):
    """route_mode 1: the plugin takes the raw router logits and the selection bias and routes like Inkling 2 itself."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("The Marlin kernels need SM80 or newer")
    case = _INKLING_CASE
    fixture = _make_qwen_fixture(torch.float16, case)
    runner = _build(case, words=True, route=True, route_mode=1)
    dense = _reference_dense_weights(fixture)
    packed = {k: v.to("cuda").contiguous() for k, v in fixture.packed_inputs.items()}
    for key in ("fc1_qweights", "fc1_block_scales", "fc2_qweights", "fc2_block_scales"):
        packed[key] = packed[key].view(torch.int32)
    generator = torch.Generator().manual_seed(900 + num_tokens)
    hidden = (torch.randn((num_tokens, case.hidden_size), generator=generator) * 0.25).half().cuda()
    router = torch.randn((num_tokens, case.num_experts), generator=generator).cuda()
    bias = (torch.randn((case.num_experts, ), generator=generator) * 0.01).cuda()
    expected = _moe_reference(fixture, hidden, router, bias, dense)
    actual = torch.empty_like(hidden)
    runner.execute({
        "router_logits": router,
        "hidden_states": hidden,
        "fc_gate_up_qweights": packed["fc1_qweights"],
        "fc_gate_up_scales": packed["fc1_block_scales"],
        "fc_down_qweights": packed["fc2_qweights"],
        "fc_down_scales": packed["fc2_block_scales"],
        "fc1_global_scales": packed["fc1_global_scales"],
        "fc2_global_scales": packed["fc2_global_scales"],
        "route_bias": bias,
        "output": actual,
    })
    assert bool(torch.isfinite(actual.float()).all())
    assert_close(f"plugin routing T={num_tokens}", expected, actual * case.routed_scaling_factor, atol=0.05, rtol=0.02,
                 cos_threshold=0.99)


def test_sigmoid_routing_rejects_missing_bias_input_and_bad_mode():
    case = _INKLING_CASE
    for kw, route in (({"route_mode": 1}, False), ({"route_mode": 2}, True)):
        specs = _specs(case, False, route)
        runner = PluginRunner()
        with pytest.raises((PluginUnsupportedError, RuntimeError)):
            runner.build(input_specs=specs, output_names=["output"], plugin_name="Int4MoePlugin", plugin_version="1",
                         plugin_fields=_fields(case, **kw), profiles=_profiles(specs), expect_unsupported=True)
