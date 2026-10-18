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

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")


def _reference(gate_up):
    gate, up = gate_up.to(torch.float32).chunk(2, dim=-1)
    out = torch.nn.functional.gelu(gate, approximate="tanh") * up
    return out.clamp(-2048.0, 2048.0).to(torch.float16)


# widths: 128 and 15360 take the 128-bit path, 100 and 7 the scalar one
@pytest.mark.parametrize("num_tokens", [1, 16, 62])
@pytest.mark.parametrize("width", [128, 15360, 100, 7])
def test_gelu_tanh_mul_matches_reference(num_tokens, width):
    runner = PluginRunner()
    runner.build(input_specs=[("gate_up", trt.float16, (-1, 2 * width))],
                 output_names=["activated"],
                 plugin_name="GeluTanhMulPlugin",
                 plugin_version="1",
                 plugin_fields=[],
                 profiles={"gate_up": ((1, 2 * width), (16, 2 * width), (64, 2 * width))})

    generator = torch.Generator().manual_seed(2026 + num_tokens + width)
    # scale 8 pushes products past the +-2048 clamp for some elements
    gate_up = (torch.randn((num_tokens, 2 * width), generator=generator) * 8).to(torch.float16).to("cuda")
    output = torch.empty((num_tokens, width), dtype=torch.float16, device="cuda")
    runner.execute({"gate_up": gate_up, "activated": output})

    expected = _reference(gate_up)
    # FP32 arithmetic with different tanh implementations: allow one FP16 ulp
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
    assert output.abs().max().item() <= 2048.0


def test_gelu_tanh_mul_rank3_input():
    width = 256
    runner = PluginRunner()
    runner.build(input_specs=[("gate_up", trt.float16, (-1, -1, 2 * width))],
                 output_names=["activated"],
                 plugin_name="GeluTanhMulPlugin",
                 plugin_version="1",
                 plugin_fields=[],
                 profiles={"gate_up": ((1, 1, 2 * width), (2, 8, 2 * width), (3, 64, 2 * width))})
    gate_up = torch.randn((3, 7, 2 * width), dtype=torch.float16).to("cuda")
    output = torch.empty((3, 7, width), dtype=torch.float16, device="cuda")
    runner.execute({"gate_up": gate_up, "activated": output})
    torch.testing.assert_close(output, _reference(gate_up), rtol=2e-3, atol=2e-3)
