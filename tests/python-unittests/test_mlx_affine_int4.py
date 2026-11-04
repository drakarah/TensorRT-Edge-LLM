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
"""MLX affine INT4 (group 64) Python wiring: quant-type parsing, weight repack, linear, loader.

CPU only; no TensorRT or checkpoint needed.
"""
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "kernelSrcs",
                           "int4_affine_cutedsl")


def _quant_config(tmp_path, config, sidecar=None):
    from tensorrt_edgellm.config import _parse_checkpoint_quant
    (tmp_path / "config.json").write_text(json.dumps(config))
    if sidecar is not None:
        (tmp_path / "model-int4.json").write_text(json.dumps(sidecar))
    return _parse_checkpoint_quant(str(tmp_path), config)


def test_parse_flint_model_type(tmp_path):
    from tensorrt_edgellm.config import QUANT_INT4_MLX_AFFINE
    q = _quant_config(tmp_path, {
        "model_type": "flint",
        "quantization": {
            "bits": 4,
            "group_size": 64
        }
    })
    assert q.quant_type == QUANT_INT4_MLX_AFFINE and q.group_size == 64


def test_parse_sidecar_marker(tmp_path):
    from tensorrt_edgellm.config import QUANT_INT4_MLX_AFFINE
    cfg = {"model_type": "other", "quantization": {"bits": 4, "group_size": 64}}
    q = _quant_config(tmp_path, cfg, {"weight_format": "mlx-affine-int4"})
    assert q.quant_type == QUANT_INT4_MLX_AFFINE


def test_parse_requires_marker_and_group_64(tmp_path):
    cfg = {"model_type": "other", "quantization": {"bits": 4, "group_size": 64}}
    assert _quant_config(tmp_path, cfg).quant_type == "fp16"
    with pytest.raises(ValueError, match="group_size 64"):
        _quant_config(tmp_path, {
            "model_type": "flint",
            "quantization": {
                "bits": 4,
                "group_size": 32
            }
        })


def _mlx_tensors(n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
    scales = (torch.rand(n, k // 64, generator=g) * 0.02 +
              0.002).to(torch.bfloat16)
    biases = (-8 * scales.float() + torch.randn(n, k // 64, generator=g) * 0.003
              ).to(torch.bfloat16)
    return weight, scales, biases


@pytest.mark.parametrize("n,k", [(128, 64), (256, 192), (136, 128)])
def test_repack_matches_kernel_reference(n, k):
    """The package repack is bit-identical to the kernel test's reference repack."""
    sys.path.insert(0, _KERNEL_DIR)
    try:
        import int4_affine_reference as ref
    finally:
        sys.path.remove(_KERNEL_DIR)
    from tensorrt_edgellm.checkpoint.mlx_affine_repack import (
        mlx_affine_fragment_rows, repack_mlx_affine_scales_biases,
        repack_mlx_affine_weights)
    weight, scales, biases = _mlx_tensors(n, k)
    packed = repack_mlx_affine_weights(weight)
    assert packed.dtype == torch.int8
    assert packed.shape == (mlx_affine_fragment_rows(n, k), 512)
    expected = ref.repack_weights_for_tile(weight)
    assert torch.equal(packed.view(torch.int32), expected)
    s_t, b_t = repack_mlx_affine_scales_biases(scales, biases)
    s_ref, b_ref = ref.repack_scales_biases(scales, biases)
    assert torch.equal(s_t, s_ref) and torch.equal(b_t, b_ref)
    assert s_t.shape == (k // 64, n) and s_t.dtype == torch.float16


def test_linear_repack_and_idempotence():
    from tensorrt_edgellm.checkpoint.repacking import _repack_mlx_affine_linears
    from tensorrt_edgellm.models.linear import MlxAffineInt4Linear
    lin = MlxAffineInt4Linear(128, 256)
    weight, scales, biases = _mlx_tensors(256, 128)
    lin._buffers["weight"] = weight
    lin._buffers["scales"] = scales.to(torch.float16)  # the loader casts bf16 -> fp16
    lin._buffers["biases"] = biases.to(torch.float16)
    model = torch.nn.Sequential(lin)
    _repack_mlx_affine_linears(model)
    assert lin.weight.dtype == torch.int8 and lin.weight.shape == (32, 512)
    assert lin.scales.shape == (2, 256) and lin.biases.shape == (2, 256)
    first = lin.weight.clone()
    _repack_mlx_affine_linears(model)  # second pass is a no-op
    assert torch.equal(first, lin.weight)


def test_linear_rejects_unsupported_geometry():
    from tensorrt_edgellm.models.linear import MlxAffineInt4Linear
    with pytest.raises(ValueError):
        MlxAffineInt4Linear(100, 256)
    with pytest.raises(ValueError):
        MlxAffineInt4Linear(128, 250)
    with pytest.raises(ValueError):
        MlxAffineInt4Linear(128, 256, group_size=32)


def test_loader_reads_index_less_shards(tmp_path):
    from safetensors.torch import save_file

    from tensorrt_edgellm.checkpoint.loader import _build_shard_map
    save_file({"a.weight": torch.zeros(2)},
              str(tmp_path / "model-00000.safetensors"))
    save_file({"b.weight": torch.zeros(2)},
              str(tmp_path / "model-00001.safetensors"))
    shard_map = _build_shard_map(str(tmp_path))
    assert set(shard_map) == {"a.weight", "b.weight"}
    assert shard_map["b.weight"].endswith("model-00001.safetensors")
