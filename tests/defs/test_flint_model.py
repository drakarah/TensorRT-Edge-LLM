# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Flint model structure tests: config parsing and module tree (no weights, no GPU)."""
import json

import pytest

torch = pytest.importorskip("torch")

# 6 layers: two dense leading layers, NoPE full-attention layers every 4th.
_LAYER_TYPES = [
    "full_attention", "full_attention", "sliding_attention",
    "sliding_attention", "sliding_attention", "full_attention"
]
_NO_ROPE = [0, 1, 5]


def _write_config(tmp_path, **overrides):
    cfg = {
        "architectures": ["FlintForCausalLM"],
        "model_type": "flint",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "use_sliding_window": True,
        "sliding_window": 17,
        "layer_types": _LAYER_TYPES,
        "no_rope_layer_indices": _NO_ROPE,
        "mlp_only_layers": [0, 1],
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 128,
        "norm_topk_prob": True,
        "torch_dtype": "bfloat16",
    }
    cfg.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


def _load(tmp_path, **overrides):
    import tensorrt_edgellm._export_api  # noqa: F401  (registers model classes)
    from tensorrt_edgellm.config import ModelConfig
    from tensorrt_edgellm.model import _MODEL_REGISTRY, standard_attention_scale
    cfg = ModelConfig.from_pretrained(_write_config(tmp_path, **overrides),
                                      standard_attention_scale)
    with torch.device("meta"):
        model = _MODEL_REGISTRY["flint"](cfg)
    return cfg, model


def test_flint_config(tmp_path):
    cfg, _ = _load(tmp_path)
    assert cfg.model_type == "flint"
    assert cfg.has_qk_norm
    assert cfg.sliding_window_size == 17
    assert cfg.attention_layer_types == _LAYER_TYPES
    assert cfg.full_rope_config["rope_scaling"]["rope_type"] == "nope"
    assert cfg.sliding_rope_config["rope_scaling"] is None
    assert cfg.moe_shared_expert_intermediate_size == 128


def test_flint_rejects_nope_mismatch(tmp_path):
    with pytest.raises(ValueError, match="no_rope_layer_indices"):
        _load(tmp_path, no_rope_layer_indices=[0, 1])


def test_flint_module_tree(tmp_path):
    from tensorrt_edgellm.models.flint import FlintSparseMoeBlock
    _, model = _load(tmp_path)
    layers = model.model.layers
    assert len(layers) == 6
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        full = _LAYER_TYPES[i] == "full_attention"
        assert attn.sliding_window_size == (-1 if full else 17)
        assert attn.softmax_offset.shape == (8, )
        assert isinstance(layer.mlp, FlintSparseMoeBlock) == (i >= 2)
    moe = layers[2].mlp
    # routed experts are placeholders (streamed into the sidecar at export), not weights
    assert moe.expert_gate_up_qweights.numel() == 4
    assert not any("switch_mlp" in k for k in model.state_dict())
    assert model.should_skip_checkpoint_key("model.layers.2.mlp.switch_mlp.up_proj.weight")
    assert not model.should_skip_checkpoint_key("model.layers.2.mlp.shared_expert.up_proj.weight")
    assert not hasattr(moe, "shared_expert_gate")  # shared expert is ungated


def test_flint_sinks_captured_after_load(tmp_path):
    _, model = _load(tmp_path)
    attn = model.model.layers[2].self_attn
    attn.softmax_offset = torch.arange(8, dtype=torch.float32)
    attn._capture_attention_sinks()
    assert attn._attention_sinks_list == [float(i) for i in range(8)]
