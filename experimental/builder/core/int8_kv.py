"""INT8 KV-cache scales for the direct builder (``EDGELLM_INT8_KV_SCALES``).

The same contract as ``tensorrt_edgellm/models/int8_kv_cache.py`` of the ONNX export path: a JSON file
``{"k_scale": [...], "v_scale": [...]}`` indexed by absolute decoder-layer index holds one symmetric
dequant scale (``amax / 127``) per layer for K and for V. Setting the variable switches the engine to an
INT8 KV cache; Q stays FP16 (scale 1.0).
"""
import json
import os
import re
from functools import lru_cache
from typing import Tuple

ENV_VAR = "EDGELLM_INT8_KV_SCALES"
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def requested() -> bool:
    """Whether the engine should use an INT8 KV cache."""
    return bool(os.environ.get(ENV_VAR))


@lru_cache(maxsize=1)
def _load() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    path = os.environ.get(ENV_VAR, "")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    k_scales = tuple(float(x) for x in data["k_scale"])
    v_scales = tuple(float(x) for x in data["v_scale"])
    if len(k_scales) != len(v_scales) or not k_scales:
        raise ValueError(
            f"{path}: k_scale and v_scale must be non-empty and equally long")
    if any(x <= 0.0 for x in k_scales + v_scales):
        raise ValueError(f"{path}: INT8 KV scales must be positive")
    return k_scales, v_scales


def qkv_scales_for_prefix(attn_prefix: str) -> Tuple[float, float, float]:
    """Plugin ``qkv_scales`` ``(q, k, v)`` of the attention layer named by its module prefix."""
    match = _LAYER.search(attn_prefix)
    if match is None:
        raise ValueError(
            f"INT8 KV cache: no decoder layer index in {attn_prefix!r}")
    layer = int(match.group(1))
    k_scales, v_scales = _load()
    if layer >= len(k_scales):
        raise ValueError(f"INT8 KV scales have no entry for layer {layer}")
    return (1.0, k_scales[layer], v_scales[layer])
