# SPDX-License-Identifier: Apache-2.0
"""INT8 KV-cache scales for export (``EDGELLM_INT8_KV_SCALES``).

The INT8 KV cache stores K and V as symmetric int8 with one dequant scale (quant -> orig) per
layer for K and one for V, measured offline (tools/calibrate_int8_kv.py in slim-gemma4-orin):
``scale = amax / 127``. The JSON file holds ``{"k_scale": [...], "v_scale": [...]}`` indexed by
absolute decoder-layer index. Setting the environment variable to that file's path switches the
export to an INT8 KV cache.
"""

import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

ENV_VAR = "EDGELLM_INT8_KV_SCALES"


def int8_kv_cache_requested() -> bool:
    """Whether the export should use an INT8 KV cache."""
    return bool(os.environ.get(ENV_VAR))


@lru_cache(maxsize=1)
def _load_scales() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    path = os.environ.get(ENV_VAR, "")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    k_scales = tuple(float(x) for x in data["k_scale"])
    v_scales = tuple(float(x) for x in data["v_scale"])
    if len(k_scales) != len(v_scales) or not k_scales:
        raise ValueError(f"{path}: k_scale and v_scale must be non-empty and equally long")
    if any(x <= 0.0 for x in k_scales + v_scales):
        raise ValueError(f"{path}: INT8 KV scales must be positive")
    return k_scales, v_scales


def int8_kv_qkv_scales(storage_layer: int) -> List[float]:
    """Plugin ``qkv_scales`` [q, k, v] for a layer whose KV lives in ``storage_layer``.

    Q stays FP16 with an INT8 KV cache, so its scale is 1.0. KV-sharing layers and the MTP
    assistant must pass the scales of the layer that owns the cache they read.
    """
    k_scales, v_scales = _load_scales()
    if not 0 <= storage_layer < len(k_scales):
        raise ValueError(f"INT8 KV scales have no entry for layer {storage_layer}")
    return [1.0, k_scales[storage_layer], v_scales[storage_layer]]


def kv_cache_torch_dtype(kv_cache_quant: Optional[str], default):
    """Torch dtype of the paged KV pool for the export dummy inputs."""
    import torch
    if kv_cache_quant == "fp8":
        return torch.float8_e4m3fn
    if kv_cache_quant == "int8":
        return torch.int8
    return default
