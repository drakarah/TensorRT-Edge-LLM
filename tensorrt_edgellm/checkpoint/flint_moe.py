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
"""Streamed export of Flint's routed-expert weights (MLX affine INT4 -> affine Marlin).

The routed experts of ``flint-int4`` are ~21.5 GB; they never enter the torch model or
the ONNX graph. The model traces the ``Int4MoeAffinePlugin`` node with tiny *placeholder*
weight inputs (:func:`placeholder`), each uniquely encoding ``(layer, slot)``. After the
ONNX export :func:`externalize_flint_moe` (1) turns every placeholder into a static-shape
graph input named ``flint_moe.layers.{L}.{slot}``, and (2) streams the real weights, one
layer at a time (~0.5 GB resident), from the checkpoint through the repack below into the
safetensors sidecar the runtime binds to those inputs (``external_weight_files``).

Per projection the plugin wants (kernel ``marlin_moe_wna16_affine``, ``w = (q-8)*s + b'``)::

    qweight  int32 [E, K/16, 2N]   Marlin tile layout of the raw nibbles (shipped as int8 [E, K/16, 8N])
    scales   fp16  [E, K/64, N]    Marlin-permuted
    b'       fp16  [E, K/64, N]    fp16(float32(bias) + 8 * float32(scale)), same permutation

gate|up are fused along N (gate columns first), so ``N = 2 * moe_inter_size`` for gate|up
and ``N = hidden_size`` for down.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

GROUP = 64
SLOTS = ("gate_up_qweights", "gate_up_scales", "gate_up_bias_prime",
         "down_qweights", "down_scales", "down_bias_prime")
_Q_SLOTS = (0, 3)
INPUT_PREFIX = "flint_moe"
PLUGIN_OP = "Int4MoeAffinePlugin"
_MAGIC = 90


def external_input_name(layer: int, slot: int) -> str:
    return f"{INPUT_PREFIX}.layers.{layer}.{SLOTS[slot]}"


# ---------------------------------------------------------------------------
# Placeholders (tiny, unique per (layer, slot) so the ONNX optimiser cannot merge them)
# ---------------------------------------------------------------------------
def placeholder(layer: int, slot: int) -> torch.Tensor:
    if slot in _Q_SLOTS:
        return torch.tensor([layer & 127, layer >> 7, slot, _MAGIC],
                            dtype=torch.int8)
    return torch.tensor([layer, slot, _MAGIC, 0], dtype=torch.float16)


def decode_placeholder(array: np.ndarray) -> "Tuple[int, int] | None":
    a = np.asarray(array).reshape(-1)
    if a.size != 4:
        return None
    if a.dtype == np.int8 and int(a[3]) == _MAGIC and int(a[2]) in _Q_SLOTS:
        return int(a[0]) + (int(a[1]) << 7), int(a[2])
    if a.dtype == np.float16 and int(a[2]) == _MAGIC and int(a[1]) not in (
            _Q_SLOTS):
        return int(a[0]), int(a[1])
    return None


# ---------------------------------------------------------------------------
# Repack (torch; ~0.7 s per layer on the GPU)
# ---------------------------------------------------------------------------
def _pack_index(i):
    return 2 * (i % 4) + i // 4


def _output_index(slot):
    return 4 * (slot % 32) + slot // 32


def _row_index(slot, pc):
    return 2 * (slot % 4) + (pc % 2) + 8 * ((pc // 2) % 2)


def _col_index(slot, pc):
    block = slot // 4
    return (block // 8) * 16 + block % 8 + (0 if pc < 4 else 8)


def _tile_tables():
    rr = np.zeros((128, 8), np.int64)
    cc = np.zeros((128, 8), np.int64)
    for slot in range(128):
        for p in range(8):
            ps = _pack_index(p)
            rr[slot, p] = _row_index(slot, ps)
            cc[slot, p] = _col_index(slot, ps)
    return rr, cc, np.array([_output_index(s) for s in range(128)])


def unpack_q(w_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> int32 [..., K] (low nibble = even k)."""
    return torch.stack([w_u8 & 0xF, w_u8 >> 4], dim=-1).flatten(-2).to(
        torch.int32)


def marlin_repack_q(q_ekn: torch.Tensor, e_chunk: int = 8) -> torch.Tensor:
    """q int32 [E, N, K] (0..15) -> int32 [E, K/16, 2N] Marlin tile layout."""
    E, N, K = q_ekn.shape
    dev = q_ekn.device
    rr, cc, out_pos = _tile_tables()
    flat = torch.from_numpy(rr * 64 + cc).to(dev)
    shifts = 4 * torch.arange(8, device=dev, dtype=torch.int32)
    perm = torch.from_numpy(out_pos).to(dev)
    out = torch.empty((E, K // 16, N * 2), dtype=torch.int32, device=dev)
    for e0 in range(0, E, e_chunk):
        q = q_ekn[e0:e0 + e_chunk].transpose(1, 2)  # [e, K, N]
        e = q.shape[0]
        t = q.reshape(e, K // 16, 16, N // 64, 64).permute(
            0, 1, 3, 2, 4).reshape(e, K // 16, N // 64, 1024)
        g = t[..., flat.reshape(-1)].reshape(e, K // 16, N // 64, 128, 8)
        words = (g << shifts).sum(-1, dtype=torch.int32)
        res = torch.empty_like(words)
        res[..., perm] = words
        out[e0:e0 + e_chunk] = res.reshape(e, K // 16, N * 2)
    return out


def marlin_permute_scales(s_egn: torch.Tensor) -> torch.Tensor:
    """[E, G, N]: permute n inside each block of 64 (int4MoeScaleRepack)."""
    N = s_egn.shape[2]
    n = torch.arange(N, device=s_egn.device)
    out_n = (n // 64) * 64 + 8 * (n % 8) + (n % 64) // 8
    out = torch.empty_like(s_egn)
    out[:, :, out_n] = s_egn
    return out


def make_scale_bias_prime(scales: torch.Tensor, biases: torch.Tensor):
    """fp16 scale and b' = fp16(float32(bias) + 8 * float32(scale)); [E, N, G]."""
    s32 = scales.to(torch.float32)
    bp32 = biases.to(torch.float32) + 8.0 * s32
    return s32.to(torch.float16), bp32.to(torch.float16)


def repack_projection(parts: List[Tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor]],
                      e_chunk: int = 8):
    """Repack one projection, or fused gate|up (``parts`` = [gate, up]).

    Each part is ``(weight uint8 [E, N, K/2], scales [E, N, K/64], biases)`` on the
    compute device. Returns ``(qweight int32 [E, K/16, 2N], scales, b')`` with
    scales/b' fp16 ``[E, K/64, N]``, Marlin-permuted.
    """
    ss, bs = [], []
    for _, s, b in parts:
        s16, bp16 = make_scale_bias_prime(s, b)
        ss.append(s16.transpose(1, 2))
        bs.append(bp16.transpose(1, 2))
    scales = torch.cat(ss, dim=2)
    bprime = torch.cat(bs, dim=2)
    num_experts = parts[0][0].shape[0]
    outs = []
    for e0 in range(0, num_experts, e_chunk):
        q = torch.cat([unpack_q(w[e0:e0 + e_chunk]) for w, _, _ in parts],
                      dim=1)
        outs.append(marlin_repack_q(q, e_chunk))
    return (torch.cat(outs, 0), marlin_permute_scales(scales).contiguous(),
            marlin_permute_scales(bprime).contiguous())


def slot_shapes(num_experts: int, hidden: int, inter: int):
    """(dtype_str, shape) of the six engine inputs, in slot order."""
    return [
        ("I8", (num_experts, hidden // 16, 16 * inter)),
        ("F16", (num_experts, hidden // GROUP, 2 * inter)),
        ("F16", (num_experts, hidden // GROUP, 2 * inter)),
        ("I8", (num_experts, inter // 16, 8 * hidden)),
        ("F16", (num_experts, inter // GROUP, hidden)),
        ("F16", (num_experts, inter // GROUP, hidden)),
    ]


def repack_layer(get, layer: int, device) -> List[torch.Tensor]:
    """Repack the experts of one layer. ``get(key)`` returns a CPU checkpoint tensor."""

    def proj(name):
        base = f"model.layers.{layer}.mlp.switch_mlp.{name}"
        return tuple(
            get(f"{base}.{suffix}").to(device)
            for suffix in ("weight", "scales", "biases"))

    gate, up = proj("gate_proj"), proj("up_proj")
    gu_q, gu_s, gu_b = repack_projection([gate, up])
    del gate, up
    dn_q, dn_s, dn_b = repack_projection([proj("down_proj")])
    return [
        gu_q.view(torch.int8), gu_s, gu_b,
        dn_q.view(torch.int8), dn_s, dn_b
    ]


# ---------------------------------------------------------------------------
# ONNX post-processing + sidecar
# ---------------------------------------------------------------------------
def _attr_int(node, name):
    for a in node.attribute:
        if a.name == name:
            return int(a.i)
    raise KeyError(name)


def externalize_flint_moe(onnx_path: str,
                          model_dir: str,
                          sidecar_name: str = "external_flint_moe_weights.safetensors",
                          device: str = "cuda") -> List[dict]:
    """Rewire placeholder weights to graph inputs and stream the sidecar.

    Returns the ``external_weight_files`` manifest entries (empty if the graph has no
    Int4MoeAffinePlugin node).
    """
    import onnx
    from onnx import TensorProto, external_data_helper, helper, numpy_helper

    model = onnx.load(onnx_path, load_external_data=False)
    base_dir = os.path.dirname(os.path.abspath(onnx_path))
    inits = {i.name: i for i in model.graph.initializer}
    nodes = [n for n in model.graph.node if n.op_type == PLUGIN_OP]
    if not nodes:
        return []

    dims = None
    wiring: Dict[Tuple[int, int], str] = {}
    for node in nodes:
        d = (_attr_int(node, "num_experts"), _attr_int(node, "hidden_size"),
             _attr_int(node, "moe_inter_size"))
        if dims is None:
            dims = d
        assert d == dims, "all Flint MoE layers share one geometry"
        layer = None
        for idx in range(2, 8):
            init = inits.get(node.input[idx])
            if init is None:
                raise RuntimeError(
                    f"{PLUGIN_OP} input {idx} ({node.input[idx]}) is not a "
                    "placeholder initializer")
            if init.external_data:
                external_data_helper.load_external_data_for_tensor(
                    init, base_dir)
            found = decode_placeholder(numpy_helper.to_array(init))
            if found is None or found[1] != idx - 2:
                raise RuntimeError(
                    f"{node.input[idx]} is not the expected placeholder")
            if layer is None:
                layer = found[0]
            assert found[0] == layer
            name = external_input_name(layer, idx - 2)
            wiring[(layer, idx - 2)] = name
            node.input[idx] = name
    num_experts, hidden, inter = dims
    shapes = slot_shapes(num_experts, hidden, inter)
    elem = {"I8": TensorProto.INT8, "F16": TensorProto.FLOAT16}
    layers = sorted({l for l, _ in wiring})

    used = {i for n in model.graph.node for i in n.input}
    keep = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    existing = {i.name for i in model.graph.input}
    for layer in layers:
        for slot in range(6):
            name = wiring[(layer, slot)]
            if name not in existing:
                dt, shape = shapes[slot]
                model.graph.input.append(
                    helper.make_tensor_value_info(name, elem[dt],
                                                  list(shape)))
    onnx.save_model(model, onnx_path)

    path = os.path.join(base_dir, sidecar_name)
    write_sidecar(path, model_dir, layers, dims, device)
    names = [wiring[(l, s)] for l in layers for s in range(6)]
    return [{
        "file": sidecar_name,
        "kind": "flint_moe_weights",
        "tensors": names
    }]


def write_sidecar(path: str, model_dir: str, layers: List[int], dims,
                  device: str = "cuda") -> None:
    """Write the safetensors sidecar layer by layer (peak ~1 GB host RAM)."""
    import time

    from safetensors import safe_open

    from .loader import _build_shard_map

    num_experts, hidden, inter = dims
    shapes = slot_shapes(num_experts, hidden, inter)
    itemsize = {"I8": 1, "F16": 2}
    header, offset = {}, 0
    for layer in layers:
        for slot in range(6):
            dt, shape = shapes[slot]
            nbytes = int(np.prod(shape)) * itemsize[dt]
            header[external_input_name(layer, slot)] = {
                "dtype": dt,
                "shape": list(shape),
                "data_offsets": [offset, offset + nbytes]
            }
            offset += nbytes
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)

    shard_map = _build_shard_map(model_dir)
    handles: Dict[str, object] = {}

    def get(key: str) -> torch.Tensor:
        shard = shard_map[key]
        if shard not in handles:
            handles[shard] = safe_open(shard, framework="pt", device="cpu")
        return handles[shard].get_tensor(key)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    with open(path, "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for n, layer in enumerate(layers):
            for slot, t in enumerate(repack_layer(get, layer, dev)):
                dt, shape = shapes[slot]
                assert tuple(t.shape) == shape, (layer, slot, t.shape, shape)
                f.write(memoryview(t.cpu().contiguous().numpy()).cast("B"))
            # Drop the checkpoint mappings: their resident file pages count as RSS.
            handles.clear()
            f.flush()
            os.fsync(f.fileno())
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            if n % 8 == 0:
                logger.info("flint moe sidecar: layer %d (%d/%d) %.0fs", layer,
                            n + 1, len(layers), time.time() - t0)
    logger.info("flint moe sidecar written: %s (%.2f GB)", path,
                os.path.getsize(path) / 1e9)
