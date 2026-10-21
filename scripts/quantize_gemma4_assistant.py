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
"""INT4 (symmetric, group 128) weights for the linears of a Gemma 4 MTP assistant, written as a GPTQ-format checkpoint.

The assistant's layers are read about three times per decode round in FP16. INT4 reads a quarter of the bytes, and the
Int4GroupwiseGemm plugin runs them through the same path as the target's linears. Round-to-nearest with an MSE-optimal clip
per group, no calibration (the assistant's real inputs are the target's hidden states, so GPTQ would need a target forward
pass). Quality is judged by MTP acceptance (tokens per round); the target verifies every drafted token, so its outputs do not
depend on the drafter.

``--mlp-only`` leaves self_attn.q_proj / o_proj in bf16 and quantises the MLP only. Measured on the Gemma 4 12B assistant
(tree 16, INT4 candidate head, Jetson AGX Orin): 99% of the FP16 drafter's tokens per round and a 4.0% shorter decode round
(-3.3% per output token); quantising q/o as well lost 4.2% acceptance and cancelled the round saving.
``--projections`` additionally quantises pre_projection / post_projection (not measured).

Usage::

    python3 scripts/quantize_gemma4_assistant.py SRC_DIR DST_DIR --mlp-only

then export the pair as usual with ``--mtp-draft-dir DST_DIR`` (the INT4 candidate head is a separate export switch,
``EDGELLM_GEMMA4_DRAFT_HEAD_INT4=1``). Only the draft engine needs rebuilding.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP = 128
CLIP_RATIOS = (1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76)
LAYER_LINEARS = ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj",
                 "mlp.up_proj", "mlp.down_proj")
PROJECTIONS = ("pre_projection.weight", "post_projection.weight")


def quantize_rows(weight: torch.Tensor, group_size: int):
    """Symmetric INT4 per group with an MSE-optimal clip: (q [out, in] int8 in [-8, 7], scales [out, in/G], relative error)."""
    out_features, in_features = weight.shape
    groups = weight.reshape(out_features, in_features // group_size,
                            group_size)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    best_error = torch.full(amax.shape, float("inf"), device=weight.device)
    best_scale = amax / 7
    for ratio in CLIP_RATIOS:
        scale = amax * ratio / 7
        q = torch.clamp(torch.round(groups / scale), -8, 7)
        error = ((q * scale - groups)**2).sum(dim=-1, keepdim=True)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
    q = torch.clamp(torch.round(groups / best_scale), -8, 7).to(torch.int8)
    relative_error = (best_error.sum() /
                      (groups**2).sum().clamp(min=1e-12)).item()
    return q.reshape(out_features,
                     in_features), best_scale.squeeze(-1), relative_error


def quantize_groupwise(weight: torch.Tensor,
                       group_size: int,
                       chunk_rows: int = 8192):
    """quantize_rows over row chunks, to bound the temporary memory of the clip search."""
    q_parts, scale_parts, error_sum, energy_sum = [], [], 0.0, 0.0
    for start in range(0, weight.shape[0], chunk_rows):
        rows = weight[start:start + chunk_rows].float()
        q, scales, relative_error = quantize_rows(rows, group_size)
        energy = (rows**2).sum().item()
        error_sum += relative_error * energy
        energy_sum += energy
        q_parts.append(q)
        scale_parts.append(scales)
    return torch.cat(q_parts), torch.cat(scale_parts), error_sum / max(
        energy_sum, 1e-12)


def pack_gptq(q: torch.Tensor, scales: torch.Tensor) -> dict:
    """Pack signed int4 [out, in] into GPTQ tensors (qweight int32, qzeros int32, scales fp16)."""
    out_features, in_features = q.shape
    unsigned = (q.to(torch.int64) + 8).t().contiguous().reshape(
        in_features // 8, 8, out_features)
    shifts = torch.arange(0, 32, 4, device=q.device,
                          dtype=torch.int64).view(1, 8, 1)
    qweight = (unsigned << shifts).sum(dim=1)
    qweight = ((qweight + 2**31) % 2**32 - 2**31).to(
        torch.int32)  # wrap to signed int32
    zero_word = torch.tensor(0x88888888 - 2**32, dtype=torch.int32)
    qzeros = zero_word.repeat(scales.shape[1], out_features // 8)
    return {
        "qweight": qweight.cpu(),
        "qzeros": qzeros.cpu(),
        "scales": scales.t().contiguous().to(torch.float16).cpu()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "src",
        help=
        "assistant checkpoint directory (model.safetensors, config.json, ...)")
    parser.add_argument("dst", help="output directory")
    parser.add_argument("--mlp-only",
                        action="store_true",
                        help="leave self_attn.q_proj / o_proj in bf16")
    parser.add_argument("--projections",
                        action="store_true",
                        help="also quantise pre_projection / post_projection")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    linears = tuple(
        n for n in LAYER_LINEARS
        if n.startswith("mlp.")) if args.mlp_only else LAYER_LINEARS
    os.makedirs(args.dst, exist_ok=True)
    out, errors, names = {}, {}, set()
    with safe_open(os.path.join(args.src, "model.safetensors"),
                   framework="pt") as f:
        for key in f.keys():
            module = key[:-len(".weight")] if key.endswith(".weight") else None
            is_layer = (module is not None and ".layers." in module
                        and module.split(".layers.")[1].split(".",
                                                              1)[1] in linears)
            is_projection = args.projections and key in PROJECTIONS
            if is_layer or is_projection:
                q, scales, errors[module] = quantize_groupwise(
                    f.get_tensor(key).to(args.device), GROUP)
                for name, tensor in pack_gptq(q, scales).items():
                    out[f"{module}.{name}"] = tensor
                names.add(
                    module.split(".layers.")[1].
                    split(".", 1)[1] if is_layer else module)
            else:
                out[key] = f.get_tensor(key)
    save_file(out,
              os.path.join(args.dst, "model.safetensors"),
              metadata={"format": "pt"})

    with open(os.path.join(args.src, "config.json")) as f:
        config = json.load(f)
    config["quantization_config"] = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": GROUP,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
        "modules_in_block_to_quantize": sorted(names),
    }
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for name in os.listdir(args.src):
        if name not in ("model.safetensors", "config.json"):
            shutil.copy(os.path.join(args.src, name),
                        os.path.join(args.dst, name))
    print(
        json.dumps({
            "modules":
            len(errors),
            "mean_rel_sq_error":
            sum(errors.values()) / max(len(errors), 1),
            "worst":
            sorted(errors.items(), key=lambda item: -item[1])[:3]
        }))


if __name__ == "__main__":
    main()
