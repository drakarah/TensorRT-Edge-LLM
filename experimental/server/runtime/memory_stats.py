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
"""Memory figures for the stats page: the GPU, the runtime's own pools and the host."""

import subprocess
import threading
import time
from typing import Any, Dict, Optional

_MIB = 1024 * 1024
_DEVICE_TTL_SECONDS = 1.0
_MIB_FIELDS = {
    "kv_mapped_bytes": "kv_mapped_mib",
    "kv_capacity_bytes": "kv_capacity_mib",
    "expert_bytes_per_slot_all_layers": "expert_slot_mib",
    "expert_cache_bytes": "expert_cache_mib",
    "host_expert_weights_bytes": "host_expert_weights_mib",
    "host_embedding_bytes": "host_embedding_mib",
    "host_snapshot_bytes": "host_snapshot_mib",
    "host_kv_tier_bytes": "host_kv_tier_mib",
    "host_kv_tier_used_bytes": "host_kv_tier_used_mib",
}
# Fields a binding older than the server may not have; the page then shows the coarse host bar.
_OPTIONAL_RUNTIME_FIELDS = ("host_expert_weights_bytes", "host_embedding_bytes",
                            "host_snapshot_bytes", "host_kv_tier_bytes",
                            "host_kv_tier_used_bytes")
_RUNTIME_FIELDS = ("kv_growable", "kv_mapped_tokens", "kv_capacity_tokens",
                   "kv_mapped_bytes", "kv_capacity_bytes", "kv_pages_usable",
                   "kv_pages_free", "kv_pages_high_water", "context_records",
                   "recurrent_snapshots_free", "recurrent_snapshots_capacity",
                   "partial_kv_snapshots_free",
                   "partial_kv_snapshots_capacity", "expert_layers",
                   "expert_slots_per_layer", "expert_max_slots_per_layer",
                   "expert_bytes_per_slot_all_layers", "expert_cache_bytes",
                   "expert_cache_alloc_failures")

_device_lock = threading.Lock()
_device_cache: Dict[str, Any] = {"at": 0.0, "value": None}


def parse_nvidia_smi(output: str) -> Optional[Dict[str, float]]:
    """First GPU of ``--query-gpu=memory.total,memory.used,memory.free`` (MiB)."""
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            total, used, free = (float(p) for p in parts)
        except ValueError:
            return None
        if total <= 0 or used < 0 or free < 0:
            return None
        return {"total_mib": total, "used_mib": used, "free_mib": free}
    return None


def device_memory() -> Optional[Dict[str, float]]:
    now = time.monotonic()
    with _device_lock:
        if now - _device_cache["at"] < _DEVICE_TTL_SECONDS:
            return _device_cache["value"]
    try:
        out = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits"
        ],
                             capture_output=True,
                             text=True,
                             timeout=2,
                             check=True).stdout
        value = parse_nvidia_smi(out)
    except (OSError, subprocess.SubprocessError):
        value = None
    with _device_lock:
        _device_cache.update(at=time.monotonic(), value=value)
    return value


def read_kib(text: str, key: str) -> Optional[int]:
    """Value of a ``Key:   123 kB`` line of /proc/meminfo or /proc/self/status."""
    for line in text.splitlines():
        if line.startswith(key + ":"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1])
    return None


def host_memory(
        meminfo_path: str = "/proc/meminfo",
        status_path: str = "/proc/self/status") -> Optional[Dict[str, float]]:
    try:
        with open(meminfo_path, encoding="utf-8") as f:
            meminfo = f.read()
        with open(status_path, encoding="utf-8") as f:
            status = f.read()
    except OSError:
        return None
    total = read_kib(meminfo, "MemTotal")
    available = read_kib(meminfo, "MemAvailable")
    rss = read_kib(status, "VmRSS")
    if total is None or available is None or rss is None:
        return None
    return {
        "total_mib": total / 1024,
        "available_mib": available / 1024,
        "server_rss_mib": rss / 1024,
    }


def runtime_memory(llm: Any) -> Optional[Dict[str, Any]]:
    """The runtime's own figures as of the last request boundary, in MiB where bytes."""
    getter = getattr(getattr(llm, "_runtime", None), "get_memory_metrics",
                     None)
    if getter is None:
        return None
    metrics = getter()
    if not getattr(metrics, "valid", False):
        return None
    out = {}
    for name in _RUNTIME_FIELDS + _OPTIONAL_RUNTIME_FIELDS:
        value = getattr(metrics, name, None)
        if value is None and name in _OPTIONAL_RUNTIME_FIELDS:
            continue
        if name in _MIB_FIELDS:
            out[_MIB_FIELDS[name]] = value / _MIB
        else:
            out[name] = value
    return out


def memory_snapshot(llm: Any) -> Dict[str, Any]:
    try:
        runtime = runtime_memory(llm)
    except Exception:  # a binding older than the server must not break the page
        runtime = None
    return {
        "at": round(time.time(), 3),
        "device": device_memory(),
        "runtime": runtime,
        "host": host_memory(),
    }
