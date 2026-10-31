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
"""Memory figures behind /api/stats: the GPU reading, the runtime's own pools, the host."""

import subprocess
import types

import pytest
from test_server_requests import _create_app, _FakeLLM

from experimental.server.runtime import memory_stats

_MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def _fresh_device_cache():
    memory_stats._device_cache.update(at=0.0, value=None)
    yield
    memory_stats._device_cache.update(at=0.0, value=None)


def _metrics(**overrides):
    values = {name: 1 for name in memory_stats._RUNTIME_FIELDS}
    values.update(valid=True,
                  kv_mapped_bytes=2 * _MIB,
                  kv_capacity_bytes=8 * _MIB,
                  expert_bytes_per_slot_all_layers=3 * _MIB,
                  expert_cache_bytes=12 * _MIB)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _llm(metrics):
    runtime = types.SimpleNamespace(get_memory_metrics=lambda: metrics)
    return types.SimpleNamespace(_runtime=runtime)


def test_nvidia_smi_first_gpu_is_parsed():
    assert memory_stats.parse_nvidia_smi("8188, 7759, 47\n") == {
        "total_mib": 8188.0,
        "used_mib": 7759.0,
        "free_mib": 47.0
    }
    assert memory_stats.parse_nvidia_smi(
        "100, 40, 60\n200, 1, 199\n")["total_mib"] == 100.0


@pytest.mark.parametrize("output", [
    "", "\n", "N/A, N/A, N/A", "1, 2", "1, 2, 3, 4", "0, 0, 0", "100, -1, 101",
    "abc"
])
def test_nvidia_smi_unusable_output_is_none(output):
    assert memory_stats.parse_nvidia_smi(output) is None


def test_device_memory_is_cached_for_a_second(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(stdout="8188, 100, 8088\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert memory_stats.device_memory()["used_mib"] == 100.0
    assert memory_stats.device_memory()["used_mib"] == 100.0
    assert len(calls) == 1


@pytest.mark.parametrize("error", [
    FileNotFoundError(),
    subprocess.TimeoutExpired("nvidia-smi", 2),
    subprocess.CalledProcessError(9, "nvidia-smi"),
])
def test_device_memory_failure_is_none(monkeypatch, error):

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    assert memory_stats.device_memory() is None


def test_read_kib():
    text = "MemTotal:       32588 kB\nVmRSS:\t  120 kB\nBad:  x kB\n"
    assert memory_stats.read_kib(text, "MemTotal") == 32588
    assert memory_stats.read_kib(text, "VmRSS") == 120
    assert memory_stats.read_kib(text, "Bad") is None
    assert memory_stats.read_kib(text, "Missing") is None
    assert memory_stats.read_kib(text, "Mem") is None


def test_host_memory_from_proc_files(tmp_path):
    meminfo = tmp_path / "meminfo"
    status = tmp_path / "status"
    meminfo.write_text("MemTotal: 2048 kB\nMemAvailable: 1024 kB\n")
    status.write_text("VmRSS: 512 kB\n")
    assert memory_stats.host_memory(str(meminfo), str(status)) == {
        "total_mib": 2.0,
        "available_mib": 1.0,
        "server_rss_mib": 0.5
    }


def test_host_memory_missing_file_or_key_is_none(tmp_path):
    meminfo = tmp_path / "meminfo"
    status = tmp_path / "status"
    assert memory_stats.host_memory(str(meminfo), str(status)) is None
    meminfo.write_text("MemTotal: 2048 kB\n")
    status.write_text("VmRSS: 512 kB\n")
    assert memory_stats.host_memory(str(meminfo), str(status)) is None


def test_runtime_memory_converts_bytes_to_mib_and_keeps_counts():
    out = memory_stats.runtime_memory(_llm(_metrics()))
    assert out["kv_mapped_mib"] == 2.0
    assert out["kv_capacity_mib"] == 8.0
    assert out["expert_slot_mib"] == 3.0
    assert out["expert_cache_mib"] == 12.0
    assert out["expert_layers"] == 1
    assert not any(name.endswith("_bytes") for name in out)


@pytest.mark.parametrize("llm", [
    types.SimpleNamespace(),
    types.SimpleNamespace(_runtime=None),
    types.SimpleNamespace(_runtime=types.SimpleNamespace()),
    _llm(types.SimpleNamespace(valid=False)),
])
def test_runtime_memory_is_none_without_figures(llm):
    assert memory_stats.runtime_memory(llm) is None


def test_snapshot_survives_a_failing_runtime_getter(monkeypatch):

    def broken():
        raise RuntimeError("binding mismatch")

    monkeypatch.setattr(memory_stats, "device_memory",
                        lambda: {"total_mib": 1.0})
    monkeypatch.setattr(memory_stats, "host_memory",
                        lambda: {"total_mib": 2.0})
    llm = types.SimpleNamespace(_runtime=types.SimpleNamespace(
        get_memory_metrics=broken))
    snapshot = memory_stats.memory_snapshot(llm)
    assert snapshot["runtime"] is None
    assert snapshot["device"] == {"total_mib": 1.0}
    assert snapshot["host"] == {"total_mib": 2.0}
    assert snapshot["at"] > 0


def _client(tmp_path, monkeypatch, enabled):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    monkeypatch.setattr(
        memory_stats, "device_memory", lambda: {
            "total_mib": 8188.0,
            "used_mib": 7000.0,
            "free_mib": 1188.0
        })
    if enabled:
        monkeypatch.setenv("EDGELLM_REQUEST_STATS", "1")
    return TestClient(_create_app(_FakeLLM(tmp_path)))


@pytest.mark.parametrize("enabled", [False, True])
def test_api_stats_carries_memory_with_or_without_request_stats(
        tmp_path, monkeypatch, enabled):
    body = _client(tmp_path, monkeypatch, enabled).get("/api/stats").json()
    assert body["enabled"] is enabled
    assert body["memory"]["device"]["total_mib"] == 8188.0
    assert body["memory"]["runtime"] is None


def test_stats_page_draws_the_memory_bars(tmp_path, monkeypatch):
    page = _client(tmp_path, monkeypatch, enabled=True).get("/stats").text
    for needle in ("renderMemory", 'id="gpu"', 'id="kv"', 'id="experts"',
                   'id="host"', 'id="snaps"'):
        assert needle in page
