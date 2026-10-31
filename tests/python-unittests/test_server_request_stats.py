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
"""Per-request timing ring (EDGELLM_REQUEST_STATS) and its /stats routes."""

import threading
import time

import pytest
from test_server_requests import _create_app, _FakeLLM

from experimental.server.runtime.request_stats import (RequestStats,
                                                       RequestTimer)

_ENV = ("EDGELLM_REQUEST_STATS", "EDGELLM_REQUEST_STATS_CAPACITY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _entry(**fields):
    entry = {
        "status": "ok",
        "stream": True,
        "prompt_tokens": 100,
        "reused_tokens": 0,
        "completion_tokens": 11,
        "ttft_ms": 100.0,
        "decode_ms": 1000.0,
        "decode_tokens": 10,
        "decode_tok_s": 10.0,
    }
    entry.update(fields)
    return entry


@pytest.mark.parametrize("value", ["", "0", "off", "no", "abc"])
def test_disabled_unless_the_flag_is_truthy(monkeypatch, value):
    monkeypatch.setenv("EDGELLM_REQUEST_STATS", value)
    assert RequestStats.from_env() is None


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_truthy_flag_enables_with_the_default_capacity(monkeypatch, value):
    monkeypatch.setenv("EDGELLM_REQUEST_STATS", value)
    assert RequestStats.from_env().capacity == 500


@pytest.mark.parametrize("value,expected",
                         [("20", 20), ("0", 500), ("-3", 500), ("x", 500),
                          ("100001", 500), ("100000", 100000)])
def test_capacity_falls_back_outside_its_range(monkeypatch, value, expected):
    monkeypatch.setenv("EDGELLM_REQUEST_STATS", "1")
    monkeypatch.setenv("EDGELLM_REQUEST_STATS_CAPACITY", value)
    assert RequestStats.from_env().capacity == expected


def test_ring_keeps_the_newest_entries_with_increasing_ids():
    stats = RequestStats(3)
    for _ in range(5):
        stats.record(_entry())
    ids = [e["id"] for e in stats.snapshot()["requests"]]
    assert ids == [3, 4, 5]


def test_reset_clears_the_ring_but_not_the_id_sequence():
    stats = RequestStats(3)
    stats.record(_entry())
    stats.reset()
    assert stats.snapshot()["requests"] == []
    stats.record(_entry())
    assert stats.snapshot()["requests"][0]["id"] == 2


@pytest.mark.parametrize("limit,count", [(None, 4), (2, 2), (0, 0), (-1, 0),
                                         (10, 4)])
def test_snapshot_limit(limit, count):
    stats = RequestStats(10)
    for _ in range(4):
        stats.record(_entry())
    assert len(stats.snapshot(limit)["requests"]) == count


def test_concurrent_records_keep_unique_ids():
    stats = RequestStats(1000)

    def work():
        for _ in range(100):
            stats.record(_entry())

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ids = [e["id"] for e in stats.snapshot()["requests"]]
    assert sorted(ids) == list(range(1, 801))


def test_streamed_timer_splits_prefill_from_decode():
    timer = RequestTimer(stream=True)
    timer.prompt_tokens = 1000
    timer.reused_tokens = 900
    time.sleep(0.05)
    timer.tokens(1)
    time.sleep(0.1)
    timer.tokens(9)
    entry = timer.entry("ok")
    assert entry["completion_tokens"] == 10
    assert entry["decode_tokens"] == 9
    assert entry["ttft_ms"] >= 50
    assert entry["decode_ms"] >= 100
    assert entry["prefill_tok_s"] == pytest.approx(100 /
                                                   (entry["ttft_ms"] / 1000),
                                                   rel=0.01)
    assert entry["decode_tok_s"] == pytest.approx(9 /
                                                  (entry["decode_ms"] / 1000),
                                                  rel=0.01)


def test_non_streamed_timer_has_only_a_total():
    timer = RequestTimer(stream=False)
    timer.prompt_tokens = 10
    timer.tokens(5)
    entry = timer.entry("ok")
    assert entry["total_ms"] >= 0
    assert entry["ttft_ms"] is None
    assert entry["decode_ms"] is None
    assert entry["prefill_tok_s"] is None
    assert entry["decode_tok_s"] is None


def test_timer_without_tokens_reports_no_rates():
    entry = RequestTimer(stream=True).entry("error")
    assert entry["completion_tokens"] == 0
    assert entry["ttft_ms"] is None
    assert entry["decode_tok_s"] is None


def test_a_single_token_has_no_decode_rate():
    timer = RequestTimer(stream=True)
    timer.prompt_tokens = 5
    timer.tokens(1)
    entry = timer.entry("ok")
    assert entry["decode_tokens"] == 0
    assert entry["decode_tok_s"] is None
    assert entry["prefill_tok_s"] is not None


def test_empty_token_deltas_do_not_start_the_decode_clock():
    timer = RequestTimer(stream=True)
    timer.tokens(0)
    timer.tokens(-1)
    assert timer.entry("ok")["ttft_ms"] is None


def test_unknown_prompt_leaves_the_prefill_rate_empty():
    timer = RequestTimer(stream=True)
    timer.tokens(1)
    assert timer.entry("ok")["prefill_tok_s"] is None
    timer = RequestTimer(stream=True)
    timer.prompt_tokens = 50
    timer.tokens(1)
    assert timer.entry("ok")["prefill_tok_s"] is not None


def test_reuse_larger_than_the_prompt_gives_no_prefill_rate():
    timer = RequestTimer(stream=True)
    timer.prompt_tokens = 10
    timer.reused_tokens = 12
    timer.tokens(1)
    assert timer.entry("ok")["prefill_tok_s"] is None


def test_summary_weights_by_tokens_and_ignores_failed_requests():
    stats = RequestStats(10)
    stats.record(_entry(prompt_tokens=100, reused_tokens=0, ttft_ms=100.0))
    stats.record(
        _entry(prompt_tokens=1000,
               reused_tokens=800,
               ttft_ms=100.0,
               decode_tokens=30,
               decode_ms=1000.0,
               decode_tok_s=30.0))
    stats.record(_entry(status="error", prompt_tokens=10**6))
    stats.record(_entry(status="aborted", prompt_tokens=10**6))
    summary = stats.snapshot()["summary"]
    assert summary["requests"] == 4
    assert summary["ok"] == 2
    assert summary["prompt_tokens"] == 1100
    assert summary["prefill_tok_s"] == pytest.approx(300 / 0.2, rel=0.001)
    assert summary["decode_tok_s"] == pytest.approx(40 / 2.0, rel=0.001)
    assert summary["reuse_share"] == pytest.approx(800 / 1100)


def test_summary_of_an_empty_ring_is_all_empty():
    summary = RequestStats(5).snapshot()["summary"]
    assert summary["requests"] == 0
    assert summary["prefill_tok_s"] is None
    assert summary["decode_tok_s"] is None
    assert summary["reuse_share"] is None


def test_summary_skips_unknown_reuse_and_non_streamed_requests():
    stats = RequestStats(5)
    stats.record(
        _entry(stream=False,
               ttft_ms=None,
               decode_ms=None,
               decode_tokens=None,
               decode_tok_s=None,
               reused_tokens=None))
    summary = stats.snapshot()["summary"]
    assert summary["ok"] == 1
    assert summary["streamed"] == 0
    assert summary["prefill_tok_s"] is None
    assert summary["reuse_share"] is None


def _client(tmp_path, monkeypatch, enabled):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    if enabled:
        monkeypatch.setenv("EDGELLM_REQUEST_STATS", "1")
    return TestClient(_create_app(_FakeLLM(tmp_path)))


def test_route_reports_disabled(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, enabled=False)
    client.post("/v1/chat/completions",
                json={"messages": [{
                    "role": "user",
                    "content": "hi"
                }]})
    assert client.get("/stats").json() == {"enabled": False}
    assert client.post("/stats/reset").json() == {"enabled": False}


def test_route_records_a_non_streamed_request(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, enabled=True)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{
            "role": "user",
            "content": "hi"
        }]})
    assert response.status_code == 200
    stats = client.get("/stats").json()
    (entry, ) = stats["requests"]
    assert (entry["status"], entry["stream"]) == ("ok", False)
    assert (entry["prompt_tokens"], entry["completion_tokens"]) == (7, 3)
    assert entry["ttft_ms"] is None


def test_route_records_a_streamed_request(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, enabled=True)
    response = client.post("/v1/chat/completions",
                           json={
                               "messages": [{
                                   "role": "user",
                                   "content": "hi"
                               }],
                               "stream": True
                           })
    assert response.status_code == 200
    (entry, ) = client.get("/stats", params={"limit": 5}).json()["requests"]
    assert (entry["status"], entry["stream"]) == ("ok", True)
    assert entry["prompt_tokens"] == 7
    assert entry["completion_tokens"] == 2
    assert entry["ttft_ms"] is not None


def test_route_reset_empties_the_ring(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, enabled=True)
    client.post("/v1/chat/completions",
                json={"messages": [{
                    "role": "user",
                    "content": "hi"
                }]})
    assert client.post("/stats/reset").json() == {"enabled": True}
    assert client.get("/stats").json()["requests"] == []
