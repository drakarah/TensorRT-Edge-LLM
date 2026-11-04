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
"""Per-request timing of the served endpoint, kept in a bounded ring."""

import itertools
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

_ENABLE_ENV = "EDGELLM_REQUEST_STATS"
_CAPACITY_ENV = "EDGELLM_REQUEST_STATS_CAPACITY"
_DEFAULT_CAPACITY = 500
_MAX_CAPACITY = 100000
# Below this many uncached prompt tokens the time to first token is mostly fixed cost (cache restore, first decode step).
MIN_PREFILL_TOKENS = 256


def _capacity_from_env() -> int:
    try:
        capacity = int(os.environ.get(_CAPACITY_ENV, ""))
    except ValueError:
        return _DEFAULT_CAPACITY
    return capacity if 1 <= capacity <= _MAX_CAPACITY else _DEFAULT_CAPACITY


def _rate(count: float, seconds: float) -> Optional[float]:
    return round(count / seconds, 1) if count > 0 and seconds > 0 else None


class RequestTimer:
    """Timestamps of one request; fills the entry the ring stores."""

    def __init__(self, stream: bool) -> None:
        self.stream = stream
        self.started_at = time.time()
        self._start = time.perf_counter()
        self._first: Optional[float] = None
        self._first_tokens = 0
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens = 0
        self.reused_tokens: Optional[int] = None
        self.reuse_before: Optional[int] = None

    def tokens(self, count: int) -> None:
        if count <= 0:
            return
        if self._first is None:
            self._first = time.perf_counter()
            self._first_tokens = count
        self.completion_tokens += count

    def entry(self, status: str) -> Dict[str, Any]:
        end = time.perf_counter()
        total = end - self._start
        ttft = self._first - self._start if self.stream and self._first is not None else None
        decode_seconds = end - self._first if ttft is not None else None
        decode_tokens = self.completion_tokens - self._first_tokens
        new_prompt = None
        if self.prompt_tokens is not None:
            new_prompt = self.prompt_tokens - (self.reused_tokens or 0)
        return {
            "started_at":
            round(self.started_at, 3),
            "status":
            status,
            "stream":
            self.stream,
            "prompt_tokens":
            self.prompt_tokens,
            "reused_tokens":
            self.reused_tokens,
            "completion_tokens":
            self.completion_tokens,
            "ttft_ms":
            None if ttft is None else round(ttft * 1000, 1),
            "decode_ms":
            None if decode_seconds is None else round(decode_seconds *
                                                      1000, 1),
            "decode_tokens":
            None if decode_seconds is None else decode_tokens,
            "total_ms":
            round(total * 1000, 1),
            "prefill_tok_s":
            None if ttft is None or new_prompt is None
            or new_prompt < MIN_PREFILL_TOKENS else _rate(new_prompt, ttft),
            "decode_tok_s":
            None if decode_seconds is None else _rate(decode_tokens,
                                                      decode_seconds),
        }


class RequestStats:
    """Thread-safe ring of the last ``capacity`` finished requests."""

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> Optional["RequestStats"]:
        if os.environ.get(_ENABLE_ENV,
                          "").strip().lower() not in ("1", "true", "yes",
                                                      "on"):
            return None
        return cls(_capacity_from_env())

    @property
    def capacity(self) -> int:
        return self._entries.maxlen

    def record(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            entry["id"] = next(self._ids)
            self._entries.append(entry)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()

    def snapshot(self, limit: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            entries: List[Dict[str, Any]] = [dict(e) for e in self._entries]
        shown = entries if limit is None else entries[
            -limit:] if limit > 0 else []
        return {
            "enabled": True,
            "capacity": self.capacity,
            "summary": self._summary(entries),
            "requests": shown,
        }

    @staticmethod
    def _summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        done = [e for e in entries if e["status"] == "ok"]
        streamed = [e for e in done if e["ttft_ms"] is not None]
        prefill_tokens = prefill_seconds = decode_tokens = decode_seconds = 0.0
        for e in streamed:
            if e["prefill_tok_s"] is not None:
                prefill_tokens += e["prompt_tokens"] - (e["reused_tokens"]
                                                        or 0)
                prefill_seconds += e["ttft_ms"] / 1000
            if e["decode_tokens"]:
                decode_tokens += e["decode_tokens"]
                decode_seconds += e["decode_ms"] / 1000
        known_reuse = [
            e for e in done
            if e["reused_tokens"] is not None and e["prompt_tokens"]
        ]
        reused = sum(e["reused_tokens"] for e in known_reuse)
        prompts = sum(e["prompt_tokens"] for e in known_reuse)
        return {
            "requests": len(entries),
            "ok": len(done),
            "streamed": len(streamed),
            "prompt_tokens": sum(e["prompt_tokens"] or 0 for e in done),
            "completion_tokens": sum(e["completion_tokens"] for e in done),
            "reuse_share": reused / prompts if prompts else None,
            "prefill_tok_s": _rate(prefill_tokens, prefill_seconds),
            "decode_tok_s": _rate(decode_tokens, decode_seconds),
        }
