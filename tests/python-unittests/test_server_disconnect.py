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
"""A client that disconnects must free the engine: request guards, the cancel path of non-streaming routes, and the two settings around it."""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from test_server_requests import _create_app, _FakeLLM

from experimental.server.config import ApiConfig
from experimental.server.runtime.engine import CompletionOutput


class _BlockingLLM(_FakeLLM):
    """Generation that runs until the engine handle it published is cancelled."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.cancelled = threading.Event()
        self.started = threading.Event()
        self.published = False
        self.block = True

    def _complete_prepared_request(self, request, params, tool_config, **kwargs):
        on_handle = kwargs.get("on_handle")
        if self.block:
            if on_handle is not None:
                self.published = True
                on_handle(SimpleHandle(self.cancelled))
            self.started.set()
            self.cancelled.wait(timeout=20.0)
            return CompletionOutput(text="partial", token_ids=[1],
                                    prompt_tokens=7, finish_reason="cancelled")
        return super()._complete_prepared_request(request, params, tool_config,
                                                  **kwargs)


class SimpleHandle:

    def __init__(self, event: threading.Event):
        self._event = event

    def cancel(self):
        self._event.set()


def _chat_body(**extra) -> bytes:
    body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return json.dumps(body).encode()


class _Connection:
    """Drives the ASGI app by hand: a request body, then a disconnect on demand."""

    def __init__(self, app, path="/v1/chat/completions", body=b"",
                 chunks=None, headers=None):
        raw = chunks if chunks is not None else [body]
        total = sum(len(c) for c in raw)
        self.app = app
        self.sent = []
        self._disconnect = asyncio.Event()
        self._messages = [{
            "type": "http.request",
            "body": c,
            "more_body": i < len(raw) - 1,
        } for i, c in enumerate(raw)]
        self.scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("client", 1),
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(total).encode())] +
                       list(headers or []),
        }
        self.task = None

    async def _receive(self):
        if self._messages:
            return self._messages.pop(0)
        await self._disconnect.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message):
        self.sent.append(message)

    def start(self):
        self.task = asyncio.ensure_future(
            self.app(self.scope, self._receive, self._send))

    def disconnect(self):
        self._disconnect.set()

    @property
    def status(self):
        for message in self.sent:
            if message["type"] == "http.response.start":
                return message["status"]
        return None

    @property
    def body(self) -> bytes:
        return b"".join(m.get("body", b"") for m in self.sent
                        if m["type"] == "http.response.body")


def _run(coro):
    return asyncio.run(coro)


async def _within(predicate, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


def test_disconnect_cancels_a_running_non_streaming_request(tmp_path):
    llm = _BlockingLLM(tmp_path)
    app = _create_app(llm)

    async def scenario():
        conn = _Connection(app, body=_chat_body())
        conn.start()
        assert await _within(llm.started.is_set, 5.0), "generation never started"
        assert not llm.cancelled.is_set()
        conn.disconnect()
        assert await _within(llm.cancelled.is_set, 5.0), "the engine request was not cancelled"
        await asyncio.wait_for(conn.task, 5.0)
        assert conn.status == 499

    _run(scenario())
    assert llm.published


def test_a_request_that_completes_before_any_disconnect_is_answered_normally(tmp_path):
    llm = _BlockingLLM(tmp_path)
    llm.block = False
    app = _create_app(llm)

    async def scenario():
        conn = _Connection(app, body=_chat_body())
        conn.start()
        await asyncio.wait_for(conn.task, 5.0)
        assert conn.status == 200
        assert json.loads(conn.body)["choices"][0]["message"]["content"] == "answer"
        conn.disconnect()

    _run(scenario())
    assert not llm.cancelled.is_set()


def test_disconnect_while_the_body_is_still_arriving_runs_no_handler(tmp_path):
    llm = _BlockingLLM(tmp_path)
    app = _create_app(llm)

    async def scenario():
        conn = _Connection(app, body=_chat_body())
        # the first chunk says more is coming; the client then goes away
        conn._messages = [{"type": "http.request", "body": b'{"model": "fake-model", ', "more_body": True}]
        conn.start()
        await asyncio.sleep(0.1)
        conn.disconnect()
        await asyncio.wait_for(conn.task, 5.0)
        assert conn.status is None, "no response is written to a client that left"

    _run(scenario())
    assert not llm.started.is_set()


def test_a_body_delivered_in_several_chunks_is_reassembled(tmp_path):
    llm = _BlockingLLM(tmp_path)
    llm.block = False
    app = _create_app(llm)
    body = _chat_body()

    async def scenario():
        conn = _Connection(app, chunks=[body[:7], body[7:20], body[20:]])
        conn.start()
        await asyncio.wait_for(conn.task, 5.0)
        assert conn.status == 200
        conn.disconnect()

    _run(scenario())


def test_an_oversized_body_is_refused_with_413(tmp_path, monkeypatch):
    from experimental.server.api import app as app_module
    monkeypatch.setattr(app_module, "MAX_REQUEST_BODY_BYTES", 64)
    llm = _BlockingLLM(tmp_path)
    app = _create_app(llm)

    async def scenario():
        conn = _Connection(app, chunks=[b"x" * 40, b"x" * 40])
        conn.start()
        await asyncio.wait_for(conn.task, 5.0)
        assert conn.status == 413
        conn.disconnect()

    _run(scenario())
    assert not llm.started.is_set()


def test_the_api_key_is_still_checked_before_the_body_is_read(tmp_path):
    llm = _BlockingLLM(tmp_path)
    app = _create_app(llm, ApiConfig(api_key="secret"))

    async def scenario():
        denied = _Connection(app, body=_chat_body())
        denied.start()
        await asyncio.wait_for(denied.task, 5.0)
        assert denied.status == 401
        allowed = _Connection(app, body=_chat_body(), headers=[(b"authorization", b"Bearer secret")])
        llm.block = False
        allowed.start()
        await asyncio.wait_for(allowed.task, 5.0)
        assert allowed.status == 200
        denied.disconnect()
        allowed.disconnect()

    _run(scenario())


def test_non_http_scopes_pass_through_the_guards(tmp_path):
    llm = _BlockingLLM(tmp_path)
    app = _create_app(llm)

    async def scenario():
        received = []

        async def receive():
            return {"type": "lifespan.startup"} if not received else {"type": "lifespan.shutdown"}

        async def send(message):
            received.append(message["type"])

        await asyncio.wait_for(app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send), 5.0)
        return received

    types = _run(scenario())
    assert "lifespan.startup.complete" in types


def test_each_disconnect_frees_the_seat_for_the_next_request(tmp_path):
    llm = _BlockingLLM(tmp_path)
    app = _create_app(llm)

    async def scenario():
        for _ in range(3):
            llm.cancelled.clear()
            llm.started.clear()
            conn = _Connection(app, body=_chat_body())
            conn.start()
            assert await _within(llm.started.is_set, 5.0)
            conn.disconnect()
            assert await _within(llm.cancelled.is_set, 5.0)
            await asyncio.wait_for(conn.task, 5.0)
        llm.block = False
        ok = _Connection(app, body=_chat_body())
        ok.start()
        await asyncio.wait_for(ok.task, 5.0)
        assert ok.status == 200
        ok.disconnect()

    _run(scenario())


def test_the_body_of_a_cancelled_request_is_kept_when_a_directory_is_set(tmp_path, monkeypatch):
    target = tmp_path / "kept" / "nested"
    monkeypatch.setenv("EDGELLM_LOG_CANCELLED_DIR", str(target))
    llm = _BlockingLLM(tmp_path)
    app = _create_app(llm)

    async def scenario():
        conn = _Connection(app, body=_chat_body(max_tokens=77))
        conn.start()
        assert await _within(llm.started.is_set, 5.0)
        conn.disconnect()
        await asyncio.wait_for(conn.task, 5.0)

    _run(scenario())
    files = list(target.glob("cancelled_*.json"))
    assert len(files) == 1
    kept = json.loads(files[0].read_text())
    assert kept["max_tokens"] == 77
    assert kept["messages"][0]["content"] == "hi"


def test_nothing_is_written_for_a_cancelled_request_without_the_setting(tmp_path, monkeypatch):
    monkeypatch.delenv("EDGELLM_LOG_CANCELLED_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    llm = _BlockingLLM(tmp_path)
    app = _create_app(llm)

    async def scenario():
        conn = _Connection(app, body=_chat_body())
        conn.start()
        assert await _within(llm.started.is_set, 5.0)
        conn.disconnect()
        await asyncio.wait_for(conn.task, 5.0)

    _run(scenario())
    assert not list(tmp_path.rglob("cancelled_*.json"))


def test_a_request_that_completes_leaves_no_file_even_with_the_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGELLM_LOG_CANCELLED_DIR", str(tmp_path / "kept"))
    llm = _BlockingLLM(tmp_path)
    llm.block = False
    app = _create_app(llm)

    async def scenario():
        conn = _Connection(app, body=_chat_body())
        conn.start()
        await asyncio.wait_for(conn.task, 5.0)
        assert conn.status == 200
        conn.disconnect()

    _run(scenario())
    assert not (tmp_path / "kept").exists()


def _default_max_tokens(tmp_path):
    app = _create_app(_FakeLLM(tmp_path))
    return app.state.openai_serving_chat._default_max_tokens


@pytest.mark.parametrize("value, expected", [
    ("50", 50),
    ("1", 1),
    ("0", None),
    ("-5", None),
    ("many", None),
    ("", None),
    ("12.5", None),
    ("99999999", None),
])
def test_default_max_tokens_cap(tmp_path, monkeypatch, value, expected):
    monkeypatch.delenv("EDGELLM_DEFAULT_MAX_TOKENS", raising=False)
    baseline = _default_max_tokens(tmp_path / "base")
    monkeypatch.setenv("EDGELLM_DEFAULT_MAX_TOKENS", value)
    capped = _default_max_tokens(tmp_path / "capped")
    # an unusable value, or one above the capacity, leaves the default at the KV capacity
    assert capped == (min(baseline, expected) if expected is not None else baseline)


def test_default_max_tokens_cap_applies_only_to_requests_without_max_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGELLM_DEFAULT_MAX_TOKENS", "5")
    llm = _FakeLLM(tmp_path)
    app = _create_app(llm)

    async def scenario():
        for extra, expected in (({}, 5), ({"max_tokens": 100}, 100), ({"max_tokens": 2}, 2)):
            conn = _Connection(app, body=_chat_body(**extra))
            conn.start()
            await asyncio.wait_for(conn.task, 5.0)
            assert conn.status == 200
            assert llm.last_sampling_params.max_tokens == expected
            conn.disconnect()

    _run(scenario())
