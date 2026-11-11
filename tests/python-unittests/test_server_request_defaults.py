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
"""Server-wide defaults for request fields that the client leaves out."""

import pytest

from experimental.server.api.protocol import ChatCompletionRequest

_ENV = ("EDGELLM_DEFAULT_TEMPERATURE", "EDGELLM_DEFAULT_TOP_P",
        "EDGELLM_DEFAULT_TOP_K", "EDGELLM_DEFAULT_ENABLE_THINKING",
        "EDGELLM_DEFAULT_PRESENCE_PENALTY", "EDGELLM_DEFAULT_REASONING_BUDGET",
        "EDGELLM_DEFAULT_REASONING_BUDGET_MESSAGE")


def _request(**fields):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}],
                                 **fields)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def test_built_in_defaults_without_environment():
    request = _request()
    assert (request.temperature, request.top_p, request.top_k) == (0.7, 0.9, 50)
    assert request.enable_thinking is False


def test_environment_defaults_apply_when_the_client_is_silent(monkeypatch):
    monkeypatch.setenv("EDGELLM_DEFAULT_TEMPERATURE", "0.6")
    monkeypatch.setenv("EDGELLM_DEFAULT_TOP_P", "0.95")
    monkeypatch.setenv("EDGELLM_DEFAULT_TOP_K", "20")
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", "1")
    request = _request()
    assert (request.temperature, request.top_p, request.top_k) == (0.6, 0.95, 20)
    assert request.enable_thinking is True


def test_client_values_win_over_the_environment(monkeypatch):
    monkeypatch.setenv("EDGELLM_DEFAULT_TEMPERATURE", "0.6")
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", "1")
    request = _request(temperature=0.0, top_p=0.5, top_k=5, enable_thinking=False)
    assert (request.temperature, request.top_p, request.top_k) == (0.0, 0.5, 5)
    assert request.enable_thinking is False


def test_template_kwarg_overrides_the_thinking_default(monkeypatch):
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", "1")
    request = _request(chat_template_kwargs={"enable_thinking": False})
    assert request.enable_thinking is False


@pytest.mark.parametrize("value", ["abc", "1.5x", " ", ""])
def test_unparsable_or_empty_values_fall_back(monkeypatch, value):
    monkeypatch.setenv("EDGELLM_DEFAULT_TEMPERATURE", value)
    monkeypatch.setenv("EDGELLM_DEFAULT_TOP_K", value)
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", value)
    request = _request()
    assert request.temperature == 0.7
    assert request.top_k == 50
    assert request.enable_thinking is False


@pytest.mark.parametrize("name,value,field,expected", [
    ("EDGELLM_DEFAULT_TEMPERATURE", "3.5", "temperature", 0.7),
    ("EDGELLM_DEFAULT_TEMPERATURE", "-0.1", "temperature", 0.7),
    ("EDGELLM_DEFAULT_TOP_P", "0", "top_p", 0.9),
    ("EDGELLM_DEFAULT_TOP_P", "1.5", "top_p", 0.9),
    ("EDGELLM_DEFAULT_TOP_K", "-2", "top_k", 50),
])
def test_out_of_range_environment_defaults_fall_back(monkeypatch, name, value, field, expected):
    monkeypatch.setenv(name, value)
    assert getattr(_request(), field) == expected


def test_zero_temperature_default_is_greedy(monkeypatch):
    monkeypatch.setenv("EDGELLM_DEFAULT_TEMPERATURE", "0")
    assert _request().temperature == 0.0


def test_presence_penalty_default_comes_from_the_environment(monkeypatch):
    assert _request().presence_penalty == 0.0
    monkeypatch.setenv("EDGELLM_DEFAULT_PRESENCE_PENALTY", "1.5")
    assert _request().presence_penalty == 1.5
    assert _request(presence_penalty=0.5).presence_penalty == 0.5


@pytest.mark.parametrize("value", ["3", "-2.5", "x", ""])
def test_invalid_presence_penalty_default_falls_back(monkeypatch, value):
    monkeypatch.setenv("EDGELLM_DEFAULT_PRESENCE_PENALTY", value)
    assert _request().presence_penalty == 0.0


def test_reasoning_budget_default_comes_from_the_environment(monkeypatch):
    assert _request().reasoning_budget_tokens == -1
    assert _request().reasoning_budget_message == ""
    monkeypatch.setenv("EDGELLM_DEFAULT_REASONING_BUDGET", "8192")
    monkeypatch.setenv("EDGELLM_DEFAULT_REASONING_BUDGET_MESSAGE", "Time is up.")
    assert _request().reasoning_budget_tokens == 8192
    assert _request().reasoning_budget_message == "Time is up."
    assert _request(reasoning_budget_tokens=0).reasoning_budget_tokens == 0


@pytest.mark.parametrize("value", ["-2", "x", ""])
def test_invalid_reasoning_budget_default_falls_back(monkeypatch, value):
    monkeypatch.setenv("EDGELLM_DEFAULT_REASONING_BUDGET", value)
    assert _request().reasoning_budget_tokens == -1


def _req(**extra):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **extra)


@pytest.mark.parametrize("extra", [
    {"reasoning_effort": "none"},
    {"reasoning_effort": "off"},
    {"enable_thinking": False},
    {"chat_template_kwargs": {"enable_thinking": False}},
    {"thinking": {"type": "disabled"}},
    {"reasoning": {"enabled": False}},
    {"reasoning": {"effort": "none"}},
    {"reasoning": "off"},
])
def test_every_spelling_of_reasoning_off_turns_thinking_off(monkeypatch, extra):
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", "1")
    request = _req(**extra)
    assert request.enable_thinking is False
    assert request.reasoning_effort is None


@pytest.mark.parametrize("extra,effort", [
    ({"reasoning_effort": "low"}, "low"),
    ({"reasoning_effort": "high"}, "xhigh"),
    ({"reasoning": {"effort": "medium"}}, "medium"),
    ({"reasoning": "high"}, "xhigh"),
    ({"chat_template_kwargs": {"reasoning_effort": "low"}}, "low"),
])
def test_a_reasoning_level_turns_thinking_on_even_when_the_default_is_off(monkeypatch, extra, effort):
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", "0")
    request = _req(**extra)
    assert request.enable_thinking is True
    assert request.reasoning_effort == effort


@pytest.mark.parametrize("extra", [{"thinking": {"type": "enabled"}}, {"reasoning": {"enabled": True}}])
def test_thinking_enabled_without_a_level_turns_thinking_on(monkeypatch, extra):
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", "0")
    assert _req(**extra).enable_thinking is True


def test_explicit_enable_thinking_false_beats_a_level(monkeypatch):
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", "1")
    request = _req(enable_thinking=False, reasoning_effort="high")
    assert request.enable_thinking is False


def test_a_silent_client_gets_the_server_default(monkeypatch):
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", "0")
    assert _req().enable_thinking is False
    monkeypatch.setenv("EDGELLM_DEFAULT_ENABLE_THINKING", "1")
    assert _req().enable_thinking is True


def test_thinking_budget_tokens_become_the_reasoning_budget(monkeypatch):
    assert _req(thinking={"type": "enabled", "budget_tokens": 2048}).reasoning_budget_tokens == 2048
    assert _req(reasoning={"effort": "low", "max_tokens": 512}).reasoning_budget_tokens == 512
    assert _req(thinking={"type": "enabled", "budget_tokens": 2048}, reasoning_budget_tokens=64).reasoning_budget_tokens == 64
