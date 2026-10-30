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
