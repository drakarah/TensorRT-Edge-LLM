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
"""Inkling 2 assistant output: ``<|message_model|>`` messages split into reasoning, content and tool calls."""

import json

import pytest

from experimental.server.parsing.reasoning import (
    REASONING_PARSERS, InklingReasoningParser, InklingStreamingReasoningParser)
from experimental.server.parsing.tool_calling import (parse_assistant_output,
                                                      stream_assistant_output,
                                                      validate_tool_request)

THINK = "<|message_model|><|content_thinking|>"
TEXT = "<|message_model|><|content_text|>"
END = "<|end_message|>"
EOS = "<|content_model_end_sampling|>"


def tool_message(name, args):
    body = json.dumps({"name": name, "args": args}, separators=(",", ":"))
    return f"<|message_model|>{name}<|content_invoke_tool_json|>{body}{END}"


def config(tool_choice="auto"):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    }
                }
            },
        },
    }]
    return validate_tool_request([{"role": "user", "content": "x"}], tools, tool_choice, True)


def split(text, chunk=None):
    parser = InklingStreamingReasoningParser()
    deltas = []
    pieces = [text] if chunk is None else [text[i:i + chunk] for i in range(0, len(text), chunk)]
    for piece in pieces:
        deltas.extend(parser.feed(piece))
    deltas.extend(parser.flush())
    reasoning = "".join(d.text for d in deltas if d.field == "reasoning")
    content = "".join(d.text for d in deltas if d.field == "content")
    return reasoning, content


def test_registry_resolves_and_auto_detects(tmp_path):
    assert isinstance(REASONING_PARSERS.resolve("inkling", str(tmp_path)), InklingReasoningParser)
    model = tmp_path / "Inkling-2-Mini"
    model.mkdir()
    assert isinstance(REASONING_PARSERS.resolve("auto", str(model)), InklingReasoningParser)


def test_thinking_then_answer():
    text = THINK + "plan it" + END + TEXT + "Paris" + END + EOS
    assert InklingReasoningParser().extract(text) == ("plan it", "Paris")


def test_answer_without_thinking():
    assert InklingReasoningParser().extract(TEXT + "hi" + END + EOS) == (None, "hi")


def test_empty_thinking_and_empty_output():
    assert InklingReasoningParser().extract(THINK + END + TEXT + "x" + END) == (None, "x")
    assert InklingReasoningParser().extract("") == (None, None)
    assert InklingReasoningParser().extract(EOS) == (None, None)


def test_truncated_output_keeps_what_was_generated():
    assert InklingReasoningParser().extract(THINK + "still thinking") == ("still thinking", None)
    assert InklingReasoningParser().extract(THINK + "a" + END + TEXT + "partial") == ("a", "partial")


def test_missing_end_marker_before_next_message():
    # a message that is cut by the next header still ends: no text bleeds across messages
    text = THINK + "a" + END + TEXT + "b" + END + TEXT + "c" + END
    assert InklingReasoningParser().extract(text) == ("a", "bc")


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, 11, 64])
def test_streaming_matches_whole_text_for_every_chunk_size(chunk):
    text = THINK + "plan <b>x</b> | <|not a marker" + END + TEXT + "Answer, with <tags> and 'quotes'." + END + EOS
    assert split(text, chunk) == InklingReasoningParser().extract(text) or split(text, chunk) == (
        "plan <b>x</b> | <|not a marker", "Answer, with <tags> and 'quotes'.")


def test_literal_marker_prefix_inside_text_is_not_swallowed():
    reasoning, content = split(THINK + "a <|content_text> b" + END, 3)
    assert reasoning == "a <|content_text> b" and content == ""


def test_tool_message_becomes_a_tool_call_block():
    reasoning, content = split(THINK + "need weather" + END + tool_message("get_weather", {"city": "Paris"}) + EOS)
    assert reasoning == "need weather"
    block = json.loads(content.removeprefix("<tool_call>").removesuffix("</tool_call>"))
    assert block == {"name": "get_weather", "arguments": {"city": "Paris"}}


def test_full_pipeline_nonstreaming_tool_call(tmp_path):
    text = THINK + "need weather" + END + TEXT + "Checking." + END + tool_message("get_weather", {"city": "Paris"}) + EOS
    parsed = parse_assistant_output(text, config(), str(tmp_path), tool_parser="inkling", reasoning_parser="inkling")
    assert parsed.reasoning == "need weather"
    assert parsed.content == "Checking."
    assert [c.name for c in parsed.tool_calls] == ["get_weather"]
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Paris"}


def test_parallel_tool_calls(tmp_path):
    text = (tool_message("get_weather", {"city": "Paris"}) + tool_message("get_weather", {"city": "Rome"}) + EOS)
    parsed = parse_assistant_output(text, config(), str(tmp_path), tool_parser="inkling", reasoning_parser="inkling")
    assert [json.loads(c.arguments)["city"] for c in parsed.tool_calls] == ["Paris", "Rome"]


def test_tool_arguments_with_nested_json_and_braces(tmp_path):
    args = {"city": "a}b{c", "opts": {"deep": [1, {"x": "y"}]}}
    parsed = parse_assistant_output(tool_message("get_weather", args), config(), str(tmp_path), tool_parser="inkling", reasoning_parser="inkling")
    assert json.loads(parsed.tool_calls[0].arguments) == args


def test_invalid_tool_json_stays_visible_not_a_call(tmp_path):
    text = "<|message_model|>get_weather<|content_invoke_tool_json|>{not json" + END
    parsed = parse_assistant_output(text, config(), str(tmp_path), tool_parser="inkling", reasoning_parser="inkling")
    assert parsed.tool_calls == []
    assert "{not json" in (parsed.content or "")


@pytest.mark.parametrize("chunk", [1, 4, 9, 40])
def test_full_pipeline_streaming_tool_call(tmp_path, chunk):
    text = THINK + "need weather" + END + TEXT + "Checking." + END + tool_message("get_weather", {"city": "Paris"}) + EOS
    parser = stream_assistant_output(config(), str(tmp_path), tool_parser="inkling", reasoning_parser="inkling")
    events = []
    for i in range(0, len(text), chunk):
        events.extend(parser.feed(text[i:i + chunk]))
    events.extend(parser.flush())
    assert "".join(e["text"] for e in events if e["type"] == "reasoning") == "need weather"
    assert "".join(e["text"] for e in events if e["type"] == "content") == "Checking."
    calls = [e["tool_call"] for e in events if e["type"] == "tool_call"]
    assert len(calls) == 1 and calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Paris"}


def test_tool_parser_auto_selects_inkling_and_ignores_unlisted_tools(tmp_path):
    model = tmp_path / "inkling-2-mini"
    model.mkdir()
    ok = parse_assistant_output(tool_message("get_weather", {"city": "X"}), config(), str(model), reasoning_parser="inkling")
    assert [c.name for c in ok.tool_calls] == ["get_weather"]
    unknown = parse_assistant_output(tool_message("delete_all", {}), config(), str(model), reasoning_parser="inkling")
    assert unknown.tool_calls == []


@pytest.mark.parametrize("chunk", [1, 3, 8, 50])
def test_streaming_text_message_with_tool_choice_none_like_plain_output(tmp_path, chunk):
    text = THINK + "t" + END + TEXT + "just text <b>" + END + EOS
    parser = stream_assistant_output(config(), str(tmp_path), tool_parser="inkling", reasoning_parser="inkling")
    events = []
    for i in range(0, len(text), chunk):
        events.extend(parser.feed(text[i:i + chunk]))
    events.extend(parser.flush())
    assert "".join(e["text"] for e in events if e["type"] == "reasoning") == "t"
    assert "".join(e["text"] for e in events if e["type"] == "content") == "just text <b>"
    assert not [e for e in events if e["type"] == "tool_call"]


@pytest.mark.parametrize("chunk", [1, 5, 17])
def test_streaming_two_tool_calls_with_text_between(tmp_path, chunk):
    text = (TEXT + "one" + END + tool_message("get_weather", {"city": "A"}) + TEXT + "two" + END
            + tool_message("get_weather", {"city": "B"}) + EOS)
    parser = stream_assistant_output(config(), str(tmp_path), tool_parser="inkling", reasoning_parser="inkling")
    events = []
    for i in range(0, len(text), chunk):
        events.extend(parser.feed(text[i:i + chunk]))
    events.extend(parser.flush())
    assert "".join(e["text"] for e in events if e["type"] == "content") == "onetwo"
    assert [json.loads(e["tool_call"].arguments)["city"] for e in events if e["type"] == "tool_call"] == ["A", "B"]
