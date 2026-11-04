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
"""Keep control-token strings in message content from becoming control tokens.

The rendered chat prompt is tokenised with added tokens parsed atomically, so a
tool result or user text that contains ``<|im_start|>`` or ``<think>`` would
become the real token: forged turns, a closed or reopened thinking block, or an
image placeholder with no image behind it. A zero-width space after the first
character makes the tokenizer see ordinary text.
"""

import os
import re

_ZERO_WIDTH_SPACE = "​"

_CONTROL_TOKEN = re.compile(
    r"<\|[^\s|<>]{1,40}\|>"
    r"|</?(?:think|tool_call|tool_response)>"
    r"|<(?:bos|eos|pad|start_of_turn|end_of_turn)>")


def defang_enabled() -> bool:
    return os.environ.get("EDGELLM_ALLOW_CONTROL_TOKENS_IN_CONTENT", "0") in ("", "0")


def defang(text: str) -> str:
    return _CONTROL_TOKEN.sub(lambda m: m.group(0)[0] + _ZERO_WIDTH_SPACE + m.group(0)[1:], text)
