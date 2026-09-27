"""The <tool_call> / <tool_response> text format the assistant is trained on and served with.

Kept in a module with no Flask or training imports so both ``krishidisha.services.toolcalls`` (serving) and
the LLM training-data renderer can use it.
"""
from __future__ import annotations

import json
from typing import Any


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Render a call in the training/serving format."""
    body = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return "<tool_call>" + chr(10) + body + chr(10) + "</tool_call>"


def format_tool_response(name: str, content: str) -> str:
    body = "{" + f"\"name\": {json.dumps(name)}, \"content\": {json.dumps(content, ensure_ascii=False)}" + "}"
    return "<tool_response>" + chr(10) + body + chr(10) + "</tool_response>"
