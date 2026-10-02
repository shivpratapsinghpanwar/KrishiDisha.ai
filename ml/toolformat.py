"""The <tool_call> / <tool_response> text format the assistant is trained on and served with.

Kept in a module with no Flask or training imports so both ``krishidisha.services.toolcalls`` (serving) and
the LLM training-data renderer can use it.
"""
from __future__ import annotations

import json
from typing import Any

# The short production system prompt every fine-tuning example is rendered with. The app sends exactly this
# prompt (and the bare question, no retrieval passages or tags) when it serves a KrishiDisha fine-tune, so the
# model sees the same context at inference as in training.
COMPACT_SYSTEM_PROMPT = (
    "You are KrishiDisha Sahayak, an expert agricultural advisor for Indian farmers. Be practical and specific "
    "(doses in kg/acre or ml/litre, timings, product names, INR costs). Use the tools for numbers, forecasts, prices "
    "and catalogue facts instead of guessing; ask briefly for missing inputs. Give prevention and an organic/IPM "
    "option first, remind about safety gear and pre-harvest interval for sprays, never recommend banned pesticides. "
    "Reply in the farmer's language. Keep answers under 250 words with bullets or small tables."
)


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Render a call in the training/serving format."""
    body = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return "<tool_call>" + chr(10) + body + chr(10) + "</tool_call>"


def format_tool_response(name: str, content: str) -> str:
    body = "{" + f"\"name\": {json.dumps(name)}, \"content\": {json.dumps(content, ensure_ascii=False)}" + "}"
    return "<tool_response>" + chr(10) + body + chr(10) + "</tool_response>"
