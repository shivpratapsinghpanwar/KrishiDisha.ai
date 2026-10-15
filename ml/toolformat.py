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


QWEN_TOOLS_HEAD = (
    "# Tools" + chr(10) + chr(10) +
    "You may call one or more functions to assist with the user query." + chr(10) + chr(10) +
    "You are provided with function signatures within <tools></tools> XML tags:" + chr(10) + "<tools>" + chr(10))
QWEN_TOOLS_TAIL = (
    chr(10) + "</tools>" + chr(10) + chr(10) +
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:" + chr(10) +
    "<tool_call>" + chr(10) + '{"name": <function-name>, "arguments": <args-json-object>}' + chr(10) + "</tool_call>")


def qwen_tools_block(tools: list[dict]) -> str:
    """The tools section exactly as Qwen2.5's chat template renders it (Jinja ``tojson`` keeps key order).

    Serving a KrishiDisha fine-tune through Ollama with the OpenAI ``tools`` parameter makes Ollama parse the
    model's ``<tool_call>`` block itself, and Ollama 0.34 drops it (empty reply, no tool_calls). Rendering the
    same block into the system message and sending no ``tools`` parameter gives the model the prompt it was
    trained on and returns the ``<tool_call>`` text verbatim for ``services.toolcalls.extract_tool_calls``.
    """
    body = chr(10).join(json.dumps(t, ensure_ascii=False) for t in tools)
    return QWEN_TOOLS_HEAD + body + QWEN_TOOLS_TAIL


def system_with_tools(system_prompt: str, tools: list[dict]) -> str:
    return system_prompt + chr(10) + chr(10) + qwen_tools_block(tools) if tools else system_prompt

