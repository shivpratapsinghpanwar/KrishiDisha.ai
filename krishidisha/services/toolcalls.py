"""Model-agnostic tool-call parsing for locally served LLMs.

Qwen, Gemma, Llama and most fine-tuned small models emit tool calls as text in the Hermes format::

    <tool_call>
    {"name": "fertilizer_calculator", "arguments": {"crop": "wheat", "area": 2}}
    </tool_call>

Ollama / vLLM parse these into ``tool_calls`` only when the served model's chat template supports it.
When it does not (or the template is wrong), the call arrives as plain text. :func:`extract_tool_calls`
recovers such calls so ``AgriAssistant._chat_openai`` can execute them anyway; :func:`strip_tool_calls`
removes the blocks from the visible reply. The same format is what ``ml/llm`` training data uses, so
fine-tuned models are consistent whatever the serving stack does.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OPEN_ONLY = re.compile(r"<tool_call>\s*(\{.*\})\s*$", re.DOTALL)          # unterminated block at the end
_FENCED = re.compile(r"```(?:json|tool_call)?\s*(\{\s*\"name\"\s*:.*?\})\s*```", re.DOTALL)
_TAGGED_JSON = re.compile(r"(?:^|\n)\s*(\{\s*\"name\"\s*:\s*\"[a-zA-Z0-9_]+\"\s*,\s*\"(?:arguments|parameters)\"\s*:\s*\{.*?\}\s*\})\s*(?:\n|$)", re.DOTALL)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")

    def as_openai(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)}}


def _loads_lenient(text: str) -> dict | None:
    text = text.strip()
    no_trailing = re.sub(r",\s*([}\]])", r"\1", text)
    for candidate in (text, no_trailing, text.replace("'", '"'), no_trailing.replace("'", '"')):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


def _to_call(obj: dict, known_tools: set[str] | None) -> ParsedToolCall | None:
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    if not isinstance(name, str):
        return None
    args = obj.get("arguments", obj.get("parameters", obj.get("input", {})))
    if isinstance(args, str):
        args = _loads_lenient(args) or {}
    if not isinstance(args, dict):
        args = {}
    if known_tools is not None and name not in known_tools:
        return None
    return ParsedToolCall(name=name, arguments=args)


def extract_tool_calls(text: str, known_tools: set[str] | None = None) -> list[ParsedToolCall]:
    """Return every tool call found in ``text`` (Hermes tags, fenced JSON or bare JSON lines)."""
    if not text or "name" not in text:
        return []
    calls: list[ParsedToolCall] = []
    seen: set[str] = set()
    for pattern in (_BLOCK, _OPEN_ONLY, _FENCED, _TAGGED_JSON):
        for m in pattern.finditer(text):
            obj = _loads_lenient(m.group(1))
            if not obj:
                continue
            call = _to_call(obj, known_tools)
            if call is None:
                continue
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            if key in seen:
                continue
            seen.add(key)
            calls.append(call)
        if calls and pattern is _BLOCK:
            break  # proper blocks found; do not double-parse the same JSON via looser patterns
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove tool-call blocks (and dangling tags) from a reply meant for the user."""
    if not text:
        return text
    out = _BLOCK.sub("", text)
    out = _OPEN_ONLY.sub("", out)
    out = _FENCED.sub("", out)
    out = _TAGGED_JSON.sub("\n", out)
    out = re.sub(r"</?tool_call>", "", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Render a call in the training/serving format."""
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}\n</tool_call>"


def format_tool_response(name: str, content: str) -> str:
    return f"<tool_response>\n{{\"name\": {json.dumps(name)}, \"content\": {json.dumps(content, ensure_ascii=False)}}}\n</tool_response>"
