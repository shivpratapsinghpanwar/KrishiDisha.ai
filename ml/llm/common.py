"""Shared helpers for the LLM data pipeline: app context, tool schemas, chat-JSONL I/O, Hermes rendering."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

DATA_DIR = Path("data/llm")
LANGS = ["en", "hi", "hinglish", "mr", "pa", "gu", "bn", "ta", "te", "kn", "ml", "or"]

# Short production system prompt used for training data; must match what the app sends at serving
# time (see krishidisha/services/llm.py SYSTEM_PROMPT) so the fine-tuned model sees the same context.
COMPACT_SYSTEM_PROMPT = (
    "You are KrishiDisha Sahayak, an expert agricultural advisor for Indian farmers. Be practical and specific "
    "(doses in kg/acre or ml/litre, timings, product names, INR costs). Use the tools for numbers, forecasts, prices "
    "and catalogue facts instead of guessing; ask briefly for missing inputs. Give prevention and an organic/IPM "
    "option first, remind about safety gear and pre-harvest interval for sprays, never recommend banned pesticides. "
    "Reply in the farmer's language. Keep answers under 250 words with bullets or small tables."
)


@contextmanager
def app_context(offline: bool = True) -> Iterator[Any]:
    """Yield the Flask app with services attached (rules assistant, stub disease model by default)."""
    if offline:
        os.environ.setdefault("LLM_PROVIDER", "rules")
        os.environ.setdefault("DISEASE_MODEL_BACKEND", "stub")
        os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    from krishidisha import create_app

    app = create_app()
    with app.app_context():
        yield app


def openai_tool_schemas(app) -> list[dict]:
    return [t.openai() for t in app.tools]


def tool_by_name(app, name: str):
    return next(t for t in app.tools if t.name == name)


def run_tool_json(app, name: str, args: dict) -> tuple[dict | list | str, bool]:
    from krishidisha.services.tools import run_tool

    out, is_err = run_tool(app.tools, name, args)
    try:
        return json.loads(out), is_err
    except ValueError:
        return out, is_err


# ------------------------------------------------------------------ JSONL
def example_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_example(source: str, language: str, messages: list[dict], tools: list[dict] | None = None,
                 meta: dict | None = None, system: str = COMPACT_SYSTEM_PROMPT) -> dict:
    msgs = [{"role": "system", "content": system}] + messages
    ex = {"id": example_id(source, language, json.dumps(messages, ensure_ascii=False, sort_keys=True)),
          "source": source, "language": language, "messages": msgs}
    if tools:
        ex["tools"] = tools
    if meta:
        ex["meta"] = meta
    return ex


def assistant_tool_call(name: str, arguments: dict, call_id: str | None = None) -> dict:
    """Assistant turn that only calls a tool (OpenAI shape)."""
    call_id = call_id or f"call_{example_id(name, json.dumps(arguments, sort_keys=True))[:10]}"
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}


def tool_result_turn(call: dict, content: Any) -> dict:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": call["tool_calls"][0]["id"], "name": call["tool_calls"][0]["function"]["name"],
            "content": text[:6000]}


# ------------------------------------------------------------------ Hermes rendering
def render_hermes(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Convert OpenAI-shaped messages into plain text turns with <tool_call>/<tool_response> blocks.

    Used for models whose chat template has no native tool support (Gemma) and as a fallback for
    any template: every assistant tool call becomes text, every tool result becomes a user turn.
    """
    from krishidisha.services.toolcalls import format_tool_call, format_tool_response

    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system" and tools:
            tool_desc = "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools)
            out.append({"role": "system", "content": m["content"] + "\n\n# Tools\nYou may call these functions by "
                        "writing <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>:\n" + tool_desc})
        elif role == "assistant" and m.get("tool_calls"):
            blocks = [format_tool_call(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
                      for c in m["tool_calls"]]
            content = (m.get("content") or "").strip()
            out.append({"role": "assistant", "content": (content + "\n" if content else "") + "\n".join(blocks)})
        elif role == "tool":
            out.append({"role": "user", "content": format_tool_response(m.get("name", "tool"), m["content"])})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    # merge consecutive user turns (tool responses after a user message) for templates that require alternation
    merged: list[dict] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] == "user":
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


def seeded(seed: int = 42) -> random.Random:
    return random.Random(seed)
