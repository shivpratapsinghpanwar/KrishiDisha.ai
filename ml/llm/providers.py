"""Teacher-model providers for ``distill.py``.

``AnthropicBatches``  - Message Batches API (50 % off, async). The stages were written against Anthropic's
                         response shape (content blocks with ``.type``/``.text``/``.input``, ``stop_reason``, ``usage``).
``OpenAICompatible``  - any OpenAI-style chat endpoint, called sequentially with rate limiting and retries, and
                         its responses converted into the same Anthropic-like shape so the stages need no changes.
                         Used for Gemini's free tier (``GEMINI_API_KEY``, endpoint
                         ``https://generativelanguage.googleapis.com/v1beta/openai/``), Groq, OpenRouter or a local
                         Ollama/vLLM model.

Both expose ``run_batch(requests, label) -> {custom_id: result}`` where ``result.type`` is ``"succeeded"`` /
``"errored"`` and ``result.message`` mimics ``anthropic.types.Message``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

log = logging.getLogger(__name__)

KNOWN_BASES = {"gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "groq": "https://api.groq.com/openai/v1",
               "openrouter": "https://openrouter.ai/api/v1",
               "ollama": "http://localhost:11434/v1"}
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ------------------------------------------------------------------ Anthropic
class AnthropicBatches:
    name = "anthropic"

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=3, timeout=120.0)

    def run_batch(self, requests_: list[dict], label: str, poll: int = 30) -> dict[str, Any]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        batch = self.client.messages.batches.create(requests=[Request(custom_id=r["custom_id"],
                                                                      params=MessageCreateParamsNonStreaming(**r["params"]))
                                                              for r in requests_])
        print(f"[{label}] submitted batch {batch.id} with {len(requests_)} requests")
        while True:
            b = self.client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            c = b.request_counts
            done = c.succeeded + c.errored + c.expired + c.canceled
            print(f"  {batch.id}: {b.processing_status} ({done}/{done + c.processing})", flush=True)
            time.sleep(poll)
        return {res.custom_id: res.result for res in self.client.messages.batches.results(batch.id)}


# ------------------------------------------------------------------ OpenAI-compatible
def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(getattr(b, "text", "") if not isinstance(b, dict) else b.get("text", "")
                   for b in content if (getattr(b, "type", None) or (isinstance(b, dict) and b.get("type"))) == "text")


def anthropic_to_openai_messages(system, messages: list[dict]) -> list[dict]:
    """Anthropic-shaped request (system blocks, tool_use/tool_result content) -> OpenAI chat messages."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": _text_of(system) if not isinstance(system, str) else system})
    for m in messages:
        role, content = m["role"], m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        blocks = [b if isinstance(b, dict) else {"type": getattr(b, "type", None), "text": getattr(b, "text", None),
                                                  "id": getattr(b, "id", None), "name": getattr(b, "name", None),
                                                  "input": getattr(b, "input", None)} for b in content]
        if role == "assistant":
            text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"] or {}, ensure_ascii=False)}}
                     for b in blocks if b.get("type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # user: text and/or tool_result blocks
            results = [b for b in blocks if b.get("type") == "tool_result"]
            for b in results:
                out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": _text_of(b.get("content", ""))})
            text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
            if text:
                out.append({"role": "user", "content": text})
    return out


def anthropic_tools_to_openai(tools: list[dict] | None) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t.get("input_schema", {"type": "object", "properties": {}})}}
            for t in (tools or [])]


def openai_to_anthropic_message(choice_message, usage, model: str):
    blocks = []
    if choice_message.content:
        blocks.append(SimpleNamespace(type="text", text=choice_message.content))
    for c in getattr(choice_message, "tool_calls", None) or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except ValueError:
            args = {}
        blocks.append(SimpleNamespace(type="tool_use", id=c.id, name=c.function.name, input=args))
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    u = SimpleNamespace(input_tokens=getattr(usage, "prompt_tokens", 0) or 0, output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=blocks, stop_reason=stop, usage=u, model=model)


class OpenAICompatible:
    """Sequential calls with a requests-per-minute budget; free tiers are slow but cost nothing."""
    name = "openai"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, rpm: float = 10.0, max_retries: int = 40,
                 model: str | None = None, label: str | None = None, stop_on_daily_quota: bool = False):
        from openai import OpenAI

        self.model = model                      # when set, overrides params["model"] (multi-provider runs)
        self.label = label or "openai"
        self.stop_on_daily_quota = stop_on_daily_quota

        base_url = base_url or os.getenv("TEACHER_BASE_URL") or (GEMINI_OPENAI_BASE if os.getenv("GEMINI_API_KEY") else os.getenv("OPENAI_BASE_URL"))
        api_key = api_key or os.getenv("TEACHER_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or "none"
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0, max_retries=0)
        self.min_interval = 60.0 / max(rpm, 0.1)
        self.max_retries = max_retries
        self._last = 0.0
        self.base_url = base_url

    def _one(self, params: dict):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        messages = anthropic_to_openai_messages(params.get("system"), params["messages"])
        kwargs: dict[str, Any] = {"model": self.model or params["model"], "messages": messages, "max_tokens": params.get("max_tokens", 1024),
                                  "temperature": params.get("temperature", 0.7)}
        tools = anthropic_tools_to_openai(params.get("tools"))
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._last = time.time()
                return openai_to_anthropic_message(resp.choices[0].message, resp.usage, kwargs["model"])
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                low = msg.lower()
                # per-minute limits clear in seconds; per-day quota ("PerDay", "daily") needs a long pause
                daily = "day" in low or "daily" in low
                retry_after = min(60 * (attempt + 1), 900) if daily else min(20 * (attempt + 1), 180)
                if "429" in msg or "rate" in low or "quota" in low or "503" in msg or "overloaded" in low or "resource" in low:
                    if daily and self.stop_on_daily_quota:
                        raise DailyQuotaExhausted(f"{self.label}: {msg[:160]}")
                    quota = re.findall(r"quotaId': '([^']*)'", msg)
                    hint = re.findall(r"retryDelay': '(\d+)", msg)
                    if hint:  # honour the server's own delay when it gives one (plus a little slack)
                        retry_after = max(retry_after if quota and "Day" in quota[0] else 0, int(hint[0]) + 5)
                    print(f"  provider throttled ({quota[0] if quota else msg[:90]}); sleeping {retry_after}s", flush=True)
                    time.sleep(retry_after)
                    continue
                if "tool" in msg.lower() and tools:
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    continue
                raise
        raise RuntimeError("provider kept throttling")

    def run_batch(self, requests_: list[dict], label: str, poll: int = 0) -> dict[str, Any]:
        """Sequential requests with an append-only cache so an interrupted run resumes where it stopped.

        Cache: ``data/llm/.cache/<label>.jsonl`` (one line per succeeded request, keyed by custom_id).
        Delete the file to force a fresh run.
        """
        cache_path = _batch_cache_path(label)
        cached = _load_batch_cache(cache_path)
        out: dict[str, Any] = {cid: SimpleNamespace(type="succeeded", message=_message_from_json(m))
                               for cid, m in cached.items() if any(r["custom_id"] == cid for r in requests_)}
        todo = [r for r in requests_ if r["custom_id"] not in out]
        print(f"[{label}] {len(todo)} sequential requests via {self.base_url} (~{self.min_interval:.0f}s apart)"
              + (f"; {len(out)} restored from {cache_path}" if out else ""))
        t0 = time.time()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(todo):
            try:
                message = self._one(r["params"])
                out[r["custom_id"]] = SimpleNamespace(type="succeeded", message=message)
                with cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"custom_id": r["custom_id"], "message": _message_to_json(message)}, ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001
                log.warning("%s failed: %s", r["custom_id"], exc)
                out[r["custom_id"]] = SimpleNamespace(type="errored", error=str(exc)[:200])
                msg = str(exc)
                if i == 0 and ("404" in msg or "not found" in msg.lower() or "no longer available" in msg.lower()):
                    raise SystemExit(f"model {r['params'].get('model')} is not available on this endpoint: {msg[:160]}")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)} done ({time.time() - t0:.0f}s)", flush=True)
        return out


def _batch_cache_path(label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return Path(os.getenv("KRISHIDISHA_LLM_CACHE", "data/llm/.cache")) / f"{safe}.jsonl"


def _load_batch_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rows[d["custom_id"]] = d["message"]
            except (ValueError, KeyError):
                continue  # a partially written last line from a killed process
    return rows


def _message_to_json(m) -> dict:
    blocks = []
    for b in m.content:
        if b.type == "text":
            blocks.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    u = m.usage
    return {"content": blocks, "stop_reason": m.stop_reason, "model": m.model,
            "usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                      "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0),
                      "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0)}}


def _message_from_json(d: dict):
    blocks = [SimpleNamespace(**b) for b in d["content"]]
    return SimpleNamespace(content=blocks, stop_reason=d["stop_reason"], model=d.get("model"),
                           usage=SimpleNamespace(**d["usage"]))


class DailyQuotaExhausted(RuntimeError):
    """Raised by OpenAICompatible(stop_on_daily_quota=True) so a multi-provider run can hand the request to another API."""


class MultiProvider:
    """Several OpenAI-compatible providers draining one request queue concurrently (one thread each).

    Each worker honours its own requests-per-minute budget and uses its own model; a worker that hits its
    daily quota puts the request back and exits, so the remaining providers finish the batch. Results are
    cached per request exactly like OpenAICompatible.run_batch (shared cache file, guarded by a lock).
    """
    name = "multi"

    def __init__(self, providers: list[OpenAICompatible]):
        assert providers, "MultiProvider needs at least one provider"
        self.providers = providers
        self.base_url = " + ".join(f"{p.label}:{p.model or 'default'}" for p in providers)

    def run_batch(self, requests_: list[dict], label: str, poll: int = 0) -> dict[str, Any]:
        import queue
        import threading

        cache_path = _batch_cache_path(label)
        cached = _load_batch_cache(cache_path)
        wanted = {r["custom_id"] for r in requests_}
        out: dict[str, Any] = {cid: SimpleNamespace(type="succeeded", message=_message_from_json(m))
                               for cid, m in cached.items() if cid in wanted}
        todo = [r for r in requests_ if r["custom_id"] not in out]
        print(f"[{label}] {len(todo)} requests across {len(self.providers)} providers ({self.base_url})"
              + (f"; {len(out)} restored from {cache_path}" if out else ""), flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        q: queue.Queue = queue.Queue()
        for r in todo:
            q.put(r)
        lock = threading.Lock()
        done_count = {"n": 0}
        t0 = time.time()
        per_provider: dict[str, int] = {p.label: 0 for p in self.providers}

        def worker(p: OpenAICompatible) -> None:
            while True:
                try:
                    r = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    message = p._one(r["params"])
                except DailyQuotaExhausted as exc:
                    q.put(r)
                    print(f"  {p.label}: daily quota reached, leaving the pool ({str(exc)[:100]})", flush=True)
                    return
                except SystemExit as exc:  # model not found on this endpoint: this provider is misconfigured
                    q.put(r)
                    print(f"  {p.label}: {exc}; leaving the pool", flush=True)
                    return
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s failed on %s: %s", r["custom_id"], p.label, exc)
                    with lock:
                        out[r["custom_id"]] = SimpleNamespace(type="errored", error=str(exc)[:200])
                    continue
                with lock:
                    out[r["custom_id"]] = SimpleNamespace(type="succeeded", message=message)
                    with cache_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"custom_id": r["custom_id"], "message": _message_to_json(message)}, ensure_ascii=False) + chr(10))
                    done_count["n"] += 1
                    per_provider[p.label] += 1
                    if done_count["n"] % 25 == 0:
                        print(f"  {done_count['n']}/{len(todo)} done ({time.time() - t0:.0f}s) {per_provider}", flush=True)

        threads = [threading.Thread(target=worker, args=(p,), daemon=True, name=p.label) for p in self.providers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        left = q.qsize()
        if left:
            print(f"  {left} requests left undone: every provider hit its daily quota; rerun tomorrow (cache resumes)", flush=True)
            while not q.empty():
                r = q.get_nowait()
                out[r["custom_id"]] = SimpleNamespace(type="errored", error="daily quota exhausted on all providers")
        print(f"  finished {done_count['n']} requests in {time.time() - t0:.0f}s: {per_provider}", flush=True)
        return out


FREE_TIER_RPM = {"gemini": 10.0, "groq": 3.0, "openrouter": 4.0, "ollama": 10.0}  # ollama = cloud models via the local server


def _single(name: str, base_url: str | None, rpm: float | None, model: str | None, multi: bool) -> OpenAICompatible:
    base_url = base_url or KNOWN_BASES.get(name)
    api_key = None
    if name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
    elif name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
    elif name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    return OpenAICompatible(base_url=base_url, api_key=api_key, rpm=rpm or FREE_TIER_RPM.get(name, 10.0),
                            model=model, label=name, stop_on_daily_quota=multi)


def make_provider(name: str, dry_run: bool = False, base_url: str | None = None, rpm: float | None = None,
                  models: dict[str, str] | None = None):
    """``name`` may be a comma-separated list ("groq,gemini,openrouter") -> MultiProvider; ``models`` maps each
    provider name to the model it should use (required for a multi run, since one request cannot name them all)."""
    if dry_run:
        return None
    names = [n.strip() for n in name.split(",") if n.strip()]
    if len(names) > 1:
        if "anthropic" in names:
            raise SystemExit("anthropic (Message Batches) cannot be pooled with sequential providers")
        return MultiProvider([_single(n, None, None, (models or {}).get(n), multi=True) for n in names])
    name = names[0]
    if name == "anthropic":
        return AnthropicBatches()
    if name in ("openai", "gemini", "groq", "openrouter", "ollama"):
        return _single(name, base_url, rpm, None, multi=False)
    raise SystemExit(f"unknown provider {name}")
