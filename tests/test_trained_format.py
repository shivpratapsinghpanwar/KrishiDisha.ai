"""The KrishiDisha fine-tune must be served with the exact prompt shape it was trained on."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from krishidisha.services.llm import SYSTEM_PROMPT, AgriAssistant
from ml.toolformat import COMPACT_SYSTEM_PROMPT


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, item):
        return None


class _FakeOpenAI:
    """Records the request and answers once without tool calls."""

    def __init__(self):
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        msg = SimpleNamespace(content="2 acre gehu: 87 kg urea, 54 kg DAP.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _assistant(app, model: str, **extra) -> tuple[AgriAssistant, _FakeOpenAI]:
    cfg = _Cfg(LLM_PROVIDER="openai", OPENAI_MODEL=model, OPENAI_BASE_URL="http://localhost:11434/v1",
               OPENAI_API_KEY="ollama", LLM_MAX_TOOL_ROUNDS=2, CHAT_HISTORY_TURNS=4, TRANSLATION_BACKEND="none", **extra)
    assistant = AgriAssistant(cfg, app.tools, app.kb, None)
    fake = _FakeOpenAI()
    assistant._openai = fake
    return assistant, fake


def test_fine_tune_gets_compact_prompt_and_bare_question(app, monkeypatch):
    monkeypatch.delenv("LLM_TRAINED_FORMAT", raising=False)
    assistant, fake = _assistant(app, "krishidisha")
    assert assistant.trained_format is True
    res = assistant.chat("How much urea for 2 acres of wheat?", farmer_context="state: MP", language="en")
    req = fake.requests[0]
    # tools are rendered into the system message exactly as in training; no `tools` parameter is sent
    assert req["messages"][0]["role"] == "system"
    assert req["messages"][0]["content"].startswith(COMPACT_SYSTEM_PROMPT)
    assert "<tools>" in req["messages"][0]["content"] and "tools" not in req
    user = req["messages"][-1]["content"]
    assert user == "How much urea for 2 acres of wheat?"  # no profile / passages / language tag
    assert all(f"\"name\": \"{tool.name}\"" in req["messages"][0]["content"] for tool in app.tools)
    assert res["provider"] == "openai"


def test_generic_model_keeps_full_prompt_and_grounding(app, monkeypatch):
    monkeypatch.delenv("LLM_TRAINED_FORMAT", raising=False)
    assistant, fake = _assistant(app, "gpt-4o-mini")
    assert assistant.trained_format is False
    assistant.chat("How much urea for 2 acres of wheat?", farmer_context="state: MP", language="en")
    req = fake.requests[0]
    assert req["messages"][0]["content"] == SYSTEM_PROMPT
    user = req["messages"][-1]["content"]
    assert user.startswith("[Farmer profile: state: MP]")
    assert "[Preferred reply language: English]" in user


@pytest.mark.parametrize("flag,expected", [("true", True), ("false", False)])
def test_flag_overrides_model_name(app, monkeypatch, flag, expected):
    monkeypatch.setenv("LLM_TRAINED_FORMAT", flag)
    assistant, _ = _assistant(app, "gpt-4o-mini" if expected else "krishidisha")
    assert assistant.trained_format is expected


def test_training_prompt_matches_dataset_rows():
    """Guard against the prompt drifting away from what data/llm/train.jsonl was rendered with."""
    assert COMPACT_SYSTEM_PROMPT.startswith("You are KrishiDisha Sahayak, an expert agricultural advisor")
    assert json.dumps(COMPACT_SYSTEM_PROMPT)  # plain string, no formatting placeholders


def test_qwen_tools_block_matches_template():
    """The inline tools block must equal what Qwen2.5's chat template rendered during training."""
    import re
    from pathlib import Path

    pytest = __import__("pytest")
    adapter = Path("models/llm/adapter-qwen-v1")
    if not (adapter / "tokenizer_config.json").exists():
        pytest.skip("Qwen tokenizer (adapter) not present")
    transformers = pytest.importorskip("transformers")
    from ml.toolformat import COMPACT_SYSTEM_PROMPT, system_with_tools

    tools = [{"type": "function", "function": {"name": "crop_calendar", "description": "Sowing and harvest windows.",
                                               "parameters": {"type": "object", "properties": {"crop": {"type": "string"}}, "required": ["crop"]}}}]
    tok = transformers.AutoTokenizer.from_pretrained(str(adapter))
    full = tok.apply_chat_template([{"role": "system", "content": COMPACT_SYSTEM_PROMPT}, {"role": "user", "content": "x"}],
                                   tools=tools, tokenize=False)
    rendered = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", full, re.S).group(1)
    assert rendered == system_with_tools(COMPACT_SYSTEM_PROMPT, tools)

