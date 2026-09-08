"""Tool-call parsing for locally served models and the pesticide safety guard."""
from __future__ import annotations

import json

from krishidisha.services.safety import check_reply, safety_check_tool
from krishidisha.services.toolcalls import extract_tool_calls, format_tool_call, strip_tool_calls

TOOLS = {"fertilizer_calculator", "get_weather", "recommend_crop", "search_products"}


def test_hermes_block_single():
    text = 'Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"place": "Indore", "days": 3}}\n</tool_call>'
    calls = extract_tool_calls(text, TOOLS)
    assert len(calls) == 1 and calls[0].name == "get_weather" and calls[0].arguments == {"place": "Indore", "days": 3}
    assert strip_tool_calls(text) == "Let me check."
    oa = calls[0].as_openai()
    assert oa["type"] == "function" and json.loads(oa["function"]["arguments"])["place"] == "Indore"


def test_hermes_multiple_blocks_and_dedup():
    text = ('<tool_call>{"name": "fertilizer_calculator", "arguments": {"crop": "wheat", "area": 2}}</tool_call>\n'
            '<tool_call>{"name": "get_weather", "arguments": {"place": "Ludhiana"}}</tool_call>\n'
            '<tool_call>{"name": "fertilizer_calculator", "arguments": {"crop": "wheat", "area": 2}}</tool_call>')
    calls = extract_tool_calls(text, TOOLS)
    assert [c.name for c in calls] == ["fertilizer_calculator", "get_weather"]


def test_fenced_json_and_bare_json_fallbacks():
    fenced = 'Calling tool:\n```json\n{"name": "recommend_crop", "arguments": {"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82, "ph": 6.5, "rainfall": 200}}\n```'
    calls = extract_tool_calls(fenced, TOOLS)
    assert len(calls) == 1 and calls[0].arguments["ph"] == 6.5
    bare = '{"name": "search_products", "parameters": {"query": "urea"}}\n'
    calls = extract_tool_calls(bare, TOOLS)
    assert len(calls) == 1 and calls[0].arguments == {"query": "urea"}


def test_unknown_tool_and_plain_text_ignored():
    assert extract_tool_calls('<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>', TOOLS) == []
    assert extract_tool_calls("Namaste! Wheat needs 120 kg N per hectare.", TOOLS) == []
    assert extract_tool_calls("", TOOLS) == []


def test_lenient_json_single_quotes_and_trailing_comma():
    text = "<tool_call>{'name': 'get_weather', 'arguments': {'place': 'Nashik',}}</tool_call>"
    calls = extract_tool_calls(text, TOOLS)
    assert len(calls) == 1 and calls[0].arguments["place"] == "Nashik"


def test_unterminated_block_at_end():
    text = 'Sure.\n<tool_call>\n{"name": "get_weather", "arguments": {"place": "Pune"}}'
    calls = extract_tool_calls(text, TOOLS)
    assert len(calls) == 1 and calls[0].arguments["place"] == "Pune"


def test_format_roundtrip():
    text = format_tool_call("get_weather", {"place": "Indore"})
    calls = extract_tool_calls(text, TOOLS)
    assert calls[0].name == "get_weather" and strip_tool_calls(text) == ""


# ------------------------------------------------------------------ safety
def test_safety_flags_banned_molecule():
    res = check_reply("Spray monocrotophos 36 SL at 1.5 ml per litre against aphids.")
    assert not res["ok"] and res["banned"][0]["molecule"] == "monocrotophos"
    assert "banned or restricted" in res["annotated"]


def test_safety_flags_excessive_dose_but_allows_normal():
    bad = check_reply("Use mancozeb 75 WP at 25 g per litre of water.")
    assert bad["doses"] and bad["doses"][0]["value"] == 25.0 and "Check the dose" in bad["annotated"]
    ok = check_reply("Use mancozeb 75 WP at 2.5 g per litre of water, repeat after 10 days. Apply urea 40 kg per acre.")
    assert ok["ok"] and ok["annotated"].endswith("acre.")


def test_safety_clean_text_and_tool_shape():
    res = safety_check_tool("Sow wheat by 15 November and irrigate at crown root initiation.")
    assert res == {"ok": True, "flags": [], "banned": [], "doses": []}
