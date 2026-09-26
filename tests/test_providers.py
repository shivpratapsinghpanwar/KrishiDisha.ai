"""Teacher-provider adapters: Anthropic-shaped requests <-> OpenAI chat format (no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from ml.llm.providers import anthropic_to_openai_messages, anthropic_tools_to_openai, openai_to_anthropic_message


def test_request_conversion_covers_system_tools_and_tool_results():
    system = [{"type": "text", "text": "You are KrishiDisha.", "cache_control": {"type": "ephemeral"}}]
    messages = [
        {"role": "user", "content": "urea for 2 acres of wheat?"},
        {"role": "assistant", "content": [SimpleNamespace(type="text", text="Let me calculate."),
                                          SimpleNamespace(type="tool_use", id="t1", name="fertilizer_calculator", input={"crop": "wheat", "area": 2})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"fertilizers_kg": {"Urea": 170}}'}]},
    ]
    out = anthropic_to_openai_messages(system, messages)
    assert out[0] == {"role": "system", "content": "You are KrishiDisha."}
    assert out[1]["role"] == "user"
    assert out[2]["role"] == "assistant" and out[2]["tool_calls"][0]["function"]["name"] == "fertilizer_calculator"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"crop": "wheat", "area": 2}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"fertilizers_kg": {"Urea": 170}}'}
    tools = anthropic_tools_to_openai([{"name": "get_weather", "description": "forecast", "input_schema": {"type": "object", "properties": {}}}])
    assert tools[0]["function"]["name"] == "get_weather" and tools[0]["type"] == "function"


def test_response_conversion_marks_tool_use_and_usage():
    msg = SimpleNamespace(content="", tool_calls=[SimpleNamespace(id="c9", function=SimpleNamespace(name="get_weather", arguments='{"place": "Indore"}'))])
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30)
    m = openai_to_anthropic_message(msg, usage, "gemini-2.5-flash")
    assert m.stop_reason == "tool_use" and m.content[0].type == "tool_use" and m.content[0].input == {"place": "Indore"}
    assert m.usage.input_tokens == 120 and m.usage.output_tokens == 30
    plain = openai_to_anthropic_message(SimpleNamespace(content="Sow by 15 Nov.", tool_calls=None), usage, "x")
    assert plain.stop_reason == "end_turn" and plain.content[0].text == "Sow by 15 Nov."
