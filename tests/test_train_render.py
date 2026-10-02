"""Training-side rendering guards (the v1 run lost every tool-call turn to truncation; this keeps it from recurring)."""
from __future__ import annotations

import json

from ml.llm.common import normalise_for_training


def _tool_example() -> list[dict]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "How much urea for 2 acres of wheat?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "fertilizer_calculator", "arguments": json.dumps({"crop": "wheat", "area": 2, "unit": "acre"})}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "fertilizer_calculator", "content": "x" * 5000},
        {"role": "assistant", "content": "Urea 87 kg, DAP 54 kg."},
    ]


def test_arguments_become_objects_and_tool_output_is_capped():
    out = normalise_for_training(_tool_example(), max_tool_chars=100)
    call = out[2]["tool_calls"][0]["function"]
    assert call["arguments"] == {"crop": "wheat", "area": 2, "unit": "acre"}
    assert len(out[3]["content"]) < 130 and out[3]["content"].endswith("...(truncated)")
    # the input is not mutated
    assert isinstance(_tool_example()[2]["tool_calls"][0]["function"]["arguments"], str)


def test_bad_argument_json_is_kept_not_dropped():
    msgs = _tool_example()
    msgs[2]["tool_calls"][0]["function"]["arguments"] = "{not json"
    out = normalise_for_training(msgs)
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"_raw": "{not json"}


def test_plain_messages_pass_through():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "namaste"}]
    assert normalise_for_training(msgs) == msgs
