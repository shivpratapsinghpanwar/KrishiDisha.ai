# Assistant v2 (Qwen2.5-3B-Instruct + QLoRA), 2026-10-02, evaluated 2026-10-15

Same data as v1 (4,417 examples) but trained with the truncation fix (`--max-seq 6144`, 0 of 574 tool examples cut):
2 epochs, 554 steps, 5 h on a T4 x2; train loss 2.29 -> 1.03 (mean), **eval loss 0.394** on the 606-row v1 eval
set (v1: 0.607 on the same set). GGUF q4_k_m 1.93 GB, private; Ollama tag `krishidisha-v2`.

## Frozen-set evaluation (606 examples, Kaggle T4, Ollama in-kernel, greedy decoding, no judge)

| System | Tool match | Tool match (tool rows) | Args valid | Language ok | hi ok | hinglish ok | en ok | Safety flags/ex | Latency s |
|---|---|---|---|---|---|---|---|---|---|
| rules | 77.2% | 46.5% | - | 39.1% | 4.1% | 0.0% | 100.0% | 0.02 | 0.02 |
| ollama:krishidisha-v1 | 74.3% | 1.3% | - | 85.5% | 88.4% | 68.8% | 99.6% | 0.03 | 10.29 |
| ollama:krishidisha-v2 | 57.8% | 86.6% | - | 66.0% | 76.2% | 23.1% | 100.0% | 0.06 | 29.58 |

Tool rows = eval examples whose reference trajectory calls at least one tool (157 of them).

Raw per-example results: `models/reports/llm_eval_v2/*.json`. The "Args valid" column is empty because the assistant does not yet return the parsed tool arguments to the harness; argument validity was not measured.

### What the numbers say

- **v2 learned tool calling; v1 and v0 never did.** On the 157 examples whose reference calls a tool, v2 calls the
  exact tool set 86.6% of the time (v1 1.3%, rules bot 46.5%). Per tool: fertilizer_calculator 47/47,
  predict_yield 26/26, recommend_crop 23/23, crop_calendar 19/19, crop_guide 18/20, government_schemes 9/14,
  get_disease_info 5/7, search_knowledge 0/10, mandi/weather 0/2.
- **v2 over-calls.** On the 449 plain questions it still calls a tool 52% of the time (crop_guide 59, search_knowledge 51,
  crop_calendar 50, get_disease_info 41, search_products 36). The answers are then grounded in the tool output rather
  than memory, which is the behaviour we want for facts, but the harness scores them as misses, and each such reply
  costs a second model turn (35 s mean on plain rows vs 13 s on tool rows; Kaggle T4 inside the kernel with the mandi API timing out, not laptop numbers).
- **Hinglish regressed (23% vs 69% for v1)**, for two reasons that a five-question local probe separates: after a tool
  turn v2 often does switch to English (the PM-Kisan and crop-calendar replies came back fully in English), and the
  harness's keyword heuristic also labels genuinely Hinglish tool-grounded replies as English when they quote English
  tool output ("3 acre cotton ke liye: Urea 212 kg..." scores as `en`). The real part is a data gap: the teacher
  trajectories carried the "reply in Hinglish" instruction only in the teacher prompt, not the training prompt. Hindi
  held at 76% (v1 88%), English 100%. On the same probe, the tool-grounded PM-Kisan answer quoted the correct
  Rs 6,000 per year, where the from-memory answer had fabricated an amount.
- The harness measures tool selection, language and banned-molecule hits only. It does not score factual accuracy:
  probes showed v2 fabricating numbers when it answers from memory (a PM-Kisan amount, "2 acres = 8 hectares"),
  and mapping "gulabi sundi" (pink bollworm) to leaf curl virus before calling get_disease_info. A judge run is the
  next measurement.
- Serving note: Ollama 0.34 drops the model's `<tool_call>` block when the request carries the OpenAI `tools`
  parameter; the app now renders the tools block into the system message (byte-identical to training) and parses
  the call text itself (`ml/toolformat.py`, commit d38520f).

### v3 plan (data, not architecture)

1. Finish distillation rounds 1-2 (cache now r1 864 / r2 264 of ~1,400) so the final assistant turn after each tool
   result is in the training data, with the user's language tagged in the *training* prompt.
2. Add "no tool needed" negatives from the KB pairs with the tools block present, so the model learns when not to call.
3. Canonicalise crop/pest names in tool arguments (Hinglish spellings such as "gehhu") through the glossary.
4. Judge scoring (Kaggle Model Proxy, Gemini 3 Flash) on accuracy and actionability, alongside this harness.
