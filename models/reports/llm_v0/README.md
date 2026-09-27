# Assistant v0 (Qwen2.5-3B-Instruct + QLoRA), 2026-09-27

First end-to-end run of the own-LLM pipeline. Trained on Kaggle (T4 x2, Unsloth, 4-bit, r=16) on the 3,595
examples that existed before teacher distillation: knowledge-base pairs (EN/HI/Hinglish) + KisanVaani Q&A.
No tool-call trajectories were in this set yet.

| | |
|---|---|
| Examples / epochs | 3,595 / 2 |
| Train loss | 2.33 -> 0.67 |
| Eval loss (300 held-out) | 0.384 |
| Train time | 43 min |
| Export | q4_k_m 1.93 GB, q8_0 3.29 GB (llama.cpp), private |
| Local speed (RTX 2050, q4_k_m via Ollama) | 15-30 tokens/s |

What v0 does well: answers KB-covered questions in the asked language (Hindi in Devanagari, Hinglish
romanised, English) with the right crop facts and the "confirm with your KVK" habit.

What v0 does not do yet (expected, and what the distilled data is for):
- does not emit `<tool_call>` blocks: no tool trajectories in the training set, so a dose question is
  answered from memory instead of via `fertilizer_calculator`;
- degenerate repetition on out-of-distribution Hindi questions (disease symptoms), because the KB pairs
  are templated and short;
- occasional wrong crop/stage mix-ups in numbers.

v1 will add the teacher trajectories (Groq gpt-oss-120b, Gemini, OpenRouter, Ollama cloud pool) after
`dedup_filter`, then be scored on the frozen 500-example eval set against the rules bot and the base model.

## Eval harness, 100 frozen prompts through the real assistant (2026-09-27)

| System | Tool match | Language ok | Safety flags/ex | Latency s |
|---|---|---|---|---|
| rules bot | 83% | 36% | 0.02 | 0.11 |
| v0 (ollama:krishidisha) | 100%* | 67% | 0.02 | 7.9 |

*The frozen set has almost no tool-call references yet, so "tool match" is not informative until the
distilled trajectories add them. Language compliance by language: en 34/34, hinglish 29/34, **hi 4/32**.

Root cause of the Hindi failure is the training data, not the model: 85% of the "Hindi" KB-pair targets
(587 rows) were English facts inside a Hindi sentence frame (median Devanagari share 0.14). Fix in v1: those
answers are rewritten into real Hindi by the teacher pool (`distill rewrite --target-language hi`), and
`dedup_filter` now drops any hi-tagged example whose answer is under 50% Devanagari.
