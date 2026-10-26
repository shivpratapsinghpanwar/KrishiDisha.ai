# Assistant v1 (Qwen2.5-3B-Instruct + QLoRA), 2026-10-02

Second run of the own-LLM pipeline, trained on Kaggle (kernel v10, T4 x2, Unsloth, 4-bit base, LoRA r=16,
lr 2e-4, 2 epochs, 370 steps, 40 min). The difference from v0 is the data: the v1 SFT set adds the first
teacher-distilled tool trajectories, the real-Hindi rewrites of the knowledge-base pairs, and the
tool-call templates, after `dedup_filter` (36 rows dropped by the safety gate, 147 duplicates).

| | v0 (2026-09-27) | v1 (2026-10-02) |
|---|---|---|
| Training examples | 3,595 | 4,417 |
| Sources | KB pairs (EN/HI/Hinglish) + KisanVaani | KB pairs 1,432 (376 real-Hindi rewrites) + KisanVaani 1,517 + distilled trajectories 994 + tool templates ~470 |
| Languages (train) | en / hi / hinglish | en 2,852 · hinglish 691 · hi 682 · bn 73 · ta 48 · te 37 · mr 34 |
| Tool-call examples | 0 | ~1,460 (distilled + templates) |
| Train loss | 2.33 -> 0.67 | 2.29 -> 0.87 (mean 1.09) |
| Eval loss | 0.384 on the 300-row v0 eval | 0.743 at step 200 -> **0.607** at step 370 on the 606-row v1 eval |
| Export | q4_k_m 1.93 GB, q8_0 3.29 GB | q4_k_m 1.93 GB, q8_0 3.29 GB (private, see README "Model weights") |
| Ollama tag | `krishidisha-v0` | `krishidisha` (`models/llm/krishidisha-v1-q4_k_m.gguf`) |

The two eval losses are **not comparable**: the v1 eval set (`data/llm/eval.jsonl`, 606 rows) keeps the
500 v0 prompts but rewrites 125 Hindi references into real Hindi and adds a 120-example tool-trajectory
slice (157 rows carry tool calls), which is harder than the templated v0 set.

Local artefacts (gitignored): `models/llm/adapter-qwen-v1/` (LoRA adapter, 120 MB),
`models/llm/krishidisha-v1-q4_k_m.gguf`; the v0 files were renamed to `krishidisha-v0-*.gguf` and the
v0 Ollama model is kept as `krishidisha-v0` for side-by-side evaluation.

## Frozen-set evaluation (606 examples, no judge)

Run with `uv run python -m ml.llm.eval_llm --systems rules ollama:krishidisha-v0 ollama:krishidisha`
through the real `AgriAssistant` (same tools, retrieval and safety guard as the app).

See `models/reports/llm_v2/README.md` for the completed comparison (rules bot, v1, v2 on the 606-set, run on Kaggle on 2026-10-15): v1 calls the exact tool set on 1.3% of tool examples (v2: 86.6%), confirming the truncation diagnosis; v1 keeps the best language compliance (85.5%, Hinglish 68.8%).

## Sample generations

See `sample_generations.md` (12 held-out prompts generated right after training). The answers keep
the v0 strengths (crop facts in the asked language, the "confirm with your KVK" habit). Whether the
tool trajectories taught the model to emit `<tool_call>` blocks is what the tool-match column of the
frozen-set evaluation measures.

## Root cause: v1 never saw its tool-call turns (found 2026-10-02)

Teacher-forcing v1 on one of its own training examples shows it cannot even continue
`<tool_call>
{"name": "` - it learned nothing from the 574 tool examples. The rendered system prompt plus the
13-tool JSON schema block is **~2,500 tokens**, and `train_llm.py` ran with `--max-seq 2048`: `SFTTrainer`
truncates the tail, so every tool example lost its tool call, tool result and final answer (median rendered
length 3,040 tokens, p90 4,817). The same cut removed the answer turn of the 894 no-tool rows that carry the
tools field. v1 therefore trained on the 2,949 plain rows only - it is v0 plus the real-Hindi rewrites.

Fixed for v2 in the same commit:
- `--max-seq` default 6144 (p98 of the normalised data is 4,829 tokens; 0 rows over 6,144), batch 1 x accum 16;
- a guard that aborts training when more than 2 % of examples would be truncated;
- `normalise_for_training`: tool-call `arguments` rendered as JSON objects (the dataset stores OpenAI-style
  strings, which Qwen's template `tojson`-escaped into a quoted string target), tool outputs capped at 1,800 chars;
- serving: the app now sends a KrishiDisha fine-tune exactly the training prompt (compact system prompt, bare
  question - no retrieval passages, profile or language tags) via `LLM_TRAINED_FORMAT=auto`.

## Known gaps carried into v2

- Distillation rounds 1/2 are incomplete (r0 2,904/2,904, r1 566, r2 143): about 800 tool questions still
  lack their final assistant turn. `dedup_filter` keeps only complete trajectories, so just 172 distilled
  tool conversations (plus ~400 templated ones) made it into the v1/v2 training set. One more editor-started
  Kaggle distillation session (Model Proxy quota) would roughly triple the real tool examples for v3.
- Regional languages other than Hindi/Hinglish have under 80 examples each.
- No LLM-as-judge scores yet (needs an `ANTHROPIC_API_KEY` for `--judge`).
