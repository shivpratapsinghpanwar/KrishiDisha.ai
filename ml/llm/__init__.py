"""KrishiDisha's own assistant model: data building, fine-tuning, export and evaluation.

Data pipeline (all outputs under ``data/llm/``, gitignored except the frozen eval set):

1. ``build_kb_pairs``   - deterministic instruction pairs + tool trajectories from the project's
                          knowledge base and tools (free, ~8-10k examples)
2. ``public_data``      - Kisan Call Centre transcripts (data.gov.in) and open agri Q&A sets
3. ``distill``          - Claude-generated trajectories through the real AgriAssistant (Batch API)
4. ``translate_sft``    - IndicTrans2 translations for regional languages (Kaggle GPU)
5. ``dedup_filter``     - MinHash dedup, language tags, safety filter, frozen eval split
6. ``train_llm``        - QLoRA (Unsloth + TRL) on Kaggle/Colab
7. ``export_gguf``      - merge -> GGUF -> Ollama Modelfile
8. ``eval_llm``         - tool-selection accuracy, judge scores, per-language table

Every example is one JSON line in the OpenAI chat format::

    {"id": "...", "source": "kb|kcc|distill|translated|feedback", "language": "en|hi|hinglish|mr|...",
     "tools": [<openai tool schemas>], "messages": [{"role": "system"|"user"|"assistant"|"tool", ...}]}

Assistant tool calls are stored *both* as ``tool_calls`` (OpenAI shape) and rendered into the
assistant ``content`` as Hermes ``<tool_call>`` blocks by ``common.render_hermes`` at training time,
so the same file serves Qwen (native template) and Gemma (text tool calls).
"""
