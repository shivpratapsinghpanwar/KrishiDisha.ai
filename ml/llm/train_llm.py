"""QLoRA fine-tuning of the KrishiDisha assistant (runs on a Kaggle / Colab T4; not on the 4 GB laptop).

    pip install "unsloth[colab-new]" trl peft bitsandbytes datasets
    python -m ml.llm.train_llm --data data/llm/train.jsonl --eval data/llm/eval.jsonl \
        --base unsloth/Qwen2.5-3B-Instruct-bnb-4bit --out /kaggle/working/kd-qwen --epochs 2 --resume
    python -m ml.llm.train_llm ... --base unsloth/gemma-3-4b-it-unsloth-bnb-4bit --out /kaggle/working/kd-gemma

Two candidate bases, same data:
* Qwen2.5-3B-Instruct  - native Hermes tool template (``apply_chat_template(..., tools=...)``), Apache-2.0
* Gemma-3-4B-it        - broadest Indic coverage; no tool template, so tool calls are rendered as text with
                         ``common.render_hermes`` and the same ``<tool_call>`` blocks are learnt

Loss is computed on assistant turns only. Checkpoints every ``--save-steps`` so a dead session resumes.
Outputs: LoRA adapter in ``<out>/adapter``, ``train_log.json``, and ``sample_generations.md`` on 12 eval prompts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .common import read_jsonl, render_hermes

FAMILY = {"qwen": {"instruction": "<|im_start|>user\n", "response": "<|im_start|>assistant\n", "native_tools": True},
          "gemma": {"instruction": "<start_of_turn>user\n", "response": "<start_of_turn>model\n", "native_tools": False},
          "llama": {"instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
                    "response": "<|start_header_id|>assistant<|end_header_id|>\n\n", "native_tools": True}}


def family_of(base: str) -> str:
    b = base.lower()
    return "qwen" if "qwen" in b else "gemma" if "gemma" in b else "llama" if "llama" in b else "qwen"


def render_example(tokenizer, ex: dict, fam: dict) -> str:
    messages, tools = ex["messages"], ex.get("tools")
    if tools and fam["native_tools"]:
        try:
            return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
        except Exception:  # noqa: BLE001 - template without tool support after all
            pass
    plain = render_hermes(messages, tools)
    # Gemma has no system role: fold it into the first user turn
    if fam is FAMILY["gemma"] and plain and plain[0]["role"] == "system":
        sys_text = plain.pop(0)["content"]
        if plain and plain[0]["role"] == "user":
            plain[0]["content"] = sys_text + "\n\n" + plain[0]["content"]
    return tokenizer.apply_chat_template(plain, tokenize=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--eval", type=Path)
    p.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="use only the first N examples (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    fam = FAMILY[family_of(args.base)]
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.base, max_seq_length=args.max_seq,
                                                         load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    texts = [render_example(tokenizer, ex, fam) for ex in rows]
    lengths = [len(tokenizer(t).input_ids) for t in texts[:500]]
    print(f"{len(texts)} examples; token length p50={sorted(lengths)[len(lengths) // 2]} max={max(lengths)} (first 500)")
    train_ds = Dataset.from_list([{"text": t} for t in texts]).shuffle(seed=args.seed)
    eval_ds = None
    if args.eval and args.eval.exists():
        eval_ds = Dataset.from_list([{"text": render_example(tokenizer, ex, fam)} for ex in read_jsonl(args.eval)[:300]])

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = SFTConfig(
        output_dir=str(args.out), per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_steps=args.save_steps, save_total_limit=2, eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps, bf16=bf16, fp16=not bf16, optim="adamw_8bit", weight_decay=0.01, seed=args.seed,
        max_seq_length=args.max_seq, dataset_text_field="text", packing=False, report_to="none")
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=eval_ds, args=cfg)
    trainer = train_on_responses_only(trainer, instruction_part=fam["instruction"], response_part=fam["response"])

    t0 = time.time()
    resume = args.resume and any(Path(args.out).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume)
    adapter_dir = args.out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (args.out / "train_log.json").write_text(json.dumps({
        "base": args.base, "family": family_of(args.base), "examples": len(texts), "epochs": args.epochs, "lr": args.lr,
        "rank": args.rank, "seconds": round(time.time() - t0), "log_history": trainer.state.log_history[-50:]}, indent=1))

    # a few sample generations for a quick sanity check
    FastLanguageModel.for_inference(model)
    samples = read_jsonl(args.eval)[:12] if args.eval and args.eval.exists() else rows[:6]
    lines = ["# Sample generations", ""]
    for ex in samples:
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        prompt_msgs = [m for m in ex["messages"] if m["role"] in ("system", "user")][:2]
        if fam is FAMILY["gemma"] and prompt_msgs[0]["role"] == "system":
            prompt_msgs = [{"role": "user", "content": prompt_msgs[0]["content"] + "\n\n" + prompt_msgs[1]["content"]}]
        ids = tokenizer.apply_chat_template(prompt_msgs, tools=ex.get("tools") if fam["native_tools"] else None,
                                            add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=300, temperature=0.3, do_sample=True)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        lines += [f"**User ({ex.get('language')}):** {user[:300]}", "", f"**Model:** {text[:900]}", "", "---", ""]
    (args.out / "sample_generations.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"adapter saved to {adapter_dir}; {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
