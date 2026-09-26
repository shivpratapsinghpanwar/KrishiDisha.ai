"""Machine-translate a slice of the SFT data into regional languages with IndicTrans2 (Kaggle/Colab GPU).

    pip install IndicTransToolkit sentencepiece sacrebleu
    python -m ml.llm.translate_sft --inp data/llm/train.jsonl --out data/llm/translated.jsonl \
        --langs mr bn te ta --per-lang 2500 --model ai4bharat/indictrans2-en-indic-1B

Only the *user* and *assistant text* turns are translated; tool calls, tool results and the system prompt
stay in English (that is what the model will see at serving time, because the language layer translates
the farmer's message to English before the tools run and translates the reply back afterwards). Numbers,
units, product grades, links and tool JSON are masked with ``krishidisha.services.lang.mask`` so the
translator cannot mangle them. A round-trip chrF check on a sample drops sentences that translate badly.

Output examples carry ``source: translated``, ``language: <code>`` and ``meta.origin_id``.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

from .common import DATA_DIR, example_id, read_jsonl, write_jsonl

FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva", "pa": "pan_Guru", "gu": "guj_Gujr", "bn": "ben_Beng",
          "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym", "or": "ory_Orya"}


class IT2:
    def __init__(self, model: str, device: str = "cuda"):
        import torch
        from IndicTransToolkit.processor import IndicProcessor  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, trust_remote_code=True,
                                                           torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()
        self.ip = IndicProcessor(inference=True)

    def translate(self, sentences: list[str], src: str, tgt: str, batch: int = 32) -> list[str]:
        import torch

        out: list[str] = []
        for i in range(0, len(sentences), batch):
            chunk = sentences[i:i + batch]
            pre = self.ip.preprocess_batch(chunk, src_lang=FLORES[src], tgt_lang=FLORES[tgt])
            with torch.no_grad():
                enc = self.tok(pre, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(self.device)
                gen = self.model.generate(**enc, max_length=256, num_beams=4, num_return_sequences=1)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out += self.ip.postprocess_batch(dec, lang=FLORES[tgt])
        return out


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+|\n", text) if s.strip()]


def translate_text(it2: IT2, text: str, tgt: str) -> str:
    """Translate a markdown-ish text line by line, keeping bullets/tables and masked spans."""
    from krishidisha.services.lang import mask

    lines = text.split("\n")
    masked = [mask(ln) for ln in lines]
    todo = [(i, m.text) for i, m in enumerate(masked) if m.text.strip() and not m.text.strip().startswith("|--")]
    translated = it2.translate([t for _, t in todo], "en", tgt) if todo else []
    out = list(lines)
    for (i, _), tr in zip(todo, translated):
        prefix = re.match(r"^(\s*(?:[-*]\s+|\d+\.\s+|\|\s*)?)", lines[i]).group(1)
        out[i] = prefix + masked[i].restore(tr).lstrip() if prefix else masked[i].restore(tr)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inp", type=Path, default=DATA_DIR / "train.jsonl")
    p.add_argument("--out", type=Path, default=DATA_DIR / "translated.jsonl")
    p.add_argument("--langs", nargs="+", default=["mr", "bn", "te", "ta"])
    p.add_argument("--per-lang", type=int, default=2500)
    p.add_argument("--model", default="ai4bharat/indictrans2-en-indic-1B")
    p.add_argument("--back-model", default="ai4bharat/indictrans2-indic-en-dist-200M", help="for the round-trip check")
    p.add_argument("--check-sample", type=int, default=200)
    p.add_argument("--min-chrf", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rows = [r for r in read_jsonl(args.inp) if r.get("language") == "en"]
    rng = random.Random(args.seed)
    it2 = IT2(args.model)
    back = None
    out_rows: list[dict] = []
    for lang in args.langs:
        sample = rng.sample(rows, min(args.per_lang, len(rows)))
        print(f"[{lang}] translating {len(sample)} examples", flush=True)
        produced = []
        for k, ex in enumerate(sample):
            msgs = []
            for m in ex["messages"]:
                if m["role"] in ("user", "assistant") and m.get("content") and not m.get("tool_calls") \
                        and not m["content"].startswith("<tool_response>"):
                    msgs.append(dict(m, content=translate_text(it2, m["content"], lang)))
                else:
                    msgs.append(m)
            new = dict(ex, id=example_id("tr", lang, ex["id"]), source="translated", language=lang, messages=msgs,
                       meta=dict(ex.get("meta") or {}, origin_id=ex["id"], origin_source=ex["source"], mt_model=args.model))
            produced.append(new)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(sample)}", flush=True)
        # round-trip quality check on a sample of user turns
        if args.check_sample and produced:
            try:
                import sacrebleu

                back = back or IT2(args.back_model)
                chk = rng.sample(produced, min(args.check_sample, len(produced)))
                src = [next(m["content"] for m in rows_by_id(rows)[c["meta"]["origin_id"]]["messages"] if m["role"] == "user") for c in chk]
                tr = [next(m["content"] for m in c["messages"] if m["role"] == "user") for c in chk]
                rt = back.translate(tr, lang, "en")
                scores = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(rt, src)]
                mean = sum(scores) / len(scores)
                bad = {c["id"] for c, s in zip(chk, scores) if s < args.min_chrf}
                print(f"[{lang}] round-trip chrF mean {mean:.1f}; dropping {len(bad)} of {len(chk)} checked below {args.min_chrf}")
                produced = [c for c in produced if c["id"] not in bad]
            except ImportError:
                print("sacrebleu not installed; skipping the round-trip check")
        out_rows += produced
    write_jsonl(args.out, out_rows)
    print(f"wrote {args.out}: {len(out_rows)} translated examples")
    return 0


def rows_by_id(rows: list[dict]) -> dict[str, dict]:
    global _BY_ID
    try:
        return _BY_ID
    except NameError:
        _BY_ID = {r["id"]: r for r in rows}
        return _BY_ID


if __name__ == "__main__":
    sys.exit(main())
