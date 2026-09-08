"""Merge every data source, deduplicate, tag languages, apply the safety filter and freeze the splits.

    python -m ml.llm.dedup_filter --inputs data/llm/kb_pairs.jsonl data/llm/kisanvaani.jsonl data/llm/distill.jsonl \
        --train data/llm/train.jsonl --eval data/llm/eval.jsonl --eval-size 600

Steps
-----
1. MinHash-LSH near-duplicate removal over (user turn + assistant turn), 5-gram shingles, Jaccard > 0.8
   (``datasketch``; falls back to exact normalised-text dedup when it is not installed).
2. Language tag per example: keeps the source's tag if present, else a cheap script/keyword detector
   (Devanagari -> hi, other Indic scripts by Unicode block, romanised Hindi keywords -> hinglish, else en).
3. Safety filter: drop any assistant turn that names a banned/restricted pesticide (``services/safety.py``)
   or contains a rupee amount for a scheme that does not appear in ``government_schemes.json``.
4. Frozen, stratified eval split (by source x language) written once; later runs keep the same eval ids
   (``--keep-eval`` reads the existing eval file and excludes those ids from train).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .common import DATA_DIR, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

SCRIPT_LANG = {"DEVANAGARI": "hi", "GURMUKHI": "pa", "GUJARATI": "gu", "BENGALI": "bn", "TAMIL": "ta", "TELUGU": "te",
               "KANNADA": "kn", "MALAYALAM": "ml", "ORIYA": "or"}
HINGLISH_WORDS = {"kya", "hai", "hain", "kaise", "kab", "kitna", "kitni", "mein", "me", "ke", "ki", "ka", "liye", "fasal",
                  "khad", "khet", "kheti", "beej", "paani", "dawa", "keeda", "rog", "bimari", "batao", "bataye", "chahiye",
                  "karein", "karna", "lagta", "lagega", "kaun", "kaunsi", "acha", "accha", "aur", "nahi", "bhi"}


def detect_language(text: str) -> str:
    counts: Counter = Counter()
    for ch in text:
        if ch.isalpha():
            try:
                name = unicodedata.name(ch, "")
            except ValueError:
                continue
            block = name.split(" ")[0]
            counts[SCRIPT_LANG.get(block, "latin")] += 1
    if counts:
        top, n = counts.most_common(1)[0]
        if top != "latin" and n >= 3:
            return top
    words = re.findall(r"[a-z]+", text.lower())
    if words and sum(w in HINGLISH_WORDS for w in words) / len(words) >= 0.12:
        return "hinglish"
    return "en"


def _user_assistant_text(ex: dict) -> tuple[str, str]:
    user = " ".join(m.get("content", "") for m in ex["messages"] if m["role"] == "user" and not m.get("content", "").startswith("<tool_response>"))
    assistant = " ".join(m.get("content", "") for m in ex["messages"] if m["role"] == "assistant" and m.get("content"))
    return user, assistant


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9ऀ-෿ ]+", " ", text.lower()).strip()


def dedupe(examples: list[dict], threshold: float = 0.8) -> tuple[list[dict], int]:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log.warning("datasketch not installed; exact dedup only")
        seen, kept = set(), []
        for ex in examples:
            key = _normalise(" ".join(_user_assistant_text(ex)))[:400]
            if key in seen:
                continue
            seen.add(key)
            kept.append(ex)
        return kept, len(examples) - len(kept)

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept: list[dict] = []
    for i, ex in enumerate(examples):
        text = _normalise(" ".join(_user_assistant_text(ex)))
        tokens = text.split()
        shingles = {" ".join(tokens[j:j + 5]) for j in range(max(len(tokens) - 4, 1))} or {text}
        m = MinHash(num_perm=128)
        for s in shingles:
            m.update(s.encode("utf-8"))
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        kept.append(ex)
    return kept, len(examples) - len(kept)


def safety_ok(ex: dict, scheme_amounts: set[str]) -> bool:
    from krishidisha.services.safety import check_reply

    _, assistant = _user_assistant_text(ex)
    res = check_reply(assistant)
    if res["banned"]:
        return False
    # scheme figures: any "Rs 6,000"-style amount near the word scheme/yojana must be a known amount
    if re.search(r"\b(scheme|yojana|pm-?kisan|kcc|pmfby)\b", assistant, re.I):
        for amt in re.findall(r"(?:rs\.?|₹|inr)\s*([\d,]{3,})", assistant, re.I):
            if amt.replace(",", "") not in scheme_amounts:
                return False
    return True


def load_scheme_amounts(path: Path = Path("data/knowledge/government_schemes.json")) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    return {a.replace(",", "") for a in re.findall(r"(?:Rs\.?|₹|INR)\s*([\d,]{3,})", text)}


def stratified_eval(examples: list[dict], size: int, seed: int, keep_ids: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    if keep_ids:
        ev = [e for e in examples if e["id"] in keep_ids]
        tr = [e for e in examples if e["id"] not in keep_ids]
        return tr, ev
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in examples:
        groups[(e["source"].split("_")[0], e["language"])].append(e)
    per_group = max(1, size // max(len(groups), 1))
    ev_ids: set[str] = set()
    for items in groups.values():
        rng.shuffle(items)
        for e in items[:per_group]:
            ev_ids.add(e["id"])
    # top up to the requested size from the largest groups
    if len(ev_ids) < size:
        pool = [e for e in examples if e["id"] not in ev_ids]
        rng.shuffle(pool)
        for e in pool[: size - len(ev_ids)]:
            ev_ids.add(e["id"])
    return [e for e in examples if e["id"] not in ev_ids], [e for e in examples if e["id"] in ev_ids]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    p.add_argument("--eval", type=Path, default=DATA_DIR / "eval.jsonl")
    p.add_argument("--eval-size", type=int, default=600)
    p.add_argument("--keep-eval", action="store_true", help="reuse the ids in the existing --eval file")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-raw", action="store_true", help="drop kcc_raw rows that were not rewritten by the teacher")
    args = p.parse_args(argv)

    examples: list[dict] = []
    for path in args.inputs:
        if path.exists():
            rows = read_jsonl(path)
            print(f"{path}: {len(rows)}")
            examples += rows
    if args.drop_raw:
        examples = [e for e in examples if not (e.get("meta") or {}).get("raw")]
    n0 = len(examples)
    for e in examples:
        if not e.get("language"):
            user, _ = _user_assistant_text(e)
            e["language"] = detect_language(user)
    amounts = load_scheme_amounts()
    examples = [e for e in examples if safety_ok(e, amounts)]
    n_safety = n0 - len(examples)
    examples, n_dup = dedupe(examples, args.threshold)

    keep_ids = {e["id"] for e in read_jsonl(args.eval)} if args.keep_eval and args.eval.exists() else None
    train, ev = stratified_eval(examples, args.eval_size, args.seed, keep_ids)
    write_jsonl(args.train, train)
    write_jsonl(args.eval, ev)
    summary = {"input": n0, "dropped_safety": n_safety, "dropped_duplicates": n_dup, "train": len(train), "eval": len(ev),
               "by_source": dict(Counter(e["source"] for e in train)), "by_language": dict(Counter(e["language"] for e in train)),
               "eval_by_language": dict(Counter(e["language"] for e in ev))}
    (args.train.parent / "dataset_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
