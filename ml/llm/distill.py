"""Teacher-generated training data through the Anthropic Message Batches API (50 % off, async).

    python -m ml.llm.distill questions    --n 3000 --out data/llm/questions.jsonl          # seed farmer questions
    python -m ml.llm.distill trajectories --questions data/llm/questions.jsonl --out data/llm/distill.jsonl
    python -m ml.llm.distill rewrite      --inp data/llm/kcc.jsonl --out data/llm/kcc_rewritten.jsonl
    python -m ml.llm.distill status       # cost ledger

Every stage:
* is a *batch* (results within 24 h, usually well under an hour), submitted then polled;
* reuses the app's real tool registry (``app.tools``) so tool calls in the data are exactly the ones the
  product supports, and tool results come from the real tools running locally between rounds;
* caches the system prompt + tool schemas (``cache_control``) so per-example input cost is small;
* records token usage and dollars in ``data/llm/cost_ledger.json`` and refuses to submit a batch whose
  estimated cost would push the ledger past ``--max-usd`` (default $100, ~Rs 8k).

Models: ``--model claude-sonnet-5`` for volume, ``--hard-model claude-opus-5`` for the share of examples
tagged hard (multi-tool, safety, regional language). Requires ``ANTHROPIC_API_KEY`` or ``ant auth login``.
Read Anthropic's Commercial Terms before running: outputs may be used to train models for your own
product but not to build a competing general-purpose model.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

from .common import (DATA_DIR, app_context, assistant_tool_call, example_id, make_example, openai_tool_schemas,
                     read_jsonl, run_tool_json, tool_result_turn, write_jsonl)

log = logging.getLogger(__name__)

# Batch prices per million tokens (50 % of list); cache reads 0.1x, cache writes 1.25x of input.
PRICES = {"claude-sonnet-5": (1.0, 5.0), "claude-opus-5": (2.5, 12.5), "claude-haiku-4-5": (0.5, 2.5),
          "claude-sonnet-4-6": (1.5, 7.5)}
LEDGER = DATA_DIR / "cost_ledger.json"

PERSONAS = {
    "en": ["a smallholder in Madhya Pradesh with 2 acres", "a Punjab wheat-rice farmer with 10 acres",
           "a Maharashtra cotton and soybean farmer", "a Tamil Nadu paddy farmer", "a young farmer starting a vegetable plot",
           "a woman farmer growing chickpea and mustard in Rajasthan", "an orchard owner with mango and banana in Andhra",
           "a sugarcane grower in Uttar Pradesh", "a Gujarat groundnut farmer", "a Karnataka ragi and maize farmer"],
    "hi": ["मध्य प्रदेश का 2 एकड़ वाला छोटा किसान", "पंजाब का गेहूं-धान किसान", "राजस्थान की महिला किसान (चना, सरसों)",
           "उत्तर प्रदेश का गन्ना किसान", "बिहार का सब्जी उगाने वाला किसान"],
    "hinglish": ["MP ka 2 acre wala kisan", "UP ka ganna kisan", "Maharashtra ka kapas aur soybean kisan",
                 "Haryana ka gehu kisan", "Bihar ka sabzi kisan"],
    "mr": ["महाराष्ट्रातील कापूस आणि सोयाबीन शेतकरी", "नाशिकचा द्राक्ष उत्पादक"],
    "bn": ["পশ্চিমবঙ্গের ধান চাষি", "একজন সবজি চাষি"],
    "te": ["ఆంధ్రప్రదేశ్ వరి రైతు", "తెలంగాణ పత్తి రైతు"],
    "ta": ["தமிழ்நாட்டு நெல் விவசாயி", "வாழை மற்றும் கரும்பு விவசாயி"],
}
LANG_NAMES = {"en": "English", "hi": "Hindi (Devanagari script)", "hinglish": "Hinglish (Hindi written in Latin letters)",
              "mr": "Marathi", "bn": "Bengali", "te": "Telugu", "ta": "Tamil"}
LANG_MIX = [("en", 0.45), ("hi", 0.25), ("hinglish", 0.10), ("mr", 0.05), ("bn", 0.05), ("te", 0.05), ("ta", 0.05)]
TOPICS = ["crop choice for my soil and season", "fertilizer dose and timing", "a disease I see on leaves", "insect pest damage",
          "when to sow and harvest", "weather this week and what to do", "mandi prices and when to sell", "a government scheme",
          "irrigation scheduling", "seed variety selection", "organic / natural farming inputs", "weed control",
          "soil health card values", "post-harvest storage", "buying inputs (which product and dose)", "crop insurance claim",
          "yield expectation for my field", "intercropping", "nursery raising", "livestock fodder crop"]


def _client(dry_run: bool = False):
    if dry_run:
        return None
    import anthropic

    return anthropic.Anthropic(max_retries=3, timeout=120.0)


# ------------------------------------------------------------------ ledger
def ledger_load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"usd": 0.0, "batches": []}


def ledger_add(entry: dict) -> None:
    led = ledger_load()
    led["usd"] = round(led["usd"] + entry["usd"], 4)
    led["batches"].append(entry)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=1), encoding="utf-8")


def cost_of(model: str, usage) -> float:
    pin, pout = PRICES.get(model, (2.5, 12.5))
    u = usage
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    return (inp * pin + cr * pin * 0.1 + cw * pin * 1.25 + out * pout) / 1e6


def estimate(model: str, n: int, in_tokens: int, out_tokens: int, cached: int = 0) -> float:
    pin, pout = PRICES.get(model, (2.5, 12.5))
    return n * (in_tokens * pin + cached * pin * 0.1 + out_tokens * pout) / 1e6


def guard(max_usd: float, est: float, label: str) -> None:
    spent = ledger_load()["usd"]
    print(f"[{label}] estimated ${est:.2f}; ledger so far ${spent:.2f}; cap ${max_usd:.2f}")
    if spent + est > max_usd:
        raise SystemExit(f"refusing: ${spent:.2f} + ${est:.2f} would exceed --max-usd {max_usd}")


# ------------------------------------------------------------------ batch helpers
def submit_batch(client, requests_: list[dict], label: str) -> str:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch = client.messages.batches.create(requests=[Request(custom_id=r["custom_id"],
                                                             params=MessageCreateParamsNonStreaming(**r["params"]))
                                                     for r in requests_])
    print(f"[{label}] submitted batch {batch.id} with {len(requests_)} requests")
    return batch.id


def wait_batch(client, batch_id: str, poll: int = 30) -> dict:
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        c = b.request_counts
        print(f"  {batch_id}: {b.processing_status} (done {c.succeeded + c.errored + c.expired + c.canceled}/"
              f"{c.processing + c.succeeded + c.errored + c.expired + c.canceled})", flush=True)
        time.sleep(poll)
    results = {}
    for res in client.messages.batches.results(batch_id):
        results[res.custom_id] = res.result
    return results


def _system_blocks(app, system_prompt: str, tools: bool) -> tuple[list[dict], list[dict]]:
    tool_defs = [t.anthropic() for t in app.tools] if tools else []
    if tool_defs:
        tool_defs[-1] = dict(tool_defs[-1], cache_control={"type": "ephemeral"})
    return [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}], tool_defs


# ------------------------------------------------------------------ stage 1: questions
def stage_questions(args) -> int:
    rng = random.Random(args.seed)
    client = _client(args.dry_run)
    with app_context() as app:
        crops = sorted(app.kb.crop_guides)
        pests = [p["name"] for p in app.kb.pests]
        schemes = [s["name"] for s in app.kb.schemes]
    requests_ = []
    per_call = 8
    n_calls = max(1, args.n // per_call)
    for i in range(n_calls):
        lang = rng.choices([l for l, _ in LANG_MIX], weights=[w for _, w in LANG_MIX])[0]
        persona = rng.choice(PERSONAS[lang])
        topic = rng.choice(TOPICS)
        crop = rng.choice(crops)
        extra = rng.choice([f"mention the pest '{rng.choice(pests)}'", f"mention the scheme '{rng.choice(schemes)}'",
                            "include numbers from a soil health card (N, P, K, pH)", "include the farmer's district",
                            "ask a follow-up in the same message", "be very short, like an SMS", "", ""])
        prompt = (f"Write {per_call} different, realistic questions that {persona} might ask an agricultural assistant about "
                  f"'{topic}', focused on the crop '{crop}'. Language: {LANG_NAMES[lang]}. {extra}. Vary phrasing, detail "
                  f"and spelling like real farmers do (some typos allowed). Return ONLY a JSON array of strings.")
        requests_.append({"custom_id": f"q-{i}-{lang}", "params": {
            "model": args.model, "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt}]}})
    guard(args.max_usd, estimate(args.model, len(requests_), 180, 500), "questions")
    if args.dry_run:
        return 0
    batch_id = submit_batch(client, requests_, "questions")
    results = wait_batch(client, batch_id)
    rows, usd = [], 0.0
    for cid, res in results.items():
        if res.type != "succeeded":
            continue
        usd += cost_of(args.model, res.message.usage)
        text = "".join(b.text for b in res.message.content if b.type == "text")
        lang = cid.rsplit("-", 1)[-1]
        try:
            arr = json.loads(text[text.index("["): text.rindex("]") + 1])
        except ValueError:
            continue
        for q in arr:
            if isinstance(q, str) and 8 <= len(q) <= 600:
                rows.append({"id": example_id("q", q), "language": lang, "question": q.strip()})
    ledger_add({"stage": "questions", "batch": batch_id, "model": args.model, "n": len(results), "usd": round(usd, 4),
                "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}: {len(rows)} questions (${usd:.2f})")
    return 0


# ------------------------------------------------------------------ stage 2: trajectories
def _to_openai_history(content_blocks, tool_results: dict[str, str]) -> list[dict]:
    """Anthropic assistant content (text + tool_use) + our tool results -> OpenAI-shaped turns."""
    text = "".join(b.text for b in content_blocks if b.type == "text").strip()
    calls = [b for b in content_blocks if b.type == "tool_use"]
    turns: list[dict] = []
    if calls:
        assistant = {"role": "assistant", "content": text,
                     "tool_calls": [{"id": c.id, "type": "function",
                                     "function": {"name": c.name, "arguments": json.dumps(dict(c.input), ensure_ascii=False)}}
                                    for c in calls]}
        turns.append(assistant)
        for c in calls:
            turns.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": tool_results.get(c.id, "")[:6000]})
    elif text:
        turns.append({"role": "assistant", "content": text})
    return turns


def stage_trajectories(args) -> int:
    from krishidisha.services.llm import SYSTEM_PROMPT
    from krishidisha.services.tools import run_tool

    questions = read_jsonl(args.questions)
    rng = random.Random(args.seed)
    rng.shuffle(questions)
    questions = questions[: args.n] if args.n else questions
    client = _client(args.dry_run)
    with app_context() as app:
        system, tool_defs = _system_blocks(app, SYSTEM_PROMPT, tools=True)
        tools_openai = openai_tool_schemas(app)
        hard_share = args.hard_share
        # conversation state per question: anthropic messages + openai turns
        state = {}
        for q in questions:
            hard = (q["language"] not in ("en", "hi", "hinglish")) or rng.random() < hard_share
            model = args.hard_model if hard else args.model
            passages = app.kb.search(q["question"], k=3)
            grounding = ""
            if passages:
                grounding = "\n\n[Knowledge base passages that may help (cite them when used):\n" + "\n".join(
                    f"- ({p['source']}) {p['title']}: {p['text'][:500]}" for p in passages) + "]"
            user = q["question"] + grounding + f"\n\n[Preferred reply language: {LANG_NAMES.get(q['language'], q['language'])}]"
            state[q["id"]] = {"q": q, "model": model, "anthropic": [{"role": "user", "content": user}],
                              "openai": [{"role": "user", "content": q["question"]}], "done": False, "tools_used": []}

        for round_no in range(args.max_rounds + 1):
            pending = [sid for sid, s in state.items() if not s["done"]]
            if not pending:
                break
            reqs = []
            for sid in pending:
                s = state[sid]
                reqs.append({"custom_id": sid, "params": {"model": s["model"], "max_tokens": 1500, "system": system,
                                                          "tools": tool_defs, "messages": s["anthropic"]}})
            n_hard = sum(1 for sid in pending if state[sid]["model"] == args.hard_model)
            est = estimate(args.model, len(pending) - n_hard, 1200, 450, cached=2500) + estimate(args.hard_model, n_hard, 1200, 450, cached=2500)
            guard(args.max_usd, est, f"trajectories round {round_no}")
            if args.dry_run:
                return 0
            batch_id = submit_batch(client, reqs, f"trajectories r{round_no}")
            results = wait_batch(client, batch_id)
            usd = 0.0
            for sid, res in results.items():
                s = state[sid]
                if res.type != "succeeded":
                    s["done"] = True
                    s["failed"] = True
                    continue
                msg = res.message
                usd += cost_of(s["model"], msg.usage)
                if msg.stop_reason == "tool_use" and round_no < args.max_rounds:
                    tool_results: dict[str, str] = {}
                    anth_results = []
                    for b in msg.content:
                        if b.type == "tool_use":
                            out, is_err = run_tool(app.tools, b.name, dict(b.input))
                            tool_results[b.id] = out
                            s["tools_used"].append(b.name)
                            item = {"type": "tool_result", "tool_use_id": b.id, "content": out}
                            if is_err:
                                item["is_error"] = True
                            anth_results.append(item)
                    s["anthropic"].append({"role": "assistant", "content": msg.content})
                    s["anthropic"].append({"role": "user", "content": anth_results})
                    s["openai"] += _to_openai_history(msg.content, tool_results)
                else:
                    s["openai"] += _to_openai_history(msg.content, {})
                    s["done"] = True
            ledger_add({"stage": f"trajectories_r{round_no}", "batch": batch_id, "n": len(results), "usd": round(usd, 4),
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S")})

        rows = []
        for sid, s in state.items():
            if s.get("failed") or not s["done"] or s["openai"][-1]["role"] != "assistant" or not s["openai"][-1].get("content"):
                continue
            rows.append(make_example("distill", s["q"]["language"], s["openai"], tools=tools_openai,
                                     meta={"teacher": s["model"], "tools_used": s["tools_used"], "question_id": sid}))
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}: {len(rows)} trajectories from {len(questions)} questions")
    return 0


# ------------------------------------------------------------------ stage 3: rewrite KCC answers
REWRITE_PROMPT = (
    "You are an Indian agricultural extension expert. A farmer asked the Kisan Call Centre the question below and the "
    "operator logged a terse answer. Rewrite the answer as a complete, practical reply (80-180 words) in the same "
    "language as the question: keep every fact and dose from the operator's note, add the missing specifics a good "
    "advisor would give (timing, quantity per acre or per litre, safety, a non-chemical option when sensible), and do "
    "not invent scheme amounts or banned pesticides. Reply with the answer only.\n\nQuestion: {q}\nOperator note: {a}")


def stage_rewrite(args) -> int:
    rows = read_jsonl(args.inp)
    if args.n:
        rows = rows[: args.n]
    client = _client(args.dry_run)
    reqs = []
    for r in rows:
        q, a = r["messages"][1]["content"], r["messages"][2]["content"]
        reqs.append({"custom_id": r["id"], "params": {"model": args.model, "max_tokens": 600,
                                                      "messages": [{"role": "user", "content": REWRITE_PROMPT.format(q=q, a=a)}]}})
    guard(args.max_usd, estimate(args.model, len(reqs), 260, 260), "rewrite")
    if args.dry_run:
        return 0
    batch_id = submit_batch(client, reqs, "rewrite")
    results = wait_batch(client, batch_id)
    out_rows, usd = [], 0.0
    by_id = {r["id"]: r for r in rows}
    for cid, res in results.items():
        if res.type != "succeeded":
            continue
        usd += cost_of(args.model, res.message.usage)
        text = "".join(b.text for b in res.message.content if b.type == "text").strip()
        src = by_id[cid]
        if len(text) < 60:
            continue
        meta = dict(src.get("meta") or {}, raw=False, rewritten_by=args.model)
        out_rows.append(make_example("kcc", src.get("language", "en"), [src["messages"][1], {"role": "assistant", "content": text}],
                                     meta=meta))
    ledger_add({"stage": "rewrite", "batch": batch_id, "model": args.model, "n": len(results), "usd": round(usd, 4),
                "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    write_jsonl(args.out, out_rows)
    print(f"wrote {args.out}: {len(out_rows)} rewritten answers (${usd:.2f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="claude-sonnet-5")
    common.add_argument("--max-usd", type=float, default=100.0)
    common.add_argument("--dry-run", action="store_true", help="estimate cost, submit nothing")
    common.add_argument("--seed", type=int, default=42)
    q = sub.add_parser("questions", parents=[common])
    q.add_argument("--n", type=int, default=3000)
    q.add_argument("--out", type=Path, default=DATA_DIR / "questions.jsonl")
    t = sub.add_parser("trajectories", parents=[common])
    t.add_argument("--questions", type=Path, default=DATA_DIR / "questions.jsonl")
    t.add_argument("--out", type=Path, default=DATA_DIR / "distill.jsonl")
    t.add_argument("--n", type=int, default=0, help="limit number of questions (0 = all)")
    t.add_argument("--hard-model", default="claude-opus-5")
    t.add_argument("--hard-share", type=float, default=0.12, help="share of EN/HI questions routed to the hard model")
    t.add_argument("--max-rounds", type=int, default=2)
    r = sub.add_parser("rewrite", parents=[common])
    r.add_argument("--inp", type=Path, default=DATA_DIR / "kcc.jsonl")
    r.add_argument("--out", type=Path, default=DATA_DIR / "kcc_rewritten.jsonl")
    r.add_argument("--n", type=int, default=0)
    sub.add_parser("status")
    args = p.parse_args(argv)
    if args.stage == "status":
        print(json.dumps(ledger_load(), indent=1))
        return 0
    return {"questions": stage_questions, "trajectories": stage_trajectories, "rewrite": stage_rewrite}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
