"""Public agricultural Q&A sources -> chat JSONL.

    python -m ml.llm.public_data kisanvaani --out data/llm/kisanvaani.jsonl [--max 6000]
    python -m ml.llm.public_data kcc --out data/llm/kcc.jsonl --states "MADHYA PRADESH" "PUNJAB" --max 8000

Sources
-------
* **KisanVaani/agriculture-qa-english-only** (Hugging Face, Apache-2.0, ~22k rows of question/answer).
  Pulled through the datasets-server rows API (no extra dependency), filtered for length and India
  relevance, deduplicated.
* **Kisan Call Centre transcripts** (data.gov.in, Open Government Data Licence India - attribute).
  Resource ``cef25fe2-9231-4128-8aec-2c948fedd43f`` ("Kisan Call Centre (KCC) - Transcripts of farmers
  queries & answers"). Needs your own data.gov.in API key (``DATA_GOV_API_KEY``); the public sample
  key is rate-limited. Answers are terse operator notes ("spray 2 ml/l imida"), so they are kept as
  *raw* pairs here and rewritten into full answers by the teacher model in ``distill.py --rewrite``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests

from .common import DATA_DIR, make_example, write_jsonl

log = logging.getLogger(__name__)

HF_ROWS = "https://datasets-server.huggingface.co/rows"
KCC_RESOURCE = "cef25fe2-9231-4128-8aec-2c948fedd43f"
DATA_GOV = f"https://api.data.gov.in/resource/{KCC_RESOURCE}"

INDIA_HINTS = re.compile(r"\b(india|indian|kharif|rabi|zaid|acre|bigha|quintal|mandi|kvk|krishi|kisan|rupee|rs\.?|₹|"
                         r"punjab|haryana|maharashtra|karnataka|tamil|andhra|telangana|gujarat|rajasthan|bihar|bengal|"
                         r"odisha|madhya|uttar|kerala|assam|paddy|arhar|tur|moong|urad|gram|bajra|jowar|ragi|sugarcane|"
                         r"cotton|soybean|mustard|groundnut|wheat|rice|maize|chilli|onion|potato|tomato)\b", re.I)
BAD_ANSWER = re.compile(r"^(?:\s*(?:not|no|nil|n/?a|none|test|-+)\s*$)|call\s*(?:again|back)|contact\s+(?:kvk|aeo|ado)\s*$", re.I)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text


# ------------------------------------------------------------------ KisanVaani
def fetch_kisanvaani(max_rows: int, min_answer_chars: int = 80, india_only: bool = True, sleep: float = 0.2) -> list[dict]:
    dataset = "KisanVaani/agriculture-qa-english-only"
    rows, offset, seen = [], 0, set()
    session = requests.Session()
    while len(rows) < max_rows:
        payload = None
        for attempt in range(6):
            try:
                r = session.get(HF_ROWS, params={"dataset": dataset, "config": "default", "split": "train",
                                                 "offset": offset, "length": 100}, timeout=30)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                r.raise_for_status()
                payload = r.json()
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                wait = 3 * (attempt + 1)
                print(f"  kisanvaani: {type(exc).__name__} at offset {offset}; retry in {wait}s", flush=True)
                time.sleep(wait)
                session = requests.Session()
        if payload is None:
            print(f"  kisanvaani: giving up at offset {offset} after repeated failures; keeping {len(rows)} rows")
            break
        batch = payload.get("rows", [])
        if not batch:
            break
        for item in batch:
            row = item["row"]
            q = _clean(row.get("question") or row.get("instruction") or row.get("input") or "")
            a = _clean(row.get("answers") or row.get("answer") or row.get("output") or row.get("response") or "")
            if len(q) < 12 or len(a) < min_answer_chars or BAD_ANSWER.search(a):
                continue
            if india_only and not INDIA_HINTS.search(q + " " + a):
                continue
            key = q.lower()[:120]
            if key in seen:
                continue
            seen.add(key)
            rows.append(make_example("kisanvaani", "en", [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                     meta={"licence": "Apache-2.0", "dataset": dataset}))
            if len(rows) >= max_rows:
                break
        offset += 100
        if offset >= payload.get("num_rows_total", 10**9):
            break
        time.sleep(sleep)
        if offset % 2000 == 0:
            print(f"  kisanvaani offset {offset}: kept {len(rows)}", flush=True)
    return rows


# ------------------------------------------------------------------ KCC
KCC_FIELDS = {"state": ["StateName", "state_name", "State"], "district": ["DistrictName", "district_name"],
              "season": ["Season", "season"], "sector": ["Sector", "sector"], "category": ["Category", "category"],
              "crop": ["Crop", "crop"], "qtype": ["QueryType", "query_type"], "query": ["QueryText", "query_text", "Query"],
              "answer": ["KccAns", "kcc_ans", "Answer", "KCCAns"], "date": ["CreatedOn", "created_on", "Date"]}


def _pick(rec: dict, names: list[str]) -> str:
    for n in names:
        if n in rec and rec[n] not in (None, ""):
            return str(rec[n])
    low = {k.lower(): v for k, v in rec.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] not in (None, ""):
            return str(low[n.lower()])
    return ""


def fetch_kcc(api_key: str, states: list[str] | None, max_rows: int, min_answer_chars: int = 50,
              page: int = 500, sleep: float = 1.0) -> list[dict]:
    if not api_key:
        raise SystemExit("set DATA_GOV_API_KEY (free key from https://data.gov.in/user/register); the sample key is rate-limited")
    rows, seen = [], set()
    targets = states or [None]
    for state in targets:
        offset, empty_pages = 0, 0
        while len(rows) < max_rows and empty_pages < 2:
            params = {"api-key": api_key, "format": "json", "limit": page, "offset": offset}
            if state:
                params["filters[StateName]"] = state
            for attempt in range(6):
                r = requests.get(DATA_GOV, params=params, timeout=60)
                if r.status_code == 200 and "Rate limit" not in r.text[:200]:
                    break
                wait = 10 * (attempt + 1)
                print(f"  data.gov.in throttled; waiting {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise SystemExit("data.gov.in keeps rate-limiting; try again later or with a smaller --page")
            payload = r.json()
            recs = payload.get("records", [])
            if not recs:
                empty_pages += 1
                offset += page
                continue
            for rec in recs:
                q, a = _clean(_pick(rec, KCC_FIELDS["query"])), _clean(_pick(rec, KCC_FIELDS["answer"]))
                if len(q) < 10 or len(a) < min_answer_chars or BAD_ANSWER.search(a):
                    continue
                crop = _pick(rec, KCC_FIELDS["crop"])
                qtype = _pick(rec, KCC_FIELDS["qtype"])
                key = (crop.lower(), qtype.lower(), q.lower()[:80])
                if key in seen:
                    continue
                seen.add(key)
                meta = {"state": _pick(rec, KCC_FIELDS["state"]), "district": _pick(rec, KCC_FIELDS["district"]),
                        "crop": crop, "query_type": qtype, "season": _pick(rec, KCC_FIELDS["season"]),
                        "date": _pick(rec, KCC_FIELDS["date"]), "licence": "OGDL-India (attribute: Kisan Call Centre, DA&FW)",
                        "raw": True}
                user = q if not crop or crop.lower() in q.lower() else f"[{crop}] {q}"
                rows.append(make_example("kcc_raw", "en", [{"role": "user", "content": user}, {"role": "assistant", "content": a}],
                                         meta=meta))
                if len(rows) >= max_rows:
                    break
            offset += page
            print(f"  kcc {state or 'all'} offset {offset}: kept {len(rows)}", flush=True)
            time.sleep(sleep)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=["kisanvaani", "kcc"])
    p.add_argument("--out", type=Path)
    p.add_argument("--max", type=int, default=6000)
    p.add_argument("--states", nargs="*", help="KCC: StateName filters, e.g. 'MADHYA PRADESH' 'PUNJAB'")
    p.add_argument("--api-key", default=os.getenv("DATA_GOV_API_KEY", ""))
    p.add_argument("--all-countries", action="store_true", help="KisanVaani: keep non-India rows too")
    args = p.parse_args(argv)
    out = args.out or DATA_DIR / f"{args.source}.jsonl"
    if args.source == "kisanvaani":
        rows = fetch_kisanvaani(args.max, india_only=not args.all_countries)
    else:
        rows = fetch_kcc(args.api_key, args.states, args.max)
    write_jsonl(out, rows)
    print(f"wrote {out}: {len(rows)} examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
