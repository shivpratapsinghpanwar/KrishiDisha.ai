"""Merge per-system eval_llm JSON outputs (one run per system, see memory: laptop runs them separately) into
one comparison report.

    python scripts/merge_llm_eval.py models/reports/llm_eval_v1_part_*.json --out models/reports/llm_eval_v1.md
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(bool(x) for x in xs) / len(xs):.1f}%" if xs else "-"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    results: dict[str, list[dict]] = {}
    for f in a.inputs:
        d = json.load(open(f, encoding="utf-8"))
        for name, rows in d["results"].items():
            results.setdefault(name, []).extend(rows)
    lines = [f"# Assistant evaluation, merged ({len(next(iter(results.values())))} examples per system)", "",
             "| System | Tool match | Tool match (tool rows) | Args valid | Language ok | hi ok | hinglish ok | en ok | Safety flags/ex | Latency s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, rows in results.items():
        tool_rows = [r for r in rows if r.get("ref_tools")]
        by_lang = collections.defaultdict(list)
        for r in rows:
            by_lang[r.get("language")].append(r.get("language_ok"))
        lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
        flags = sum(len(r.get("safety_flags") or []) for r in rows) / max(len(rows), 1)
        lines.append(f"| {name} | {pct([r.get('tool_match') for r in rows])} | {pct([r.get('tool_match') for r in tool_rows])} "
                     f"| {pct([r.get('args_valid') for r in rows if r.get('args_valid') is not None])} | {pct([r.get('language_ok') for r in rows])} "
                     f"| {pct(by_lang.get('hi', []))} | {pct(by_lang.get('hinglish', []))} | {pct(by_lang.get('en', []))} "
                     f"| {flags:.2f} | {sum(lat) / max(len(lat), 1):.2f} |")
    lines += ["", "Tool rows = eval examples whose reference trajectory calls at least one tool "
              f"({len([r for r in next(iter(results.values())) if r.get('ref_tools')])} of them)."]
    Path(a.out).write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print(chr(10).join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
