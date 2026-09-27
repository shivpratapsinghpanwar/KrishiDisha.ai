"""Keep the teacher-reply cache in sync between this machine and the private Kaggle dataset the distillation
kernel resumes from.

    python scripts/kaggle_sync_cache.py push [--create]   # local data/llm/.cache + questions/train/eval -> new dataset version
    python scripts/kaggle_sync_cache.py pull              # latest dataset version -> merged into local data/llm/.cache

The cache files are append-only JSONL keyed by custom_id, so merging = union by custom_id (local wins on ties).
Dataset: sspanwar/krishidisha-distill-cache (private). Never contains keys; never contains weights.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DATASET = "sspanwar/krishidisha-distill-cache"
CACHE = HERE / "data" / "llm" / ".cache"
EXTRA = ["questions.jsonl", "train.jsonl", "eval.jsonl", "kisanvaani.jsonl", "kb_pairs.jsonl"]


def kaggle_bin() -> str:
    for cand in (HERE / ".venv-train" / "Scripts" / "kaggle.exe", HERE / ".venv-train" / "bin" / "kaggle"):
        if cand.exists():
            return str(cand)
    return shutil.which("kaggle") or sys.exit("kaggle CLI not found")


def merge_jsonl(dst: Path, src: Path) -> int:
    seen = set()
    rows = []
    for p in (dst, src):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("custom_id") in seen:
                continue
            seen.add(d.get("custom_id"))
            rows.append(line)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(chr(10).join(rows) + (chr(10) if rows else ""), encoding="utf-8")
    return len(rows)


def push(create: bool) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="kd-cache-"))
    (tmp / "cache").mkdir()
    n = 0
    for p in CACHE.glob("*.jsonl"):
        shutil.copy2(p, tmp / "cache" / p.name)
        n += 1
    for name in EXTRA:
        p = HERE / "data" / "llm" / name
        if p.exists():
            shutil.copy2(p, tmp / name)
    (tmp / "dataset-metadata.json").write_text(json.dumps({
        "title": "KrishiDisha distillation cache", "id": DATASET, "licenses": [{"name": "other"}],
        "description": "Teacher-reply cache (append-only JSONL per stage) and seed data for the KrishiDisha assistant "
                       "distillation kernel. Private. No keys, no weights."}, indent=1))
    cmd = [kaggle_bin(), "datasets", "create" if create else "version", "-p", str(tmp), "--dir-mode", "zip"]
    if not create:
        cmd += ["-m", f"{n} cache files"]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)


def pull() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="kd-cache-"))
    subprocess.run([kaggle_bin(), "datasets", "download", "-p", str(tmp), "--unzip", DATASET], check=True)
    found = list(tmp.rglob("*.jsonl"))
    total = 0
    for p in found:
        if p.parent.name == "cache":
            total += merge_jsonl(CACHE / p.name, p)
            print(f"merged {p.name}: {total} rows total so far")
    shutil.rmtree(tmp, ignore_errors=True)
    print("local cache:", {p.name: sum(1 for _ in p.open(encoding='utf-8')) for p in CACHE.glob('*.jsonl')})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["push", "pull"])
    p.add_argument("--create", action="store_true")
    a = p.parse_args(argv)
    (push(a.create) if a.action == "push" else pull())
    return 0


if __name__ == "__main__":
    sys.exit(main())
