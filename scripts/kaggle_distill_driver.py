"""Driver executed by the Kaggle distillation notebook (``exec(open(...).read())`` after the source unpack and the
secrets cell). Keeping the logic here means changes ship through ``scripts/kaggle_sync_src.py`` and never need
an edit in the Kaggle notebook editor (whose saved versions are the only ones that carry the secrets).

Expects: cwd = /tmp/krishidisha (unpacked source), env with the teacher keys, W = /kaggle/working.
Optional env: DISTILL_MAX_HOURS (default 11), DISTILL_PROVIDERS (default groq,gemini,openrouter[,ollama]).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time

W = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
MAX_HOURS = float(os.environ.get("DISTILL_MAX_HOURS", "11"))
PROVIDERS = os.environ.get("DISTILL_PROVIDERS") or "kaggle*4,groq,gemini,openrouter"  # kaggle*4 = four workers on the Model Proxy ($10/day)
if os.environ.get("OLLAMA_API_KEY") and "ollama" not in PROVIDERS:
    os.environ["TEACHER_OLLAMA_BASE_URL"] = "https://ollama.com/v1"
    PROVIDERS += ",ollama"
os.environ.setdefault("TRANSLATION_BACKEND", "none")
os.environ.setdefault("KB_EMBEDDING_MODEL", "")

n_keys = {b: len([1 for k in [b] + [f"{b}_{i}" for i in range(2, 10)] if os.environ.get(k)])
          for b in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")}
assert any(n_keys.values()) or os.environ.get("KAGGLE_KEY"), "no teacher API key found: attach the secrets in the Kaggle editor (Add-ons > Secrets)"
print("providers:", PROVIDERS, "keys per provider:", n_keys, flush=True)

subprocess.run("pip install -q flask flask-sqlalchemy python-dotenv openai anthropic datasketch reportlab 2>&1 | tail -1", shell=True)

# seed data + cache from the cache dataset (Kaggle extracts the zip)
src_root = glob.glob("/kaggle/input/**/krishidisha-distill-cache", recursive=True)[0]
os.makedirs("data/llm/.cache", exist_ok=True)
for p in glob.glob(src_root + "/**/*.jsonl", recursive=True):
    dst = "data/llm/.cache/" + os.path.basename(p) if "/cache/" in p.replace(chr(92), "/") else "data/llm/" + os.path.basename(p)
    shutil.copy2(p, dst)


def counts():
    return {os.path.basename(p): sum(1 for _ in open(p, encoding="utf-8")) for p in sorted(glob.glob("data/llm/.cache/*.jsonl"))}


def push_cache(note: str) -> None:
    """Version the cache dataset from inside the kernel (needs KAGGLE_USERNAME/KAGGLE_KEY secrets)."""
    if not os.environ.get("KAGGLE_KEY"):
        print("no KAGGLE_KEY secret: cache stays in /kaggle/working only", flush=True)
        return
    tmp = "/tmp/cache_upload"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp + "/cache")
    for p in glob.glob("data/llm/.cache/*.jsonl"):
        shutil.copy2(p, tmp + "/cache/")
    for n in ["questions.jsonl", "train.jsonl", "eval.jsonl", "kisanvaani.jsonl", "kb_pairs.jsonl"]:
        if os.path.exists("data/llm/" + n):
            shutil.copy2("data/llm/" + n, tmp + "/")
    json.dump({"title": "KrishiDisha distillation cache", "id": "sspanwar/krishidisha-distill-cache",
               "licenses": [{"name": "other"}]}, open(tmp + "/dataset-metadata.json", "w"))
    r = subprocess.run(["kaggle", "datasets", "version", "-p", tmp, "-m", note, "--dir-mode", "zip"], capture_output=True, text=True)
    print("cache push:", (r.stdout + r.stderr).strip()[-160:], flush=True)


print("start:", counts(), "questions:", sum(1 for _ in open("data/llm/questions.jsonl", encoding="utf-8")), flush=True)
t0 = time.time()
passes = 0
while time.time() - t0 < MAX_HOURS * 3600:
    passes += 1
    before = counts()
    for stage in [f"python -u -m ml.llm.distill trajectories --provider {PROVIDERS} --questions data/llm/questions.jsonl --out {W}/distill.jsonl",
                  f"python -u -m ml.llm.distill rewrite --provider {PROVIDERS} --inp data/llm/train.jsonl --filter-language hi --target-language hi --out {W}/kb_hi_rewritten.jsonl"]:
        subprocess.run(stage + f" 2>&1 | grep -vE 'INFO|Warning' | tee -a {W}/distill.log | tail -6", shell=True)
    after = counts()
    print(f"pass {passes} at {(time.time() - t0) / 3600:.1f}h: {after}", flush=True)
    shutil.copytree("data/llm/.cache", f"{W}/cache", dirs_exist_ok=True)
    if after != before:
        push_cache(f"pass {passes}: {after}")
    else:
        wait = 1800  # every provider is out of daily quota; Google/Groq reset at 00:00 PT
        if time.time() - t0 + wait > MAX_HOURS * 3600:
            break
        print(f"no progress; sleeping {wait // 60} min", flush=True)
        time.sleep(wait)
print("done:", counts(), flush=True)
