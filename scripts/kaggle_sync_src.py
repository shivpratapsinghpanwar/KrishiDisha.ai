"""Push the lab source tree to the private Kaggle dataset the training kernels read from.

    python scripts/kaggle_sync_src.py            # new dataset version from the current git HEAD
    python scripts/kaggle_sync_src.py --create   # first time only

Kaggle kernels cannot clone a private GitHub repository without a token, so the notebooks unpack
``/kaggle/input/datasets/sspanwar/krishidisha-lab-src/src.zip`` instead of running ``git clone``. The zip is
``git archive HEAD`` (tracked files only: no data, weights, caches or .env). Requires the Kaggle CLI in
``.venv-train`` or on PATH.
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
DATASET = "sspanwar/krishidisha-lab-src"


def kaggle_bin() -> str:
    for cand in (HERE / ".venv-train" / "Scripts" / "kaggle.exe", HERE / ".venv-train" / "bin" / "kaggle"):
        if cand.exists():
            return str(cand)
    return shutil.which("kaggle") or sys.exit("kaggle CLI not found")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--create", action="store_true")
    p.add_argument("--message", default=None)
    args = p.parse_args(argv)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("warning: uncommitted changes are NOT included (git archive HEAD)", file=sys.stderr)
    tmp = Path(tempfile.mkdtemp(prefix="kd-src-"))
    zip_path = tmp / "src.zip"
    with zip_path.open("wb") as fh:
        subprocess.run(["git", "archive", "--format=zip", "--prefix=krishidisha/", "HEAD"], cwd=HERE, stdout=fh, check=True)
    (tmp / "dataset-metadata.json").write_text(json.dumps({
        "title": "KrishiDisha lab source", "id": DATASET, "licenses": [{"name": "other"}],
        "description": f"Tracked source of the private KrishiDisha lab repo at {sha} (git archive; no data or weights). Private."}, indent=1))
    (tmp / "VERSION.txt").write_text(sha + chr(10))
    cmd = [kaggle_bin(), "datasets", "create" if args.create else "version", "-p", str(tmp)]
    if not args.create:
        cmd += ["-m", args.message or f"source at {sha}"]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{DATASET} <- {sha} ({zip_path.name} {zip_path.stat().st_size // 1024 if zip_path.exists() else '?'} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
