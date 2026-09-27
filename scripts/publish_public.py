"""Mirror this (private) lab repository to the public app repository with the training paths stripped from history.

    python scripts/publish_public.py [--public https://github.com/shivpratapsinghpanwar/KrishiDisha.ai.git] [--dry-run]

Steps: fresh clone of the current checkout -> ``git filter-repo --invert-paths`` with scripts/public_exclude.txt ->
force-push ``main``. Requires git-filter-repo on PATH. Never run it the other way round.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC = "https://github.com/shivpratapsinghpanwar/KrishiDisha.ai.git"


def run(cmd, cwd=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--public", default=DEFAULT_PUBLIC)
    p.add_argument("--branch", default="main")
    p.add_argument("--dry-run", action="store_true", help="build the filtered clone, print its summary, push nothing")
    args = p.parse_args(argv)
    if shutil.which("git-filter-repo") is None:
        sys.exit("git-filter-repo not found on PATH (pip install git-filter-repo)")
    excludes = [l.strip() for l in (HERE / "scripts" / "public_exclude.txt").read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")]
    tmp = Path(tempfile.mkdtemp(prefix="kd-public-"))
    clone = tmp / "mirror"
    run(["git", "clone", "--no-local", "--branch", args.branch, str(HERE), str(clone)])
    filt = ["git", "filter-repo", "--invert-paths"]
    for e in excludes:
        filt += ["--path", e]
    run(filt, cwd=clone)
    run(["git", "log", "--oneline", "-3"], cwd=clone)
    run(["git", "ls-files"], cwd=clone)
    leftovers = subprocess.run(["git", "ls-files"], cwd=clone, capture_output=True, text=True).stdout.splitlines()
    bad = [f for f in leftovers if any(f == e or f.startswith(e) for e in excludes)]
    if bad:
        sys.exit(f"excluded paths still present: {bad[:5]}")
    if args.dry_run:
        print(f"dry run: filtered clone at {clone}")
        return 0
    run(["git", "remote", "add", "public", args.public], cwd=clone)
    run(["git", "push", "--force", "public", f"{args.branch}:{args.branch}"], cwd=clone)
    shutil.rmtree(tmp, ignore_errors=True)
    print("public mirror updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
