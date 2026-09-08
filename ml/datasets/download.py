"""Fetch every registered image source into ``$KRISHIDISHA_DATA_ROOT/<name>/``.

    python -m ml.datasets.download --all                # every `enabled: true` source
    python -m ml.datasets.download --name paddy_doctor  # one source (enabled or not)
    python -m ml.datasets.download --list

Each source directory gets a ``_source.json`` with the registry entry, the archive checksum and the
extraction time, so :mod:`ml.datasets.build_disease_manifest` can skip unchanged sources.

Kinds: ``kaggle_dataset`` / ``kaggle_competition`` (Kaggle API, credentials in ``~/.kaggle``),
``git`` (shallow clone), ``http`` (streamed download), ``local_zip`` (already on disk),
``local_dir`` (own photos; nothing to download).

Zip extraction is done with :mod:`zipfile` member-by-member so that ``include_prefix`` /
``exclude`` filters apply and *case-colliding* paths (the PlantVillage zip has a duplicate
lower-case tree that breaks on NTFS) are skipped instead of aborting the extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

from . import SOURCES_FILE, data_root


def load_sources() -> dict[str, dict]:
    with open(SOURCES_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["sources"]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ extraction
def extract_zip(zip_path: Path, dest: Path, include_prefix: str | None = None, exclude: list[str] | None = None,
                strip_prefix: bool = True) -> int:
    """Extract ``zip_path`` into ``dest``; returns the number of files written."""
    dest.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for info in members:
            name = info.filename
            if info.is_dir():
                continue
            if include_prefix and not name.startswith(include_prefix):
                continue
            if exclude and any(x in name for x in exclude):
                continue
            rel = name[len(include_prefix):] if (include_prefix and strip_prefix) else name
            key = rel.lower()
            if key in written:          # case collision on NTFS: keep the first one
                continue
            written.add(key)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            n += 1
            if n % 5000 == 0:
                print(f"  extracted {n} files...", flush=True)
    return n


def _extract_all_archives(folder: Path) -> None:
    """Kaggle downloads arrive as one zip; some datasets nest more zips inside."""
    for _ in range(3):
        zips = [p for p in folder.rglob("*.zip") if p.is_file()]
        if not zips:
            return
        for z in zips:
            print(f"  unzip {z.name}")
            extract_zip(z, z.parent)
            z.unlink()


# ------------------------------------------------------------------- fetchers
def _kaggle_cmd() -> list[str]:
    exe = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    return [str(exe)] if exe.exists() else [sys.executable, "-m", "kaggle.cli"]


def fetch_kaggle_dataset(ref: str, dest: Path) -> None:
    subprocess.run(_kaggle_cmd() + ["datasets", "download", "-d", ref, "-p", str(dest), "--force"], check=True)
    _extract_all_archives(dest)


def fetch_kaggle_competition(ref: str, dest: Path) -> None:
    subprocess.run(_kaggle_cmd() + ["competitions", "download", "-c", ref, "-p", str(dest), "--force"], check=True)
    _extract_all_archives(dest)


def fetch_git(url: str, dest: Path) -> None:
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
        return
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def fetch_http(url: str, dest: Path) -> Path:
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    fname = url.rsplit("/", 1)[-1] or "download.bin"
    if "." not in fname:
        fname += ".zip"
    target = dest / fname
    if target.exists():
        return target
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(target, "wb") as out:
            for chunk in r.iter_content(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total and done % (50 << 20) < (1 << 20):
                    print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
    return target


# ------------------------------------------------------------------- driver
def download_source(name: str, spec: dict, root: Path, force: bool = False) -> Path:
    dest = root / name
    marker = dest / "_source.json"
    if marker.exists() and not force:
        print(f"[{name}] already present ({dest}); use --force to re-download")
        return dest
    kind = spec["kind"]
    print(f"[{name}] {kind}: {spec['ref']}")
    t0 = time.time()
    checksum = None
    if kind == "kaggle_dataset":
        fetch_kaggle_dataset(spec["ref"], dest)
    elif kind == "kaggle_competition":
        fetch_kaggle_competition(spec["ref"], dest)
    elif kind == "git":
        fetch_git(spec["ref"], dest)
    elif kind == "http":
        archive = fetch_http(spec["ref"], dest)
        checksum = sha256_of(archive)
        if archive.suffix.lower() == ".zip":
            extract_zip(archive, dest, spec.get("include_prefix"), spec.get("exclude"))
            archive.unlink()
    elif kind == "local_zip":
        archive = Path(spec["ref"])
        if not archive.is_absolute():
            archive = root / archive
        if not archive.exists():
            raise FileNotFoundError(archive)
        checksum = sha256_of(archive)
        n = extract_zip(archive, dest, spec.get("include_prefix"), spec.get("exclude"))
        print(f"  {n} files")
    elif kind == "local_dir":
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  local folder; drop labelled photos under {dest}/<Crop___Condition>/")
    else:
        raise ValueError(f"unknown kind {kind}")
    n_images = sum(1 for p in dest.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    marker.write_text(json.dumps({"name": name, "spec": spec, "sha256": checksum, "images": n_images,
                                  "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                  "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[{name}] done: {n_images} images in {time.time() - t0:.0f}s")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="download every enabled source")
    p.add_argument("--name", action="append", help="download this source (repeatable)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--from-zip", type=Path, help="override the archive path for a local_zip source")
    args = p.parse_args(argv)

    sources = load_sources()
    root = data_root()
    if args.list or not (args.all or args.name):
        for name, s in sources.items():
            present = (root / name / "_source.json").exists()
            print(f"{'*' if s.get('enabled') else ' '} {'[ok]' if present else '[  ]'} {name:20s} {s['kind']:18s} "
                  f"{s['domain']:5s} {s.get('crop', ''):12s} {s.get('licence', '')[:50]}")
        return 0
    names = args.name or [n for n, s in sources.items() if s.get("enabled")]
    failures = []
    for name in names:
        spec = dict(sources[name])
        if args.from_zip and spec["kind"] == "local_zip":
            spec["ref"] = str(args.from_zip)
        try:
            download_source(name, spec, root, force=args.force)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            print(f"[{name}] FAILED: {exc}")
            failures.append(name)
    if failures:
        print("failed:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
