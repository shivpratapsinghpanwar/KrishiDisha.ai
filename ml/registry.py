"""Tiny model registry: ``models/registry.json`` lists every shipped artefact with provenance.

Each entry: name, version (monotonic per name), path (relative to models/), sha256, size, git sha,
trained_at, arch, metrics, extra, card. The app surfaces the latest entry per name in
``/api/v1/health`` so it is always clear which model is serving.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path("models/registry.json")


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(registry_path: Path = DEFAULT_REGISTRY) -> list[dict]:
    if not Path(registry_path).exists():
        return []
    try:
        return json.loads(Path(registry_path).read_text(encoding="utf-8"))
    except ValueError:
        return []


def latest(name: str, registry_path: Path = DEFAULT_REGISTRY) -> dict | None:
    entries = [e for e in load(registry_path) if e.get("name") == name]
    return max(entries, key=lambda e: e.get("version", 0)) if entries else None


def register(name: str, artefact: Path, arch: str | None = None, metrics: dict | None = None,
             extra: dict | None = None, card: str | None = None, registry_path: Path = DEFAULT_REGISTRY) -> dict:
    artefact = Path(artefact)
    entries = load(registry_path)
    version = max([e.get("version", 0) for e in entries if e.get("name") == name], default=0) + 1
    try:
        rel = artefact.resolve().relative_to(Path(registry_path).resolve().parent).as_posix()
    except ValueError:
        rel = artefact.as_posix()
    entry: dict[str, Any] = {
        "name": name, "version": version, "path": rel, "sha256": _sha256(artefact), "size_mb": round(artefact.stat().st_size / 1e6, 2),
        "git_sha": _git_sha(), "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "arch": arch,
        "metrics": metrics or {}, "extra": extra or {}, "card": card,
    }
    entries.append(entry)
    Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
    Path(registry_path).write_text(json.dumps(entries, indent=1), encoding="utf-8")
    print(f"registered {name} v{version} -> {rel}")
    return entry
