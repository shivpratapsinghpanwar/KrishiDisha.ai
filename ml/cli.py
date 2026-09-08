"""One entry point for the ML tooling:  ``python -m ml <command> [args]``

    python -m ml datasets list
    python -m ml datasets download --all
    python -m ml build disease-manifest --resize 320
    python -m ml train disease --manifest <...>/manifest.csv --arch timm:efficientnet_b0
    python -m ml train tabular --tasks crop yield
    python -m ml calibrate disease --manifest <...>
    python -m ml eval disease --manifest <...>
    python -m ml export disease
    python -m ml registry

Every sub-command forwards the remaining arguments to the underlying module's ``main``.
"""
from __future__ import annotations

import json
import sys


def _forward(module: str, argv: list[str]) -> int:
    import importlib

    mod = importlib.import_module(module)
    return int(mod.main(argv) or 0)


COMMANDS = {
    ("datasets", "download"): "ml.datasets.download",
    ("datasets", "list"): "ml.datasets.download",
    ("build", "disease-manifest"): "ml.datasets.build_disease_manifest",
    ("train", "disease"): "ml.vision.train",
    ("train", "tabular"): "ml.train_tabular",
    ("calibrate", "disease"): "ml.vision.calibrate",
    ("eval", "disease"): "ml.vision.eval",
    ("export", "disease"): "ml.vision.export",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "registry":
        from .registry import load

        for e in load():
            print(f"{e['name']:16s} v{e['version']:<3d} {e['path']:40s} {e.get('registered_at', '')}  "
                  f"{json.dumps(e.get('metrics', {}))[:80]}")
        return 0
    key = tuple(argv[:2])
    if key == ("datasets", "list"):
        return _forward("ml.datasets.download", ["--list"])
    if key not in COMMANDS:
        print(f"unknown command {' '.join(argv[:2])!r}\n{__doc__}")
        return 2
    return _forward(COMMANDS[key], argv[2:])


if __name__ == "__main__":
    sys.exit(main())
