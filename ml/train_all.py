"""Retrain everything KrishiDisha serves: tabular models, then the field disease classifier.

The tabular stage always runs (its CSVs ship with the repo). The vision stage runs when the unified
field manifest exists (``<KRISHIDISHA_DATA_ROOT>/disease_unified/manifest.csv``, built by
``python -m ml.datasets.build_disease_manifest`` from the downloaded sources) - otherwise it is skipped
with a note rather than failing the whole run. PlantVillage is never used (owner's decision).

Any arguments after a bare ``--`` are forwarded verbatim to :mod:`ml.vision.train`, e.g.::

    python -m ml.train_all -- --epochs 12 --arch timm:efficientnet_b0 --batch-size 32 --ema

Other examples::

    python -m ml.train_all                       # tabular (+ vision if the manifest exists)
    python -m ml.train_all --skip-vision         # tabular only
    python -m ml.train_all --skip-tabular --build-manifest -- --limit-per-class 40 --epochs 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    forwarded: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, forwarded = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("models"))
    p.add_argument("--manifest", type=Path, help="override the disease manifest path")
    p.add_argument("--build-manifest", action="store_true", help="(re)build the unified manifest from downloaded sources first")
    p.add_argument("--skip-tabular", action="store_true")
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--no-export", action="store_true", help="skip ONNX export after training")
    args = p.parse_args(argv)

    t0 = time.time()
    if not args.skip_tabular:
        print("=== tabular ===")
        from ml.train_tabular import main as tabular_main

        rc = tabular_main(["--data-dir", str(args.data_dir), "--output", str(args.output)])
        if rc:
            return rc

    if args.skip_vision:
        return 0
    from ml.datasets import data_root

    manifest = args.manifest or data_root() / "disease_unified" / "manifest.csv"
    if args.build_manifest or not manifest.exists():
        if args.build_manifest or any((data_root() / n / "_source.json").exists() for n in ("rice_leaf_4", "paddy_doctor", "mango_leaf")):
            print("=== building manifest ===")
            from ml.datasets.build_disease_manifest import main as build_main

            build_main(["--out", str(manifest.parent)])
    if not manifest.exists():
        print(f"SKIPPED vision stage: no manifest at {manifest}. Run `python -m ml.datasets.download --all` then "
              f"`python -m ml.datasets.build_disease_manifest`.")
        return 0

    print("=== vision ===")
    from ml.vision.train import main as vision_main

    vargs = ["--manifest", str(manifest), "--output", str(args.output)]
    if "--calibrate" not in forwarded:
        vargs.append("--calibrate")
    rc = vision_main(vargs + forwarded)
    if rc:
        return rc
    ckpt = args.output / "plant_disease_model.pt"
    if ckpt.exists():
        from ml.vision.eval import run_eval
        from ml.datasets import SOURCES_FILE

        run_eval(ckpt, manifest, args.output, sources_yaml=SOURCES_FILE)
        if not args.no_export:
            from ml.vision.export import main as export_main

            export_main(["--checkpoint", str(ckpt), "--output", str(args.output)])
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
