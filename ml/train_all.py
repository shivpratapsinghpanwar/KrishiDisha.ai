"""Retrain everything KrishiDisha serves: tabular models, then the disease CNN.

The tabular stage always runs (its CSVs ship with the repo).  The image stage
runs only when a PlantVillage-style ``ImageFolder`` dataset can be found - it
is several gigabytes and is not part of the repository - otherwise it is
skipped with a note rather than failing the whole run.

Dataset lookup order for the image stage:

1. ``--disease-data-dir`` if given
2. ``$KRISHIDISHA_DISEASE_DATA``
3. the first of a few conventional local paths that contains ``train/`` and a
   validation folder

Any arguments after a bare ``--`` are forwarded verbatim to
:mod:`ml.train_disease`, e.g.::

    python -m ml.train_all -- --epochs 6 --arch efficientnet_b0 --batch-size 32

Other examples::

    python -m ml.train_all                       # tabular (+ disease if found)
    python -m ml.train_all --skip-disease        # tabular only
    python -m ml.train_all --skip-tabular -- --limit-per-class 40 --epochs 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_DISEASE_PATHS = [
    Path("data/plantvillage"),
    Path("datasets/plantvillage"),
    Path("../datasets/plantvillage"),
    Path("C:/Shivpratap_Singh_Official_Work/datasets/plantvillage"),
]


def locate_disease_dataset(explicit: Path | None) -> Path | None:
    """Return a folder that (somewhere below it) holds train/ + valid/, or None."""
    from ml.train_disease import find_dataset_root

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("KRISHIDISHA_DISEASE_DATA")
    if env:
        candidates.append(Path(env))
    candidates.extend(DEFAULT_DISEASE_PATHS)

    for cand in candidates:
        if not cand.is_dir():
            continue
        try:
            return find_dataset_root(cand)
        except FileNotFoundError:
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the tabular stage and, when the dataset exists, the disease stage."""
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="folder with the tabular CSVs")
    p.add_argument("--output", type=Path, default=Path("models"), help="folder for all artefacts")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--disease-data-dir", type=Path, default=None,
                   help="PlantVillage dataset root (skips the image stage if absent)")
    p.add_argument("--skip-tabular", action="store_true")
    p.add_argument("--skip-disease", action="store_true")
    args = p.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".gitkeep").touch()

    if not args.skip_tabular:
        from ml.train_tabular import main as tabular_main

        print("=" * 70)
        print("STAGE 1/2  tabular models (crop / fertilizer / yield)")
        print("=" * 70)
        rc = tabular_main(["--data-dir", str(args.data_dir), "--output", str(args.output),
                           "--cv-folds", str(args.cv_folds)])
        if rc != 0:
            return rc
    else:
        print("skipping tabular stage (--skip-tabular)")

    if args.skip_disease:
        print("\nskipping disease stage (--skip-disease)")
        return 0

    root = locate_disease_dataset(args.disease_data_dir)
    if root is None:
        print("\n" + "=" * 70)
        print("STAGE 2/2  plant-disease CNN - SKIPPED")
        print("No ImageFolder dataset with train/ and valid/ found. Download the")
        print("Kaggle 'New Plant Diseases Dataset (Augmented)' and rerun with")
        print("  python -m ml.train_all --disease-data-dir <path>")
        print("=" * 70)
        return 0

    from ml.train_disease import main as disease_main

    print("\n" + "=" * 70)
    print(f"STAGE 2/2  plant-disease CNN  (dataset: {root})")
    print("=" * 70)
    return disease_main(["--data-dir", str(root), "--output", str(args.output), *passthrough])


if __name__ == "__main__":
    raise SystemExit(main())
