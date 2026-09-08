"""Unify every downloaded image source into one manifest + pre-resized images.

    python -m ml.datasets.build_disease_manifest --out <DATA_ROOT>/disease_unified --resize 320
    python -m ml.datasets.build_disease_manifest --out ... --sources paddy_doctor rice_leaf_4 --dry-run

Outputs under ``--out``:

* ``manifest.csv``   one row per image: path (relative to --out), label, crop, condition, source,
                     domain (field|lab|ood), split (train|valid|test), width, height, sha1, phash
* ``images/<source>/<label>/<hash>.jpg``   JPEG q90, short side = --resize
* ``classes.json``   sorted class list + per-class / per-source counts
* ``unmapped.json``  raw labels the taxonomy could not place (fix them in taxonomy.py, re-run)
* ``build_report.md`` human summary

Rules (from the plan):

* only sources with ``domain: field`` or ``ood`` are included unless ``--include-lab`` is given;
* exact duplicates (sha1) and near-duplicates (perceptual hash, Hamming <= 4) are removed
  *across* sources and splits so a test image never has a twin in train;
* each source is split 70/15/15 stratified by class unless it ships its own splits
  (``splits:`` in sources.yaml) or is a hold-out (``holdout: true`` -> everything is ``test``);
* classes with fewer than ``--min-per-class`` images (default 40) are dropped and listed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

from . import SOURCES_FILE, data_root
from .taxonomy import NOT_A_LEAF, TAXONOMY_VERSION, canonical_label, split as split_label

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = {"train": "train", "training": "train", "valid": "valid", "val": "valid", "validation": "valid",
               "test": "test", "testing": "test"}


# ------------------------------------------------------------------ discovery
def _iter_imagefolder(src_dir: Path, spec: dict):
    """Yield (path, raw_label, parent_label, split_or_None) for class-folder layouts."""
    base = src_dir / spec["subdir"] if spec.get("subdir") else src_dir
    if not base.exists():
        # some archives nest one extra folder
        cands = [p for p in src_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))]
        base = cands[0] if len(cands) == 1 else src_dir
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXT:
            continue
        rel = path.relative_to(base)
        parts = list(rel.parts[:-1])
        if not parts:
            continue
        split = None
        # strip split folders wherever they appear (train/valid/test at any depth)
        cleaned = []
        for part in parts:
            key = part.lower()
            if key in SPLIT_NAMES:
                split = SPLIT_NAMES[key]
            else:
                cleaned.append(part)
        if not cleaned:
            continue
        raw = cleaned[-1]
        parent = cleaned[-2] if len(cleaned) >= 2 else None
        yield path, raw, parent, split


def _iter_csv(src_dir: Path, spec: dict):
    csv_path = src_dir / spec["csv"]
    img_dir = src_dir / spec.get("image_dir", "")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = row[spec["csv_label_col"]]
            name = row[spec["csv_image_col"]]
            cand = img_dir / label / name
            if not cand.exists():
                cand = img_dir / name
            if cand.exists():
                yield cand, label, None, None


def _iter_flat(src_dir: Path, spec: dict, sample: int | None, seed: int):
    base = src_dir / spec["subdir"] if spec.get("subdir") else src_dir
    files = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    if sample and len(files) > sample:
        files = random.Random(seed).sample(files, sample)
    for path in files:
        yield path, "not_a_leaf", None, None


def discover(name: str, spec: dict, src_dir: Path, seed: int):
    layout = spec.get("layout", "imagefolder")
    if layout == "csv":
        return _iter_csv(src_dir, spec)
    if layout == "flat":
        return _iter_flat(src_dir, spec, spec.get("sample"), seed)
    return _iter_imagefolder(src_dir, spec)


# ------------------------------------------------------------------ hashing
def phash(img: Image.Image, size: int = 8) -> int:
    """64-bit perceptual hash (DCT-free average-hash on a 4x downsample; good enough for twins)."""
    small = ImageOps.grayscale(img).resize((size * 4, size * 4), Image.Resampling.BILINEAR).resize((size, size), Image.Resampling.BOX)
    px = list(small.getdata())
    mean = sum(px) / len(px)
    bits = 0
    for v in px:
        bits = (bits << 1) | (1 if v > mean else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ------------------------------------------------------------------ processing
def process_image(src: Path, dst: Path, resize: int) -> tuple[int, int, str, int] | None:
    """Resize (short side = resize), save JPEG q90. Returns (w, h, sha1, phash) or None if unreadable."""
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            if resize and min(w, h) > resize:
                scale = resize / min(w, h)
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
            ph = phash(im)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=90, optimize=True)
    except (UnidentifiedImageError, OSError) as exc:
        print(f"  unreadable {src}: {exc}")
        return None
    sha = hashlib.sha1(dst.read_bytes()).hexdigest()
    return im.size[0], im.size[1], sha, ph


def assign_splits(rows: list[dict], seed: int, ratios=(0.70, 0.15, 0.15)) -> None:
    """Stratified per (source, label) split for rows whose split is still empty."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if not r["split"]:
            groups[(r["source"], r["label"])].append(r)
    for (_, _), items in groups.items():
        rng.shuffle(items)
        n = len(items)
        n_valid = max(1, round(n * ratios[1])) if n >= 5 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 5 else 0
        for i, r in enumerate(items):
            r["split"] = "test" if i < n_test else ("valid" if i < n_test + n_valid else "train")


def dedupe(rows: list[dict], max_hamming: int = 4) -> tuple[list[dict], int, int]:
    """Drop exact (sha1) and near (phash) duplicates. Test rows win over valid over train."""
    order = {"test": 0, "valid": 1, "train": 2}
    rows = sorted(rows, key=lambda r: order[r["split"]])
    seen_sha: set[str] = set()
    kept: list[dict] = []
    exact = near = 0
    # bucket phashes by top 16 bits to keep the near-dup scan fast
    buckets: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["sha1"] in seen_sha:
            exact += 1
            continue
        ph = r["phash"]
        bucket = buckets[ph >> 48]
        if any(hamming(ph, other) <= max_hamming for other in bucket):
            near += 1
            continue
        seen_sha.add(r["sha1"])
        bucket.append(ph)
        kept.append(r)
    return kept, exact, near


# ------------------------------------------------------------------ main
def build(out: Path, sources: dict[str, dict], names: list[str], resize: int, min_per_class: int, seed: int,
          include_lab: bool, dry_run: bool) -> dict:
    root = data_root()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    unmapped: dict[str, Counter] = defaultdict(Counter)
    per_source: dict[str, Counter] = {}
    t0 = time.time()

    for name in names:
        spec = sources[name]
        src_dir = root / name
        if not (src_dir / "_source.json").exists() and spec["kind"] != "local_dir":
            print(f"[{name}] not downloaded, skipping")
            continue
        if spec["domain"] == "lab" and not include_lab:
            print(f"[{name}] lab data excluded (owner's rule); pass --include-lab to override")
            continue
        holdout = bool(spec.get("holdout"))
        only_crops = {c.lower() for c in spec.get("only_crops", [])}
        counts: Counter = Counter()
        n_seen = 0
        for path, raw, parent, split in discover(name, spec, src_dir, seed):
            n_seen += 1
            label = canonical_label(spec["mapping"], raw, parent)
            if label is None:
                unmapped[name][f"{parent}/{raw}" if parent else raw] += 1
                continue
            crop, cond = split_label(label)
            if only_crops and crop.lower() not in only_crops:
                continue
            if holdout:
                split = "test"
            rel = Path("images") / name / label / f"{hashlib.md5(str(path).encode()).hexdigest()[:16]}.jpg"
            if dry_run:
                counts[label] += 1
                continue
            dst = out / rel
            if dst.exists():
                try:
                    with Image.open(dst) as im:
                        w, h = im.size
                        ph = phash(im.convert("RGB"))
                    sha = hashlib.sha1(dst.read_bytes()).hexdigest()
                    meta = (w, h, sha, ph)
                except OSError:
                    meta = process_image(path, dst, resize)
            else:
                meta = process_image(path, dst, resize)
            if meta is None:
                continue
            w, h, sha, ph = meta
            rows.append({"path": rel.as_posix(), "label": label, "crop": crop, "condition": cond, "source": name,
                         "domain": spec["domain"], "split": split or "", "width": w, "height": h, "sha1": sha,
                         "phash": ph})
            counts[label] += 1
            if len(rows) % 2000 == 0:
                print(f"  {len(rows)} images processed ({time.time() - t0:.0f}s)", flush=True)
        per_source[name] = counts
        print(f"[{name}] {sum(counts.values())} images, {len(counts)} classes, {n_seen - sum(counts.values())} skipped")

    if dry_run:
        return {"per_source": {k: dict(v) for k, v in per_source.items()},
                "unmapped": {k: dict(v) for k, v in unmapped.items()}}

    # not-a-leaf class always gets the ood domain label
    for r in rows:
        if r["label"] == NOT_A_LEAF:
            r["domain"] = "ood"

    # drop small classes (computed over field+ood rows only)
    class_counts = Counter(r["label"] for r in rows)
    small = {c for c, n in class_counts.items() if n < min_per_class}
    rows = [r for r in rows if r["label"] not in small]

    assign_splits(rows, seed)
    rows, n_exact, n_near = dedupe(rows)
    rows.sort(key=lambda r: (r["source"], r["label"], r["path"]))

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["path"])
        writer.writeheader()
        writer.writerows(rows)

    classes = sorted({r["label"] for r in rows})
    by_class = Counter(r["label"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    by_src_split = Counter((r["source"], r["split"]) for r in rows)
    field_test = sum(1 for r in rows if r["split"] == "test" and r["domain"] == "field")
    summary = {
        "taxonomy_version": TAXONOMY_VERSION, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "resize": resize,
        "seed": seed, "n_images": len(rows), "n_classes": len(classes), "classes": classes,
        "per_class": dict(sorted(by_class.items())), "per_split": dict(by_split),
        "per_source_split": {f"{s}/{sp}": n for (s, sp), n in sorted(by_src_split.items())},
        "dropped_small_classes": sorted(small), "duplicates_removed": {"exact": n_exact, "near": n_near},
        "field_test_images": field_test,
        "sources": {n: {"domain": sources[n]["domain"], "licence": sources[n].get("licence"),
                        "country": sources[n].get("country"), "url": sources[n].get("url")} for n in per_source},
    }
    (out / "classes.json").write_text(json.dumps(summary, indent=2))
    (out / "unmapped.json").write_text(json.dumps({k: dict(v) for k, v in unmapped.items()}, indent=2))

    lines = [f"# Disease manifest build ({summary['built_at']})", "",
             f"{len(rows)} images, {len(classes)} classes, taxonomy {TAXONOMY_VERSION}, resize {resize}px", "",
             f"Splits: {dict(by_split)}; field test images: {field_test}",
             f"Duplicates removed: exact {n_exact}, near {n_near}; dropped small classes: {sorted(small) or 'none'}", "",
             "| Source | Domain | Images | Classes | Licence |", "|---|---|---|---|---|"]
    for n, counts in per_source.items():
        lines.append(f"| {n} | {sources[n]['domain']} | {sum(counts.values())} | {len(counts)} | {sources[n].get('licence', '')} |")
    lines += ["", "| Class | Images |", "|---|---|"] + [f"| {c} | {by_class[c]} |" for c in classes]
    if any(unmapped.values()):
        lines += ["", "## Unmapped raw labels (fix in taxonomy.py)"]
        for n, c in unmapped.items():
            for raw, k in c.most_common():
                lines.append(f"- {n}: `{raw}` x{k}")
    (out / "build_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out / 'manifest.csv'}: {len(rows)} images, {len(classes)} classes in {time.time() - t0:.0f}s")
    if any(unmapped.values()):
        print("UNMAPPED labels:", json.dumps({k: dict(v) for k, v in unmapped.items()}, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None, help="default: <DATA_ROOT>/disease_unified")
    p.add_argument("--sources", nargs="*", help="subset of source names (default: all downloaded)")
    p.add_argument("--resize", type=int, default=320, help="short side in px (0 = keep)")
    p.add_argument("--min-per-class", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-lab", action="store_true", help="include lab-domain sources (PlantVillage)")
    p.add_argument("--dry-run", action="store_true", help="only map labels and count; write nothing")
    args = p.parse_args(argv)

    with open(SOURCES_FILE, encoding="utf-8") as fh:
        sources = yaml.safe_load(fh)["sources"]
    names = args.sources or list(sources)
    out = args.out or data_root() / "disease_unified"
    result = build(out, sources, names, args.resize, args.min_per_class, args.seed, args.include_lab, args.dry_run)
    if args.dry_run:
        print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
