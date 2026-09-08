"""Evaluate a checkpoint on the manifest's test split and write the report + model card.

    python -m ml.vision.eval --checkpoint models/plant_disease_model.pt --manifest <...>/manifest.csv

Outputs (under ``--output``, default ``models``):
* ``metrics_disease.json``          everything below as data
* ``reports/disease_eval.md``       headline (own field set, public field test), per-source, per-crop tables
* ``reports/disease_confusion_<crop>.png``  one confusion matrix per crop (not one giant blob)
* ``reports/plant_disease_card.md`` model card with dataset provenance and licences
* crop tiers written back into the checkpoint (``crop_tiers``) so the app can flag experimental crops
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..datasets.taxonomy import NOT_A_LEAF, TAXONOMY_VERSION, split as split_label
from .calibrate import collect_logits, expected_calibration_error
from .data import ManifestDataset, read_manifest
from .model import get_transforms, load_checkpoint


def tier_for(n_train_field: int, top1: float | None) -> str:
    if top1 is None:
        return "C"
    if n_train_field >= 1000 and top1 >= 0.90:
        return "A"
    if n_train_field >= 300 and top1 >= 0.80:
        return "B"
    return "C"


def _accuracy(pred, y, idx):
    if not idx.any():
        return None
    return float((pred[idx] == y[idx]).mean())


def _top3(probs, y, idx):
    if not idx.any():
        return None
    top3 = np.argsort(-probs[idx], axis=1)[:, :3]
    return float((top3 == y[idx][:, None]).any(1).mean())


def confusion_png(classes_sub: list[int], class_names: list[str], pred, y, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(classes_sub)
    idx = {c: i for i, c in enumerate(classes_sub)}
    cm = np.zeros((k, k), dtype=int)
    for p_, t_ in zip(pred, y):
        if t_ in idx:
            cm[idx[t_], idx.get(p_, -1) if p_ in idx else 0] += 0 if p_ not in idx else 1
    row_sum = cm.sum(1, keepdims=True).clip(min=1)
    norm = cm / row_sum
    fig, ax = plt.subplots(figsize=(max(5, k * 0.6 + 2), max(4, k * 0.5 + 2)))
    ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    labels = [class_names[c].split("___", 1)[1].replace("_", " ") for c in classes_sub]
    ax.set_xticks(range(k), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k), labels, fontsize=8)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_eval(ckpt_path: Path, manifest: Path, output: Path, batch_size: int = 64, workers: int = 2,
             sources_yaml: Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(ckpt_path)
    classes: list[str] = ckpt["classes"]
    T = float(ckpt.get("temperature", 1.0))
    rows_all = read_manifest(manifest)
    rows = [r for r in rows_all if r["split"] == "test" and r["label"] in set(classes)]
    train_field = Counter(r["crop"] for r in rows_all if r["split"] == "train" and r["domain"] == "field")
    ds = ManifestDataset(rows, manifest.parent, classes,
                         get_transforms(False, ckpt["img_size"], ckpt["mean"], ckpt["std"], ckpt["crop_pct"]))
    t0 = time.time()
    logits, y = collect_logits(model, ds, device, batch_size, workers)
    latency_ms = (time.time() - t0) / max(len(ds), 1) * 1000
    probs = torch.softmax(logits / T, 1).numpy()
    yn = y.numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    src = np.array([r["source"] for r in ds.rows])
    dom = np.array([r["domain"] for r in ds.rows])
    crop = np.array([r["crop"] for r in ds.rows])

    field = dom == "field"
    own = src == "own_photos"
    leaf = np.array([classes[t] != NOT_A_LEAF for t in yn])
    ood_thr = float(ckpt.get("ood_threshold", 0.0))
    # OOD: how well does max-softmax separate not-a-leaf from leaves (AUROC via rank statistic)
    auroc = None
    if leaf.any() and (~leaf).any():
        pos, neg = conf[leaf], conf[~leaf]
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        auroc = float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    fpr95 = float((conf[~leaf] >= ood_thr).mean()) if (~leaf).any() and ood_thr else None

    per_source = {}
    for s in sorted(set(src)):
        m = src == s
        per_source[s] = {"n": int(m.sum()), "top1": _accuracy(pred, yn, m), "top3": _top3(probs, yn, m)}
    per_crop = {}
    tiers = {}
    for c in sorted(set(crop)):
        m = (crop == c) & field
        if not m.any():
            continue
        acc = _accuracy(pred, yn, m)
        per_crop[c] = {"n_test": int(m.sum()), "n_train_field": int(train_field.get(c, 0)), "top1": acc,
                       "top3": _top3(probs, yn, m), "classes": sorted({classes[t] for t in yn[m]})}
        tiers[c] = tier_for(train_field.get(c, 0), acc)
        per_crop[c]["tier"] = tiers[c]
    per_class = {}
    for i, name in enumerate(classes):
        m = yn == i
        if m.any():
            tp = int(((pred == i) & m).sum())
            fp = int(((pred == i) & ~m).sum())
            fn = int(((pred != i) & m).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            per_class[name] = {"support": int(m.sum()), "precision": round(prec, 4), "recall": round(rec, 4),
                               "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0}
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else None

    metrics = {
        "checkpoint": str(ckpt_path), "arch": ckpt["arch"], "img_size": ckpt["img_size"], "taxonomy_version": TAXONOMY_VERSION,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_test": int(len(yn)), "temperature": T,
        "headline": {
            "own_field_test": {"n": int(own.sum()), "top1": _accuracy(pred, yn, own), "top3": _top3(probs, yn, own)},
            "public_field_test": {"n": int((field & ~own).sum()), "top1": _accuracy(pred, yn, field & ~own),
                                  "top3": _top3(probs, yn, field & ~own)},
            "all_test": {"n": int(len(yn)), "top1": _accuracy(pred, yn, np.ones_like(field)),
                         "top3": _top3(probs, yn, np.ones_like(field)), "macro_f1": macro_f1},
        },
        "ece": round(expected_calibration_error(probs, yn), 4),
        "ood": {"auroc": auroc, "threshold": ood_thr, "fpr_at_threshold": fpr95,
                "n_not_leaf": int((~leaf).sum())},
        "latency_ms_per_image": round(latency_ms, 2), "device": str(device),
        "per_source": per_source, "per_crop": per_crop, "per_class": per_class, "crop_tiers": tiers,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics_disease.json").write_text(json.dumps(metrics, indent=1))
    ckpt["crop_tiers"] = tiers
    ckpt["metrics_summary"] = metrics["headline"]
    torch.save(ckpt, ckpt_path)

    rep = output / "reports"
    rep.mkdir(exist_ok=True)
    for c in per_crop:
        cls_idx = [i for i, n in enumerate(classes) if split_label(n)[0] == c]
        if 1 < len(cls_idx) <= 20:
            confusion_png(cls_idx, classes, pred, yn, rep / f"disease_confusion_{c.lower()}.png", f"{c} (test)")

    def pct(v):
        return "-" if v is None else f"{v * 100:.1f}%"

    h = metrics["headline"]
    lines = [f"# Disease classifier evaluation ({metrics['evaluated_at']})", "",
             f"Checkpoint `{ckpt_path.name}`, arch `{ckpt['arch']}`, {ckpt['img_size']}px, {len(classes)} classes, "
             f"temperature {T:.3f}, {latency_ms:.1f} ms/image on {device}.", "",
             "## Headline", "", "| Set | Images | Top-1 | Top-3 |", "|---|---|---|---|",
             f"| **KrishiDisha own field test** | {h['own_field_test']['n']} | {pct(h['own_field_test']['top1'])} | {pct(h['own_field_test']['top3'])} |",
             f"| Public field test (all sources) | {h['public_field_test']['n']} | {pct(h['public_field_test']['top1'])} | {pct(h['public_field_test']['top3'])} |",
             f"| All test incl. not-a-leaf | {h['all_test']['n']} | {pct(h['all_test']['top1'])} | {pct(h['all_test']['top3'])} |",
             "", f"Macro-F1 {pct(macro_f1)}; ECE {metrics['ece']}; not-a-leaf AUROC {pct(auroc)}"
             + (f", leaf-pass threshold {ood_thr:.3f} lets {pct(fpr95)} of non-leaves through" if fpr95 is not None else ""),
             "", "## Per source", "", "| Source | Images | Top-1 | Top-3 |", "|---|---|---|---|"]
    lines += [f"| {s} | {v['n']} | {pct(v['top1'])} | {pct(v['top3'])} |" for s, v in per_source.items()]
    lines += ["", "## Per crop (field test)", "", "| Crop | Tier | Train field imgs | Test imgs | Top-1 | Top-3 | Classes |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {c} | {v['tier']} | {v['n_train_field']} | {v['n_test']} | {pct(v['top1'])} | {pct(v['top3'])} | {len(v['classes'])} |"
              for c, v in per_crop.items()]
    lines += ["", "Tier A: >= 1,000 field training images and >= 90% top-1. B: >= 300 and >= 80%. C: experimental "
              "(shown with a warning in the app).", "", "## Per class", "", "| Class | Support | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {v['support']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |" for n, v in per_class.items()]
    (rep / "disease_eval.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------------------------------------------------------- model card
    sources_meta = {}
    if sources_yaml and sources_yaml.exists():
        import yaml

        sources_meta = yaml.safe_load(open(sources_yaml, encoding="utf-8"))["sources"]
    card = [f"# Model card: KrishiDisha field disease classifier", "",
            f"* Architecture: `{ckpt['arch']}`, input {ckpt['img_size']}px, {len(classes)} classes (taxonomy {TAXONOMY_VERSION})",
            f"* Trained: {ckpt.get('trained_at', '?')} on {ckpt.get('train_images', '?')} field images from {', '.join(ckpt.get('sources', []))}",
            f"* Validation top-1 {pct(ckpt.get('val_accuracy'))}; calibrated with temperature {T:.3f}",
            f"* Headline: own field test {pct(h['own_field_test']['top1'])} top-1 ({h['own_field_test']['n']} images); "
            f"public field test {pct(h['public_field_test']['top1'])} top-1 ({h['public_field_test']['n']} images)",
            "* PlantVillage (lab photography) is NOT part of the training set or any number above.", "",
            "## Intended use", "Decision support for Indian farmers photographing a single leaf in daylight. Tier C crops are experimental; "
            "every prediction below the uncertainty threshold is shown as uncertain. Not a substitute for a KVK diagnosis.", "",
            "## Data sources and licences", "", "| Source | Domain | Country | Licence |", "|---|---|---|---|"]
    for s in per_source:
        m = sources_meta.get(s, {})
        card.append(f"| {s} | {m.get('domain', '?')} | {m.get('country', '?')} | {m.get('licence', '?')} |")
    card += ["", "## Crop coverage", "", "| Crop | Tier | Top-1 |", "|---|---|---|"]
    card += [f"| {c} | {v['tier']} | {pct(v['top1'])} |" for c, v in per_crop.items()]
    (rep / "plant_disease_card.md").write_text("\n".join(card), encoding="utf-8")
    print("\n".join(lines[:16]))
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("models/plant_disease_model.pt"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("models"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args(argv)
    from ..datasets import SOURCES_FILE

    run_eval(args.checkpoint, args.manifest, args.output, args.batch_size, args.num_workers, SOURCES_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
