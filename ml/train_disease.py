"""Fine-tune a torchvision CNN on the PlantVillage leaf-disease dataset.

Expects an ``ImageFolder``-shaped dataset::

    <data-dir>/train/<class name>/*.jpg
    <data-dir>/valid/<class name>/*.jpg     (``val`` and ``validation`` also work)

``--data-dir`` may point either directly at the folder containing ``train`` and
``valid``, or at any ancestor of it (the Kaggle "New Plant Diseases Dataset
(Augmented)" zip nests the real root two levels deep) - the script walks down
to find it.

Outputs (all under ``--output``, default ``models/``)
----------------------------------------------------
``plant_disease_model.pt``
    ``torch.save`` dict with keys ``arch``, ``state_dict``, ``val_accuracy``
    (float in 0-1), ``classes``, ``img_size``, ``epoch`` - exactly what
    :meth:`krishidisha.services.disease.DiseaseDetector._load_local` reads.
``plant_disease_classes.json``
    JSON list of class names in label-index order.
``disease_training_log.json``
    Per-epoch loss / accuracy / lr / duration, appended as training proceeds so
    an interrupted run still leaves a readable history.
``metrics_disease.json``
    Final per-class precision / recall / F1 from ``classification_report``.
``reports/disease_confusion_matrix.png``

Training recipe: AdamW + OneCycleLR, label smoothing 0.1, mixed precision
(``torch.amp`` autocast + ``GradScaler``) and ``channels_last`` memory format -
tuned for a 4 GB RTX 2050 at batch size 32-48.

Examples
--------
Smoke test::

    python -m ml.train_disease --data-dir <root> --limit-per-class 40 --epochs 1

Full run::

    python -m ml.train_disease --data-dir <root> --epochs 6 --batch-size 48
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

from ml.disease_model import build_model, get_transforms

VALID_DIR_NAMES = ("valid", "val", "validation")


# --------------------------------------------------------------------------
# dataset discovery
# --------------------------------------------------------------------------
def find_dataset_root(start: Path, max_depth: int = 4) -> Path:
    """Walk down from ``start`` until a folder holding train/ + valid/ is found.

    Raises
    ------
    FileNotFoundError
        If no such folder exists within ``max_depth`` levels.
    """
    start = Path(start)
    queue: list[tuple[Path, int]] = [(start, 0)]
    while queue:
        node, depth = queue.pop(0)
        if not node.is_dir():
            continue
        if (node / "train").is_dir() and any((node / v).is_dir() for v in VALID_DIR_NAMES):
            return node
        if depth < max_depth:
            for child in sorted(node.iterdir()):
                if child.is_dir():
                    queue.append((child, depth + 1))
    raise FileNotFoundError(f"No folder with train/ and valid/ subfolders found under {start}")


def valid_dir(root: Path) -> Path:
    """Return the validation split directory inside ``root``."""
    for name in VALID_DIR_NAMES:
        if (root / name).is_dir():
            return root / name
    raise FileNotFoundError(f"No validation split in {root}")


def limit_dataset(ds: ImageFolder, per_class: int, seed: int = 42) -> Subset:
    """Deterministically down-sample ``ds`` to ``per_class`` images per class."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(ds.samples):
        by_class.setdefault(label, []).append(idx)
    keep: list[int] = []
    for label, idxs in by_class.items():
        arr = np.asarray(idxs)
        if len(arr) > per_class:
            arr = rng.choice(arr, size=per_class, replace=False)
        keep.extend(int(i) for i in arr)
    keep.sort()
    return Subset(ds, keep)


# --------------------------------------------------------------------------
# train / eval loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None,
              scaler=None, use_amp: bool = False, collect: bool = False):
    """Run one train (``optimizer`` given) or eval pass.

    Returns ``(mean_loss, accuracy, y_true, y_pred)``; the label arrays are
    empty unless ``collect`` is True.
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_n = 0
    correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(amp_device, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        preds = out.argmax(1)
        bs = y.size(0)
        total_loss += float(loss.detach()) * bs
        total_n += bs
        correct += int((preds == y).sum())
        if collect:
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
        if step % 50 == 0:
            phase = "train" if train_mode else "valid"
            print(f"    [{phase}] step {step}/{len(loader)} loss={total_loss / max(total_n, 1):.4f} "
                  f"acc={correct / max(total_n, 1):.4f}", flush=True)

    return total_loss / max(total_n, 1), correct / max(total_n, 1), y_true, y_pred


def plot_confusion(cm: np.ndarray, labels: list[str], out: Path) -> None:
    """Save a normalised confusion-matrix heatmap."""
    n = len(labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(7, n * 0.32)))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    short = [lbl.replace("___", " | ").replace("_", " ")[:38] for lbl in labels]
    ax.set_xticks(range(n), short, rotation=90, fontsize=6)
    ax.set_yticks(range(n), short, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Plant disease confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse arguments, fine-tune the CNN and write all artefacts."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="dataset root (or any ancestor of the folder holding train/ and valid/)")
    p.add_argument("--arch", default="mobilenet_v3_large",
                   choices=["mobilenet_v3_large", "efficientnet_b0", "resnet50"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--limit-per-class", type=int, default=0,
                   help="cap images per class in each split (0 = use everything); for smoke tests")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("models"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch (not recommended)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f"device={device} amp={use_amp} arch={args.arch} epochs={args.epochs} "
          f"batch={args.batch_size} img={args.img_size}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)} "
              f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB", flush=True)
        torch.backends.cudnn.benchmark = True

    root = find_dataset_root(args.data_dir)
    print(f"dataset root: {root}", flush=True)

    train_ds = ImageFolder(root / "train", transform=get_transforms(True, args.img_size))
    val_ds = ImageFolder(valid_dir(root), transform=get_transforms(False, args.img_size))
    classes = list(train_ds.classes)
    if val_ds.classes != classes:
        raise RuntimeError("train/valid class lists differ; refusing to train")
    print(f"classes={len(classes)} train={len(train_ds)} valid={len(val_ds)}", flush=True)

    train_data: Any = train_ds
    val_data: Any = val_ds
    if args.limit_per_class > 0:
        train_data = limit_dataset(train_ds, args.limit_per_class)
        val_data = limit_dataset(val_ds, max(args.limit_per_class // 2, 4))
        print(f"limited to {len(train_data)} train / {len(val_data)} valid images", flush=True)

    workers = args.num_workers if device.type == "cuda" else min(args.num_workers, 2)
    loader_kw = dict(num_workers=workers, pin_memory=(device.type == "cuda"))
    if workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_data) > args.batch_size, **loader_kw)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **loader_kw)

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained)
    model = model.to(device, memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 10, epochs=args.epochs,
        steps_per_epoch=max(len(train_loader), 1), pct_start=0.3,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = out_dir / "plant_disease_model.pt"
    classes_path = out_dir / "plant_disease_classes.json"
    log_path = out_dir / "disease_training_log.json"
    classes_path.write_text(json.dumps(classes, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_acc = -1.0
    best_epoch = -1
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n--- epoch {epoch}/{args.epochs} ---", flush=True)
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device,
                                          optimizer=optimizer, scheduler=scheduler,
                                          scaler=scaler, use_amp=use_amp)
        va_loss, va_acc, y_true, y_pred = run_epoch(model, val_loader, criterion, device,
                                                    use_amp=use_amp, collect=True)
        entry = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 5), "train_accuracy": round(tr_acc, 5),
            "val_loss": round(va_loss, 5), "val_accuracy": round(va_acc, 5),
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(entry)
        log_path.write_text(json.dumps(
            {"arch": args.arch, "epochs": args.epochs, "batch_size": args.batch_size,
             "img_size": args.img_size, "device": str(device), "amp": use_amp,
             "num_classes": len(classes), "train_images": len(train_data),
             "val_images": len(val_data), "history": history}, indent=2), encoding="utf-8")
        print(f"  epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} ({entry['seconds']}s)", flush=True)

        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save({
                "arch": args.arch,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "val_accuracy": float(va_acc),
                "classes": classes,
                "img_size": args.img_size,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  saved best checkpoint -> {ckpt_path} (val_acc={va_acc:.4f})", flush=True)
            # keep this epoch's predictions for the final report (best epoch only)
            best_true, best_pred = y_true, y_pred

    total = round(time.time() - started, 1)
    print(f"\ntraining done in {total}s; best val_accuracy={best_acc:.4f} @ epoch {best_epoch}", flush=True)

    # ---------------------------------------------------------- final report
    report = classification_report(best_true, best_pred, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(best_true, best_pred, labels=list(range(len(classes))))
    plot_confusion(cm, classes, reports / "disease_confusion_matrix.png")

    metrics = {
        "arch": args.arch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr": args.lr,
        "device": str(device),
        "amp": use_amp,
        "limit_per_class": args.limit_per_class,
        "num_classes": len(classes),
        "classes": classes,
        "train_images": len(train_data),
        "val_images": len(val_data),
        "best_val_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "total_seconds": total,
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "metrics_disease.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'metrics_disease.json'} and {reports / 'disease_confusion_matrix.png'}", flush=True)
    print(f"macro F1={report['macro avg']['f1-score']:.4f} "
          f"weighted F1={report['weighted avg']['f1-score']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
