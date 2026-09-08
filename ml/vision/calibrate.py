"""Post-hoc calibration: temperature scaling + not-a-leaf / uncertainty thresholds.

    python -m ml.vision.calibrate --checkpoint models/plant_disease_model.pt --manifest <...>/manifest.csv

Fits a single temperature ``T`` on the ``valid`` split (all domains) by minimising NLL, then picks:

* ``ood_threshold``: max-softmax value below which the app should say "not confident / not a leaf",
  chosen so that 95 % of real leaf images in ``valid`` still pass;
* ``uncertain_threshold``: calibrated top-1 probability below which the UI shows "uncertain" (fixed 0.45,
  reported alongside the accuracy of the images it would flag).

Both numbers and the ECE before/after are written back into the checkpoint (``format`` 2 keys
``temperature``, ``ood_threshold``, ``uncertain_threshold``, ``calibration``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets.taxonomy import NOT_A_LEAF
from .data import ManifestDataset, read_manifest
from .model import get_transforms, load_checkpoint


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


@torch.no_grad()
def collect_logits(model, ds: ManifestDataset, device: torch.device, batch_size: int, workers: int):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model = model.to(device).eval()
    out, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def calibrate_checkpoint(ckpt_path: Path, manifest: Path, device: torch.device | None = None, batch_size: int = 64,
                         workers: int = 2, uncertain_threshold: float = 0.45, leaf_pass_rate: float = 0.95) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(ckpt_path)
    classes = ckpt["classes"]
    rows = [r for r in read_manifest(manifest) if r["split"] == "valid"]
    ds = ManifestDataset(rows, manifest.parent, classes,
                         get_transforms(False, ckpt["img_size"], ckpt["mean"], ckpt["std"], ckpt["crop_pct"]))
    logits, y = collect_logits(model, ds, device, batch_size, workers)
    T = fit_temperature(logits, y)
    probs_raw = torch.softmax(logits, 1).numpy()
    probs = torch.softmax(logits / T, 1).numpy()
    yn = y.numpy()
    ece_before, ece_after = expected_calibration_error(probs_raw, yn), expected_calibration_error(probs, yn)

    leaf_idx = np.array([classes[t] != NOT_A_LEAF for t in yn])
    leaf_conf = probs.max(1)[leaf_idx]
    ood_threshold = float(np.quantile(leaf_conf, 1 - leaf_pass_rate)) if leaf_idx.any() else 0.2
    flagged = probs.max(1) < uncertain_threshold
    acc_flagged = float((probs.argmax(1)[flagged] == yn[flagged]).mean()) if flagged.any() else None
    acc_confident = float((probs.argmax(1)[~flagged] == yn[~flagged]).mean()) if (~flagged).any() else None

    calib = {"n_valid": int(len(yn)), "temperature": round(T, 4), "ece_before": round(ece_before, 4),
             "ece_after": round(ece_after, 4), "ood_threshold": round(ood_threshold, 4),
             "uncertain_threshold": uncertain_threshold, "share_flagged_uncertain": round(float(flagged.mean()), 4),
             "acc_when_flagged": None if acc_flagged is None else round(acc_flagged, 4),
             "acc_when_confident": None if acc_confident is None else round(acc_confident, 4)}
    ckpt.update({"temperature": T, "ood_threshold": ood_threshold, "uncertain_threshold": uncertain_threshold,
                 "calibration": calib})
    torch.save(ckpt, ckpt_path)
    print(json.dumps(calib, indent=1))
    return calib


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("models/plant_disease_model.pt"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--uncertain-threshold", type=float, default=0.45)
    args = p.parse_args(argv)
    calibrate_checkpoint(args.checkpoint, args.manifest, batch_size=args.batch_size, workers=args.num_workers,
                         uncertain_threshold=args.uncertain_threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
