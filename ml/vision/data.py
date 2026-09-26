"""Manifest-driven dataset, balanced sampling and Mixup/CutMix for the disease classifier."""
from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["phash"] = int(r["phash"]) if r.get("phash") else 0
    return rows


class ManifestDataset(Dataset):
    """Rows of ``manifest.csv`` (already filtered to one split) -> (image tensor, class index)."""

    def __init__(self, rows: list[dict], root: Path, classes: list[str], transform, limit_per_class: int = 0,
                 seed: int = 42):
        self.root = Path(root)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        rows = [r for r in rows if r["label"] in self.class_to_idx]
        if limit_per_class:
            rng = random.Random(seed)
            by_cls: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by_cls[r["label"]].append(r)
            rows = []
            for items in by_cls.values():
                rng.shuffle(items)
                rows.extend(items[:limit_per_class])
        self.rows = rows
        self.transform = transform
        self.targets = [self.class_to_idx[r["label"]] for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        with Image.open(self.root / r["path"]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


def balanced_sampler(ds: ManifestDataset, field_weight: float = 1.0, own_weight: float = 3.0,
                     max_class_share: float = 0.03, max_source_share: float = 0.25, seed: int = 42,
                     num_samples: int | None = None) -> WeightedRandomSampler:
    """Per-sample weights: 1/sqrt(class_count), x field_weight for field photos, x own_weight for the
    KrishiDisha collection; then capped so no class exceeds ``max_class_share`` and no source exceeds
    ``max_source_share`` of an epoch's expected draws."""
    class_counts = Counter(ds.targets)
    w = []
    for r, t in zip(ds.rows, ds.targets):
        wt = 1.0 / math.sqrt(class_counts[t])
        if r.get("domain") == "field":
            wt *= field_weight
        if r.get("source") == "own_photos":
            wt *= own_weight
        w.append(wt)
    total = sum(w)
    # cap class share
    cls_share: dict[int, float] = defaultdict(float)
    for wt, t in zip(w, ds.targets):
        cls_share[t] += wt / total
    n_cls = len(class_counts)
    cap = max(max_class_share, 1.0 / n_cls) if n_cls else 1.0
    scale_cls = {t: min(1.0, cap / s) if s > 0 else 1.0 for t, s in cls_share.items()}
    w = [wt * scale_cls[t] for wt, t in zip(w, ds.targets)]
    # cap source share
    total = sum(w)
    src_share: dict[str, float] = defaultdict(float)
    for wt, r in zip(w, ds.rows):
        src_share[r["source"]] += wt / total
    n_src = len(src_share)
    cap_s = max(max_source_share, 1.0 / n_src) if n_src else 1.0
    scale_src = {s: min(1.0, cap_s / sh) if sh > 0 else 1.0 for s, sh in src_share.items()}
    w = [wt * scale_src[r["source"]] for wt, r in zip(w, ds.rows)]
    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples or len(ds), replacement=True,
                                 generator=g)


class MixupCutmix:
    """Batch-level Mixup / CutMix producing soft targets (label smoothing folded in)."""

    def __init__(self, num_classes: int, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, prob: float = 0.5,
                 switch_prob: float = 0.5, label_smoothing: float = 0.1):
        self.n = num_classes
        self.mixup_alpha, self.cutmix_alpha, self.prob, self.switch_prob = mixup_alpha, cutmix_alpha, prob, switch_prob
        self.smoothing = label_smoothing

    def _one_hot(self, y: torch.Tensor) -> torch.Tensor:
        off = self.smoothing / self.n
        on = 1.0 - self.smoothing + off
        return torch.full((y.size(0), self.n), off, device=y.device).scatter_(1, y.unsqueeze(1), on)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        y1 = self._one_hot(y)
        if self.prob <= 0 or random.random() > self.prob:
            return x, y1
        perm = torch.randperm(x.size(0), device=x.device)
        y2 = y1[perm]
        if random.random() < self.switch_prob and self.cutmix_alpha > 0:
            lam = random.betavariate(self.cutmix_alpha, self.cutmix_alpha)
            H, W = x.shape[-2:]
            rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
            cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
            y0, y1_, x0, x1_ = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
            x = x.clone()
            x[:, :, y0:y1_, x0:x1_] = x[perm][:, :, y0:y1_, x0:x1_]
            lam = 1 - ((y1_ - y0) * (x1_ - x0) / (H * W))
        else:
            lam = random.betavariate(self.mixup_alpha, self.mixup_alpha)
            x = x * lam + x[perm] * (1 - lam)
        return x, y1 * lam + y2 * (1 - lam)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(-target * torch.log_softmax(logits, dim=-1), dim=-1).mean()
