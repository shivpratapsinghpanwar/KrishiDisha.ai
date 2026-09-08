"""Backbone factory and preprocessing shared by training, export and the app.

``build_model("timm:efficientnet_b0", n)`` or ``build_model("mobilenet_v3_large", n)`` (torchvision).
The checkpoint written by :mod:`ml.vision.train` records everything the app needs to reproduce
inference exactly: ``arch, img_size, mean, std, crop_pct, classes, temperature, ood_threshold``.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TORCHVISION_ARCHS = {"mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0", "efficientnet_b1", "resnet50",
                     "resnet18"}

# Sensible defaults per backbone for a 4 GB GPU (batch at 224 px with AMP) - informational.
VRAM_HINTS = {
    "timm:mobilenetv3_large_100": "bs 48 @224 ~2.0 GB", "timm:efficientnet_b0": "bs 32 @224 ~2.4 GB",
    "timm:efficientnet_b2": "bs 24 @256 ~3.2 GB", "timm:efficientnetv2_rw_s": "bs 16 @256 + grad ckpt ~3.6 GB",
    "timm:convnext_tiny": "T4/Colab only (bs 64 @224 ~9 GB)", "timm:vit_small_patch14_dinov2": "T4/Colab only",
}


def build_model(arch: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2,
                drop_path_rate: float = 0.1) -> nn.Module:
    """Create a classifier. ``arch`` is ``timm:<name>`` or a torchvision model name."""
    if arch.startswith("timm:"):
        import timm

        name = arch[5:]
        kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": num_classes, "drop_rate": drop_rate}
        try:
            return timm.create_model(name, drop_path_rate=drop_path_rate, **kwargs)
        except TypeError:  # backbone without stochastic depth
            return timm.create_model(name, **kwargs)

    from torchvision import models

    if arch not in TORCHVISION_ARCHS:
        raise ValueError(f"unknown arch {arch!r}; use timm:<name> or one of {sorted(TORCHVISION_ARCHS)}")
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, arch)(weights=weights)
    if arch.startswith("mobilenet_v3") or arch.startswith("efficientnet"):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    elif arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resolve_preprocessing(arch: str, img_size: int | None = None) -> dict[str, Any]:
    """Normalisation + crop fraction the backbone was pre-trained with (timm knows; torchvision = ImageNet)."""
    cfg = {"img_size": img_size or 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "crop_pct": 0.875}
    if arch.startswith("timm:"):
        try:
            import timm

            pcfg = timm.create_model(arch[5:], pretrained=False).pretrained_cfg
            cfg["mean"] = tuple(pcfg.get("mean", IMAGENET_MEAN))
            cfg["std"] = tuple(pcfg.get("std", IMAGENET_STD))
            cfg["crop_pct"] = float(pcfg.get("crop_pct", 0.875))
            if img_size is None and pcfg.get("input_size"):
                cfg["img_size"] = int(pcfg["input_size"][-1])
        except Exception:  # noqa: BLE001 - offline or unknown model
            pass
    return cfg


def get_transforms(train: bool, img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct: float = 0.875,
                   randaug: bool = True):
    """Training: RandomResizedCrop + flips + RandAugment + colour jitter + random erasing.
    Eval: resize(short side / crop_pct) -> centre crop, exactly what the app and ONNX export use."""
    from torchvision import transforms as T

    if train:
        aug = [T.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
               T.RandomHorizontalFlip(), T.RandomVerticalFlip(p=0.2)]
        if randaug:
            aug.append(T.RandAugment(num_ops=2, magnitude=9))
        aug += [T.ColorJitter(0.25, 0.25, 0.2, 0.03), T.ToTensor(), T.Normalize(mean, std),
                T.RandomErasing(p=0.25, scale=(0.02, 0.15))]
        return T.Compose(aug)
    resize_to = int(round(img_size / crop_pct))
    return T.Compose([T.Resize(resize_to), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(mean, std)])


class ModelEma:
    """Exponential moving average of weights (timm.utils.ModelEmaV2 equivalent, torch-only)."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        import copy

        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def checkpoint_payload(model: nn.Module, arch: str, classes: list[str], prep: dict[str, Any], **extra) -> dict:
    """The dict written to ``plant_disease_model.pt``; everything the app needs to serve the model."""
    return {
        "format": 2, "arch": arch, "classes": classes, "num_classes": len(classes),
        "img_size": prep["img_size"], "mean": list(prep["mean"]), "std": list(prep["std"]), "crop_pct": prep["crop_pct"],
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        **extra,
    }


def load_checkpoint(path, map_location="cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
