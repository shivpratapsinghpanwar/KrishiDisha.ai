"""Architecture factory and image transforms for the plant-disease CNN.

This module is imported both by the training script and, at request time, by
:meth:`krishidisha.services.disease.DiseaseDetector._load_local`, which calls
``build_model(arch, num_classes, pretrained=False)`` and then loads the saved
``state_dict``.  Keep ``build_model`` free of heavy side effects and keep the
head replacement deterministic so old checkpoints stay loadable.

Supported architectures
-----------------------
``mobilenet_v3_large``
    ~5.4M params, the default: fastest to train and small enough to ship.
``efficientnet_b0``
    ~5.3M params, usually a point or two more accurate, slower.
``resnet50``
    ~25M params, the heavyweight baseline.

Inference-time preprocessing must match ``get_transforms(train=False)``:
resize to ``img_size`` x ``img_size``, ``ToTensor``, ImageNet mean/std
normalisation.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_ARCHS = ("mobilenet_v3_large", "efficientnet_b0", "resnet50")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a torchvision backbone with its classifier head resized.

    Parameters
    ----------
    arch:
        One of :data:`SUPPORTED_ARCHS`.
    num_classes:
        Number of output logits (38 for the augmented PlantVillage set).
    pretrained:
        Load ImageNet weights.  Training uses ``True``; checkpoint loading uses
        ``False`` (the fine-tuned weights are about to overwrite everything and
        we do not want a network download inside a web request).

    Returns
    -------
    torch.nn.Module
    """
    arch = arch.lower()
    if arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch {arch!r}; expected one of {SUPPORTED_ARCHS}")
    return model


def get_transforms(train: bool, img_size: int = 224) -> transforms.Compose:
    """Return the train (augmented) or eval (deterministic) transform pipeline.

    The eval pipeline is intentionally identical to the one hard-coded in
    ``DiseaseDetector._load_local`` so that offline metrics match production.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
