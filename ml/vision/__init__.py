"""KrishiDisha vision training package (field disease classifier).

Modules
-------
``ml.vision.model``     - ``build_model`` / ``get_transforms`` (timm + torchvision backbones)
``ml.vision.data``      - manifest-driven dataset, balanced sampler, augmentation
``ml.vision.train``     - training loop (AMP, EMA, Mixup/CutMix, resume, stage-B fine-tune)
``ml.vision.calibrate`` - temperature scaling + not-a-leaf / uncertainty thresholds
``ml.vision.eval``      - per-crop / per-source evaluation report and model card
``ml.vision.export``    - ONNX export + registry entry

Usage::

    python -m ml.vision.train --manifest <DATA_ROOT>/disease_unified/manifest.csv --arch timm:efficientnet_b0
    python -m ml.vision.eval  --checkpoint models/plant_disease_model.pt --manifest ...
    python -m ml.vision.export --checkpoint models/plant_disease_model.pt
"""
