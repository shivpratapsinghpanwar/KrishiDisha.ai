"""Dataset registry for KrishiDisha's own models.

* ``sources.yaml``            - every image source: where it comes from, its licence, how to extract it,
                                and which label-mapping to use. Adding a crop = one entry here + one
                                mapping in :mod:`ml.datasets.taxonomy`.
* :mod:`ml.datasets.download` - fetch (Kaggle / git / http / local zip) into ``$KRISHIDISHA_DATA_ROOT``.
* :mod:`ml.datasets.taxonomy` - canonical ``Crop___Condition`` labels shared by training and the app.
* :mod:`ml.datasets.build_disease_manifest` - unify all sources into one manifest + pre-resized images.

Owner's rule: only *field* photographs of crops Indian farmers grow are used to train the served
model. PlantVillage (lab photography) is registered so the zip on disk can be extracted, but it is
``domain: lab`` and excluded from training and from every reported metric unless explicitly opted
into as a pre-training stage.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path(os.getenv("KRISHIDISHA_DATA_ROOT", r"C:/Shivpratap_Singh_Official_Work/datasets"))
SOURCES_FILE = Path(__file__).with_name("sources.yaml")


def data_root() -> Path:
    root = Path(os.getenv("KRISHIDISHA_DATA_ROOT", DEFAULT_DATA_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    return root
