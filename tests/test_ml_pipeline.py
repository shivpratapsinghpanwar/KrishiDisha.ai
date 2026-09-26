"""End-to-end smoke test of the vision pipeline on a synthetic 4-class dataset (CPU, ~1 minute).

Builds a tiny image source -> manifest -> 1-epoch training with a small torchvision backbone ->
calibration -> evaluation -> ONNX export -> loads the export through the app's DiseaseDetector.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from PIL import Image, ImageDraw  # noqa: E402

from ml.datasets import taxonomy  # noqa: E402

CLASSES = ["Rice___Blast", "Rice___healthy", "Wheat___Leaf_rust", "Other___not_a_leaf"]


def _make_image(path: Path, cls: int, rng: random.Random) -> None:
    img = Image.new("RGB", (96, 96), (rng.randint(0, 40), 120 + cls * 30, rng.randint(0, 40)))
    d = ImageDraw.Draw(img)
    for _ in range(cls + 1):
        x, y = rng.randint(0, 70), rng.randint(0, 70)
        d.ellipse([x, y, x + 20, y + 20], fill=(200, 50 + cls * 40, 50))
    for _ in range(40):  # random texture so perceptual hashes differ between images
        x, y = rng.randint(0, 90), rng.randint(0, 90)
        d.rectangle([x, y, x + rng.randint(2, 8), y + rng.randint(2, 8)],
                    fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG")


@pytest.fixture(scope="module")
def synthetic_manifest(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    rng = random.Random(0)
    src = root / "own_photos"
    for ci, cls in enumerate(CLASSES):
        for i in range(24):
            _make_image(src / cls / f"img_{i}.jpg", ci, rng)
    (src / "_source.json").write_text("{}")
    from ml.datasets import build_disease_manifest as b

    sources = {"own_photos": {"kind": "local_dir", "domain": "field", "mapping": "canonical", "layout": "imagefolder",
                              "licence": "own"}}
    import os

    os.environ["KRISHIDISHA_DATA_ROOT"] = str(root)
    out = root / "unified"
    summary = b.build(out, sources, ["own_photos"], resize=64, min_per_class=5, seed=1, include_lab=False, dry_run=False)
    assert summary["n_classes"] == 4 and 80 <= summary["n_images"] <= 96
    assert summary["per_split"]["test"] > 0 and summary["per_split"]["valid"] > 0
    return out / "manifest.csv"


def test_taxonomy_maps_known_and_generic_labels():
    assert taxonomy.canonical_label("paddy_doctor", "bacterial_leaf_blight") == "Rice___Bacterial_leaf_blight"
    assert taxonomy.canonical_label("rice_leaf_4", "Brownspot") == "Rice___Brown_spot"
    assert taxonomy.canonical_label("plantdoc", "Tomato leaf late blight") == "Tomato___Late_blight"
    assert taxonomy.canonical_label("cotton_leaf", "Bacterial Blight") == "Cotton___Bacterial_blight"
    assert taxonomy.canonical_label("cotton_leaf", "Healthy leaf") == "Cotton___healthy"
    assert taxonomy.canonical_label("ccmt", "leaf blight", parent="Maize") == "Maize___Leaf_blight"
    assert taxonomy.canonical_label("not_leaf", "anything") == taxonomy.NOT_A_LEAF
    assert taxonomy.canonical_label("plantvillage", "Corn_(maize)___Common_rust_") == "Maize___Common_rust"
    assert taxonomy.canonical_label("canonical", "junk") is None


def test_train_calibrate_eval_export(synthetic_manifest, tmp_path):
    from ml.vision import train

    out = tmp_path / "models"
    rc = train.main(["--manifest", str(synthetic_manifest), "--arch", "mobilenet_v3_small", "--img-size", "64",
                     "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--device", "cpu", "--output", str(out),
                     "--no-pretrained", "--mix-prob", "0.3", "--ema", "--calibrate"])
    assert rc == 0
    ckpt = torch.load(out / "plant_disease_model.pt", map_location="cpu", weights_only=False)
    assert ckpt["classes"] == sorted(CLASSES) and ckpt["format"] == 2
    assert "temperature" in ckpt and "ood_threshold" in ckpt
    assert json.loads((out / "plant_disease_classes.json").read_text()) == sorted(CLASSES)

    from ml.vision.eval import run_eval

    metrics = run_eval(out / "plant_disease_model.pt", synthetic_manifest, out, batch_size=16, workers=0)
    assert metrics["headline"]["own_field_test"]["n"] > 0
    assert (out / "reports" / "disease_eval.md").exists() and (out / "reports" / "plant_disease_card.md").exists()
    assert set(metrics["crop_tiers"]) == {"Rice", "Wheat"}

    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from ml.vision.export import export_onnx

    onnx_path = export_onnx(out / "plant_disease_model.pt", out, check=True)
    assert onnx_path.exists()

    from krishidisha.services.disease import DiseaseDetector

    det = DiseaseDetector(out, backend="onnx")
    info = det.info()
    assert info["available"] and info["backend"] == "onnx" and info["num_classes"] == 4
    img = Image.new("RGB", (80, 80), (10, 150, 10))
    res = det.predict(img)
    assert res["available"] and len(res["predictions"]) == 3
    assert {"uncertain", "is_plant", "crop_tier"} <= set(res["top"].keys() | res.keys())
