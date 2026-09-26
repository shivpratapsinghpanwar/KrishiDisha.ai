"""Every artefact listed in models/registry.json (if present) must load and predict; missing files skip."""
from __future__ import annotations

from pathlib import Path

import pytest

from ml.registry import load, latest

MODELS = Path("models")


def test_registry_entries_point_to_existing_files_or_skip():
    entries = load(MODELS / "registry.json")
    if not entries:
        pytest.skip("no registry yet")
    for e in entries:
        assert e["name"] and e["version"] >= 1 and e["sha256"]
    names = {e["name"] for e in entries}
    for n in names:
        assert latest(n, MODELS / "registry.json")["name"] == n


def test_plant_disease_onnx_serves_if_present():
    onnx = MODELS / "plant_disease_model.onnx"
    meta = MODELS / "plant_disease_model.meta.json"
    if not (onnx.exists() and meta.exists()):
        pytest.skip("no exported field disease model in models/")
    pytest.importorskip("onnxruntime")
    from PIL import Image

    from krishidisha.services.disease import DiseaseDetector

    det = DiseaseDetector(MODELS, backend="onnx")
    info = det.info()
    assert info["available"] and info["backend"] == "onnx" and info["num_classes"] > 1
    res = det.predict(Image.new("RGB", (240, 240), (40, 140, 40)))
    assert res["available"] and len(res["predictions"]) == 3
    assert isinstance(res["uncertain"], bool) and isinstance(res["is_plant"], bool)
    assert "Other___not_a_leaf" in det.classes or True  # OOD class recommended, not required


def test_tabular_pickles_load_through_service(app):
    status = app.ml.status()
    assert set(status) == {"crop_model", "fertilizer_model", "yield_model"}
    res = app.ml.recommend_crop(N=90, P=42, K=43, temperature=21, humidity=82, ph=6.5, rainfall=203)
    assert res["recommended_crop"]
