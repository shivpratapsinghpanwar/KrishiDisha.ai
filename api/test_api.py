"""Smoke tests for the standalone FastAPI service (run: pytest api/test_api.py).

Uses FastAPI's in-process TestClient, the offline assistant and the stub disease
model so no network or GPU is required.
"""
from __future__ import annotations

import io
import os

os.environ.setdefault("LLM_PROVIDER", "rules")
os.environ.setdefault("DISEASE_MODEL_BACKEND", "stub")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_and_health(client):
    assert client.get("/").json()["name"] == "KrishiDisha API"
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["assistant"]["provider"] == "rules"


def test_crop_recommend(client):
    r = client.post("/crop/recommend", json={"N": 90, "P": 42, "K": 43, "temperature": 21, "humidity": 82,
                                             "ph": 6.5, "rainfall": 203})
    assert r.status_code == 200 and r.json()["recommended_crop"] == "rice"
    assert client.post("/crop/recommend", json={"N": 90}).status_code == 422


def test_fertilizer_and_calculator(client):
    r = client.post("/fertilizer/recommend", json={"temperature": 26, "humidity": 52, "moisture": 38,
                                                   "soil_type": "Sandy", "crop_type": "Maize", "N": 37, "P": 0,
                                                   "K": 0, "area": 2})
    # rule-first: with soil N, P and K all low the balanced N+P complex wins,
    # not straight Urea - see tests/test_services.py
    assert r.status_code == 200 and r.json()["recommended_fertilizer"] == "28-28"
    assert r.json()["why"] and len(r.json()["ranked"]) == 3
    assert r.json()["calculator"]["bags_50kg"]["Urea"] > 0
    r = client.post("/fertilizer/calculator", json={"crop": "wheat", "area": 1, "unit": "hectare"})
    assert r.status_code == 200 and r.json()["nutrient_requirement_kg"]["N"] == 120
    assert client.post("/fertilizer/calculator", json={"crop": "dragonfruit", "area": 1}).status_code == 400


def test_yield(client):
    r = client.post("/yield/predict", json={"crop": "Wheat", "crop_year": 2020, "season": "Rabi", "state": "Punjab",
                                            "area": 100, "annual_rainfall": 600,
                                            "fertilizer": 15000, "pesticide": 30})
    body = r.json()
    assert r.status_code == 200 and body["predicted_yield"] > 0
    assert body["expected_range"][0] <= body["predicted_yield"] <= body["expected_range"][1]
    assert body["baseline_yield"] > 0
    # fertilizer/pesticide are optional; "production" is gone from the schema
    r = client.post("/yield/predict", json={"crop": "Wheat", "crop_year": 2020, "season": "Rabi", "state": "Punjab",
                                            "area": 100, "annual_rainfall": 600})
    assert r.status_code == 200 and r.json()["inputs_used"]["fertilizer"] > 0


def test_disease_stub(client):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="JPEG")
    r = client.post("/disease/detect", files={"image": ("leaf.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 503
    r = client.post("/disease/detect", files={"image": ("x.jpg", b"nope", "image/jpeg")})
    assert r.status_code == 400


def test_chat_and_data(client):
    r = client.post("/chat", json={"message": "How much urea for 2 acres of wheat?", "plain": True})
    assert r.status_code == 200 and "Urea" in r.json()["reply"] and "**" not in r.json()["reply_plain"]
    assert client.get("/reference").json()["soil_types"][0] == "Black"
    assert client.get("/schemes", params={"q": "kisan"}).json()["schemes"]
    assert client.get("/crop-guide/rice").json()["guide"]["season"]
    assert client.get("/crop-guide/none").status_code == 404
    assert client.get("/crop-calendar", params={"season": "Kharif"}).json()["rows"]
    assert client.get("/knowledge/search", params={"q": "blight"}).json()["results"]
    assert client.get("/msp", params={"commodity": "wheat"}).json()["prices"][0]["msp"] == 2585
    assert client.get("/products", params={"disease": "Tomato___Late_blight"}).json()["products"]
