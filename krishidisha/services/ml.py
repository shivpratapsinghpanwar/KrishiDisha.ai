"""Tabular ML models: crop recommendation, fertilizer recommendation, yield prediction.

Models are loaded lazily from MODELS_DIR. If a model file is missing the
service trains it on the bundled CSV (takes seconds) so the app always works.
Use `python -m ml.train_all` to (re)train with full evaluation reports.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FERT_NUMERIC = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
FERT_CATEGORICAL = ["Soil Type", "Crop Type"]
YIELD_NUMERIC = ["Crop_Year", "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]
YIELD_CATEGORICAL = ["Crop", "Season", "State"]

SOIL_TYPES = ["Black", "Clayey", "Loamy", "Red", "Sandy"]
FERT_CROP_TYPES = ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil seeds", "Paddy", "Pulses",
                   "Sugarcane", "Tobacco", "Wheat"]

# Indicative economics per acre (INR) used by the crop recommendation page and
# reports. Mirrors the values in the original KrishiDisha app.
CROP_ECONOMICS: dict[str, dict[str, Any]] = {
    "rice": {"price_per_acre": 60000, "cost_per_acre": 6000, "image": "images/rice.jpeg"},
    "maize": {"price_per_acre": 30000, "cost_per_acre": 5000, "image": "images/maize.png"},
    "jute": {"price_per_acre": 45000, "cost_per_acre": 7000, "image": "images/jute.jpg"},
    "cotton": {"price_per_acre": 65000, "cost_per_acre": 8000, "image": "images/cotton.png"},
    "coconut": {"price_per_acre": 55000, "cost_per_acre": 4000, "image": "images/coconut.jpg"},
    "papaya": {"price_per_acre": 150000, "cost_per_acre": 30000, "image": "images/papaya.png"},
    "orange": {"price_per_acre": 120000, "cost_per_acre": 25000, "image": "images/orange.png"},
    "apple": {"price_per_acre": 200000, "cost_per_acre": 35000, "image": "images/apple.jpg"},
    "muskmelon": {"price_per_acre": 90000, "cost_per_acre": 15000, "image": "images/muskmelon.jfif"},
    "watermelon": {"price_per_acre": 100000, "cost_per_acre": 18000, "image": "images/watermelon.jfif"},
    "grapes": {"price_per_acre": 175000, "cost_per_acre": 40000, "image": "images/grapes.jfif"},
    "mango": {"price_per_acre": 120000, "cost_per_acre": 20000, "image": "images/mango.png"},
    "banana": {"price_per_acre": 140000, "cost_per_acre": 25000, "image": "images/crop4.png"},
    "pomegranate": {"price_per_acre": 160000, "cost_per_acre": 30000, "image": "images/crop5.png"},
    "lentil": {"price_per_acre": 45000, "cost_per_acre": 8000, "image": "images/crop6.png"},
    "blackgram": {"price_per_acre": 40000, "cost_per_acre": 6000, "image": "images/mothbeans.png"},
    "mungbean": {"price_per_acre": 50000, "cost_per_acre": 7000, "image": "images/mothbeans.png"},
    "mothbeans": {"price_per_acre": 35000, "cost_per_acre": 5000, "image": "images/mothbeans.png"},
    "pigeonpeas": {"price_per_acre": 55000, "cost_per_acre": 9000, "image": "images/pip.png"},
    "kidneybeans": {"price_per_acre": 60000, "cost_per_acre": 10000, "image": "images/kindneybeans.png"},
    "chickpea": {"price_per_acre": 55000, "cost_per_acre": 8500, "image": "images/chickpeas.jpg"},
    "coffee": {"price_per_acre": 200000, "cost_per_acre": 40000, "image": "images/coffee.jpg"},
}

FERTILIZER_INFO: dict[str, dict[str, Any]] = {
    "Urea": {"npk": "46-0-0", "image": "images/fertilizers/Urea.jpg",
             "use": "Primary nitrogen source. Apply in 2-3 splits; avoid application before heavy rain."},
    "DAP": {"npk": "18-46-0", "image": "images/fertilizers/dap.jpg",
            "use": "Di-ammonium phosphate. Best as a basal dose at sowing for phosphorus-hungry crops."},
    "14-35-14": {"npk": "14-35-14", "image": "images/fertilizers/14-35-14.png",
                 "use": "High-phosphorus complex for root establishment and flowering."},
    "28-28": {"npk": "28-28-0", "image": "images/fertilizers/28-28.png",
              "use": "Balanced N and P complex, good for cereals at tillering."},
    "17-17-17": {"npk": "17-17-17", "image": "images/fertilizers/17-17-17.png",
                 "use": "Fully balanced NPK. Suits vegetables, fruits and sugarcane."},
    "20-20": {"npk": "20-20-0", "image": "images/fertilizers/20-20.png",
              "use": "Balanced N and P with sulphur; good for oilseeds and pulses."},
    "10-26-26": {"npk": "10-26-26", "image": "images/fertilizers/10-26-26.png",
                 "use": "High P and K complex for tuber crops, cotton and fruit set."},
}


class MLService:
    """Lazy-loading wrapper around the three tabular models."""

    def __init__(self, data_dir: Path, models_dir: Path):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._crop = None
        self._fert = None
        self._yield = None
        self._yield_meta: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------ crop
    @property
    def crop_model(self):
        if self._crop is None:
            with self._lock:
                if self._crop is None:
                    self._crop = self._load_or_train("crop_recommendation_model.pkl", self._train_crop)
        return self._crop

    def _train_crop(self):
        from sklearn.ensemble import RandomForestClassifier

        df = pd.read_csv(self.data_dir / "Crop_recommendation.csv")
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(df[CROP_FEATURES], df["label"])
        return clf

    def recommend_crop(self, N: float, P: float, K: float, temperature: float, humidity: float,
                       ph: float, rainfall: float, top_k: int = 3) -> dict[str, Any]:
        X = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=CROP_FEATURES)
        model = self.crop_model
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        order = np.argsort(proba)[::-1][:top_k]
        ranked = [{"crop": classes[i], "probability": round(float(proba[i]), 4)} for i in order]
        best = ranked[0]["crop"]
        econ = CROP_ECONOMICS.get(best, {})
        revenue = econ.get("price_per_acre")
        cost = econ.get("cost_per_acre")
        return {
            "recommended_crop": best,
            "confidence": ranked[0]["probability"],
            "alternatives": ranked[1:],
            "revenue_per_acre": revenue,
            "cost_per_acre": cost,
            "profit_per_acre": (revenue - cost) if revenue is not None and cost is not None else None,
            "image": econ.get("image"),
        }

    # ------------------------------------------------------------ fertilizer
    @property
    def fertilizer_model(self):
        if self._fert is None:
            with self._lock:
                if self._fert is None:
                    self._fert = self._load_or_train("fertilizer_recommendation_model.pkl", self._train_fertilizer)
        return self._fert

    def _train_fertilizer(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        df = pd.read_csv(self.data_dir / "Fertilizer Prediction.csv")
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), FERT_CATEGORICAL),
            ("num", "passthrough", FERT_NUMERIC),
        ])
        pipe = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=300, random_state=42))])
        pipe.fit(df[FERT_CATEGORICAL + FERT_NUMERIC], df["Fertilizer Name"])
        return pipe

    def recommend_fertilizer(self, temperature: float, humidity: float, moisture: float, soil_type: str,
                             crop_type: str, N: float, K: float, P: float) -> dict[str, Any]:
        X = pd.DataFrame([{
            "Temparature": temperature, "Humidity": humidity, "Moisture": moisture,
            "Soil Type": soil_type, "Crop Type": crop_type,
            "Nitrogen": N, "Potassium": K, "Phosphorous": P,
        }])
        model = self.fertilizer_model
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        order = np.argsort(proba)[::-1][:3]
        ranked = [{"fertilizer": classes[i], "probability": round(float(proba[i]), 4)} for i in order]
        best = ranked[0]["fertilizer"]
        info = FERTILIZER_INFO.get(best, {})
        return {
            "recommended_fertilizer": best,
            "confidence": ranked[0]["probability"],
            "alternatives": ranked[1:],
            "npk": info.get("npk"),
            "usage_note": info.get("use"),
            "image": info.get("image"),
        }

    # ------------------------------------------------------------------ yield
    @property
    def yield_model(self):
        if self._yield is None:
            with self._lock:
                if self._yield is None:
                    self._yield = self._load_or_train("yield_prediction_pipeline.pkl", self._train_yield)
        return self._yield

    def _yield_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / "crop_yield.csv")
        for col in YIELD_CATEGORICAL:
            df[col] = df[col].astype(str).str.strip()
        return df

    def _train_yield(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        df = self._yield_df()
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), YIELD_CATEGORICAL),
            ("num", "passthrough", YIELD_NUMERIC),
        ])
        pipe = Pipeline([("pre", pre), ("gbr", GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                                                          learning_rate=0.08, random_state=42))])
        pipe.fit(df[YIELD_CATEGORICAL + YIELD_NUMERIC], df["Yield"])
        return pipe

    @property
    def yield_meta(self) -> dict[str, list[str]]:
        """Unique crops / states / seasons for form dropdowns and validation."""
        if self._yield_meta is None:
            meta_path = self.models_dir / "yield_meta.json"
            if meta_path.exists():
                self._yield_meta = json.loads(meta_path.read_text())
            else:
                df = self._yield_df()
                self._yield_meta = {
                    "crops": sorted(df["Crop"].unique().tolist()),
                    "states": sorted(df["State"].unique().tolist()),
                    "seasons": sorted(df["Season"].unique().tolist()),
                }
                meta_path.write_text(json.dumps(self._yield_meta))
        return self._yield_meta

    def predict_yield(self, crop: str, crop_year: int, season: str, state: str, area: float, production: float,
                      annual_rainfall: float, fertilizer: float, pesticide: float) -> dict[str, Any]:
        X = pd.DataFrame([{
            "Crop": crop.strip(), "Season": season.strip(), "State": state.strip(), "Crop_Year": crop_year,
            "Area": area, "Production": production, "Annual_Rainfall": annual_rainfall,
            "Fertilizer": fertilizer, "Pesticide": pesticide,
        }])
        pred = float(self.yield_model.predict(X)[0])
        pred = max(pred, 0.0)
        tips = []
        if pred < 1.0:
            tips.append("Predicted yield is low for this crop. Consider soil testing and improving organic matter.")
        if fertilizer / max(area, 1) < 50:
            tips.append("Fertilizer use per hectare is below typical levels; follow a soil-test based dose.")
        if annual_rainfall < 600:
            tips.append("Low rainfall year: plan supplemental irrigation and mulching to conserve moisture.")
        return {
            "predicted_yield": round(pred, 3),
            "unit": "tonnes per hectare",
            "estimated_production": round(pred * area, 2),
            "tips": tips,
        }

    # --------------------------------------------------------------- helpers
    def _load_or_train(self, filename: str, trainer):
        path = self.models_dir / filename
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as exc:  # pragma: no cover - corrupted file
                log.warning("Could not load %s (%s); retraining", path, exc)
        log.info("Training %s from bundled data (first run only)...", filename)
        model = trainer()
        try:
            joblib.dump(model, path, compress=3)
        except Exception as exc:  # pragma: no cover
            log.warning("Could not persist %s: %s", path, exc)
        return model

    def status(self) -> dict[str, bool]:
        return {
            "crop_model": (self.models_dir / "crop_recommendation_model.pkl").exists(),
            "fertilizer_model": (self.models_dir / "fertilizer_recommendation_model.pkl").exists(),
            "yield_model": (self.models_dir / "yield_prediction_pipeline.pkl").exists(),
        }
