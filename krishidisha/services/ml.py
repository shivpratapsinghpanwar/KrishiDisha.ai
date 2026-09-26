"""Tabular ML models: crop recommendation, fertilizer recommendation, yield prediction.

Models are loaded lazily from MODELS_DIR. If a model file is missing (or was
built against an older feature contract) the service trains it on the bundled
CSV (takes seconds) so the app always works. Use ``python -m ml.train_tabular``
to (re)train with the full evaluation protocol and reports.

Honesty notes
-------------
* **Yield** does *not* take ``Production`` as an input. In ``crop_yield.csv``
  ``Yield == Production / Area``, so a model given production is reading the
  answer off the back of the page. Every prediction is shipped with a 10-90 %
  ``expected_range`` and the Crop x State 5-year median as ``baseline_yield``.
* **Fertilizer** is decided by an agronomic rule (nutrient requirement, soil
  test adjustment, NPK ratio match), not by the classifier. The classifier is
  reported alongside as ``model_hint`` with its honest hold-out MAP@3.
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

from ml.estimators import (
    YIELD_CATEGORICAL,
    YIELD_CONTRACT,
    YIELD_DERIVED,
    YIELD_FEATURES,
    YIELD_NUMERIC,
    MedianBaseline,
    YieldBundle,
    build_yield_features,
)

from .knowledge import FERTILIZER_REQUIREMENTS, fertilizer_calculator

log = logging.getLogger(__name__)

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FERT_NUMERIC = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
FERT_CATEGORICAL = ["Soil Type", "Crop Type"]

# YIELD_CATEGORICAL / YIELD_NUMERIC / YIELD_DERIVED are re-exported from
# ml.estimators so the trainer and the serving layer cannot drift apart.
# NB: "Production" is deliberately absent from YIELD_NUMERIC - see module docstring.
__all__ = [
    "CROP_FEATURES", "FERT_CATEGORICAL", "FERT_CROP_KEY", "FERT_CROP_TYPES", "FERT_NUMERIC",
    "FERTILIZER_INFO", "CROP_ECONOMICS", "MLService", "SOIL_TYPES",
    "YIELD_CATEGORICAL", "YIELD_DERIVED", "YIELD_FEATURES", "YIELD_NUMERIC",
]

SOIL_TYPES = ["Black", "Clayey", "Loamy", "Red", "Sandy"]
FERT_CROP_TYPES = ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil seeds", "Paddy", "Pulses",
                   "Sugarcane", "Tobacco", "Wheat"]

#: Fertilizer-dataset crop labels -> :data:`knowledge.FERTILIZER_REQUIREMENTS`
#: keys. Lives here (rather than in each blueprint) so the web form, the JSON
#: API, the FastAPI service and the chat tool all resolve the same nutrient
#: schedule.
FERT_CROP_KEY: dict[str, str] = {
    "Paddy": "rice",
    "Ground Nuts": "groundnut",
    "Oil seeds": "mustard",
    "Pulses": "chickpea",
    "Millets": "millets",
    "Barley": "barley",
    "Cotton": "cotton",
    "Maize": "maize",
    "Sugarcane": "sugarcane",
    "Tobacco": "tobacco",
    "Wheat": "wheat",
}

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

_NUTRIENTS = ("N", "P2O5", "K2O")
_NUTRIENT_LABEL = {"N": "nitrogen", "P2O5": "phosphorus", "K2O": "potassium"}


def npk_vector(product: str) -> np.ndarray:
    """Nutrient percentages of a catalogue product as an ``[N, P2O5, K2O]`` array."""
    grade = FERTILIZER_INFO[product]["npk"]
    return np.array([float(p) for p in grade.split("-")], dtype=float)


def rank_fertilizers(deficit: dict[str, float]) -> list[dict[str, Any]]:
    """Rank the catalogue by how well its NPK ratio matches a nutrient deficit.

    Cosine similarity compares *ratios*, not magnitudes, which is exactly the
    question a farmer at the shop counter is asking: "which bag has the mix my
    field is short of?"  Quantity is then handled by the dose calculator.

    Ties (``20-20`` and ``28-28`` share the 1:1:0 direction) break towards the
    higher total nutrient content, i.e. fewer bags to buy and cart.
    """
    d = np.array([max(float(deficit.get(k, 0.0)), 0.0) for k in _NUTRIENTS])
    norm = float(np.linalg.norm(d))
    ranked = []
    for name in FERTILIZER_INFO:
        v = npk_vector(name)
        vn = float(np.linalg.norm(v))
        sim = float(d @ v / (norm * vn)) if norm > 0 and vn > 0 else 0.0
        ranked.append({"fertilizer": name, "npk": FERTILIZER_INFO[name]["npk"],
                       "similarity": round(sim, 4), "total_nutrient_pct": float(v.sum())})
    ranked.sort(key=lambda r: (-round(r["similarity"], 6), -r["total_nutrient_pct"], r["fertilizer"]))
    return ranked


def _why(product: str, deficit: dict[str, float], soil_status: dict[str, str], crop_key: str) -> str:
    """One honest sentence explaining the rule's pick."""
    v = npk_vector(product)
    contents = dict(zip(_NUTRIENTS, v))
    low = [_NUTRIENT_LABEL[k] for k in _NUTRIENTS if soil_status.get(k, "").startswith("low")]
    lead = max(_NUTRIENTS, key=lambda k: deficit.get(k, 0.0))
    dose = "-".join(f"{deficit.get(k, 0.0):.0f}" for k in _NUTRIENTS)
    soil_bit = f"soil {', '.join(low)} test{'s' if len(low) == 1 else ''} low, so " if low else ""
    sentence = (f"{soil_bit}{crop_key} needs about {dose} kg/ha of N-P2O5-K2O here; "
                f"{product} ({FERTILIZER_INFO[product]['npk']}) supplies {contents[lead]:.0f}% "
                f"{_NUTRIENT_LABEL[lead]} and matches that ratio most closely.")
    return sentence[0].upper() + sentence[1:]


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
        self._yield_meta: dict[str, Any] | None = None
        self._crop_meta: dict[str, Any] | None = None
        self._metrics: dict[str, Any] | None = None

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

    @property
    def crop_meta(self) -> dict[str, Any]:
        """Per-feature 1st/99th percentiles of the training data, for the range guard."""
        if self._crop_meta is None:
            path = self.models_dir / "crop_meta.json"
            if path.exists():
                try:
                    self._crop_meta = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001 - corrupted file
                    log.warning("Bad crop_meta.json (%s); recomputing", exc)
            if self._crop_meta is None:
                df = pd.read_csv(self.data_dir / "Crop_recommendation.csv")
                self._crop_meta = {"feature_ranges": {
                    f: {"p1": float(df[f].quantile(0.01)), "p99": float(df[f].quantile(0.99))}
                    for f in CROP_FEATURES}}
        return self._crop_meta

    def _range_warnings(self, values: dict[str, float]) -> list[str]:
        """Flag inputs outside the 1st-99th percentile of the training data."""
        ranges = (self.crop_meta or {}).get("feature_ranges") or {}
        out = []
        for feat, val in values.items():
            band = ranges.get(feat)
            if not band:
                continue
            lo, hi = band["p1"], band["p99"]
            if val < lo or val > hi:
                out.append(f"{feat}={val:g} is outside the range the model was trained on "
                           f"({lo:g} to {hi:g}); treat the recommendation with caution.")
        return out

    def recommend_crop(self, N: float, P: float, K: float, temperature: float, humidity: float,
                       ph: float, rainfall: float, top_k: int = 3) -> dict[str, Any]:
        values = {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity,
                  "ph": ph, "rainfall": rainfall}
        X = pd.DataFrame([values], columns=CROP_FEATURES)
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
            "warnings": self._range_warnings(values),
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

    def _model_hint(self, temperature: float, humidity: float, moisture: float, soil_type: str,
                    crop_type: str, N: float, K: float, P: float) -> dict[str, Any] | None:
        """The classifier's opinion - reported, never obeyed."""
        try:
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
        except Exception as exc:  # noqa: BLE001 - a hint must never break the page
            log.warning("fertilizer model hint unavailable: %s", exc)
            return None
        metrics = self._task_metrics("fertilizer")
        # A MAP@3 measured on the 99-row lookup table is memorisation, so it is
        # withheld rather than quoted as evidence of skill.
        trustworthy = bool(metrics.get("map3_trustworthy"))
        return {
            "fertilizer": ranked[0]["fertilizer"],
            "confidence": ranked[0]["probability"],
            "alternatives": ranked[1:],
            "map3_on_holdout": metrics.get("map3") if trustworthy else None,
            "reliability": metrics.get("reliability") or "Its accuracy on unseen fields is unmeasured.",
            "note": "Statistical hint only - the recommendation above comes from the agronomic rule.",
        }

    def recommend_fertilizer(self, temperature: float, humidity: float, moisture: float, soil_type: str,
                             crop_type: str, N: float, K: float, P: float, area: float | None = None,
                             unit: str = "acre") -> dict[str, Any]:
        """Recommend a fertilizer product from the crop's nutrient gap (rule-first).

        ``N``, ``K`` and ``P`` are read as **available soil nutrients in kg/ha**
        (the numbers printed on a Soil Health Card). They feed
        :func:`~krishidisha.services.knowledge.fertilizer_calculator`, whose
        STCR adjustment raises the standard dose by 25 % when a nutrient tests
        low and lowers it by 25 % when it tests high. The resulting per-hectare
        requirement *is* the deficit, and the catalogue is ranked by how closely
        each product's N-P2O5-K2O ratio matches it.

        The trained classifier is still evaluated and returned as
        ``model_hint``; it is labelled, not obeyed, because on a proper hold-out
        it is barely better than guessing.

        Known limitation: the STCR adjustment scales each nutrient by at most
        +-25 %, so a soil test can shift the dose but never reorder the ratio.
        The product choice is therefore driven mainly by the crop's base
        requirement - sugarcane (250-100-120) stays nitrogen-led even with soil
        N testing high and soil P testing low.
        """
        crop_key = FERT_CROP_KEY.get(crop_type, str(crop_type).strip().lower())
        # area=1 hectare gives the requirement per hectare directly.
        per_ha = fertilizer_calculator(crop_key, 1.0, "hectare", soil_n=N, soil_p=P, soil_k=K)
        if "error" in per_ha:
            raise ValueError(per_ha["error"])
        deficit = {k: round(float(v), 1) for k, v in per_ha["nutrient_requirement_kg"].items()}
        soil_status = per_ha["soil_status"]

        ranked = rank_fertilizers(deficit)
        best = ranked[0]["fertilizer"]
        info = FERTILIZER_INFO[best]
        top3 = ranked[:3]
        result: dict[str, Any] = {
            "recommended_fertilizer": best,
            # kept for backward compatibility: the rule's ranking score, 0-1
            "confidence": top3[0]["similarity"],
            "alternatives": [{"fertilizer": r["fertilizer"], "probability": r["similarity"]} for r in top3[1:]],
            "ranked": top3,
            "why": _why(best, deficit, soil_status, crop_key),
            "deficit_kg_per_ha": deficit,
            "soil_status": soil_status,
            "crop_key": crop_key,
            "npk": info["npk"],
            "usage_note": info["use"],
            "image": info["image"],
            "method": "agronomic rule (nutrient requirement + soil test) ranked by NPK ratio match",
        }
        hint = self._model_hint(temperature, humidity, moisture, soil_type, crop_type, N, K, P)
        if hint:
            result["model_hint"] = hint
        if area:
            result["calculator"] = fertilizer_calculator(crop_key, float(area), unit,
                                                         soil_n=N, soil_p=P, soil_k=K)
        return result

    # ------------------------------------------------------------------ yield
    @property
    def yield_model(self) -> YieldBundle:
        if self._yield is None:
            with self._lock:
                if self._yield is None:
                    self._yield = self._load_or_train(
                        "yield_prediction_pipeline.pkl", self._train_yield, validate=_yield_bundle_ok)
        return self._yield

    def _yield_df(self) -> pd.DataFrame:
        """Cleaned yield rows: no non-positive yields, no nuts-per-hectare crops."""
        df = pd.read_csv(self.data_dir / "crop_yield.csv")
        for col in YIELD_CATEGORICAL:
            df[col] = df[col].astype(str).str.strip()
        return clean_yield_frame(df)

    def _train_yield(self) -> YieldBundle:
        """Fallback trainer: leakage-free, same features as ``ml.train_tabular``."""
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        df = self._yield_df()
        X, y = build_yield_features(df), df["Yield"].astype(float)
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), YIELD_CATEGORICAL),
            ("num", "passthrough", YIELD_NUMERIC + YIELD_DERIVED),
        ])
        from sklearn.compose import TransformedTargetRegressor

        point = TransformedTargetRegressor(
            regressor=Pipeline([("pre", pre),
                                ("model", HistGradientBoostingRegressor(random_state=42, max_iter=300))]),
            func=np.log1p, inverse_func=np.expm1,
        )
        point.fit(X, y)
        baseline = MedianBaseline().fit(X, y)
        return YieldBundle(point, baseline=baseline,
                           meta={"winner": "HistGradientBoosting (untuned fallback)",
                                 "trained_by": "MLService fallback - run `python -m ml.train_tabular` for the "
                                               "evaluated model, quantile band and baseline comparison",
                                 "n_samples": int(len(X))})

    @property
    def yield_meta(self) -> dict[str, Any]:
        """Dropdown values plus the per-ha input medians used to fill blanks."""
        if self._yield_meta is None:
            meta_path = self.models_dir / "yield_meta.json"
            meta = None
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    log.warning("Bad yield_meta.json (%s); recomputing", exc)
            if meta is None or "defaults" not in meta:
                df = self._yield_df()
                meta = build_yield_meta(df)
                try:
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    log.warning("Could not persist %s: %s", meta_path, exc)
            self._yield_meta = meta
        return self._yield_meta

    def _default_input(self, key: str, crop: str, state: str) -> tuple[float, str]:
        """Per-hectare median for ``fert_per_ha`` / ``pest_per_ha`` and its provenance."""
        defaults = (self.yield_meta.get("defaults") or {})
        pair = defaults.get("by_state_crop", {}).get(f"{state}|{crop}")
        if pair and pair.get(key) is not None:
            return float(pair[key]), "state_crop_median"
        by_crop = defaults.get("by_crop", {}).get(crop)
        if by_crop and by_crop.get(key) is not None:
            return float(by_crop[key]), "crop_median"
        glob = defaults.get("global", {})
        return float(glob.get(key, 0.0)), "global_median"

    def predict_yield(self, crop: str, crop_year: int, season: str, state: str, area: float,
                      annual_rainfall: float, fertilizer: float | None = None,
                      pesticide: float | None = None, **_deprecated: Any) -> dict[str, Any]:
        """Predict yield in t/ha without looking at production.

        ``fertilizer`` and ``pesticide`` are *total* kilograms for the area and
        are optional: when omitted they are filled from the State x Crop median
        per-hectare rate (falling back to crop, then global) and reported under
        ``inputs_used``.

        ``production`` used to be a required argument. It is now swallowed by
        ``**_deprecated`` and ignored, because yield is derived from production
        in the training data - passing it in was target leakage.
        """
        for key in _deprecated:
            log.warning("predict_yield: ignoring deprecated argument %r "
                        "(it leaked the target into the model)", key)

        crop, season, state = str(crop).strip(), str(season).strip(), str(state).strip()
        area = float(area)
        if area <= 0:
            raise ValueError("area must be greater than 0")

        sources: dict[str, str] = {}
        if fertilizer is None:
            rate, sources["fertilizer"] = self._default_input("fert_per_ha", crop, state)
            fertilizer = rate * area
        else:
            fertilizer, sources["fertilizer"] = float(fertilizer), "provided"
        if pesticide is None:
            rate, sources["pesticide"] = self._default_input("pest_per_ha", crop, state)
            pesticide = rate * area
        else:
            pesticide, sources["pesticide"] = float(pesticide), "provided"

        raw = pd.DataFrame([{
            "Crop": crop, "Season": season, "State": state, "Crop_Year": int(crop_year),
            "Area": area, "Annual_Rainfall": float(annual_rainfall),
            "Fertilizer": fertilizer, "Pesticide": pesticide,
        }])
        X = build_yield_features(raw)
        bundle = self.yield_model
        pred = float(bundle.predict(X)[0])

        interval = bundle.predict_interval(X)
        expected_range = None
        if interval is not None:
            lo, hi = float(interval[0][0]), float(interval[1][0])
            # The trainer ships quantiles from the same estimator family as the
            # point model, so this normally holds already. Widening rather than
            # trusting it blindly means the page can never show a farmer a
            # "likely range" that excludes the headline number.
            expected_range = [round(min(lo, pred), 3), round(max(hi, pred), 3)]

        baseline = None
        if bundle.baseline is not None:
            val = bundle.baseline.lookup(crop, state, int(crop_year))
            baseline = round(float(val), 3) if val is not None else None

        tips = []
        if pred < 1.0:
            tips.append("Predicted yield is low for this crop. Consider soil testing and improving organic matter.")
        if fertilizer / area < 50:
            tips.append("Fertilizer use per hectare is below typical levels; follow a soil-test based dose.")
        if annual_rainfall < 600:
            tips.append("Low rainfall year: plan supplemental irrigation and mulching to conserve moisture.")
        if bundle.meta.get("winner") == "MedianBaseline":
            tips.append(f"On a 2017-2020 back-test no machine-learning model beat the recent regional median "
                        f"for this dataset, so that median is what you are being shown - the honest answer "
                        f"is what {crop} has actually been yielding in {state} lately.")
        elif baseline:
            delta = (pred - baseline) / baseline * 100
            if abs(delta) >= 10:
                tips.append(f"This is {abs(delta):.0f}% {'above' if delta > 0 else 'below'} the 5-year median "
                            f"for {crop} in {state} ({baseline} t/ha).")
        if sources.get("fertilizer") != "provided" or sources.get("pesticide") != "provided":
            tips.append("Fertilizer/pesticide were filled from regional medians; enter your own figures for a "
                        "closer estimate.")

        return {
            "predicted_yield": round(pred, 3),
            "unit": "tonnes per hectare",
            "estimated_production": round(pred * area, 2),
            "expected_range": expected_range,
            "expected_range_note": _range_note(bundle.meta) if expected_range else None,
            "baseline_yield": baseline,
            "model": bundle.meta.get("winner"),
            "inputs_used": {"fertilizer": round(fertilizer, 2), "pesticide": round(pesticide, 2),
                            "source": sources},
            "tips": tips,
        }

    # --------------------------------------------------------------- helpers
    def _task_metrics(self, task: str) -> dict[str, Any]:
        """Headline numbers from ``models/metrics_tabular.json`` (empty if absent)."""
        if self._metrics is None:
            path = self.models_dir / "metrics_tabular.json"
            try:
                self._metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except Exception as exc:  # noqa: BLE001
                log.warning("Bad metrics_tabular.json: %s", exc)
                self._metrics = {}
        return (self._metrics.get("tasks") or {}).get(task, {})

    def _load_or_train(self, filename: str, trainer, validate=None):
        path = self.models_dir / filename
        if path.exists():
            try:
                model = joblib.load(path)
                if validate is None or validate(model):
                    return model
                log.warning("%s was built against an older feature contract; retraining", path)
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


# --------------------------------------------------------------------------
# module-level data helpers (shared with ml.train_tabular)
# --------------------------------------------------------------------------
#: Crops whose "Yield" is not tonnes per hectare (Coconut is nuts/ha, median
#: ~8 466) are dropped: left in, they dominate every error metric.
MAX_PLAUSIBLE_MEDIAN_YIELD = 100.0


def clean_yield_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable yield rows and crops recorded in non-tonne units."""
    df = df.dropna(subset=YIELD_CATEGORICAL + YIELD_NUMERIC + ["Yield"]).copy()
    df = df[(df["Yield"] > 0) & (df["Area"] > 0)]
    medians = df.groupby("Crop")["Yield"].median()
    keep = medians[medians <= MAX_PLAUSIBLE_MEDIAN_YIELD].index
    return df[df["Crop"].isin(keep)].reset_index(drop=True)


def build_yield_meta(df: pd.DataFrame) -> dict[str, Any]:
    """Dropdown lists plus the per-hectare medians that fill omitted inputs."""
    work = df.copy()
    work["fert_per_ha"] = work["Fertilizer"] / work["Area"]
    work["pest_per_ha"] = work["Pesticide"] / work["Area"]

    def med(grp) -> dict[str, float]:
        return {"fert_per_ha": round(float(grp["fert_per_ha"].median()), 4),
                "pest_per_ha": round(float(grp["pest_per_ha"].median()), 4)}

    by_state_crop = {f"{state}|{crop}": med(g)
                     for (crop, state), g in work.groupby(["Crop", "State"])}
    by_crop = {crop: med(g) for crop, g in work.groupby("Crop")}
    return {
        "crops": sorted(work["Crop"].unique().tolist()),
        "states": sorted(work["State"].unique().tolist()),
        "seasons": sorted(work["Season"].unique().tolist()),
        "defaults": {"by_state_crop": by_state_crop, "by_crop": by_crop, "global": med(work)},
    }


def _range_note(meta: dict[str, Any]) -> str:
    """Describe the prediction band using its *measured* coverage, not its label.

    A band advertised as "80 %" that actually held half the time would be the
    kind of claim this module exists to stop making, so the sentence shown to
    the farmer quotes what the back-test found.
    """
    nominal = meta.get("interval_nominal_coverage")
    measured = meta.get("interval_coverage")
    if measured is None:
        return "Indicative range from past seasons."
    claim = f"{nominal * 100:.0f}% prediction interval" if nominal else "Prediction interval"
    return (f"{claim}; on a 2017-2020 back-test the actual yield fell inside a band like this "
            f"{measured * 100:.0f}% of the time.")


def _yield_bundle_ok(obj: Any) -> bool:
    """True when a loaded pickle matches the current leakage-free contract."""
    return isinstance(obj, YieldBundle) and getattr(obj, "contract", None) == YIELD_CONTRACT
