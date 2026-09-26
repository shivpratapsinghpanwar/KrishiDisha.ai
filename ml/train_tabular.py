"""Train, compare and persist the three KrishiDisha tabular models.

Tasks
-----
1. **Crop recommendation** - ``data/Crop_recommendation.csv``, 7 soil/climate
   features (``N, P, K, temperature, humidity, ph, rainfall``) -> 22 crops.
   Seven candidate estimators are compared, then a ``RandomizedSearchCV``-tuned
   random forest wrapped in ``CalibratedClassifierCV`` joins the contest so the
   published probabilities mean something.  Per-feature 1st/99th percentiles go
   to ``models/crop_meta.json`` for the serving-time out-of-range guard.
2. **Fertilizer recommendation** - trained on the 750 k-row Kaggle Playground
   Series S5E6 table when it is available (``--fert-data``), otherwise on the
   99-row ``data/Fertilizer Prediction.csv`` *with a loud warning*.  XGBoost is
   tuned with ``RandomizedSearchCV`` and calibrated with isotonic regression;
   top-1 accuracy **and MAP@3** are reported on a 20 % hold-out.  The app does
   not obey this model - it is served as a labelled hint next to an agronomic
   rule - because ~0.35 MAP@3 is what this problem honestly supports.
3. **Yield prediction** - ``data/crop_yield.csv`` -> ``Yield`` in t/ha.

   *Leakage*: ``Yield == Production / Area`` in the source data, so
   ``Production`` is **not** a feature.  Coconut (median 8 466 nuts/ha) and any
   other crop whose median yield exceeds 100 are dropped, as are non-positive
   yields.  Derived features: ``fert_per_ha``, ``pest_per_ha``, ``log_area``.
   The target is modelled as ``log1p(Yield)`` and inverted with ``expm1``.

   *Protocol*: an honest time split (fit on ``Crop_Year <= 2016``, tune on
   2014-2016, test on 2017-2020) **plus** a ``GroupKFold(5)`` grouped by
   State x Crop, both scored against a :class:`~ml.estimators.MedianBaseline`
   (the Crop x State median over the previous five years).  Optuna tunes
   XGBoost on the validation years; 0.1 / 0.9 quantile boosters supply the
   ``expected_range``.  **If the baseline wins the 2017-2020 test it is
   shipped**, because a model that cannot beat a district handbook has no
   business being called a prediction.

Method
------
Classification tasks are split 80/20 stratified, every candidate is scored with
5-fold cross-validation on the training split *and* on the held-out split, and
the winner is refit on 100 % of the data before being saved with joblib.

The saved objects honour exactly the input contracts of
:mod:`krishidisha.services.ml`, which shares its column lists with
:mod:`ml.estimators`:

* ``models/crop_recommendation_model.pkl`` - estimator taking a DataFrame with
  columns ``CROP_FEATURES``; exposes ``classes_`` / ``predict_proba``.
* ``models/fertilizer_recommendation_model.pkl`` - full pipeline taking a
  DataFrame with columns ``FERT_CATEGORICAL + FERT_NUMERIC``.
* ``models/yield_prediction_pipeline.pkl`` - a :class:`~ml.estimators.YieldBundle`
  taking a DataFrame with columns ``YIELD_FEATURES``.

Side artefacts: ``models/yield_meta.json`` (dropdown values + the per-hectare
input medians used to fill omitted inputs), ``models/crop_meta.json``,
``models/metrics_tabular.json`` (every comparison number, dataset provenance,
feature importances, confusion matrix), model cards under
``models/reports/*_card.md`` and PNG reports under ``models/reports/``.

Example
-------
``python -m ml.train_tabular --data-dir data --output models``
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (  # noqa: E402
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from ml.estimators import (  # noqa: E402
    YIELD_CATEGORICAL,
    YIELD_DERIVED,
    YIELD_FEATURES,
    YIELD_NUMERIC,
    ConformalBand,
    MedianBaseline,
    XGBLabelClassifier,
    YieldBundle,
    build_yield_features,
)
from krishidisha.services.ml import build_yield_meta, clean_yield_frame  # noqa: E402

# --------------------------------------------------------------------------
# Column contracts - YIELD_* come from ml.estimators (single source of truth)
# --------------------------------------------------------------------------
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FERT_NUMERIC = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
FERT_CATEGORICAL = ["Soil Type", "Crop Type"]
FERT_LABELS = ["10-26-26", "14-35-14", "17-17-17", "20-20", "28-28", "DAP", "Urea"]

RANDOM_STATE = 42
N_ESTIMATORS = 300  # capped so the compressed pickles stay small

# Yield time split.  2020 holds only 37 rows, so the test block is really
# 2017-2019 plus a tail.
YIELD_VAL_START, YIELD_FIT_END, YIELD_TEST_START = 2014, 2016, 2017

# Default location for the Kaggle Playground Series S5E6 fertilizer table.
DEFAULT_FERT_S5E6 = Path(r"C:\Shivpratap_Singh_Official_Work\datasets\fertilizer_s5e6\train.csv")
FERT_TUNE_SAMPLE = 200_000  # rows used for the RandomizedSearchCV sweep

DATASETS: dict[str, dict[str, Any]] = {
    "crop": {
        "name": "Crop Recommendation Dataset",
        "file": "data/Crop_recommendation.csv",
        "source_url": "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset",
        "licence": "CC0 1.0 Public Domain",
        "note": "Synthesised from Indian soil/rainfall statistics; the classes are near-separable, "
                "so a 99 % accuracy here does not transfer to a real field.",
    },
    "fertilizer": {
        "name": "Fertilizer Prediction (Kaggle)",
        "file": "data/Fertilizer Prediction.csv",
        "source_url": "https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction",
        "licence": "CC0 1.0 Public Domain",
        "note": "99 rows, one per (soil, crop, product) combination - effectively a lookup table.",
    },
    "fertilizer_s5e6": {
        "name": "Kaggle Playground Series S5E6 - Predicting Optimal Fertilizers",
        "file": "train.csv",
        "source_url": "https://www.kaggle.com/competitions/playground-series-s5e6",
        "licence": "Kaggle competition rules (non-commercial research use)",
        "note": "750 k synthetic rows with the same schema as the 99-row file; the winning public "
                "MAP@3 was ~0.38, so ~0.35 is a good score, not a broken model.",
    },
    "yield": {
        "name": "Crop Yield in Indian States",
        "file": "data/crop_yield.csv",
        "source_url": "https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset",
        "licence": "CC0 1.0 Public Domain (compiled from data.gov.in / Ministry of Agriculture "
                   "and Farmers Welfare releases)",
        "note": "State-level aggregates 1997-2020. Yield == Production / Area, so Production is "
                "excluded as a feature; Coconut is recorded in nuts/ha and is dropped.",
    },
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _one_hot() -> OneHotEncoder:
    """OneHotEncoder that tolerates unseen categories at inference time."""
    return OneHotEncoder(handle_unknown="ignore")


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _reg_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE / MAPE / R2 / RMSE on the **original** t/ha scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
    }


def map_at_k(y_true, proba: np.ndarray, classes, k: int = 3) -> float:
    """Mean average precision at ``k`` - the metric the S5E6 competition used."""
    classes = np.asarray(classes)
    order = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    top = classes[order]
    total = 0.0
    for truth, row in zip(np.asarray(y_true), top):
        for i, label in enumerate(row):
            if label == truth:
                total += 1.0 / (i + 1)
                break
    return float(total / len(top)) if len(top) else 0.0


def _classifiers(n_jobs: int = -1) -> dict[str, Any]:
    """Candidate classifiers shared by the crop and fertilizer tasks."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=n_jobs
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBLabelClassifier(params={
            "n_estimators": N_ESTIMATORS,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }),
        "SVM": Pipeline([("sc", StandardScaler(with_mean=False)), ("svc", SVC(probability=True, random_state=RANDOM_STATE))]),
        "KNN": Pipeline([("sc", StandardScaler(with_mean=False)), ("knn", KNeighborsClassifier(n_neighbors=5))]),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB(),
    }


def _wrap(pre: ColumnTransformer | None, estimator) -> Any:
    """Prefix ``estimator`` with a preprocessing step when one is given."""
    if pre is None:
        return estimator
    return Pipeline([("pre", pre), ("model", estimator)])


def _feature_names(model, fallback: list[str]) -> list[str]:
    """Best-effort expanded feature names for a fitted (possibly piped) model."""
    try:
        if isinstance(model, TransformedTargetRegressor):
            model = model.regressor_
        if isinstance(model, Pipeline) and "pre" in model.named_steps:
            return list(model.named_steps["pre"].get_feature_names_out())
    except Exception:  # pragma: no cover - defensive
        pass
    return fallback


def _importances(model) -> np.ndarray | None:
    """Pull ``feature_importances_`` out of a bare, piped, log-target or calibrated model."""
    target = model
    if isinstance(target, TransformedTargetRegressor):
        target = target.regressor_
    if isinstance(target, Pipeline):
        target = target.steps[-1][1]
    if isinstance(target, CalibratedClassifierCV):
        # The calibrator has no importances of its own; average the per-fold
        # base estimators it wraps so the report is not silently empty.
        per_fold = [getattr(c.estimator, "feature_importances_", None)
                    for c in getattr(target, "calibrated_classifiers_", [])]
        per_fold = [p for p in per_fold if p is not None]
        return np.mean(per_fold, axis=0) if per_fold else None
    return getattr(target, "feature_importances_", None)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def _plot_comparison(scores: dict[str, float], title: str, ylabel: str, out: Path,
                     higher_is_better: bool = True) -> None:
    names = list(scores)
    vals = [scores[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, vals, color="#2e7d32")
    best = int(np.argmax(vals) if higher_is_better else np.argmin(vals))
    bars[best].set_color("#f9a825")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.02, max(vals) * 1.12) if higher_is_better else max(vals) * 1.2)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, labels: list[str], title: str, out: Path) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.42), max(5, n * 0.38)))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    if n <= 25:
        thresh = cm.max() / 2 if cm.max() else 0.5
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_importance(names: list[str], values: np.ndarray, title: str, out: Path, top: int = 20) -> None:
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(order) * 0.32)))
    ax.barh([names[i] for i in order], values[order], color="#1565c0")
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_pred_vs_actual(y_true, y_pred, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.35, color="#2e7d32", edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", color="#c62828", linewidth=1)
    ax.set_xlabel("Actual yield (t/ha)")
    ax.set_ylabel("Predicted yield (t/ha)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_per_crop_mae(rows: list[dict], title: str, out: Path) -> None:
    """Grouped bars: model vs baseline MAE for the most common crops."""
    crops = [r["crop"] for r in rows][::-1]
    model = [r["model_mae"] for r in rows][::-1]
    base = [r["baseline_mae"] for r in rows][::-1]
    y = np.arange(len(crops))
    fig, ax = plt.subplots(figsize=(8, max(4, len(crops) * 0.34)))
    ax.barh(y - 0.2, model, height=0.4, label="model", color="#2e7d32")
    ax.barh(y + 0.2, base, height=0.4, label="5-year median baseline", color="#90a4ae")
    ax.set_yticks(y, crops, fontsize=8)
    ax.set_xlabel("MAE (t/ha), 2017-2020 test years")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# generic classification driver
# --------------------------------------------------------------------------
def _run_classification(
    task: str,
    X: pd.DataFrame,
    y: pd.Series,
    pre_factory,
    cv_folds: int,
    reports: Path,
    extra: dict[str, Any] | None = None,
    report_map3: bool = False,
    prefer: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compare classifiers, return ``(winner_refit_on_all_data, metrics)``.

    ``extra`` adds zero-argument factories (e.g. an already-tuned, calibrated
    estimator) to the contest on equal terms.

    ``prefer`` names a candidate that wins any *statistical tie* - i.e. it is
    shipped whenever its CV accuracy is within one CV standard deviation of the
    leader. On these near-separable datasets several estimators land within
    noise of each other, and when the scores cannot be told apart the model with
    trustworthy probabilities is the better thing to put in front of a farmer.
    """
    strat = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
    )
    folds = min(cv_folds, int(y_tr.value_counts().min()))
    folds = max(folds, 2)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    factories: dict[str, Any] = {name: (lambda e=est: e) for name, est in _classifiers().items()}
    factories.update(extra or {})

    results: dict[str, dict[str, Any]] = {}
    for name, factory in factories.items():
        model = _wrap(pre_factory(), factory())
        t0 = time.time()
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=1)
        model.fit(X_tr, y_tr)
        pred_te = model.predict(X_te)
        pred_tr = model.predict(X_tr)
        row = {
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
            "cv_folds": folds,
            "train_accuracy": float(accuracy_score(y_tr, pred_tr)),
            "test_accuracy": float(accuracy_score(y_te, pred_te)),
            "test_f1_macro": float(f1_score(y_te, pred_te, average="macro", zero_division=0)),
            "fit_seconds": round(time.time() - t0, 2),
        }
        if report_map3 and hasattr(model, "predict_proba"):
            row["test_map3"] = map_at_k(y_te, model.predict_proba(X_te), model.classes_, 3)
        results[name] = row
        extra_txt = f" MAP@3={row['test_map3']:.4f}" if "test_map3" in row else ""
        print(f"  {name:<24} cv={cv_scores.mean():.4f}+-{cv_scores.std():.4f} "
              f"test={row['test_accuracy']:.4f}{extra_txt} ({row['fit_seconds']}s)")

    best_name = max(results, key=lambda k: results[k]["cv_mean_accuracy"])
    tie_break = None
    if prefer and prefer in results and prefer != best_name:
        lead = results[best_name]["cv_mean_accuracy"]
        gap = lead - results[prefer]["cv_mean_accuracy"]
        if gap <= results[best_name]["cv_std_accuracy"]:
            tie_break = (f"{prefer} is within one CV standard deviation of {best_name} "
                         f"({gap:.4f} <= {results[best_name]['cv_std_accuracy']:.4f}) and its probabilities "
                         f"are calibrated, so it is shipped")
            best_name = prefer
    print(f"  -> best: {best_name}" + (f" ({tie_break})" if tie_break else ""))

    # holdout artefacts from the winner (fitted on the train split)
    best_holdout = _wrap(pre_factory(), factories[best_name]())
    best_holdout.fit(X_tr, y_tr)
    pred_te = best_holdout.predict(X_te)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_te, pred_te, labels=labels)
    report = classification_report(y_te, pred_te, labels=labels, output_dict=True, zero_division=0)

    _plot_comparison({k: v["cv_mean_accuracy"] for k, v in results.items()},
                     f"{task}: {folds}-fold CV accuracy", "accuracy",
                     reports / f"{task}_model_comparison.png")
    _plot_confusion(cm, labels, f"{task}: confusion matrix ({best_name}, holdout)",
                    reports / f"{task}_confusion_matrix.png")

    imp = _importances(best_holdout)
    importances: dict[str, float] = {}
    if imp is not None:
        names = _feature_names(best_holdout, list(X.columns))
        if len(names) == len(imp):
            importances = {n: float(v) for n, v in zip(names, imp)}
            _plot_importance(names, np.asarray(imp), f"{task}: feature importance ({best_name})",
                             reports / f"{task}_feature_importance.png")

    # refit the winner on ALL rows for production use
    final = _wrap(pre_factory(), factories[best_name]())
    final.fit(X, y)

    metrics = {
        "task": task,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": len(labels),
        "classes": labels,
        "split": {"train": int(len(X_tr)), "test": int(len(X_te)), "test_size": 0.2, "stratified": strat is not None},
        "comparison": results,
        "best_model": best_name,
        "best_metrics": results[best_name],
        "classification_report": report,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "feature_importances": importances,
        "refit_on_full_data": True,
    }
    if tie_break:
        metrics["tie_break"] = tie_break
    if report_map3:
        metrics["map3"] = results[best_name].get("test_map3")
        metrics["top1_accuracy"] = results[best_name]["test_accuracy"]
    return final, metrics


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
def train_crop(data_dir: Path, out_dir: Path, reports: Path, cv_folds: int, **_) -> dict[str, Any]:
    """Crop recommendation: 7 numeric features -> crop label."""
    print("[crop] loading", data_dir / "Crop_recommendation.csv")
    df = pd.read_csv(data_dir / "Crop_recommendation.csv")
    X = df[CROP_FEATURES].astype(float)
    y = df["label"].astype(str)

    # ---- RandomizedSearchCV (15 iterations) over a random forest ----------
    print("[crop] tuning RandomForest (RandomizedSearchCV, 15 iterations)...")
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [150, 250, 350, 500],
            "max_depth": [None, 6, 10, 16, 24],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "criterion": ["gini", "entropy"],
        },
        n_iter=15, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    best_params = {k: v for k, v in search.best_params_.items()}
    print(f"[crop] best params {best_params} (cv acc {search.best_score_:.4f})")

    # Sigmoid (Platt) calibration, not isotonic: 22 classes x ~100 rows leaves
    # too few points per class for a monotone step fit to be stable.
    def tuned_factory():
        return CalibratedClassifierCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params),
            method="sigmoid", cv=3)

    model, metrics = _run_classification("crop", X, y, lambda: None, cv_folds, reports,
                                         extra={"RandomForest (tuned+calibrated)": tuned_factory},
                                         prefer="RandomForest (tuned+calibrated)")
    metrics["tuning"] = {"estimator": "RandomForestClassifier", "search": "RandomizedSearchCV",
                         "n_iter": 15, "best_params": best_params,
                         "best_cv_accuracy": float(search.best_score_),
                         "calibration": "CalibratedClassifierCV(method='sigmoid', cv=3)"}
    metrics["dataset"] = DATASETS["crop"]

    path = out_dir / "crop_recommendation_model.pkl"
    joblib.dump(model, path, compress=3)
    metrics["artifact"] = path.name
    metrics["artifact_bytes"] = path.stat().st_size

    # ---- out-of-range guard reference ------------------------------------
    crop_meta = {
        "features": CROP_FEATURES,
        "n_samples": int(len(df)),
        "feature_ranges": {f: {"p1": float(X[f].quantile(0.01)), "p99": float(X[f].quantile(0.99)),
                               "min": float(X[f].min()), "max": float(X[f].max())}
                           for f in CROP_FEATURES},
        "classes": sorted(y.unique().tolist()),
    }
    meta_path = out_dir / "crop_meta.json"
    meta_path.write_text(json.dumps(crop_meta, indent=2), encoding="utf-8")
    metrics["meta_artifact"] = meta_path.name
    print(f"[crop] saved {path} ({path.stat().st_size / 1e6:.2f} MB) and {meta_path}")
    return metrics


def _load_fertilizer_data(data_dir: Path, fert_data: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prefer the 750 k-row S5E6 table; fall back to the 99-row file, loudly."""
    candidate = fert_data or DEFAULT_FERT_S5E6
    if candidate and Path(candidate).exists():
        df = pd.read_csv(candidate)
        missing = [c for c in FERT_CATEGORICAL + FERT_NUMERIC + ["Fertilizer Name"] if c not in df.columns]
        labels = sorted(df["Fertilizer Name"].astype(str).unique().tolist()) if not missing else []
        if missing:
            print(f"  !! {candidate} is missing columns {missing}; ignoring it")
        elif labels != FERT_LABELS:
            print(f"  !! {candidate} has labels {labels}, expected {FERT_LABELS}; ignoring it")
        else:
            print(f"[fertilizer] using S5E6 table {candidate} ({len(df):,} rows)")
            provenance = dict(DATASETS["fertilizer_s5e6"], file=str(candidate), rows=int(len(df)))
            return df[FERT_CATEGORICAL + FERT_NUMERIC + ["Fertilizer Name"]], provenance

    path = data_dir / "Fertilizer Prediction.csv"
    df = pd.read_csv(path)
    print("\n" + "!" * 78)
    print("!! FERTILIZER FALLBACK: the Kaggle Playground Series S5E6 table was not found at")
    print(f"!!   {candidate}")
    print("!! (`kaggle competitions download -c playground-series-s5e6` returns HTTP 403 until the")
    print("!!  competition rules are accepted in a browser).")
    print(f"!! Training on the {len(df)}-row '{path.name}' instead.  That file has one row per")
    print("!! (soil, crop, product) combination, so ANY accuracy measured on it is memorisation,")
    print("!! not generalisation.  The reported MAP@3 is meaningless - do not quote it.")
    print("!" * 78 + "\n")
    provenance = dict(DATASETS["fertilizer"], rows=int(len(df)),
                      warning="FALLBACK dataset: 99 rows, effectively a lookup table. Metrics are "
                              "memorisation, not generalisation.")
    return df[FERT_CATEGORICAL + FERT_NUMERIC + ["Fertilizer Name"]], provenance


def train_fertilizer(data_dir: Path, out_dir: Path, reports: Path, cv_folds: int,
                     fert_data: Path | None = None, **_) -> dict[str, Any]:
    """Fertilizer recommendation: one-hot Soil/Crop Type + 6 numeric -> product."""
    df, provenance = _load_fertilizer_data(data_dir, fert_data)
    X = df[FERT_CATEGORICAL + FERT_NUMERIC]
    y = df["Fertilizer Name"].astype(str)

    def pre_factory() -> ColumnTransformer:
        # Identical layout to MLService._train_fertilizer.
        return ColumnTransformer([
            ("cat", _one_hot(), FERT_CATEGORICAL),
            ("num", "passthrough", FERT_NUMERIC),
        ])

    # ---- RandomizedSearchCV (20 iterations) on a sample, then calibrate ---
    if len(X) > FERT_TUNE_SAMPLE:
        idx = X.sample(FERT_TUNE_SAMPLE, random_state=RANDOM_STATE).index
        Xs, ys = X.loc[idx], y.loc[idx]
    else:
        Xs, ys = X, y
    # Small fallback datasets cannot support 3 stratified folds per class.
    tune_folds = max(2, min(3, int(ys.value_counts().min())))
    calib_folds = max(2, min(3, int(y.value_counts().min()) // 2))
    print(f"[fertilizer] tuning XGBoost (RandomizedSearchCV, 20 iterations on {len(Xs):,} rows)...")
    search = RandomizedSearchCV(
        Pipeline([("pre", pre_factory()), ("model", XGBLabelClassifier())]),
        {
            "model__params": [
                {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss,
                 "colsample_bytree": cs, "min_child_weight": mcw, "tree_method": "hist",
                 "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0}
                for n in (300, 500, 800)
                for d in (4, 6, 8, 10)
                for lr in (0.03, 0.06, 0.1, 0.2)
                for ss in (0.7, 0.9, 1.0)
                for cs in (0.6, 0.8, 1.0)
                for mcw in (1, 5, 20)
            ],
        },
        n_iter=20, cv=StratifiedKFold(tune_folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy", random_state=RANDOM_STATE, n_jobs=1, verbose=0,
    )
    search.fit(Xs, ys)
    best_params = search.best_params_["model__params"]
    print(f"[fertilizer] best params {best_params} (cv acc {search.best_score_:.4f})")

    # Isotonic calibration: with 750 k rows there is plenty of data for a
    # non-parametric fit, and the raw booster is over-confident.
    def tuned_factory():
        return CalibratedClassifierCV(XGBLabelClassifier(params=best_params), method="isotonic",
                                      cv=calib_folds)

    model, metrics = _run_classification("fertilizer", X, y, pre_factory, cv_folds, reports,
                                         extra={"XGBoost (tuned+calibrated)": tuned_factory},
                                         report_map3=True,
                                         prefer="XGBoost (tuned+calibrated)")
    # A hold-out drawn from a 99-row lookup table measures memorisation. Say so
    # in the metrics file so the app never quotes the number as if it meant
    # something.
    trustworthy = "warning" not in provenance
    metrics["map3_trustworthy"] = trustworthy
    metrics["reliability"] = (
        f"MAP@3 {metrics.get('map3', float('nan')):.3f} on a 20% hold-out of "
        f"{len(X):,} rows." if trustworthy else
        "Its accuracy was measured on a 99-row lookup table with one row per soil/crop/product "
        "combination, so it is memorisation and does not generalise.")
    metrics["tuning"] = {"estimator": "XGBLabelClassifier", "search": "RandomizedSearchCV", "n_iter": 20,
                         "tune_rows": int(len(Xs)), "best_params": best_params,
                         "best_cv_accuracy": float(search.best_score_),
                         "cv_folds": tune_folds,
                         "calibration": f"CalibratedClassifierCV(method='isotonic', cv={calib_folds})"}
    metrics["dataset"] = provenance
    metrics["serving_note"] = ("The app does NOT obey this classifier. recommend_fertilizer() ranks the "
                               "catalogue against the crop's soil-test-adjusted nutrient deficit and returns "
                               "this model only as a labelled `model_hint`.")

    path = out_dir / "fertilizer_recommendation_model.pkl"
    joblib.dump(model, path, compress=3)
    metrics["artifact"] = path.name
    metrics["artifact_bytes"] = path.stat().st_size
    print(f"[fertilizer] saved {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"[fertilizer] HONEST HOLD-OUT: top-1 accuracy {metrics.get('top1_accuracy'):.4f}, "
          f"MAP@3 {metrics.get('map3'):.4f}")
    if not trustworthy:
        print("[fertilizer] ^ MEMORISATION, NOT SKILL: that hold-out came from the 99-row fallback table.")
    return metrics


# ------------------------------------------------------------------- yield
def _yield_pre() -> ColumnTransformer:
    """One-hot the three categoricals, pass the numerics and ratios through."""
    return ColumnTransformer([
        ("cat", _one_hot(), YIELD_CATEGORICAL),
        ("num", "passthrough", YIELD_NUMERIC + YIELD_DERIVED),
    ])


def _log_target(estimator) -> TransformedTargetRegressor:
    """Fit on ``log1p(Yield)`` and invert with ``expm1``.

    Yield is right-skewed across 50-odd crops; modelling the log keeps a
    3 t/ha wheat error and a 30 t/ha sugarcane error on comparable footing.
    """
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", _yield_pre()), ("model", estimator)]),
        func=np.log1p, inverse_func=np.expm1,
    )


def _xgb_regressor(params: dict[str, Any]):
    from xgboost import XGBRegressor

    return XGBRegressor(tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, **params)


def _tune_yield_xgb(X_fit, y_fit, X_val, y_val, n_trials: int) -> dict[str, Any]:
    """Optuna sweep scored by MAE on the 2014-2016 validation years."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        model = _log_target(_xgb_regressor(params))
        model.fit(X_fit, y_fit)
        return float(mean_absolute_error(y_val, np.clip(model.predict(X_val), 0, None)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  optuna best validation MAE {study.best_value:.4f} t/ha with {study.best_params}")
    return dict(study.best_params)


def _quantile_model(params: dict[str, Any], alpha: float):
    """XGBoost pinball-loss regressor for one tail of the prediction band."""
    q = {k: v for k, v in params.items() if k not in {"reg_alpha"}}
    return _log_target(_xgb_regressor({**q, "objective": "reg:quantileerror", "quantile_alpha": alpha}))


def train_yield(data_dir: Path, out_dir: Path, reports: Path, cv_folds: int,
                optuna_trials: int = 40, **_) -> dict[str, Any]:
    """Leakage-free yield regression with a time split and a real baseline."""
    print("[yield] loading", data_dir / "crop_yield.csv")
    raw = pd.read_csv(data_dir / "crop_yield.csv")
    for col in YIELD_CATEGORICAL:
        raw[col] = raw[col].astype(str).str.strip()
    n_raw = len(raw)
    dropped_crops = sorted(set(raw["Crop"].unique()) - set(clean_yield_frame(raw)["Crop"].unique()))
    df = clean_yield_frame(raw)
    print(f"[yield] {n_raw} rows -> {len(df)} after dropping non-positive yields and "
          f"non-tonne crops {dropped_crops}")

    X = build_yield_features(df)
    y = df["Yield"].astype(float)
    years = df["Crop_Year"].astype(int)
    groups = (df["State"] + " | " + df["Crop"]).to_numpy()

    tune_fit = years < YIELD_VAL_START
    tune_val = (years >= YIELD_VAL_START) & (years <= YIELD_FIT_END)
    fit_mask = years <= YIELD_FIT_END
    test_mask = years >= YIELD_TEST_START
    print(f"[yield] time split: fit<= {YIELD_FIT_END} ({fit_mask.sum()} rows), "
          f"tune {YIELD_VAL_START}-{YIELD_FIT_END} ({tune_val.sum()}), "
          f"test {YIELD_TEST_START}-{int(years.max())} ({test_mask.sum()})")

    # ---- Optuna on the validation years ----------------------------------
    print(f"[yield] tuning XGBoost with Optuna ({optuna_trials} trials)...")
    best_params = _tune_yield_xgb(X[tune_fit], y[tune_fit], X[tune_val], y[tune_val], optuna_trials)

    # ---- candidates, all judged on the SAME 2017-2020 test block ---------
    def candidates() -> dict[str, Any]:
        return {
            "MedianBaseline": MedianBaseline(),
            "RandomForest": _log_target(RandomForestRegressor(
                n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2)),
            "GradientBoosting": _log_target(GradientBoostingRegressor(
                n_estimators=N_ESTIMATORS, max_depth=5, learning_rate=0.08, random_state=RANDOM_STATE)),
            "XGBoost (Optuna)": _log_target(_xgb_regressor(best_params)),
        }

    X_fit, y_fit = X[fit_mask], y[fit_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    results: dict[str, dict[str, Any]] = {}
    test_preds: dict[str, np.ndarray] = {}
    for name, est in candidates().items():
        t0 = time.time()
        est.fit(X_fit, y_fit)
        pred = np.clip(est.predict(X_te), 0, None)
        test_preds[name] = pred
        results[name] = {
            "time_split_test": _reg_metrics(y_te, pred),
            "time_split_train": _reg_metrics(y_fit, np.clip(est.predict(X_fit), 0, None)),
            "fit_seconds": round(time.time() - t0, 2),
        }
        m = results[name]["time_split_test"]
        print(f"  {name:<20} test MAE={m['mae']:.4f} MAPE={m['mape'] * 100:.1f}% R2={m['r2']:.4f} "
              f"({results[name]['fit_seconds']}s)")

    # ---- GroupKFold(5) by State x Crop -----------------------------------
    print(f"[yield] GroupKFold({cv_folds}) grouped by State x Crop...")
    gkf = GroupKFold(n_splits=cv_folds)
    for name in candidates():
        maes, mapes, r2s = [], [], []
        for tr, te in gkf.split(X, y, groups=groups):
            est = candidates()[name]
            est.fit(X.iloc[tr], y.iloc[tr])
            pred = np.clip(est.predict(X.iloc[te]), 0, None)
            m = _reg_metrics(y.iloc[te], pred)
            maes.append(m["mae"]); mapes.append(m["mape"]); r2s.append(m["r2"])
        results[name]["group_kfold"] = {
            "folds": cv_folds, "grouped_by": "State x Crop",
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "mape_mean": float(np.mean(mapes)), "r2_mean": float(np.mean(r2s)),
        }
        g = results[name]["group_kfold"]
        print(f"  {name:<20} groupCV MAE={g['mae_mean']:.4f}+-{g['mae_std']:.4f} R2={g['r2_mean']:.4f}")

    # ---- winner: the model only ships if it beats the handbook -----------
    model_name = "XGBoost (Optuna)"
    model_mae = results[model_name]["time_split_test"]["mae"]
    base_mae = results["MedianBaseline"]["time_split_test"]["mae"]
    best_name = model_name if model_mae < base_mae else "MedianBaseline"
    verdict = (f"XGBoost beats the 5-year median baseline on 2017-2020 "
               f"({model_mae:.4f} vs {base_mae:.4f} t/ha MAE)" if best_name == model_name else
               f"the 5-year median baseline BEATS XGBoost on 2017-2020 "
               f"({base_mae:.4f} vs {model_mae:.4f} t/ha MAE), so the baseline is shipped")
    print(f"[yield] -> {verdict}")

    # ---- per-crop MAE for the 15 most common crops -----------------------
    test_crops = df.loc[test_mask, "Crop"]
    common = test_crops.value_counts().head(15).index.tolist()
    per_crop = []
    for crop in common:
        sel = (test_crops == crop).to_numpy()
        per_crop.append({
            "crop": crop, "n_test_rows": int(sel.sum()),
            "model_mae": float(mean_absolute_error(y_te[sel], test_preds[model_name][sel])),
            "baseline_mae": float(mean_absolute_error(y_te[sel], test_preds["MedianBaseline"][sel])),
            "mean_actual_yield": float(y_te[sel].mean()),
        })
    print("[yield] per-crop MAE (t/ha) on 2017-2020, model vs baseline:")
    for r in per_crop:
        print(f"    {r['crop']:<22} n={r['n_test_rows']:<5} model={r['model_mae']:.3f} "
              f"baseline={r['baseline_mae']:.3f} (mean actual {r['mean_actual_yield']:.2f})")

    # ---- expected_range: three candidate bands, judged on the test block --
    # The band must (a) hit its advertised level and (b) contain the point
    # estimate it brackets. The two nominal-quantile options fail one or both:
    # XGBoost's 0.1/0.9 pinball band is fitted around XGBoost's own (losing)
    # predictions, and the baseline's 10th/90th percentile over five yearly
    # observations is far too tight. Split conformal calibrates the width from
    # held-out residuals of whichever point model actually ships.
    def build_band(kind: str):
        """Return unfitted ``(low, high)`` estimators for one band strategy."""
        if kind == "baseline_quantile":
            return MedianBaseline(quantile=0.1), MedianBaseline(quantile=0.9)
        if kind == "xgb_quantile":
            return _quantile_model(best_params, 0.1), _quantile_model(best_params, 0.9)
        raise ValueError(kind)

    def conformal_pair(point_model, X_cal, y_cal):
        """Split-conformal band around an already-fitted ``point_model``."""
        per_crop, global_hw = ConformalBand.calibrate(point_model, X_cal, y_cal, coverage=0.8)
        return (ConformalBand(point_model, per_crop, global_hw, "low"),
                ConformalBand(point_model, per_crop, global_hw, "high"))

    def score_band(ql, qh, point_pred) -> dict[str, float]:
        a = np.clip(ql.predict(X_te), 0, None)
        b = np.clip(qh.predict(X_te), 0, None)
        a, b = np.minimum(a, b), np.maximum(a, b)
        return {
            "empirical_coverage": float(np.mean((y_te.to_numpy() >= a) & (y_te.to_numpy() <= b))),
            "mean_width_t_per_ha": float(np.mean(b - a)),
            "contains_point_estimate": float(np.mean((point_pred >= a) & (point_pred <= b))),
        }

    print("[yield] evaluating expected_range candidates (nominal 80% coverage)...")
    interval_scores: dict[str, dict[str, float]] = {}
    for kind in ("baseline_quantile", "xgb_quantile"):
        ql, qh = build_band(kind)
        ql.fit(X_fit, y_fit)
        qh.fit(X_fit, y_fit)
        family = "MedianBaseline" if kind == "baseline_quantile" else "XGBoost (Optuna)"
        interval_scores[kind] = score_band(ql, qh, test_preds[family])

    # Conformal band around the WINNING point model, calibrated on 2014-2016
    # (never on the 2017-2020 block it is then scored against).
    cal_point = candidates()[best_name]
    cal_point.fit(X[tune_fit], y[tune_fit])
    cal_low, cal_high = conformal_pair(cal_point, X[tune_val], y[tune_val])
    # score it using a copy fitted on the same <=2016 data as the other bands
    shipped_point_eval = candidates()[best_name]
    shipped_point_eval.fit(X_fit, y_fit)
    eval_low, eval_high = (ConformalBand(shipped_point_eval, cal_low.half_widths,
                                         cal_low.global_half_width, s) for s in ("low", "high"))
    interval_scores["conformal"] = score_band(eval_low, eval_high, test_preds[best_name])

    for kind, s in interval_scores.items():
        print(f"  {kind:<20} coverage {s['empirical_coverage'] * 100:.1f}%, "
              f"width {s['mean_width_t_per_ha']:.3f} t/ha, brackets its own point estimate "
              f"{s['contains_point_estimate'] * 100:.1f}% of the time")

    # Ship the band whose empirical coverage is closest to the 80 % it claims,
    # among those that always contain their own point estimate.
    usable = {k: s for k, s in interval_scores.items() if s["contains_point_estimate"] > 0.999}
    band_kind = min(usable or interval_scores,
                    key=lambda k: abs(interval_scores[k]["empirical_coverage"] - 0.8))
    coverage = interval_scores[band_kind]["empirical_coverage"]
    band = interval_scores[band_kind]["mean_width_t_per_ha"]
    print(f"[yield] shipping the '{band_kind}' band: {coverage * 100:.1f}% coverage "
          f"(claims 80%), mean width {band:.3f} t/ha")

    # ---- plots -----------------------------------------------------------
    _plot_comparison({k: v["time_split_test"]["mae"] for k, v in results.items()},
                     "yield: MAE on 2017-2020 test years (lower is better)", "MAE (t/ha)",
                     reports / "yield_model_comparison.png", higher_is_better=False)
    _plot_pred_vs_actual(y_te.to_numpy(), test_preds[best_name],
                         f"yield: predicted vs actual ({best_name}, 2017-2020)",
                         reports / "yield_pred_vs_actual.png")
    _plot_per_crop_mae(per_crop, "yield: per-crop MAE, model vs 5-year median",
                       reports / "yield_per_crop_mae.png")

    importances: dict[str, float] = {}
    holdout_model = _log_target(_xgb_regressor(best_params))
    holdout_model.fit(X_fit, y_fit)
    imp = _importances(holdout_model)
    if imp is not None:
        names = _feature_names(holdout_model, list(X.columns))
        if len(names) == len(imp):
            importances = {n: float(v) for n, v in zip(names, imp)}
            _plot_importance(names, np.asarray(imp), "yield: feature importance (XGBoost)",
                             reports / "yield_feature_importance.png")

    # ---- refit everything on ALL years and ship --------------------------
    print("[yield] refitting the shipped models on all years...")
    point = candidates()[best_name]
    point.fit(X, y)
    baseline_full = MedianBaseline().fit(X, y)
    if band_kind == "conformal":
        # Reuse the half-widths calibrated on 2014-2016, wrapped around the
        # point model refit on every year.
        q_low = ConformalBand(point, cal_low.half_widths, cal_low.global_half_width, "low")
        q_high = ConformalBand(point, cal_high.half_widths, cal_high.global_half_width, "high")
    else:
        q_low, q_high = build_band(band_kind)
        q_low.fit(X, y)
        q_high.fit(X, y)

    bundle = YieldBundle(
        point=point, low=q_low, high=q_high, baseline=baseline_full,
        meta={
            "winner": best_name,
            "verdict": verdict,
            "time_split_test_mae": results[best_name]["time_split_test"]["mae"],
            "baseline_test_mae": base_mae,
            "interval_method": band_kind,
            "interval_nominal_coverage": 0.8,
            "interval_coverage": coverage,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": DATASETS["yield"],
        },
    )
    path = out_dir / "yield_prediction_pipeline.pkl"
    joblib.dump(bundle, path, compress=3)
    print(f"[yield] saved {path} ({path.stat().st_size / 1e6:.2f} MB)")

    meta = build_yield_meta(df)
    meta_path = out_dir / "yield_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[yield] wrote {meta_path} ({len(meta['crops'])} crops / {len(meta['states'])} states / "
          f"{len(meta['seasons'])} seasons / {len(meta['defaults']['by_state_crop'])} state-crop input medians)")

    return {
        "task": "yield",
        "n_samples_raw": int(n_raw),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "dropped_crops": dropped_crops,
        "leakage_note": "Production is excluded: Yield == Production / Area in this dataset.",
        "features": {"categorical": YIELD_CATEGORICAL, "numeric": YIELD_NUMERIC, "derived": YIELD_DERIVED},
        "target_transform": "log1p / expm1",
        "split": {
            "protocol": "time split + GroupKFold(5) by State x Crop",
            "tune_fit_years": f"<= {YIELD_VAL_START - 1}",
            "tune_validation_years": f"{YIELD_VAL_START}-{YIELD_FIT_END}",
            "fit_years": f"<= {YIELD_FIT_END}", "test_years": f"{YIELD_TEST_START}-{int(years.max())}",
            "fit_rows": int(fit_mask.sum()), "test_rows": int(test_mask.sum()),
        },
        "tuning": {"library": "optuna", "n_trials": optuna_trials, "objective": "validation MAE (t/ha)",
                   "best_params": best_params},
        "comparison": results,
        "best_model": best_name,
        "best_metrics": results[best_name],
        "baseline_metrics": results["MedianBaseline"],
        "verdict": verdict,
        "per_crop_mae": per_crop,
        "prediction_interval": {"method": band_kind, "nominal_coverage": 0.8,
                                "point_model": best_name, "empirical_coverage": coverage,
                                "mean_width_t_per_ha": band, "candidates": interval_scores},
        "feature_importances": importances,
        "meta_counts": {k: len(v) for k, v in meta.items() if isinstance(v, list)},
        "refit_on_full_data": True,
        "dataset": DATASETS["yield"],
        "artifact": path.name,
        "artifact_bytes": path.stat().st_size,
    }


# --------------------------------------------------------------------------
# model cards
# --------------------------------------------------------------------------
def _card_header(task: str, m: dict[str, Any]) -> list[str]:
    ds = m.get("dataset", {})
    lines = [
        f"# {task.title()} model card",
        "",
        f"_Generated by `python -m ml.train_tabular` on {time.strftime('%Y-%m-%d')}._",
        "",
        "## Data",
        "",
        f"- **Dataset**: {ds.get('name', 'n/a')}",
        f"- **File**: `{ds.get('file', 'n/a')}`",
        f"- **Rows used**: {m.get('n_samples', 'n/a'):,}" if isinstance(m.get("n_samples"), int)
        else f"- **Rows used**: {m.get('n_samples', 'n/a')}",
        f"- **Source**: {ds.get('source_url', 'n/a')}",
        f"- **Licence**: {ds.get('licence', 'n/a')}",
    ]
    if ds.get("note"):
        lines.append(f"- **Note**: {ds['note']}")
    if ds.get("warning"):
        lines.append(f"- **WARNING**: {ds['warning']}")
    return lines


def write_model_cards(metrics: dict[str, Any], reports: Path) -> list[str]:
    """Write one short, honest markdown card per trained task."""
    written = []
    for task, m in metrics.get("tasks", {}).items():
        lines = _card_header(task, m)
        lines += ["", "## Model", "", f"- **Shipped estimator**: {m.get('best_model', 'n/a')}",
                  f"- **Artefact**: `models/{m.get('artifact', 'n/a')}`"]
        if m.get("tuning"):
            t = m["tuning"]
            lines.append(f"- **Tuning**: {t.get('search') or t.get('library')}, "
                         f"{t.get('n_iter') or t.get('n_trials')} iterations")
            if t.get("calibration"):
                lines.append(f"- **Calibration**: {t['calibration']}")

        lines += ["", "## Evaluation", ""]
        if task == "yield":
            best = m["best_metrics"]["time_split_test"]
            base = m["baseline_metrics"]["time_split_test"]
            gk = m["best_metrics"].get("group_kfold", {})
            lines += [
                f"Protocol: {m['split']['protocol']}. Fit on {m['split']['fit_years']} "
                f"({m['split']['fit_rows']:,} rows), tested on {m['split']['test_years']} "
                f"({m['split']['test_rows']:,} rows).",
                "",
                "| Model | MAE (t/ha) | MAPE | R2 |",
                "| --- | --- | --- | --- |",
                f"| Shipped ({m['best_model']}) | {best['mae']:.3f} | {best['mape'] * 100:.1f}% | {best['r2']:.3f} |",
                f"| 5-year Crop x State median | {base['mae']:.3f} | {base['mape'] * 100:.1f}% | {base['r2']:.3f} |",
                "",
                f"GroupKFold(5) by State x Crop: MAE {gk.get('mae_mean', float('nan')):.3f} "
                f"+- {gk.get('mae_std', float('nan')):.3f}, R2 {gk.get('r2_mean', float('nan')):.3f}.",
                "",
                f"**Verdict**: {m.get('verdict', 'n/a')}",
                "",
                f"Prediction interval (`expected_range`): **{m['prediction_interval']['method']}**, "
                f"claiming 80 % coverage and achieving "
                f"{m['prediction_interval']['empirical_coverage'] * 100:.1f}% on the test years "
                f"(mean width {m['prediction_interval']['mean_width_t_per_ha']:.2f} t/ha). "
                "Candidates considered:",
                "",
                "| Band | Empirical coverage | Mean width (t/ha) | Contains its own point estimate |",
                "| --- | --- | --- | --- |",
                *[f"| {k} | {v['empirical_coverage'] * 100:.1f}% | {v['mean_width_t_per_ha']:.2f} | "
                  f"{v['contains_point_estimate'] * 100:.1f}% |"
                  for k, v in m["prediction_interval"]["candidates"].items()],
                "",
                "## Limitations",
                "",
                "- Rows are **state-level annual aggregates**, not fields. A prediction is a regional "
                "expectation, never a guarantee for one farm.",
                f"- `Production` is excluded as a feature: {m['leakage_note']} Any published R2 above "
                "~0.9 for this dataset is measuring that leak.",
                f"- Crops dropped for non-tonne units: {', '.join(m.get('dropped_crops') or ['none'])}.",
                "- Fertilizer and pesticide inputs default to State x Crop medians when the farmer "
                "leaves them blank; the result reports which values were used.",
            ]
        else:
            comp = m.get("comparison", {})
            lines += ["| Model | CV accuracy | Hold-out accuracy | Hold-out MAP@3 |",
                      "| --- | --- | --- | --- |"]
            for name, row in comp.items():
                map3 = f"{row['test_map3']:.4f}" if "test_map3" in row else "-"
                mark = " **(shipped)**" if name == m.get("best_model") else ""
                lines.append(f"| {name}{mark} | {row['cv_mean_accuracy']:.4f} | {row['test_accuracy']:.4f} | {map3} |")
            lines.append("")
            if m.get("tie_break"):
                lines += [f"Selection: {m['tie_break']}.", ""]
            if task == "fertilizer":
                lines += [
                    f"Shipped model hold-out top-1 accuracy **{m.get('top1_accuracy', float('nan')):.4f}**, "
                    f"MAP@3 **{m.get('map3', float('nan')):.4f}**"
                    + ("." if m.get("map3_trustworthy") else
                       " — **but do not quote that number.** " + m.get("reliability", "")),
                    "",
                    "## Limitations",
                    "",
                    f"- {m.get('serving_note', '')}",
                    "- Soil type, crop type, temperature, humidity and moisture carry little signal about "
                    "which bag to buy; nutrient deficit does. That is why the rule leads.",
                ]
            else:
                lines += [
                    "## Limitations",
                    "",
                    "- The classes in this dataset are close to linearly separable, which is why every "
                    "estimator scores above 98 %. Treat it as a sanity check, not evidence of field accuracy.",
                    "- Inputs outside the 1st-99th percentile of the training data are flagged in "
                    "`recommend_crop(...)['warnings']` using `models/crop_meta.json`.",
                ]
        path = reports / f"{task}_card.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path.name)
    return written


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the requested tabular training tasks."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="folder holding the CSV datasets")
    p.add_argument("--output", type=Path, default=Path("models"), help="folder to write models/metrics into")
    p.add_argument("--cv-folds", type=int, default=5, help="cross-validation folds (default 5)")
    p.add_argument("--fert-data", type=Path, default=None,
                   help=f"Kaggle S5E6 train.csv (default: {DEFAULT_FERT_S5E6})")
    p.add_argument("--optuna-trials", type=int, default=40, help="Optuna trials for the yield model")
    p.add_argument("--tasks", nargs="+", default=["crop", "fertilizer", "yield"],
                   choices=["crop", "fertilizer", "yield"], help="subset of tasks to train")
    args = p.parse_args(argv)

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitkeep").touch()

    started = time.time()
    metrics_path = out_dir / "metrics_tabular.json"
    # Retraining a subset of tasks must not erase the other tasks' numbers -
    # MLService reads fertilizer.map3_trustworthy out of this file at runtime.
    previous: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  !! ignoring unreadable {metrics_path}: {exc}")
    metrics: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "random_state": RANDOM_STATE,
        "cv_folds": args.cv_folds,
        "datasets": DATASETS,
        "tasks": dict(previous.get("tasks") or {}),
    }
    runners = {"crop": train_crop, "fertilizer": train_fertilizer, "yield": train_yield}
    for task in args.tasks:
        print(f"\n=== {task} ===")
        metrics["tasks"][task] = runners[task](args.data_dir, out_dir, reports, args.cv_folds,
                                               fert_data=args.fert_data, optuna_trials=args.optuna_trials)

    metrics["tasks_trained_this_run"] = list(args.tasks)
    metrics["model_cards"] = write_model_cards(metrics, reports)
    metrics["total_seconds"] = round(time.time() - started, 1)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path} (total {metrics['total_seconds']}s)")
    for task, m in metrics["tasks"].items():
        fresh = "" if task in args.tasks else "  (carried over from a previous run)"
        print(f"  {task:<11} best={m['best_model']}{fresh}")
    print(f"  model cards: {', '.join(metrics['model_cards'])}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    raise SystemExit(main())
