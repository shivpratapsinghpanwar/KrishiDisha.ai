"""Train, compare and persist the three KrishiDisha tabular models.

Tasks
-----
1. **Crop recommendation** - ``data/Crop_recommendation.csv``, 7 soil/climate
   features (``N, P, K, temperature, humidity, ph, rainfall``) -> 22 crops.
2. **Fertilizer recommendation** - ``data/Fertilizer Prediction.csv``, two
   categorical columns one-hot encoded through a ``ColumnTransformer`` plus six
   numeric columns -> 7 fertilizer products.
3. **Yield prediction** - ``data/crop_yield.csv``, three whitespace-stripped
   categorical columns one-hot encoded plus six numeric columns -> ``Yield``
   (tonnes per hectare), a regression task.

Method
------
For every task the data is split 80/20 (stratified for the classification
tasks), each candidate estimator is scored with 5-fold cross-validation on the
training split *and* evaluated on the held-out test split, the winner is chosen
by mean CV score, then **refit on 100% of the data** and saved with joblib.

The saved objects honour exactly the input contracts of
:mod:`krishidisha.services.ml`:

* ``models/crop_recommendation_model.pkl`` - estimator taking a DataFrame with
  columns ``CROP_FEATURES``; exposes ``classes_`` / ``predict_proba``.
* ``models/fertilizer_recommendation_model.pkl`` - full pipeline taking a
  DataFrame with columns ``FERT_CATEGORICAL + FERT_NUMERIC``.
* ``models/yield_prediction_pipeline.pkl`` - full pipeline taking a DataFrame
  with columns ``YIELD_CATEGORICAL + YIELD_NUMERIC``.

Side artefacts: ``models/yield_meta.json`` (dropdown values),
``models/metrics_tabular.json`` (every comparison number, the winners' feature
importances and the crop confusion matrix) and PNG reports under
``models/reports/``.

Example
-------
``python -m ml.train_tabular --data-dir data --output models``
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
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
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from ml.estimators import XGBLabelClassifier

# --------------------------------------------------------------------------
# Column contracts - MUST match krishidisha/services/ml.py
# --------------------------------------------------------------------------
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FERT_NUMERIC = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
FERT_CATEGORICAL = ["Soil Type", "Crop Type"]
YIELD_NUMERIC = ["Crop_Year", "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]
YIELD_CATEGORICAL = ["Crop", "Season", "State"]

RANDOM_STATE = 42
N_ESTIMATORS = 300  # capped so the compressed pickles stay small


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _one_hot() -> OneHotEncoder:
    """OneHotEncoder that tolerates unseen categories at inference time."""
    return OneHotEncoder(handle_unknown="ignore")


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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
        if isinstance(model, Pipeline) and "pre" in model.named_steps:
            return list(model.named_steps["pre"].get_feature_names_out())
    except Exception:  # pragma: no cover - defensive
        pass
    return fallback


def _importances(model) -> np.ndarray | None:
    """Pull ``feature_importances_`` out of a bare or pipelined estimator."""
    target = model
    if isinstance(model, Pipeline):
        target = model.steps[-1][1]
    return getattr(target, "feature_importances_", None)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def _plot_comparison(scores: dict[str, float], title: str, ylabel: str, out: Path) -> None:
    names = list(scores)
    vals = [scores[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, vals, color="#2e7d32")
    best = int(np.argmax(vals))
    bars[best].set_color("#f9a825")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.02, max(vals) * 1.12))
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
) -> tuple[Any, dict[str, Any]]:
    """Compare classifiers, return ``(winner_refit_on_all_data, metrics)``."""
    strat = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
    )
    folds = min(cv_folds, int(y_tr.value_counts().min()))
    folds = max(folds, 2)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    results: dict[str, dict[str, Any]] = {}
    for name, est in _classifiers().items():
        model = _wrap(pre_factory(), est)
        t0 = time.time()
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=1)
        model.fit(X_tr, y_tr)
        pred_te = model.predict(X_te)
        pred_tr = model.predict(X_tr)
        results[name] = {
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
            "cv_folds": folds,
            "train_accuracy": float(accuracy_score(y_tr, pred_tr)),
            "test_accuracy": float(accuracy_score(y_te, pred_te)),
            "test_f1_macro": float(f1_score(y_te, pred_te, average="macro", zero_division=0)),
            "fit_seconds": round(time.time() - t0, 2),
        }
        print(f"  {name:<17} cv={cv_scores.mean():.4f}+-{cv_scores.std():.4f} "
              f"test={results[name]['test_accuracy']:.4f} ({results[name]['fit_seconds']}s)")

    best_name = max(results, key=lambda k: results[k]["cv_mean_accuracy"])
    print(f"  -> best: {best_name}")

    # holdout artefacts from the winner (fitted on the train split)
    best_holdout = _wrap(pre_factory(), _classifiers()[best_name])
    best_holdout.fit(X_tr, y_tr)
    pred_te = best_holdout.predict(X_te)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_te, pred_te, labels=labels)
    report = classification_report(y_te, pred_te, labels=labels, output_dict=True, zero_division=0)

    _plot_comparison({k: v["cv_mean_accuracy"] for k, v in results.items()},
                     f"{task}: 5-fold CV accuracy", "accuracy",
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
    final = _wrap(pre_factory(), _classifiers()[best_name])
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
    return final, metrics


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
def train_crop(data_dir: Path, out_dir: Path, reports: Path, cv_folds: int) -> dict[str, Any]:
    """Crop recommendation: 7 numeric features -> crop label."""
    print("[crop] loading", data_dir / "Crop_recommendation.csv")
    df = pd.read_csv(data_dir / "Crop_recommendation.csv")
    X = df[CROP_FEATURES].astype(float)
    y = df["label"].astype(str)
    model, metrics = _run_classification("crop", X, y, lambda: None, cv_folds, reports)
    path = out_dir / "crop_recommendation_model.pkl"
    joblib.dump(model, path, compress=3)
    metrics["artifact"] = path.name
    metrics["artifact_bytes"] = path.stat().st_size
    print(f"[crop] saved {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return metrics


def train_fertilizer(data_dir: Path, out_dir: Path, reports: Path, cv_folds: int) -> dict[str, Any]:
    """Fertilizer recommendation: one-hot Soil/Crop Type + 6 numeric -> product."""
    print("[fertilizer] loading", data_dir / "Fertilizer Prediction.csv")
    df = pd.read_csv(data_dir / "Fertilizer Prediction.csv")
    X = df[FERT_CATEGORICAL + FERT_NUMERIC]
    y = df["Fertilizer Name"].astype(str)

    def pre_factory() -> ColumnTransformer:
        # Identical layout to MLService._train_fertilizer.
        return ColumnTransformer([
            ("cat", _one_hot(), FERT_CATEGORICAL),
            ("num", "passthrough", FERT_NUMERIC),
        ])

    model, metrics = _run_classification("fertilizer", X, y, pre_factory, cv_folds, reports)
    path = out_dir / "fertilizer_recommendation_model.pkl"
    joblib.dump(model, path, compress=3)
    metrics["artifact"] = path.name
    metrics["artifact_bytes"] = path.stat().st_size
    print(f"[fertilizer] saved {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return metrics


def train_yield(data_dir: Path, out_dir: Path, reports: Path, cv_folds: int) -> dict[str, Any]:
    """Yield regression: one-hot Crop/Season/State + 6 numeric -> Yield."""
    print("[yield] loading", data_dir / "crop_yield.csv")
    df = pd.read_csv(data_dir / "crop_yield.csv")
    for col in YIELD_CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=YIELD_CATEGORICAL + YIELD_NUMERIC + ["Yield"])
    X = df[YIELD_CATEGORICAL + YIELD_NUMERIC]
    y = df["Yield"].astype(float)

    def pre_factory() -> ColumnTransformer:
        return ColumnTransformer([
            ("cat", _one_hot(), YIELD_CATEGORICAL),
            ("num", "passthrough", YIELD_NUMERIC),
        ])

    def candidates() -> dict[str, Any]:
        from xgboost import XGBRegressor
        return {
            "RandomForest": RandomForestRegressor(
                n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=N_ESTIMATORS, max_depth=5, learning_rate=0.08, random_state=RANDOM_STATE
            ),
            "XGBoost": XGBRegressor(
                n_estimators=N_ESTIMATORS, max_depth=7, learning_rate=0.08, subsample=0.9,
                colsample_bytree=0.9, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
            ),
        }

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    results: dict[str, dict[str, Any]] = {}
    for name, est in candidates().items():
        model = Pipeline([("pre", pre_factory()), ("model", est)])
        t0 = time.time()
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=1)
        model.fit(X_tr, y_tr)
        p_tr, p_te = model.predict(X_tr), model.predict(X_te)
        results[name] = {
            "cv_mean_r2": float(cv_scores.mean()),
            "cv_std_r2": float(cv_scores.std()),
            "cv_folds": cv_folds,
            "train_r2": float(r2_score(y_tr, p_tr)),
            "train_mae": float(mean_absolute_error(y_tr, p_tr)),
            "train_rmse": _rmse(y_tr, p_tr),
            "test_r2": float(r2_score(y_te, p_te)),
            "test_mae": float(mean_absolute_error(y_te, p_te)),
            "test_rmse": _rmse(y_te, p_te),
            "fit_seconds": round(time.time() - t0, 2),
        }
        print(f"  {name:<17} cvR2={cv_scores.mean():.4f} testR2={results[name]['test_r2']:.4f} "
              f"MAE={results[name]['test_mae']:.4f} RMSE={results[name]['test_rmse']:.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_mean_r2"])
    print(f"  -> best: {best_name}")

    best_holdout = Pipeline([("pre", pre_factory()), ("model", candidates()[best_name])])
    best_holdout.fit(X_tr, y_tr)
    p_te = best_holdout.predict(X_te)

    _plot_comparison({k: max(v["cv_mean_r2"], 0.0) for k, v in results.items()},
                     "yield: 5-fold CV R2", "R2", reports / "yield_model_comparison.png")
    _plot_pred_vs_actual(y_te.to_numpy(), p_te, f"yield: predicted vs actual ({best_name}, holdout)",
                         reports / "yield_pred_vs_actual.png")

    importances: dict[str, float] = {}
    imp = _importances(best_holdout)
    if imp is not None:
        names = _feature_names(best_holdout, list(X.columns))
        if len(names) == len(imp):
            importances = {n: float(v) for n, v in zip(names, imp)}
            _plot_importance(names, np.asarray(imp), f"yield: feature importance ({best_name})",
                             reports / "yield_feature_importance.png")

    final = Pipeline([("pre", pre_factory()), ("model", candidates()[best_name])])
    final.fit(X, y)
    path = out_dir / "yield_prediction_pipeline.pkl"
    joblib.dump(final, path, compress=3)
    print(f"[yield] saved {path} ({path.stat().st_size / 1e6:.2f} MB)")

    meta = {
        "crops": sorted(df["Crop"].unique().tolist()),
        "states": sorted(df["State"].unique().tolist()),
        "seasons": sorted(df["Season"].unique().tolist()),
    }
    meta_path = out_dir / "yield_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[yield] wrote {meta_path} "
          f"({len(meta['crops'])} crops / {len(meta['states'])} states / {len(meta['seasons'])} seasons)")

    return {
        "task": "yield",
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "split": {"train": int(len(X_tr)), "test": int(len(X_te)), "test_size": 0.2},
        "comparison": results,
        "best_model": best_name,
        "best_metrics": results[best_name],
        "feature_importances": importances,
        "meta_counts": {k: len(v) for k, v in meta.items()},
        "refit_on_full_data": True,
        "artifact": path.name,
        "artifact_bytes": path.stat().st_size,
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the requested tabular training tasks."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="folder holding the CSV datasets")
    p.add_argument("--output", type=Path, default=Path("models"), help="folder to write models/metrics into")
    p.add_argument("--cv-folds", type=int, default=5, help="cross-validation folds (default 5)")
    p.add_argument("--tasks", nargs="+", default=["crop", "fertilizer", "yield"],
                   choices=["crop", "fertilizer", "yield"], help="subset of tasks to train")
    args = p.parse_args(argv)

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitkeep").touch()

    started = time.time()
    metrics: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "random_state": RANDOM_STATE,
        "cv_folds": args.cv_folds,
        "tasks": {},
    }
    runners = {"crop": train_crop, "fertilizer": train_fertilizer, "yield": train_yield}
    for task in args.tasks:
        print(f"\n=== {task} ===")
        metrics["tasks"][task] = runners[task](args.data_dir, out_dir, reports, args.cv_folds)

    metrics["total_seconds"] = round(time.time() - started, 1)
    metrics_path = out_dir / "metrics_tabular.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path} (total {metrics['total_seconds']}s)")
    for task, m in metrics["tasks"].items():
        print(f"  {task:<11} best={m['best_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
