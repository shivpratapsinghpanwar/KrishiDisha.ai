"""Small scikit-learn compatible estimators and the shared feature contracts.

This module is the **single source of truth** for the column layouts that the
training scripts (:mod:`ml.train_tabular`) and the serving layer
(:class:`krishidisha.services.ml.MLService`) must agree on.  It deliberately
depends on nothing but numpy / pandas / scikit-learn so that both sides can
import it without pulling in the Flask app or the training stack.

It must stay importable from the application process: joblib resolves
``ml.estimators.<class>`` when a pickled model is loaded.

Contents
--------
:class:`XGBLabelClassifier`
    ``xgboost.XGBClassifier`` wrapper that accepts (and reports) string labels.
:class:`MedianBaseline`
    The honest "what did this crop do here recently?" reference model.
:class:`YieldBundle`
    Point model + optional quantile models + baseline, behind one ``predict``.
:func:`build_yield_features`
    Turns raw yield inputs into the leakage-free feature frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------
# Yield feature contract (leakage-free: ``Production`` is NOT an input)
# --------------------------------------------------------------------------
#: Bumped whenever the yield feature layout changes.  :class:`YieldBundle`
#: stamps it so ``MLService`` can detect (and retrain over) a stale pickle.
YIELD_CONTRACT = "yield-v2-no-production"

YIELD_CATEGORICAL = ["Crop", "Season", "State"]
#: Raw numeric inputs a caller supplies.  ``Production`` is excluded on
#: purpose: ``Yield == Production / Area`` in the source data, so feeding
#: production back in is target leakage, not prediction.
YIELD_NUMERIC = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
#: Cheap ratios derived from the raw inputs by :func:`build_yield_features`.
YIELD_DERIVED = ["fert_per_ha", "pest_per_ha", "log_area"]
YIELD_FEATURES = YIELD_CATEGORICAL + YIELD_NUMERIC + YIELD_DERIVED


def build_yield_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready feature frame for raw yield rows.

    ``df`` must carry ``YIELD_CATEGORICAL + YIELD_NUMERIC``.  Categoricals are
    whitespace-stripped, and three ratios are appended so the model can learn
    input *intensity* rather than field size:

    ``fert_per_ha``/``pest_per_ha``
        Fertilizer / pesticide kilograms divided by area.
    ``log_area``
        ``log1p(Area)`` - area spans five orders of magnitude.
    """
    out = df.copy()
    for col in YIELD_CATEGORICAL:
        out[col] = out[col].astype(str).str.strip()
    area = pd.to_numeric(out["Area"], errors="coerce").astype(float)
    safe_area = area.where(area > 0, np.nan)
    out["Area"] = area
    out["fert_per_ha"] = pd.to_numeric(out["Fertilizer"], errors="coerce").astype(float) / safe_area
    out["pest_per_ha"] = pd.to_numeric(out["Pesticide"], errors="coerce").astype(float) / safe_area
    out["log_area"] = np.log1p(area.clip(lower=0))
    return out[YIELD_FEATURES]


class MedianBaseline(RegressorMixin, BaseEstimator):
    """Predict a quantile of the same Crop x State yield over the last 5 years.

    With the default ``quantile=0.5`` this is the median - the number a farmer
    could work out from a district handbook, and therefore the bar any model has
    to clear.  It is shipped inside :class:`YieldBundle` (win or lose) so the app
    can always show a "vs 5-year average" reference next to the prediction.

    Setting ``quantile`` to 0.1 / 0.9 turns the same table into a prediction
    band.  That matters when the baseline is the *shipped* point model: an
    interval borrowed from a different (losing) estimator can easily exclude the
    prediction it is supposed to bracket, which is worse than no interval at all.

    Lookup order at predict time:

    1. ``(Crop, State, Crop_Year)`` - the quantile over the five years *before*
       that year, available only for years seen during ``fit``.
    2. ``(Crop, State)`` - over the last five training years.
    3. ``Crop`` - over the last five training years.
    4. Global.

    Parameters
    ----------
    window:
        Number of preceding years the statistic is taken over (default 5).
    quantile:
        Which quantile to report, in ``[0, 1]`` (default 0.5, the median).
    """

    def __init__(self, window: int = 5, quantile: float = 0.5):
        self.window = window
        self.quantile = quantile

    def fit(self, X, y):
        """Build the (crop, state, year) -> previous-``window``-years tables."""
        q = float(self.quantile)
        df = pd.DataFrame({
            "Crop": np.asarray(X["Crop"], dtype=object),
            "State": np.asarray(X["State"], dtype=object),
            "Crop_Year": np.asarray(X["Crop_Year"], dtype=float).astype(int),
            "y": np.asarray(y, dtype=float),
        })
        self.global_ = float(df["y"].quantile(q))
        last_year = int(df["Crop_Year"].max())
        recent = df[df["Crop_Year"] > last_year - self.window]

        self.by_crop_state_recent_ = {
            k: float(v) for k, v in recent.groupby(["Crop", "State"])["y"].quantile(q).items()
        }
        self.by_crop_recent_ = {
            k: float(v) for k, v in recent.groupby("Crop")["y"].quantile(q).items()
        }

        # Rolling table: for every (crop, state, year) present in the data, the
        # quantile of that pair over the five *preceding* years.
        table: dict[tuple, float] = {}
        for (crop, state), grp in df.groupby(["Crop", "State"]):
            years = sorted(grp["Crop_Year"].unique().tolist())
            for year in years:
                prev = grp[(grp["Crop_Year"] < year) & (grp["Crop_Year"] >= year - self.window)]["y"]
                if len(prev):
                    table[(crop, state, year)] = float(prev.quantile(q))
        self.table_ = table
        self.is_fitted_ = True
        return self

    def _one(self, crop, state, year) -> float:
        val = self.table_.get((crop, state, int(year)))
        if val is not None:
            return val
        val = self.by_crop_state_recent_.get((crop, state))
        if val is not None:
            return val
        val = self.by_crop_recent_.get(crop)
        if val is not None:
            return val
        return self.global_

    def predict(self, X):
        """Yield quantile (original scale, t/ha) for every row of ``X``."""
        crops = np.asarray(X["Crop"], dtype=object)
        states = np.asarray(X["State"], dtype=object)
        years = np.asarray(X["Crop_Year"], dtype=float).astype(int)
        return np.array([self._one(c, s, y) for c, s, y in zip(crops, states, years)], dtype=float)

    def lookup(self, crop: str, state: str, year: int) -> float | None:
        """Public single-row lookup used by the app; ``None`` when unseen."""
        if not getattr(self, "is_fitted_", False):
            return None
        crop, state = str(crop).strip(), str(state).strip()
        val = self.table_.get((crop, state, int(year)))
        if val is None:
            val = self.by_crop_state_recent_.get((crop, state))
        if val is None:
            val = self.by_crop_recent_.get(crop)
        return val


class ConformalBand(RegressorMixin, BaseEstimator):
    """One side of a split-conformal prediction interval around a point model.

    Quantile regressors give a band with a *nominal* level that need not match
    reality.  Both candidates tried here missed badly on the 2017-2020 block:
    XGBoost's 0.1/0.9 pinball band covered 70 % and did not even contain its own
    point estimate 10 % of the time, and the median baseline's own 10th/90th
    percentile - taken over just five yearly observations - covered 50 %.  A
    band advertised as 80 % that holds half the time is worse than no band.

    Split conformal fixes the level empirically: take the absolute residuals of
    the point model on a held-out calibration block, and use their ``coverage``
    quantile as the half-width.  Residuals are measured in ``log1p`` space so the
    band scales with the prediction - yields here run from 0.5 t/ha pulses to
    20 t/ha onions, and a single additive width would be nonsense for both. The
    quantile is taken per crop where there is enough calibration data, so a crop
    the model handles badly gets an honestly wider band.

    Because the half-width is non-negative, the interval always contains the
    point estimate it brackets.

    Parameters
    ----------
    point:
        The fitted point regressor this band wraps.
    half_widths:
        ``{crop: log-space half-width}`` from :meth:`calibrate`.
    global_half_width:
        Fallback half-width for crops absent from ``half_widths``.
    side:
        ``"low"`` or ``"high"``.
    """

    #: Minimum calibration rows before a crop gets its own half-width.
    MIN_CROP_ROWS = 20

    def __init__(self, point=None, half_widths: dict | None = None,
                 global_half_width: float = 0.0, side: str = "low"):
        self.point = point
        self.half_widths = half_widths
        self.global_half_width = global_half_width
        self.side = side

    @staticmethod
    def calibrate(point, X_cal, y_cal, coverage: float = 0.8) -> tuple[dict, float]:
        """Return ``(per-crop half-widths, global half-width)`` in log1p space."""
        pred = np.clip(np.asarray(point.predict(X_cal), dtype=float), 0.0, None)
        resid = np.abs(np.log1p(np.asarray(y_cal, dtype=float)) - np.log1p(pred))
        crops = np.asarray(X_cal["Crop"], dtype=object)
        global_hw = float(np.quantile(resid, coverage))
        per_crop = {}
        for crop in np.unique(crops):
            sel = crops == crop
            if sel.sum() >= ConformalBand.MIN_CROP_ROWS:
                per_crop[str(crop)] = float(np.quantile(resid[sel], coverage))
        return per_crop, global_hw

    def fit(self, X, y):  # noqa: D102 - the wrapped point model is already fitted
        return self

    def predict(self, X):
        """One edge of the interval, on the original t/ha scale."""
        pred = np.clip(np.asarray(self.point.predict(X), dtype=float), 0.0, None)
        widths = np.array([(self.half_widths or {}).get(str(c), self.global_half_width)
                           for c in np.asarray(X["Crop"], dtype=object)], dtype=float)
        shifted = np.log1p(pred) + (-widths if self.side == "low" else widths)
        return np.clip(np.expm1(shifted), 0.0, None)


class YieldBundle:
    """What ``models/yield_prediction_pipeline.pkl`` actually contains.

    Wrapping the shipped artefacts in one object keeps the serving code honest:
    the point estimate, the 10th/90th percentile band and the 5-year baseline
    always come from the same training run and the same feature contract.

    Parameters
    ----------
    point:
        Fitted regressor consuming :data:`YIELD_FEATURES`.  Predictions are on
        the original t/ha scale (log handling lives inside the estimator).
    low, high:
        Optional 0.1 / 0.9 quantile regressors for ``expected_range``.  They
        must come from the *same* estimator family as ``point``, otherwise the
        band can exclude the number it is meant to bracket.
    baseline:
        Fitted :class:`MedianBaseline`, kept even when the model wins.
    meta:
        Free-form provenance (winner name, headline metrics, dataset).
    """

    #: Read by ``MLService`` to spot a pickle from an older feature layout.
    contract = YIELD_CONTRACT

    def __init__(self, point, low=None, high=None, baseline=None, meta: dict | None = None):
        self.point = point
        self.low = low
        self.high = high
        self.baseline = baseline
        self.meta = meta or {}

    def predict(self, X):
        """Point prediction in tonnes per hectare, clipped at zero."""
        return np.clip(np.asarray(self.point.predict(X), dtype=float), 0.0, None)

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray] | None:
        """``(low, high)`` t/ha arrays, or ``None`` when no quantile models."""
        if self.low is None or self.high is None:
            return None
        lo = np.clip(np.asarray(self.low.predict(X), dtype=float), 0.0, None)
        hi = np.clip(np.asarray(self.high.predict(X), dtype=float), 0.0, None)
        return np.minimum(lo, hi), np.maximum(lo, hi)


class XGBLabelClassifier(ClassifierMixin, BaseEstimator):
    """``XGBClassifier`` wrapper that accepts (and reports) string labels.

    ``xgboost.XGBClassifier`` refuses non-numeric targets (it wants
    ``[0..n_classes-1]``) while :class:`krishidisha.services.ml.MLService`
    expects ``model.classes_`` to contain the *original* string labels
    (``"rice"``, ``"Urea"``, ...) and ``predict_proba`` columns to line up.

    Parameters
    ----------
    params:
        Mapping forwarded verbatim to :class:`xgboost.XGBClassifier`.  It is a
        single dict (rather than ``**kwargs``) so that ``sklearn.base.clone``
        can round-trip the estimator through ``get_params``/``__init__``, which
        ``cross_val_score`` relies on.

    Attributes
    ----------
    classes_:
        ``np.ndarray`` of the original labels, sorted, index-aligned with the
        columns of :meth:`predict_proba`.
    """

    def __init__(self, params: dict | None = None):
        self.params = params

    # ------------------------------------------------------------- sklearn
    def get_params(self, deep: bool = True) -> dict:  # noqa: D102 - sklearn API
        return {"params": self.params}

    def set_params(self, **kwargs):  # noqa: D102 - sklearn API
        if "params" in kwargs:
            self.params = kwargs.pop("params")
        if kwargs:
            self.params = {**(self.params or {}), **kwargs}
        return self

    # ----------------------------------------------------------------- fit
    def fit(self, X, y):
        """Encode ``y`` to integers, then fit the underlying booster."""
        from xgboost import XGBClassifier

        self.encoder_ = LabelEncoder()
        y_enc = self.encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.encoder_.classes_
        self.model_ = XGBClassifier(**(self.params or {}))
        self.model_.fit(X, y_enc)
        return self

    # ------------------------------------------------------------- predict
    def predict(self, X):
        """Return predictions in the original (string) label space."""
        return self.encoder_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X):
        """Class probabilities, columns aligned with :attr:`classes_`."""
        return self.model_.predict_proba(X)

    # ------------------------------------------------------------ passthru
    @property
    def feature_importances_(self):
        """Expose the booster's gain-based feature importances."""
        return self.model_.feature_importances_

    def __sklearn_tags__(self):  # pragma: no cover - sklearn >= 1.6 plumbing
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
