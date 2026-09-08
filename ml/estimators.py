"""Small scikit-learn compatible estimators used by the training scripts.

Only one class lives here today: :class:`XGBLabelClassifier`.  It exists
because ``xgboost.XGBClassifier`` refuses non-numeric targets (it wants
``[0..n_classes-1]``) while :class:`krishidisha.services.ml.MLService` expects
``model.classes_`` to contain the *original* string labels (``"rice"``,
``"Urea"``, ...) and ``predict_proba`` columns to line up with them.

This module must stay importable from the application process: if an XGBoost
model wins the model-selection contest it is pickled with joblib, and joblib
resolves ``ml.estimators.XGBLabelClassifier`` at load time.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class XGBLabelClassifier(ClassifierMixin, BaseEstimator):
    """``XGBClassifier`` wrapper that accepts (and reports) string labels.

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
