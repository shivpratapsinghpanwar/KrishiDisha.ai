"""KrishiDisha machine-learning training package.

This package holds the *reproducible* training code for every model the Flask
app consumes at runtime.  Nothing in here is imported by request handling
except :mod:`ml.disease_model` (the CNN architecture factory, needed to rebuild
the network before loading a checkpoint) and :mod:`ml.estimators` (needed to
unpickle an XGBoost-backed classifier).

Modules
-------
``ml.train_tabular``
    Trains, compares and persists the three scikit-learn / XGBoost models used
    by :class:`krishidisha.services.ml.MLService`.
``ml.disease_model``
    ``build_model`` / ``get_transforms`` for the plant-disease CNN.
``ml.train_disease``
    Fine-tunes a torchvision backbone on the PlantVillage ``ImageFolder`` tree.
``ml.train_all``
    Convenience entry point that runs the tabular training and, when the image
    dataset is present, the disease training too.

Usage::

    python -m ml.train_tabular --data-dir data --output models
    python -m ml.train_disease --data-dir <plantvillage_root> --epochs 6
    python -m ml.train_all
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
