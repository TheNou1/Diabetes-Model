"""Loads the saved model and makes predictions.

This is the ONLY src file the FastAPI layer should ever import.
"""

import joblib

from src.config import MODEL_PATH

_model = None


def _get_model():
    """Lazy-load the model once, on first use."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(features):
    """Predict on a 2D array-like / DataFrame of features."""
    model = _get_model()
    return model.predict(features)
