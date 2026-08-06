"""Run every registered model and rank the results."""

import pandas as pd

from src.evaluator import evaluate
from src.models import get_models
from src.config import SELECTION_METRIC


def compare(x_train, x_test, y_train, y_test):
    """Fit + evaluate every model, return a ranked DataFrame and the
    dict of fitted model objects (so train.py can save the winner).
    """
    rows = []
    fitted_models = {}

    for name, model in get_models().items():
        metrics = evaluate(model, x_train, x_test, y_train, y_test)
        rows.append({"Model": name, "R2": metrics["R2"], "RMSE": metrics["RMSE"]})
        fitted_models[name] = metrics["model"]

    results = pd.DataFrame(rows).sort_values(
        SELECTION_METRIC, ascending=False
    ).reset_index(drop=True)

    return results, fitted_models
