"""Train and evaluate a single model."""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def evaluate(model, x_train, x_test, y_train, y_test):
    """Fit `model` and return its metrics + predictions."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "R2": r2_score(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "Predictions": predictions,
        "model": model,
    }
