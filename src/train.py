"""Entry point: load data, compare models, save the best one.

Run with:  python -m src.train
(from the diabetes-predictor/ root, so the `src` package resolves)
"""

import joblib

from src.data_loader import load_dataset
from src.preprocessing import split
from src.comparison import compare
from src.config import MODEL_DIR, MODEL_PATH


def main():
    features, target = load_dataset()
    x_train, x_test, y_train, y_test = split(features, target)

    results, fitted_models = compare(x_train, x_test, y_train, y_test)

    print("MODEL COMPARISON (sorted by R2, best first)")
    print(results.to_string(index=False))

    best_name = results.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nSaved best model ('{best_name}') to {MODEL_PATH}")


if __name__ == "__main__":
    main()
