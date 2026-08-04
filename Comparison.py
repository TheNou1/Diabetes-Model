import pandas as pd

from evaluator import evaluate

from models import models

def compare(
    x_train,
    x_test,
    y_train,
    y_test
):

    results = []

    for name, model in models.items():

        metrics = evaluate(
            model,
            x_train,
            x_test,
            y_train,
            y_test
        )

        results.append({

            "Model": name,

            "R²":
                metrics["R²"],

            "RMSE":
                metrics["RMSE"]

        })

    results = pd.DataFrame(results)

    return results.sort_values(
        "R²",
        ascending=False
    )