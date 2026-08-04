
import Models

#comparing the data with outliers

dirty_results = []

for name, model in models.items():

    r2, rmse = evaluate_model(
        model,
        x_train,
        x_test,
        y_train,
        y_test
    )

    dirty_results.append({
        "Model": name,
        "R²": r2,
        "RMSE": rmse
    })

dirty_results = pd.DataFrame(dirty_results)

dirty_results.sort_values(
    by="R²",
    ascending=False
)

#comparing the data without outliers
clean_results = []

for name, model in models.items():

    r2, rmse = evaluate_model(
        model,
        xc_train,
        xc_test,
        yc_train,
        yc_test
    )

    clean_results.append({
        "Model": name,
        "R²": r2,
        "RMSE": rmse
    })

clean_results = pd.DataFrame(clean_results)

clean_results.sort_values(

    by="R²",
    ascending=False
)

comparison = dirty_results.merge(
    clean_results,
    on="Model",
    suffixes=("_Dirty", "_Clean")
)

comparison