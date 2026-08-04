from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from config import *

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(alpha=RIDGE_ALPHA),

    "Lasso Regression":
        Lasso(alpha=LASSO_ALPHA),

    "Elastic Net":
        ElasticNet(
            alpha=ELASTIC_ALPHA,
            l1_ratio=ELASTIC_L1
        ),

    "KNN":
        KNeighborsRegressor(
            n_neighbors=K
        ),

    "SVR":
        SVR(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=RANDOM_STATE
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=FOREST_TREES,
            random_state=RANDOM_STATE
        )

}