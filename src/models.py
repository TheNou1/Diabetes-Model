"""Registry of every candidate model.

Add a new model by adding one line to this dict — nothing else in the
project needs to change.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    RIDGE_ALPHA,
    LASSO_ALPHA,
    ELASTIC_ALPHA,
    ELASTIC_L1,
    KNN_NEIGHBORS,
    FOREST_TREES,
    RANDOM_STATE,
)


def get_models():
    """Return a fresh dict of unfitted models.

    Returned as a function (not a module-level constant) so that
    repeated calls to compare() never reuse an already-fitted estimator.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=RIDGE_ALPHA),
        "Lasso Regression": Lasso(alpha=LASSO_ALPHA),
        "Elastic Net": ElasticNet(alpha=ELASTIC_ALPHA, l1_ratio=ELASTIC_L1),
        "KNN": KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=FOREST_TREES, random_state=RANDOM_STATE
        ),
    }
