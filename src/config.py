"""Central configuration for the diabetes predictor project.

Keep every hyperparameter and path here so nothing is hard-coded
in the pipeline files.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"

# --- Train/test split ---------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Model hyperparameters -----------------------------------------------
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.1
ELASTIC_ALPHA = 0.1
ELASTIC_L1 = 0.5
KNN_NEIGHBORS = 5
FOREST_TREES = 100

# --- Which model comparison.py should pick as "the winner" -------------
SELECTION_METRIC = "R2"  # higher is better
