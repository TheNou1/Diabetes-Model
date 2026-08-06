"""Loads the dataset used for training.

Swap this file's internals when you move from a toy dataset to your
real CSV or a database — nothing downstream needs to change as long as
`load_dataset()` still returns (features_df, target_series).
"""

import pandas as pd
from sklearn.datasets import load_diabetes


def load_dataset():
    """Load the diabetes dataset.

    Returns
    -------
    features : pd.DataFrame
    target : pd.Series
    """
    data = load_diabetes(as_frame=True)
    features = data.frame.drop(columns=["target"])
    target = data.frame["target"]
    return features, target


def load_dataset_from_csv(path, target_column):
    """Example of how you'd load your own CSV instead.

    Parameters
    ----------
    path : str or Path to the CSV file
    target_column : str, name of the column to predict
    """
    df = pd.read_csv(path)
    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target
