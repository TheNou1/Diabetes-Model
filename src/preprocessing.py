"""Preprocessing: train/test split and (later) scaling/encoding."""

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def split(features, target):
    """Split features/target into train and test sets."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
