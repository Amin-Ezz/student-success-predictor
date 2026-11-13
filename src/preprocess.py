"""
Preprocessing and Data Transformation Module.
Handles feature/target extraction, train-test splitting, and feature scaling.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def extract_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Separate input features from multi-task targets."""
    target_cols = ["FutureGPA", "DropoutRisk", "PerformanceLevel"]
    x = df.drop(columns=target_cols)
    y_gpa = df["FutureGPA"]
    y_dropout = df["DropoutRisk"]
    y_perf = df["PerformanceLevel"]
    return x, y_gpa, y_dropout, y_perf


def split_data(
    x: pd.DataFrame,
    y_gpa: pd.Series,
    y_dropout: pd.Series,
    y_perf: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split features and all three targets synchronously."""
    return train_test_split(
        x,
        y_gpa,
        y_dropout,
        y_perf,
        test_size=test_size,
        random_state=random_state,
    )


def fit_and_scale(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_gpa_train: pd.Series,
    y_gpa_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Standardize feature inputs and the regression target (FutureGPA).
    Returns scaled matrices and the fitted StandardScaler objects.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_scaler = StandardScaler()
    y_gpa_train_scaled = y_scaler.fit_transform(
        y_gpa_train.values.reshape(-1, 1)
    ).ravel()
    y_gpa_test_scaled = y_scaler.transform(
        y_gpa_test.values.reshape(-1, 1)
    ).ravel()

    return (
        x_train_scaled,
        x_test_scaled,
        y_gpa_train_scaled,
        y_gpa_test_scaled,
        scaler,
        y_scaler,
    )


def compute_feature_statistics(df_features: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute min, max, mean, and datatype per feature.
    Used by Streamlit UI to dynamically build range-safe controls.
    """
    stats = {}
    for col in df_features.columns:
        stats[col] = {
            "min": float(df_features[col].min()),
            "max": float(df_features[col].max()),
            "mean": float(df_features[col].mean()),
            "dtype": "int" if str(df_features[col].dtype).startswith("int") else "float",
        }
    return stats
