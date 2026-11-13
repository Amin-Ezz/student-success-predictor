"""
Evaluation Module for Multi-Task Model Performance.
Computes regression and classification metrics across all three prediction tasks.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_predictions(
    y_gpa_true: np.ndarray,
    y_gpa_pred: np.ndarray,
    y_dropout_true: np.ndarray,
    y_dropout_pred_prob: np.ndarray,
    y_perf_true: np.ndarray,
    y_perf_pred_prob: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive performance metrics for all 3 tasks."""
    # Regression metrics
    mae = float(mean_absolute_error(y_gpa_true, y_gpa_pred))
    rmse = float(np.sqrt(mean_squared_error(y_gpa_true, y_gpa_pred)))
    r2 = float(r2_score(y_gpa_true, y_gpa_pred))

    # Binary classification metrics
    dropout_pred_class = (y_dropout_pred_prob > 0.5).astype(int)
    dropout_acc = float(accuracy_score(y_dropout_true, dropout_pred_class))
    dropout_prec = float(precision_score(y_dropout_true, dropout_pred_class, zero_division=0))
    dropout_rec = float(recall_score(y_dropout_true, dropout_pred_class, zero_division=0))
    dropout_f1 = float(f1_score(y_dropout_true, dropout_pred_class, zero_division=0))

    # Multi-class classification metrics
    perf_pred_class = np.argmax(y_perf_pred_prob, axis=1)
    perf_acc = float(accuracy_score(y_perf_true, perf_pred_class))
    perf_prec = float(
        precision_score(y_perf_true, perf_pred_class, average="weighted", zero_division=0)
    )
    perf_rec = float(
        recall_score(y_perf_true, perf_pred_class, average="weighted", zero_division=0)
    )
    perf_f1 = float(
        f1_score(y_perf_true, perf_pred_class, average="weighted", zero_division=0)
    )

    metrics = {
        "future_gpa": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
        "dropout": {
            "accuracy": dropout_acc,
            "precision": dropout_prec,
            "recall": dropout_rec,
            "f1": dropout_f1,
        },
        "performance": {
            "accuracy": perf_acc,
            "precision": perf_prec,
            "recall": perf_rec,
            "f1": perf_f1,
        },
    }

    return metrics


def print_metrics_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print nicely formatted metric tables."""
    print("=" * 55)
    print(" MULTI-TASK MODEL EVALUATION SUMMARY")
    print("=" * 55)

    gpa = metrics["future_gpa"]
    print("\n[Task 1: Future GPA (Regression)]")
    print(f"  - R² Score: {gpa['r2']:.4f} ({gpa['r2'] * 100:.1f}%)")
    print(f"  - MAE     : {gpa['mae']:.4f}")
    print(f"  - RMSE    : {gpa['rmse']:.4f}")

    dropout = metrics["dropout"]
    print("\n[Task 2: Dropout Risk (Binary Classification)]")
    print(f"  - Accuracy : {dropout['accuracy'] * 100:.2f}%")
    print(f"  - Precision: {dropout['precision'] * 100:.2f}%")
    print(f"  - Recall   : {dropout['recall'] * 100:.2f}%")
    print(f"  - F1-Score : {dropout['f1'] * 100:.2f}%")

    perf = metrics["performance"]
    print("\n[Task 3: Academic Performance Level (4-Class Classification)]")
    print(f"  - Accuracy : {perf['accuracy'] * 100:.2f}%")
    print(f"  - Precision: {perf['precision'] * 100:.2f}%")
    print(f"  - Recall   : {perf['recall'] * 100:.2f}%")
    print(f"  - F1-Score : {perf['f1'] * 100:.2f}%")
    print("=" * 55)
