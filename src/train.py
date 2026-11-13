"""
Model Training Pipeline.
Orchestrates data generation, preprocessing, model training, evaluation,
and exporting all model artifacts for local inference and Streamlit deployment.
"""

import os
import json
import joblib
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.data_generator import build_full_dataset
from src.preprocess import (
    extract_features_and_targets,
    split_data,
    fit_and_scale,
    compute_feature_statistics,
)
from src.model import build_multi_task_model
from src.evaluate import evaluate_predictions, print_metrics_summary


def run_training_pipeline(
    n_samples: int = 5000,
    epochs: int = 250,
    batch_size: int = 32,
    artifacts_dir: str = "artifacts",
    data_dir: str = "data",
) -> None:
    """Execute complete end-to-end training and export artifacts."""
    print(">>> Step 1/6: Generating student performance dataset...")
    df = build_full_dataset(n_samples=n_samples, data_dir=data_dir)
    print(f"    Dataset prepared with {df.shape[0]} samples and {df.shape[1]} columns.")

    print(">>> Step 2/6: Extracting features and targets...")
    x, y_gpa, y_dropout, y_perf = extract_features_and_targets(df)
    feature_columns = x.columns.tolist()
    feature_stats = compute_feature_statistics(x)

    print(">>> Step 3/6: Performing train/test split & standardizing features...")
    (
        x_train,
        x_test,
        y_gpa_train,
        y_gpa_test,
        y_dropout_train,
        y_dropout_test,
        y_perf_train,
        y_perf_test,
    ) = split_data(x, y_gpa, y_dropout, y_perf, test_size=0.2, random_state=42)

    (
        x_train_scaled,
        x_test_scaled,
        y_gpa_train_scaled,
        y_gpa_test_scaled,
        scaler,
        y_scaler,
    ) = fit_and_scale(x_train, x_test, y_gpa_train, y_gpa_test)

    print(f"    Train shape: {x_train_scaled.shape}, Test shape: {x_test_scaled.shape}")

    print(">>> Step 4/6: Building and compiling multi-task neural network...")
    model = build_multi_task_model(input_dim=x_train_scaled.shape[1])

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f">>> Step 5/6: Training model for up to {epochs} epochs...")
    model.fit(
        x_train_scaled,
        {
            "future_gpa": y_gpa_train_scaled,
            "dropout": y_dropout_train,
            "performance": y_perf_train,
        },
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print(">>> Step 6/6: Evaluating model on test set...")
    gpa_pred_scaled, dropout_pred_prob, perf_pred_prob = model.predict(
        x_test_scaled, verbose=0
    )
    gpa_pred = y_scaler.inverse_transform(gpa_pred_scaled.reshape(-1, 1)).ravel()

    metrics = evaluate_predictions(
        y_gpa_true=y_gpa_test.values,
        y_gpa_pred=gpa_pred,
        y_dropout_true=y_dropout_test.values,
        y_dropout_pred_prob=dropout_pred_prob.ravel(),
        y_perf_true=y_perf_test.values,
        y_perf_pred_prob=perf_pred_prob,
    )

    print_metrics_summary(metrics)

    # Save artifacts
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"\n>>> Saving all production artifacts to ./{artifacts_dir}/ ...")

    model.save(f"{artifacts_dir}/model.keras")
    joblib.dump(scaler, f"{artifacts_dir}/scaler.pkl")
    joblib.dump(y_scaler, f"{artifacts_dir}/y_scaler.pkl")

    with open(f"{artifacts_dir}/feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)

    with open(f"{artifacts_dir}/feature_stats.json", "w", encoding="utf-8") as f:
        json.dump(feature_stats, f, ensure_ascii=False, indent=2)

    with open(f"{artifacts_dir}/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(" All artifacts successfully exported and ready for Streamlit deployment!")


if __name__ == "__main__":
    run_training_pipeline()
