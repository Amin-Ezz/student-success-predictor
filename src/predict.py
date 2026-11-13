"""
Inference Pipeline Module.
Provides StudentPredictor class for loading artifacts and performing predictions.
"""

import os
import json
from typing import Dict, Any, Union, List, Tuple
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import StandardScaler


class StudentPredictor:
    """Predictor class that loads model artifacts and performs multi-task inferences."""

    PERFORMANCE_LABELS = ["Weak", "Average", "Good", "Excellent"]
    PERFORMANCE_LABELS_FA = ["ضعیف", "متوسط", "خوب", "عالی"]

    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        self.model: Model = load_model(os.path.join(artifacts_dir, "model.keras"))
        self.scaler: StandardScaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
        self.y_scaler: StandardScaler = joblib.load(os.path.join(artifacts_dir, "y_scaler.pkl"))

        with open(os.path.join(artifacts_dir, "feature_columns.json"), "r", encoding="utf-8") as f:
            self.feature_columns: List[str] = json.load(f)

        with open(os.path.join(artifacts_dir, "feature_stats.json"), "r", encoding="utf-8") as f:
            self.feature_stats: Dict[str, Dict[str, Any]] = json.load(f)

        with open(os.path.join(artifacts_dir, "test_metrics.json"), "r", encoding="utf-8") as f:
            self.test_metrics: Dict[str, Dict[str, float]] = json.load(f)

    def prepare_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Validate and construct correctly ordered feature DataFrame."""
        row_dict = {}
        for col in self.feature_columns:
            if col in data:
                row_dict[col] = data[col]
            elif col == "StudentID":
                row_dict[col] = self.feature_stats["StudentID"]["mean"]
            else:
                row_dict[col] = self.feature_stats[col]["mean"]
        return pd.DataFrame([row_dict], columns=self.feature_columns)

    def predict(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Run multi-task inference for a single student or batch.
        
        Returns
        -------
        dict with future_gpa, dropout_prob, is_dropout_risk,
                  performance_idx, performance_label, performance_probs
        """
        if isinstance(input_data, dict):
            df_input = self.prepare_input(input_data)
        else:
            df_input = input_data[self.feature_columns]

        x_scaled = self.scaler.transform(df_input)
        gpa_pred_scaled, dropout_pred, perf_pred = self.model.predict(x_scaled, verbose=0)

        # Invert scaled GPA
        gpa_raw = self.y_scaler.inverse_transform(gpa_pred_scaled.reshape(-1, 1)).ravel()[0]
        future_gpa = float(np.clip(gpa_raw, 0.0, 20.0))

        dropout_prob = float(dropout_pred.ravel()[0])
        perf_probs = perf_pred.ravel().tolist()
        perf_idx = int(np.argmax(perf_probs))

        return {
            "future_gpa": round(future_gpa, 2),
            "dropout_probability": round(dropout_prob, 4),
            "is_dropout_risk": dropout_prob >= 0.5,
            "performance_level_index": perf_idx,
            "performance_level_en": self.PERFORMANCE_LABELS[perf_idx],
            "performance_level_fa": self.PERFORMANCE_LABELS_FA[perf_idx],
            "performance_probabilities": [round(p, 4) for p in perf_probs],
        }
