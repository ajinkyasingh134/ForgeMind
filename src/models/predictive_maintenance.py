import json
from pathlib import Path

import joblib
import pandas as pd


# Resolve the ForgeMind project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


class PredictiveMaintenance:
    """Combine RUL prediction and sensor anomaly detection."""

    def __init__(self):
        # Load the trained models
        self.rul_model = joblib.load(
            MODEL_DIR / "rul_random_forest_capped.joblib"
        )

        self.anomaly_model = joblib.load(
            MODEL_DIR / "isolation_forest_model.joblib"
        )

        # Load the anomaly detection scaler
        self.anomaly_scaler = joblib.load(
            MODEL_DIR / "anomaly_scaler.joblib"
        )

        # Load model metadata
        with open(
            MODEL_DIR / "rul_model_metadata.json",
            "r",
            encoding="utf-8",
        ) as file:
            self.rul_metadata = json.load(file)

        with open(
            MODEL_DIR / "anomaly_model_metadata.json",
            "r",
            encoding="utf-8",
        ) as file:
            self.anomaly_metadata = json.load(file)

        self.rul_features = self.rul_metadata["features"]
        self.anomaly_features = self.anomaly_metadata["features"]

    def assess(self, observation):
        """
        Assess one engine observation.

        Parameters
        ----------
        observation : dict or pandas.Series
            One observation containing the required sensor features
            and operating settings.

        Returns
        -------
        dict
            Predicted RUL, anomaly status, and anomaly score.
        """
        row = pd.DataFrame([dict(observation)])

        required_features = set(
            self.rul_features + self.anomaly_features
        )

        missing_features = sorted(
            required_features - set(row.columns)
        )

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        # Validate that required feature values are numeric
        for feature in required_features:
            row[feature] = pd.to_numeric(
                row[feature],
                errors="raise",
            )

        if row[list(required_features)].isnull().any().any():
            raise ValueError(
                "Required features must not contain missing values."
            )

        # RUL prediction uses its own saved feature configuration
        rul_input = row[self.rul_features]
        predicted_rul = float(
            self.rul_model.predict(rul_input)[0]
        )

        # Anomaly detection uses its own saved scaler and features
        anomaly_input = row[self.anomaly_features]
        anomaly_scaled = self.anomaly_scaler.transform(
            anomaly_input
        )

        anomaly_prediction = int(
            self.anomaly_model.predict(anomaly_scaled)[0]
        )

        anomaly_score = float(
            self.anomaly_model.decision_function(
                anomaly_scaled
            )[0]
        )

        return {
            "predicted_rul_cycles": round(predicted_rul, 2),
            "anomaly_status": (
                "Anomaly"
                if anomaly_prediction == -1
                else "Normal"
            ),
            "anomaly_score": round(anomaly_score, 6),
        }