from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Locate the project root and model module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_MODULE = PROJECT_ROOT / "src" / "models"

sys.path.insert(0, str(MODEL_MODULE))

from predictive_maintenance import PredictiveMaintenance


app = FastAPI(
    title="ForgeMind Predictive Maintenance API",
    description="Engine RUL prediction and anomaly detection",
    version="1.0.0",
)

# Load the saved ML models once when the API starts
predictor = PredictiveMaintenance()


class EngineObservation(BaseModel):
    """Sensor readings for one engine cycle."""

    op_setting_1: float
    op_setting_2: float
    op_setting_3: float

    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


@app.get("/")
def home():
    return {
        "project": "ForgeMind",
        "message": "Predictive Maintenance API is running",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ForgeMind API",
        "models_loaded": predictor is not None,
    }


@app.post("/predict")
def predict_engine(observation: EngineObservation):
    """Return RUL and anomaly predictions for one engine observation."""

    try:
        result = predictor.assess(observation.model_dump())
        return result

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc