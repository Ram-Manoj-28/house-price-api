import json
from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_model.keras"
SCALER_PATH = BASE_DIR / "scaler.joblib"
FEATURES_PATH = BASE_DIR / "features.json"
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH, "r") as file:
    FEATURE_NAMES = json.load(file)
app = FastAPI(
    title="California Housing Price Prediction API",
    description="Predict California district median house values.",
    version="1.0.0"
)
class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in tens of thousands of USD")
    HouseAge: float = Field(..., description="Median house age in years")
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)
@app.get("/")
def home():
    return {
        "message": "California Housing Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}
@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        # Match the exact feature order used during training
        values = np.array(
            [[getattr(features, name) for name in FEATURE_NAMES]],
            dtype=np.float64
        )

        # Use the same scaler fitted during training
        scaled_values = scaler.transform(values)

        result = model.predict(scaled_values, verbose=0)
        predicted_units = float(np.asarray(result).reshape(-1)[0])

        if not np.isfinite(predicted_units):
            raise HTTPException(
                status_code=500,
                detail="The model returned an invalid prediction."
            )

        # Target is measured in units of $100,000
        predicted_usd = max(0.0, predicted_units) * 100000

        return {
            "predicted_value_dataset_units": round(predicted_units, 4),
            "predicted_house_value_usd": round(predicted_usd, 2),
            "currency": "USD",
            "note": (
                "Estimated district-level median house value, "
                "not an individual property appraisal."
            )
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs."
        )
