import os

import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import joblib

from api.schema import PredictionRequest, PredictionResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.joblib")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

app = FastAPI(
    title="Richness Predictor API",
    description="Placeholder API for the income prediction service.",
    version="0.1.0",
)


@app.get("/", tags=["status"])
def root() -> dict[str, str]:
    return {
        "status": "OK",
        "message": "Richness Predictor API is running.",
    }


@app.get("/health", tags=["status"])
def health() -> dict[str, str]:
    return {
        "status": "OK",
        "message": "Service healthy.",
    }


@app.post("/predict", tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    features = pd.DataFrame([request.model_dump()])
    prediction = int(model.predict(features)[0])

    return PredictionResponse(
        status="OK",
        prediction=">50K" if prediction == 1 else "<=50K",
    )


Instrumentator().instrument(app).expose(app)
