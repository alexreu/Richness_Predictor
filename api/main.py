import json
import os

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI(
    title="Richness Predictor API",
    description="Placeholder API for the income prediction service.",
    version="0.1.0",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_METRICS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "training_metrics.json")


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
def predict() -> dict[str, str]:
    return {
        "status": "OK",
        "message": "Prediction endpoint ready. Model not connected yet.",
    }


@app.get("/training-metrics", tags=["training"])
def training_metrics() -> dict:
    if not os.path.exists(TRAINING_METRICS_PATH):
        return {
            "status": "OK",
            "message": "No training metrics available yet. Run python -m training.train_from_db first.",
            "metrics": [],
        }

    with open(TRAINING_METRICS_PATH, encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    return {
        "status": "OK",
        "message": "Latest training metrics.",
        "metrics": metrics,
    }


Instrumentator().instrument(app).expose(app)
