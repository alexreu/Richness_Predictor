from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


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
def predict() -> dict[str, str]:
    return {
        "status": "OK",
        "message": "Prediction endpoint ready. Model not connected yet.",
    }


Instrumentator().instrument(app).expose(app)
