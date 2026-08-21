#!/usr/bin/env python
"""FastAPI inference service for the autocall duration model.

/predict takes an RFQ in the same shape as data/raw/rfqs.csv and runs it
through the training feature-engineering pipeline.

    docker compose run --rm --service-ports dev \\
        uvicorn api.main:app --host 0.0.0.0 --port 8000

    MODEL_PATH=data/processed/v2/model.joblib RAW_DATA_DIR=data/raw \\
        uvicorn api.main:app ...
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.market_data import MARKET, resolve_raw_dir
from api.model import MODEL, resolve_model_path
from api.preprocessing import build_model_row, engineer_features
from api.schemas import PredictionResponse, RawRFQRequest, ReloadRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL.load(resolve_model_path())
    MARKET.load(resolve_raw_dir())
    yield


app = FastAPI(title="Autocall Duration API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_path": str(MODEL.path), "raw_dir": str(MARKET.raw_dir)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: RawRFQRequest):
    feature_row, warnings = engineer_features(request, MARKET)
    row = build_model_row(feature_row)
    pred = float(MODEL.model.predict(row)[0])
    return PredictionResponse(
        predicted_avg_duration_months=pred,
        model_path=str(MODEL.path),
        stale_volatility=bool(warnings),
        warnings=warnings,
    )


@app.post("/reload")
def reload_model(request: ReloadRequest | None = None):
    """Swap the loaded model and/or market data at runtime without restarting.

    Body {"model_path": "...", "raw_dir": "..."} loads those; either field
    missing/absent re-resolves its env var (or the latest dataset version
    for the model).
    """
    model_path = resolve_model_path(request.model_path if request else None)
    raw_dir = resolve_raw_dir(request.raw_dir if request else None)
    try:
        MODEL.load(model_path)
        MARKET.load(raw_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"status": "reloaded", "model_path": str(MODEL.path), "raw_dir": str(MARKET.raw_dir)}
