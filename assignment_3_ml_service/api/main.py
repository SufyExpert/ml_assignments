"""
main.py
-------
The FastAPI application. It exposes:

  GET  /              - a "home" endpoint: basic info + links (Part D extra)
  GET  /health         - health check (Part D)
  POST /predict         - stroke-risk prediction (Part D)
  GET  /monitoring       - basic monitoring stats (Part F)

It also serves a small human-friendly web page at /ui (a static HTML file
that calls /health and /predict from the browser with plain JavaScript --
see static/index.html), so someone with no API knowledge can still use the
service.

The trained pipeline is loaded exactly once, at startup, and reused for
every request -- it is not re-loaded from disk on every prediction.
"""

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Make the monitoring/ package importable when running as `uvicorn api.main:app`
# from the project root.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from monitoring.monitoring import stats, logger  # noqa: E402
from api.schemas import (  # noqa: E402
    PatientFeatures, PredictionResponse, HealthResponse, HomeResponse,
)

MODEL_PATH = ROOT_DIR / "model" / "model.joblib"
METADATA_PATH = ROOT_DIR / "model" / "metadata.json"

# A small mutable holder for state set once at startup and read on every
# request afterwards. Using a plain dict (rather than globals scattered
# around the module) keeps it obvious what "application state" actually is.
app_state = {"pipeline": None, "metadata": None, "start_time": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load the model exactly once ---
    logger.info("Loading model from %s", MODEL_PATH)
    app_state["pipeline"] = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        app_state["metadata"] = json.load(f)
    logger.info(
        "Model loaded: version=%s algorithm=%s",
        app_state["metadata"]["model_version"],
        app_state["metadata"]["algorithm"],
    )
    yield
    # --- Shutdown: nothing to clean up for this simple service ---
    logger.info("Shutting down.")


app = FastAPI(
    title="Stroke Risk Prediction Service",
    description="A small local ML service that predicts stroke risk from "
                 "basic patient information, built for Assignment 3.",
    version="1.0",
    lifespan=lifespan,
)

# Serve the human-friendly web page at /ui (static/index.html and friends).
app.mount("/ui", StaticFiles(directory=str(ROOT_DIR / "static"), html=True), name="ui")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    FastAPI/Pydantic already reject malformed input automatically -- this
    handler just makes sure that (a) the error message stays clear and
    readable, and (b) it's counted in monitoring as an invalid-input
    request, per Part F of the brief.
    """
    model_version = (app_state["metadata"] or {}).get("model_version", "unknown")
    stats.record_invalid_input(model_version=model_version)
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "One or more fields in your request were missing or invalid.",
            "details": exc.errors(),
        },
    )


@app.get("/", response_model=HomeResponse)
def home():
    """A simple landing endpoint so hitting the API root gives useful info,
    not a bare 404."""
    metadata = app_state["metadata"]
    return HomeResponse(
        message="Stroke Risk Prediction Service is running.",
        model_version=metadata["model_version"] if metadata else None,
        docs_url="/docs",
        ui_url="/ui",
        endpoints=["/", "/health", "/predict", "/monitoring", "/docs", "/ui"],
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Reports whether the service is up and whether the model is loaded."""
    pipeline_loaded = app_state["pipeline"] is not None
    metadata = app_state["metadata"]
    return HealthResponse(
        status="ok" if pipeline_loaded else "degraded",
        model_loaded=pipeline_loaded,
        model_version=metadata["model_version"] if metadata else None,
        uptime_seconds=round(time.time() - app_state["start_time"], 1),
    )


@app.get("/monitoring")
def monitoring():
    """Basic monitoring information, per Part F of the brief."""
    snapshot = stats.snapshot()
    snapshot["model_version"] = (app_state["metadata"] or {}).get("model_version")
    return snapshot


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    """
    Predict stroke risk for one patient.

    Pydantic has already validated `patient` by the time this function
    runs -- if any field were missing or the wrong type, the request would
    have been rejected before reaching here (see validation_exception_handler
    above), and it would never count as a "successful" or "failed"
    prediction, only an "invalid input" one.
    """
    pipeline = app_state["pipeline"]
    metadata = app_state["metadata"]
    model_version = metadata["model_version"]

    t0 = time.perf_counter()
    try:
        # Build a one-row DataFrame with exactly the raw column names the
        # pipeline was trained on. The pipeline's own preprocessor handles
        # imputing, scaling, and encoding internally -- nothing is
        # reimplemented here.
        input_df = pd.DataFrame([patient.model_dump()])
        input_df = input_df[metadata["raw_feature_names"]]

        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])
    except Exception as exc:
        stats.record_failure(reason=str(exc), model_version=model_version)
        raise

    latency_ms = (time.perf_counter() - t0) * 1000
    stats.record_success(prediction=prediction, latency_ms=latency_ms, model_version=model_version)

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        model_version=model_version,
        latency_ms=round(latency_ms, 3),
    )
