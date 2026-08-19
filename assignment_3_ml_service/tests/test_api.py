"""
test_api.py
-----------
Basic tests for the FastAPI service. Run them with:

    pytest tests/ -v

These use FastAPI's TestClient, which runs the app in-process (no need for
the server or Docker to already be running) and does trigger the real
startup event, so the actual persisted model is loaded from model/model.joblib
-- run `python src/train.py` at least once before running these tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from api.main import app

# Using TestClient as a context manager (via the fixture below) is what
# actually triggers FastAPI's lifespan startup event -- that's what loads
# the model from disk. Without it, app_state["pipeline"] would stay None.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

VALID_PATIENT = {
    "gender": "Female",
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}


def test_home_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert "message" in body
    assert "/predict" in body["endpoints"]


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] is not None


def test_predict_valid_input(client):
    res = client.post("/predict", json=VALID_PATIENT)
    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert "model_version" in body
    assert body["latency_ms"] >= 0


def test_predict_missing_field_returns_clear_422(client):
    bad_input = VALID_PATIENT.copy()
    del bad_input["age"]  # required field missing
    res = client.post("/predict", json=bad_input)
    assert res.status_code == 422
    body = res.json()
    assert body["error"] == "validation_error"
    assert "details" in body


def test_predict_invalid_category_returns_422(client):
    bad_input = VALID_PATIENT.copy()
    bad_input["gender"] = "NotARealGender"  # not one of the allowed Literal values
    res = client.post("/predict", json=bad_input)
    assert res.status_code == 422


def test_predict_age_out_of_range_returns_422(client):
    bad_input = VALID_PATIENT.copy()
    bad_input["age"] = 999  # exceeds the ge=0, le=120 constraint
    res = client.post("/predict", json=bad_input)
    assert res.status_code == 422


def test_predict_missing_bmi_is_allowed(client):
    # bmi is optional -- the pipeline itself imputes it, same as training.
    optional_input = VALID_PATIENT.copy()
    del optional_input["bmi"]
    res = client.post("/predict", json=optional_input)
    assert res.status_code == 200


def test_monitoring_endpoint_reports_counts(client):
    res = client.get("/monitoring")
    assert res.status_code == 200
    body = res.json()
    assert body["total_requests"] >= 1  # earlier tests already hit /predict
    assert "prediction_class_distribution" in body
    assert "invalid_input_requests" in body
