"""
schemas.py
----------
Pydantic models describing exactly what the API accepts and returns.

Having these as real Pydantic models (rather than raw dicts) is what makes
FastAPI able to validate incoming requests automatically and reject
malformed input with a clear, structured error -- before my prediction
code ever runs.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class PatientFeatures(BaseModel):
    """
    The raw input fields the trained pipeline expects. These match the raw
    columns in the training data exactly (before any preprocessing) --
    the persisted pipeline itself does the imputing/scaling/encoding, so
    the API only has to validate that the *raw* values make sense.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    gender: Literal["Male", "Female", "Other"] = Field(
        ..., description="Patient's gender as recorded in the source data."
    )
    age: float = Field(..., ge=0, le=120, description="Age in years.")
    hypertension: Literal[0, 1] = Field(
        ..., description="1 if the patient has hypertension, else 0."
    )
    heart_disease: Literal[0, 1] = Field(
        ..., description="1 if the patient has a heart condition, else 0."
    )
    ever_married: Literal["Yes", "No"] = Field(...)
    work_type: Literal[
        "Private", "Self-employed", "Govt_job", "children", "Never_worked"
    ] = Field(...)
    Residence_type: Literal["Urban", "Rural"] = Field(...)
    avg_glucose_level: float = Field(
        ..., gt=0, le=400, description="Average blood glucose level (mg/dL)."
    )
    bmi: Optional[float] = Field(
        None, ge=0, le=100,
        description="Body mass index. Optional -- the trained pipeline "
                     "median-imputes this if left out, same as it did "
                     "during training.",
    )
    smoking_status: Literal[
        "formerly smoked", "never smoked", "smokes", "Unknown"
    ] = Field(...)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no stroke predicted, 1 = stroke predicted.")
    probability: float = Field(..., description="Model's predicted probability of stroke (0-1).")
    model_version: str
    latency_ms: float = Field(..., description="Time taken to produce this prediction, in milliseconds.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float


class HomeResponse(BaseModel):
    message: str
    model_version: Optional[str] = None
    docs_url: str
    ui_url: str
    endpoints: list[str]
