"""
schemas.py
==========
Pydantic schemas with strict clinical validation bounds.
Zero echo of patient input values in validation error responses (privacy requirement).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=0.0, le=100.0, description="Patient age in years (0-100)")
    avg_glucose_level: float = Field(..., ge=50.0, le=300.0, description="Average blood glucose level in mg/dL (50-300)")
    bmi: float = Field(..., ge=10.0, le=100.0, description="Body Mass Index in kg/m^2 (10-100)")
    gender: Literal["Male", "Female"]
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    Residence_type: Literal["Urban", "Rural"]
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]
    operating_point: Literal["cost_based_10", "min_recall_80pct", "high_precision"] | None = "cost_based_10"


class ShapItem(BaseModel):
    feature: str
    value: str
    contribution: float


class LocalSurrogateItem(BaseModel):
    rule: str
    weight: float


class PredictionResponse(BaseModel):
    probability: float
    threshold: float
    operating_point: str
    decision: Literal["HIGH_RISK", "BELOW_THRESHOLD"]
    base_value: float
    raw_margin: float
    shap: list[ShapItem]
    local_surrogate: list[LocalSurrogateItem] = Field(
        ...,
        description="Local surrogate (LIME-style) linear rules in calibrated probability points",
    )
    lime: list[LocalSurrogateItem] = Field(
        ...,
        description="Alias for local_surrogate (LIME-style) rules",
    )
    units: dict[str, str] = {
        "shap": "log-odds (raw model score)",
        "local_surrogate (LIME-style)": "calibrated probability points",
        "probability": "calibrated risk percentage",
    }
