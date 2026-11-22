"""Tests for model pipelines, splits, and probability calibration."""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import load_config


def test_split_indices_no_overlap():
    cfg = load_config()
    with open(cfg["data"]["split_indices_path"]) as f:
        meta = json.load(f)
    train_idx = set(meta["train_indices"])
    test_idx = set(meta["test_indices"])
    assert len(train_idx & test_idx) == 0, "Train and test indices overlap!"
    assert len(train_idx) + len(test_idx) == 5109


def test_uncalibrated_pipeline_output_shape_and_range():
    pipeline = joblib.load("models/v2/pipeline.joblib")
    sample_df = pd.DataFrame([{
        "age": 55.0, "avg_glucose_level": 120.0, "bmi": 28.5,
        "gender": "Female", "hypertension": 1, "heart_disease": 0,
        "ever_married": "Yes", "work_type": "Private",
        "Residence_type": "Urban", "smoking_status": "smokes"
    }])
    probs = pipeline.predict_proba(sample_df)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 0] <= 1.0
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)


def test_calibrated_pipeline_monotonicity_and_validity():
    cal_pipe = joblib.load("models/v2/calibrated_pipeline.joblib")
    # Patient with baseline risk
    p_base = pd.DataFrame([{
        "age": 40.0, "avg_glucose_level": 90.0, "bmi": 24.0,
        "gender": "Male", "hypertension": 0, "heart_disease": 0,
        "ever_married": "No", "work_type": "Private",
        "Residence_type": "Urban", "smoking_status": "never smoked"
    }])
    # Patient with higher age & glucose & lifestyle risks
    p_high = pd.DataFrame([{
        "age": 75.0, "avg_glucose_level": 220.0, "bmi": 35.0,
        "gender": "Male", "hypertension": 1, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Private",
        "Residence_type": "Urban", "smoking_status": "smokes"
    }])

    prob_base = cal_pipe.predict_proba(p_base)[0, 1]
    prob_high = cal_pipe.predict_proba(p_high)[0, 1]

    assert 0.0 <= prob_base <= 1.0
    assert 0.0 <= prob_high <= 1.0
    assert prob_high > prob_base, "Severe risk patient should have strictly higher predicted probability"


def test_operating_threshold_bounds():
    with open("models/v2/metadata.json") as f:
        meta = json.load(f)
    thr = meta["chosen_threshold"]
    assert 0.0 < thr < 1.0
    assert 0.01 <= thr <= 0.30
