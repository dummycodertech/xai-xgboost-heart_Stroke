"""Smoke tests for application inference logic."""
import joblib
import pandas as pd


def test_app_inference_pipeline_execution():
    cal_pipe = joblib.load("models/v2/calibrated_pipeline.joblib")
    raw_pipe = joblib.load("models/v2/pipeline.joblib")

    # Sample patient mimicking UI inputs
    patient_df = pd.DataFrame([{
        "age": 52,
        "avg_glucose_level": 115.0,
        "bmi": 27.4,
        "gender": "Female",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "smoking_status": "never smoked"
    }])

    prob_cal = cal_pipe.predict_proba(patient_df)[0][1]
    assert 0.0 <= prob_cal <= 1.0

    fe = raw_pipe.named_steps["feature_eng"]
    prep = raw_pipe.named_steps["preprocessor"]
    X_trans = prep.transform(fe.transform(patient_df))
    assert X_trans.shape[1] == 22
