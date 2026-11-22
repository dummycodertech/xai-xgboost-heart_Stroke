"""Tests for data loading, cleaning, and feature engineering."""
import numpy as np
import pandas as pd

from src.pipeline_builder import _add_engineered_features, clean_raw_df


def test_clean_raw_df_removes_other_gender():
    raw_path = "data/healthcare-dataset-stroke-data.csv"
    df = pd.read_csv(raw_path)
    assert len(df) == 5110
    
    cleaned = clean_raw_df(df)
    assert len(cleaned) == 5109
    assert "Other" not in cleaned["gender"].values
    assert "id" not in cleaned.columns


def test_risk_factors_count_correctness():
    # Case 1: hypertension=1, heart_disease=1, smoker=smokes -> count must be 3 (fixing legacy bug)
    df_test = pd.DataFrame([{
        "hypertension": 1,
        "heart_disease": 1,
        "smoking_status": "smokes",
        "bmi": 28.0,
    }])
    df_feat = _add_engineered_features(df_test)
    assert df_feat["risk_factors_count"].iloc[0] == 3
    assert df_feat["bmi_missing"].iloc[0] == 0

    # Case 2: formerly smoked
    df_test2 = pd.DataFrame([{
        "hypertension": 0,
        "heart_disease": 1,
        "smoking_status": "formerly smoked",
        "bmi": np.nan if "np" in locals() else None,
    }])
    df_feat2 = _add_engineered_features(df_test2)
    assert df_feat2["risk_factors_count"].iloc[0] == 2
    assert df_feat2["bmi_missing"].iloc[0] == 1

    # Case 3: never smoked, no conditions
    df_test3 = pd.DataFrame([{
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_status": "never smoked",
        "bmi": 22.0,
    }])
    df_feat3 = _add_engineered_features(df_test3)
    assert df_feat3["risk_factors_count"].iloc[0] == 0

    # Case 4: Unknown smoking status
    df_test4 = pd.DataFrame([{
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_status": "Unknown",
        "bmi": 25.0,
    }])
    df_feat4 = _add_engineered_features(df_test4)
    assert df_feat4["risk_factors_count"].iloc[0] == 1
