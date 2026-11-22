"""Tests for SHAP aggregation additivity and explainer outputs."""
import joblib
import numpy as np
import pandas as pd

from explainability_v2 import aggregate_shap_values, map_feature_to_orig


def test_map_feature_to_orig():
    assert map_feature_to_orig("num__age") == "age"
    assert map_feature_to_orig("bin__hypertension") == "hypertension"
    assert map_feature_to_orig("cat__gender_Female") == "gender"
    assert map_feature_to_orig("cat__work_type_Govt_job") == "work_type"
    assert map_feature_to_orig("cat__smoking_status_formerly smoked") == "smoking_status"


def test_shap_aggregation_additivity():
    # Synthetic SHAP values for 5 samples x 4 one-hot columns
    # cols: cat__gender_Female, cat__gender_Male, num__age, bin__hypertension
    feature_names = ["cat__gender_Female", "cat__gender_Male", "num__age", "bin__hypertension"]
    np.random.seed(42)
    synthetic_shap = np.random.randn(5, 4)

    agg_shap, orig_names = aggregate_shap_values(synthetic_shap, feature_names)

    # Must aggregate 4 columns into 3 unique original features: gender, age, hypertension
    assert set(orig_names) == {"gender", "age", "hypertension"}
    assert agg_shap.shape == (5, 3)

    # Additivity check: row sums must be identical
    assert np.allclose(synthetic_shap.sum(axis=1), agg_shap.sum(axis=1), atol=1e-7)


def test_shap_explainer_loading_and_inference():
    explainer = joblib.load("models/v2/shap_explainer.joblib")
    pipeline = joblib.load("models/v2/pipeline.joblib")
    prep = pipeline.named_steps["preprocessor"]
    fe = pipeline.named_steps["feature_eng"]

    sample_df = pd.DataFrame([{
        "age": 60.0, "avg_glucose_level": 150.0, "bmi": 29.0,
        "gender": "Male", "hypertension": 1, "heart_disease": 0,
        "ever_married": "Yes", "work_type": "Private",
        "Residence_type": "Rural", "smoking_status": "smokes"
    }])

    X_trans = prep.transform(fe.transform(sample_df))
    shap_vals = explainer(X_trans)
    assert shap_vals.values.shape[0] == 1
    assert not np.isnan(shap_vals.values).any()
