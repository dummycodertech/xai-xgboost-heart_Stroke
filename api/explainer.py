"""
explainer.py
============
In-process explainability engines:
1. Exact 5-Fold TreeSHAP:
   - Runs native XGBoost C++ pred_contribs across all 5 fold boosters inside CalibratedClassifierCV
   - Averages attributions, base value, and raw margins across folds
   - Aggregates one-hot dummy features back to original clinical features
   - Evaluates in raw margin (log-odds) space with strict additivity verification
2. Local Surrogate Explainer:
   - Evaluates in calibrated probability units
   - Generates human-readable clinical rules using training background quantiles
   - Exponential-kernel weighted Ridge regression with zero heavy external dependencies
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Load background quantiles
with open(ARTIFACTS_DIR / "background_quantiles.json") as f:
    BACKGROUND_QUANTILES = json.load(f)

RAW_FEATURE_ORDER = [
    "age", "avg_glucose_level", "bmi", "gender", "hypertension",
    "heart_disease", "combined_risk_factors", "ever_married", "work_type", "Residence_type", "smoking_status"
]


def compute_5fold_treeshap(
    cal_pipe,
    input_df: pd.DataFrame,
    patient_dict: dict,
) -> tuple[float, float, list[dict]]:
    """Compute exact TreeSHAP attributions averaged across all 5 fold boosters.
    
    Averages contributions, base values, and raw margins across the 5 fold boosters
    inside the CalibratedClassifierCV. Preserves exact additivity:
    avg_base_value + sum(contributions) == avg_raw_margin.
    The engineered interaction feature `risk_factors_count` is presented as its own
    row ("combined_risk_factors") rather than split arbitrarily across clinical drivers.
    """
    all_phi = []
    raw_margins = []
    feature_names_out = None

    for cc in cal_pipe.calibrated_classifiers_:
        pipe = cc.estimator
        fe_step = pipe.named_steps["feature_eng"]
        prep_step = pipe.named_steps["preprocessor"]
        clf_step = pipe.named_steps["classifier"]

        # Transform using this fold's fitted pipeline steps
        X_fe = fe_step.transform(input_df)
        X_trans = np.asarray(prep_step.transform(X_fe))
        feature_names_out = prep_step.get_feature_names_out().tolist()

        booster = clf_step.get_booster()
        dm = xgb.DMatrix(X_trans, feature_names=feature_names_out)

        phi = booster.predict(dm, pred_contribs=True)[0]
        margin = booster.predict(dm, output_margin=True)[0]

        all_phi.append(phi)
        raw_margins.append(margin)

    # Average across all 5 folds
    avg_phi = np.mean(all_phi, axis=0)
    avg_base_value = float(avg_phi[-1])
    avg_contribs = avg_phi[:-1]
    avg_raw_margin = float(np.mean(raw_margins))

    # Additivity assertion
    additivity_diff = abs((avg_base_value + np.sum(avg_contribs)) - avg_raw_margin)
    if additivity_diff > 1e-4:
        raise ValueError(f"TreeSHAP additivity violated! diff={additivity_diff:.2e}")

    # Aggregate dummy and engineered variables back to parent features
    agg_dict = {f: 0.0 for f in RAW_FEATURE_ORDER}
    for col_name, val in zip(feature_names_out, avg_contribs):
        matched = False
        if "bmi_missing" in col_name:
            agg_dict["bmi"] += val
            matched = True
        elif "risk_factors_count" in col_name:
            agg_dict["combined_risk_factors"] += val
            matched = True
        else:
            for rf in RAW_FEATURE_ORDER:
                if col_name.startswith(f"cat__{rf}_") or col_name == f"num__{rf}" or col_name == f"bin__{rf}" or col_name == rf:
                    agg_dict[rf] += val
                    matched = True
                    break

        if not matched:
            # Fallback for any other unmatched feature
            agg_dict["combined_risk_factors"] += val

    # Format values for display
    def format_val(f, v):
        if f == "age":
            return f"{int(v)}"
        if f == "avg_glucose_level":
            return f"{v:.1f} mg/dL"
        if f == "bmi":
            return f"{v:.1f}"
        if f in ["hypertension", "heart_disease"]:
            return "Yes" if v == 1 else "No"
        if f == "combined_risk_factors":
            rfc = (
                int(patient_dict["hypertension"])
                + int(patient_dict["heart_disease"])
                + (1 if patient_dict["smoking_status"] in ["smokes", "formerly smoked"] else 0)
            )
            return f"{rfc} active"
        return str(v)

    shap_items = []
    for f in RAW_FEATURE_ORDER:
        val_to_format = patient_dict.get(f, 0)
        shap_items.append({
            "feature": f,
            "value": format_val(f, val_to_format),
            "contribution": float(round(agg_dict.get(f, 0.0), 3)),
        })

    # Sort descending by absolute attribution
    shap_items.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return avg_base_value, avg_raw_margin, shap_items


def compute_local_surrogate(
    cal_pipe,
    patient_dict: dict,
    n_samples: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Compute local surrogate (LIME-style) linear model in calibrated probability units."""
    rng = np.random.default_rng(seed)

    age_val = float(patient_dict["age"])
    glu_val = float(patient_dict["avg_glucose_level"])
    bmi_val = float(patient_dict["bmi"])

    q_age = BACKGROUND_QUANTILES["age"]
    q_glu = BACKGROUND_QUANTILES["avg_glucose_level"]
    q_bmi = BACKGROUND_QUANTILES["bmi"]

    # Build human-readable conditions
    rule_defs = []

    # 1. Age condition
    if age_val > q_age[2]:
        rule_defs.append(("age > 61", lambda r: 1.0 if r["age"] > q_age[2] else 0.0))
    elif age_val > q_age[1]:
        rule_defs.append(("45 < age <= 61", lambda r: 1.0 if (r["age"] > q_age[1] and r["age"] <= q_age[2]) else 0.0))
    else:
        rule_defs.append(("age <= 45", lambda r: 1.0 if r["age"] <= q_age[1] else 0.0))

    # 2. Glucose condition
    if glu_val > q_glu[2]:
        rule_defs.append(("avg_glucose_level > 114", lambda r: 1.0 if r["avg_glucose_level"] > q_glu[2] else 0.0))
    else:
        rule_defs.append(("avg_glucose_level <= 114", lambda r: 1.0 if r["avg_glucose_level"] <= q_glu[2] else 0.0))

    # 3. BMI condition
    if bmi_val > q_bmi[2]:
        rule_defs.append(("bmi > 33.1", lambda r: 1.0 if r["bmi"] > q_bmi[2] else 0.0))
    else:
        rule_defs.append(("bmi <= 33.1", lambda r: 1.0 if r["bmi"] <= q_bmi[2] else 0.0))

    # 4. Hypertension
    if patient_dict["hypertension"] == 1:
        rule_defs.append(("hypertension = Yes", lambda r: 1.0 if r["hypertension"] == 1 else 0.0))
    else:
        rule_defs.append(("hypertension = No", lambda r: 1.0 if r["hypertension"] == 0 else 0.0))

    # 5. Heart disease
    if patient_dict["heart_disease"] == 1:
        rule_defs.append(("heart_disease = Yes", lambda r: 1.0 if r["heart_disease"] == 1 else 0.0))
    else:
        rule_defs.append(("heart_disease = No", lambda r: 1.0 if r["heart_disease"] == 0 else 0.0))

    # 6. Smoking status
    smk = patient_dict["smoking_status"]
    rule_defs.append((f"smoking_status = {smk}", lambda r: 1.0 if r["smoking_status"] == smk else 0.0))

    # Generate perturbations around the patient
    records = []
    binary_matrix = []

    for _ in range(n_samples):
        p = dict(patient_dict)
        # Jitter continuous values
        p["age"] = float(np.clip(age_val + rng.normal(0, 12), 0.0, 100.0))
        p["avg_glucose_level"] = float(np.clip(glu_val + rng.normal(0, 25), 50.0, 300.0))
        p["bmi"] = float(np.clip(bmi_val + rng.normal(0, 5), 10.0, 100.0))

        # Randomly flip binary / categoricals with low probability
        if rng.random() < 0.20:
            p["hypertension"] = 1 - int(p["hypertension"])
        if rng.random() < 0.15:
            p["heart_disease"] = 1 - int(p["heart_disease"])
        if rng.random() < 0.25:
            p["smoking_status"] = rng.choice(["never smoked", "formerly smoked", "smokes", "Unknown"])
        if rng.random() < 0.15:
            p["work_type"] = rng.choice(["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

        records.append(p)
        binary_row = [eval_fn(p) for _, eval_fn in rule_defs]
        binary_matrix.append(binary_row)

    df_pert = pd.DataFrame(records)
    # Calibrated probability output
    y_pert = cal_pipe.predict_proba(df_pert)[:, 1]

    X_mat = np.array(binary_matrix)
    x_orig = np.ones(len(rule_defs))  # The patient matches all active rule definitions

    # Exponential kernel weights based on feature distance
    dists = np.sum((X_mat - x_orig) ** 2, axis=1)
    weights = np.exp(-dists / 2.0)

    # Weighted Ridge regression in probability space
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_mat, y_pert, sample_weight=weights)

    lime_items = []
    for (rule_label, _), coef in zip(rule_defs, ridge.coef_):
        lime_items.append({
            "rule": rule_label,
            "weight": float(round(coef, 4)),
        })

    lime_items.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return lime_items[:6]
