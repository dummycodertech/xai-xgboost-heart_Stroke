"""
conformal_fairness.py
=====================
Phase 7: Uncertainty Quantification via Conformal Prediction & Subgroup Fairness Audit.

Components
----------
1. Mondrian Class-Conditional Conformal Prediction (coverage = 90%):
   - Computes conformal nonconformity scores per class out-of-fold.
   - Calculates empirical overall and per-class coverage.
   - Identifies the "ambiguous" prediction set {0, 1} where the model recommends clinical referral.
2. Subgroup Audit by Age Band (<40, 40-60, >60) and Gender (Female, Male):
   - Recall (TPR), FPR, Precision, Flag Rate, and ECE at operating threshold.
   - Bootstrap 95% CIs for each subgroup metric.
   - Equalized Odds gaps: |TPR_A - TPR_B| and |FPR_A - FPR_B|.
   - Cautions on small subgroup sample sizes.
3. Edge-Case Mining (v2):
   - False Positives (top 5 by confidence)
   - False Negatives (top 5 by confidence)
   - Near-Operating-Threshold (band around 0.0826)
   - Ambiguous Conformal Set (patients where set == {0, 1})
   - Saves rich case records to reports/edge_cases_v2.json with SHAP feature attributions.

Outputs
-------
- reports/conformal_results.json
- reports/subgroup_audit.json
- reports/edge_cases_v2.json
- plots/v2/subgroup_audit.png
- plots/v2/conformal_sets.png
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import load_config
from src.pipeline_builder import clean_raw_df
from evaluation import bootstrap_ci, compute_metrics, _convert_numpy
from calibrate import expected_calibration_error

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")
_PLOTS_DIR   = Path("plots/v2")


def mondrian_conformal_calibration(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    alpha: float = 0.10,
) -> tuple[float, float]:
    """Compute class-conditional conformal cutoff quantiles.

    For label 0: nonconformity score = p_cal (prob of label 1)
    For label 1: nonconformity score = 1 - p_cal (prob of label 0)

    Returns
    -------
    q0 : cutoff for label 0 (include 0 if p <= q0)
    q1 : cutoff for label 1 (include 1 if 1 - p <= q1 <=> p >= 1 - q1)
    """
    scores_0 = p_cal[y_cal == 0]
    scores_1 = (1.0 - p_cal)[y_cal == 1]

    n0 = len(scores_0)
    n1 = len(scores_1)

    # Conformal quantile with finite-sample correction: ceil((n+1)*(1-alpha)) / n
    level0 = min(1.0, math.ceil((n0 + 1) * (1.0 - alpha)) / n0)
    level1 = min(1.0, math.ceil((n1 + 1) * (1.0 - alpha)) / n1)

    q0 = float(np.quantile(scores_0, level0))
    q1 = float(np.quantile(scores_1, level1))

    return q0, q1


def predict_conformal_sets(
    p: np.ndarray,
    q0: float,
    q1: float,
) -> list[set[int]]:
    """Generate conformal prediction sets for an array of positive-class probabilities."""
    sets = []
    p_min_for_1 = 1.0 - q1
    p_max_for_0 = q0

    for prob in p:
        s = set()
        if prob <= p_max_for_0:
            s.add(0)
        if prob >= p_min_for_1:
            s.add(1)
        sets.append(s)
    return sets


def run_conformal_and_fairness():
    cfg = load_config()
    seed = cfg["seed"]
    _REPORTS_DIR.mkdir(exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and calibrated model
    log.info("Loading calibrated pipeline and metadata...")
    cal_pipe = joblib.load("models/v2/calibrated_pipeline.joblib")
    with open("models/v2/metadata.json") as f:
        meta = json.load(f)
    operating_threshold = float(meta.get("chosen_threshold", 0.0826))
    log.info("Operating threshold from Phase 5: %.4f", operating_threshold)

    df_raw = clean_raw_df(pd.read_csv(cfg["data"]["raw_path"]))
    with open(cfg["data"]["split_indices_path"]) as f:
        split_meta = json.load(f)
    train_idx = split_meta["train_indices"]

    X_train = df_raw.drop("stroke", axis=1).loc[train_idx].reset_index(drop=True)
    y_train = df_raw["stroke"].loc[train_idx].reset_index(drop=True).values

    # Get calibrated probabilities on train data via 5-fold CV to avoid overfitting
    log.info("Computing out-of-fold calibrated probabilities for conformal evaluation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.full(len(y_train), np.nan)
    
    for tr_i, val_i in cv.split(X_train, y_train):
        pipe_cv = joblib.load("models/v2/pipeline.joblib")
        # Calibrate on tr_i, predict on val_i
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import clone
        c_pipe = CalibratedClassifierCV(estimator=clone(pipe_cv), method=meta.get("calibration_method", "isotonic"), cv=3)
        c_pipe.fit(X_train.iloc[tr_i], y_train[tr_i])
        oof_probs[val_i] = c_pipe.predict_proba(X_train.iloc[val_i])[:, 1]

    # 2. Conformal Prediction Calibration & Evaluation
    log.info("Computing Mondrian Conformal Prediction bounds (alpha=0.10, 90% coverage)...")
    alpha = 0.10
    # Use 2-fold split of OOF to test calibration
    cal_idx, eval_idx = next(StratifiedKFold(n_splits=2, shuffle=True, random_state=seed).split(X_train, y_train))
    q0, q1 = mondrian_conformal_calibration(y_train[cal_idx], oof_probs[cal_idx], alpha=alpha)

    log.info("Conformal cutoffs: q0=%.4f (include 0 if p <= %.4f), q1=%.4f (include 1 if p >= %.4f)",
             q0, q0, q1, 1.0 - q1)

    eval_sets = predict_conformal_sets(oof_probs[eval_idx], q0, q1)
    y_eval = y_train[eval_idx]

    # Measure coverage on eval set
    covered = [y in s for y, s in zip(y_eval, eval_sets)]
    covered_0 = [y in s for y, s in zip(y_eval, eval_sets) if y == 0]
    covered_1 = [y in s for y, s in zip(y_eval, eval_sets) if y == 1]

    n_ambiguous = sum(1 for s in eval_sets if s == {0, 1})
    n_only_0 = sum(1 for s in eval_sets if s == {0})
    n_only_1 = sum(1 for s in eval_sets if s == {1})
    n_empty = sum(1 for s in eval_sets if len(s) == 0)

    conformal_results = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "q0_cutoff": float(q0),
        "q1_cutoff": float(q1),
        "p_threshold_for_label_0": float(q0),
        "p_threshold_for_label_1": float(1.0 - q1),
        "empirical_overall_coverage": float(np.mean(covered)),
        "empirical_class_0_coverage": float(np.mean(covered_0)),
        "empirical_class_1_coverage": float(np.mean(covered_1)),
        "n_evaluated": len(eval_idx),
        "set_composition": {
            "ambiguous_refer_both": int(n_ambiguous),
            "pct_ambiguous": float(n_ambiguous / len(eval_idx)),
            "confidently_no_stroke_only_0": int(n_only_0),
            "pct_only_0": float(n_only_0 / len(eval_idx)),
            "confidently_stroke_only_1": int(n_only_1),
            "pct_only_1": float(n_only_1 / len(eval_idx)),
            "empty": int(n_empty),
        },
    }
    with open(_REPORTS_DIR / "conformal_results.json", "w") as f:
        json.dump(conformal_results, f, indent=2)
    log.info("Conformal Coverage: Overall=%.1f%%, Class 0=%.1f%%, Class 1=%.1f%% | Ambiguous referrals: %d (%.1f%%)",
             conformal_results["empirical_overall_coverage"] * 100,
             conformal_results["empirical_class_0_coverage"] * 100,
             conformal_results["empirical_class_1_coverage"] * 100,
             n_ambiguous, conformal_results["set_composition"]["pct_ambiguous"] * 100)

    # Plot Conformal Sets distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    categories = ["No Stroke {0}", "Ambiguous {0, 1}\n(Clinical Refer)", "Stroke {1}", "Empty {}"]
    counts = [n_only_0, n_ambiguous, n_only_1, n_empty]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"]
    ax.bar(categories, counts, color=colors)
    for i, c in enumerate(counts):
        ax.text(i, c + 15, f"{c} ({c/len(eval_idx):.1%})", ha="center", fontsize=9)
    ax.set_ylabel("Patient Count")
    ax.set_title("Mondrian Conformal Prediction Sets (90% Target Coverage)")
    plt.tight_layout()
    plt.savefig(_PLOTS_DIR / "conformal_sets.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 3. Subgroup Audit by Age Band and Gender
    log.info("Running Subgroup Audit by Age Bands and Gender at threshold=%.4f...", operating_threshold)
    ages = X_train["age"].values
    genders = X_train["gender"].values

    subgroups = {
        "Age < 40": ages < 40,
        "Age 40-60": (ages >= 40) & (ages <= 60),
        "Age > 60": ages > 60,
        "Gender: Female": genders == "Female",
        "Gender: Male": genders == "Male",
    }

    subgroup_report: dict[str, dict] = {}
    for name, mask in subgroups.items():
        n_sub = int(mask.sum())
        n_pos_sub = int(y_train[mask].sum())
        p_sub = oof_probs[mask]
        y_sub = y_train[mask]

        if n_pos_sub == 0:
            log.warning("No positive cases in subgroup %s", name)
            continue

        m_sub = compute_metrics(y_sub, p_sub, threshold=operating_threshold)
        ci_sub = bootstrap_ci(y_sub, p_sub, threshold=operating_threshold, n_resamples=500, seed=seed)
        ece_sub = expected_calibration_error(y_sub, p_sub)
        fpr_sub = m_sub["fp"] / (m_sub["fp"] + m_sub["tn"]) if (m_sub["fp"] + m_sub["tn"]) > 0 else 0.0

        subgroup_report[name] = {
            "n_patients": n_sub,
            "n_positives": n_pos_sub,
            "prevalence": float(n_pos_sub / n_sub),
            "recall": float(m_sub["recall"]),
            "recall_ci_95": [float(ci_sub.get("recall", {}).get("lo", 0)), float(ci_sub.get("recall", {}).get("hi", 0))],
            "precision": float(m_sub["precision"]),
            "precision_ci_95": [float(ci_sub.get("precision", {}).get("lo", 0)), float(ci_sub.get("precision", {}).get("hi", 0))],
            "fpr": float(fpr_sub),
            "flag_rate": float(m_sub["flag_rate"]),
            "ece": float(ece_sub),
        }

    # Equalized Odds Gaps:
    # Female vs Male
    tpr_f = subgroup_report["Gender: Female"]["recall"]
    tpr_m = subgroup_report["Gender: Male"]["recall"]
    fpr_f = subgroup_report["Gender: Female"]["fpr"]
    fpr_m = subgroup_report["Gender: Male"]["fpr"]

    eo_gender = {
        "tpr_gap": float(abs(tpr_f - tpr_m)),
        "fpr_gap": float(abs(fpr_f - fpr_m)),
        "tpr_female": tpr_f,
        "tpr_male": tpr_m,
        "fpr_female": fpr_f,
        "fpr_male": fpr_m,
    }

    audit_payload = {
        "operating_threshold": operating_threshold,
        "subgroups": subgroup_report,
        "equalized_odds_gender_gap": eo_gender,
        "caution_note": (
            "Caution: Subgroup sizes vary significantly. In particular, age < 40 has very few stroke positives (n=2), "
            "leading to extremely wide confidence intervals. Findings must be interpreted with clinical humility."
        ),
    }
    with open(_REPORTS_DIR / "subgroup_audit.json", "w") as f:
        json.dump(audit_payload, f, indent=2)
    log.info("Subgroup Audit saved. Gender TPR gap = %.3f, FPR gap = %.3f", eo_gender["tpr_gap"], eo_gender["fpr_gap"])

    # Plot Subgroup Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    names = list(subgroup_report.keys())
    recalls = [subgroup_report[k]["recall"] for k in names]
    fprs = [subgroup_report[k]["fpr"] for k in names]
    r_err_lo = [subgroup_report[k]["recall"] - subgroup_report[k]["recall_ci_95"][0] for k in names]
    r_err_hi = [subgroup_report[k]["recall_ci_95"][1] - subgroup_report[k]["recall"] for k in names]

    ax1.barh(names, recalls, xerr=[r_err_lo, r_err_hi], color="#1f77b4", alpha=0.8, capsize=4)
    ax1.set_xlabel("Recall (TPR) @ Operating Threshold")
    ax1.set_title("Subgroup Recall (with 95% CIs)")
    ax1.axvline(float(meta.get("chosen_threshold", 0.0826)), color="gray", linestyle=":")

    ax2.barh(names, fprs, color="#ff7f0e", alpha=0.8)
    ax2.set_xlabel("False Positive Rate (FPR)")
    ax2.set_title("Subgroup False Positive Rate")

    plt.tight_layout()
    plt.savefig(_PLOTS_DIR / "subgroup_audit.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Subgroup audit plot saved.")

    # 4. Regenerate Edge-Case Mining (v2)
    log.info("Regenerating Edge-Case Mining (v2) against final model...")
    pipe_uncal = joblib.load("models/v2/pipeline.joblib")
    feature_eng = pipe_uncal.named_steps["feature_eng"]
    preproc = pipe_uncal.named_steps["preprocessor"]
    clf = pipe_uncal.named_steps["classifier"]
    explainer = joblib.load("models/v2/shap_explainer.joblib")

    X_fe = feature_eng.transform(X_train)
    X_proc = preproc.transform(X_fe)
    all_conformal_sets = predict_conformal_sets(oof_probs, q0, q1)

    # Binary predictions at operating threshold
    y_preds = (oof_probs >= operating_threshold).astype(int)

    # 1. False Positives (high confidence false alarms)
    fp_indices = np.where((y_preds == 1) & (y_train == 0))[0]
    fp_sorted = fp_indices[np.argsort(oof_probs[fp_indices])[::-1]][:5]

    # 2. False Negatives (missed strokes)
    fn_indices = np.where((y_preds == 0) & (y_train == 1))[0]
    fn_sorted = fn_indices[np.argsort(oof_probs[fn_indices])][:5]

    # 3. Near Operating Threshold (band: threshold +- 0.015)
    band_lo = max(0.0, operating_threshold - 0.015)
    band_hi = min(1.0, operating_threshold + 0.015)
    near_indices = np.where((oof_probs >= band_lo) & (oof_probs <= band_hi))[0][:5]

    # 4. Conformal Ambiguous Set ({0, 1})
    amb_indices = [i for i, s in enumerate(all_conformal_sets) if s == {0, 1}][:5]

    def build_case_dict(idx: int, case_type: str) -> dict:
        row_raw = X_train.iloc[idx].to_dict()
        prob = float(oof_probs[idx])
        actual = int(y_train[idx])
        c_set = sorted(list(all_conformal_sets[idx]))

        # Get SHAP top 3 features
        row_proc = X_proc.iloc[idx:idx+1] if isinstance(X_proc, pd.DataFrame) else X_proc[idx:idx+1]
        shap_val = explainer(row_proc).values[0]
        proc_names = preproc.get_feature_names_out().tolist()
        top_feat_indices = np.argsort(np.abs(shap_val))[::-1][:3]
        top_shap = [
            {"feature": proc_names[fi], "shap_value": round(float(shap_val[fi]), 4)}
            for fi in top_feat_indices
        ]

        return {
            "index": int(idx),
            "case_type": case_type,
            "calibrated_prob": round(prob, 4),
            "actual_stroke": actual,
            "predicted_binary": int(prob >= operating_threshold),
            "conformal_set": c_set,
            "patient_features": row_raw,
            "top_shap_factors": top_shap,
        }

    edge_cases_v2 = {
        "operating_threshold": operating_threshold,
        "near_threshold_band": [band_lo, band_hi],
        "false_positives": [build_case_dict(i, "false_positive") for i in fp_sorted],
        "false_negatives": [build_case_dict(i, "false_negative") for i in fn_sorted],
        "near_threshold": [build_case_dict(i, "near_threshold") for i in near_indices],
        "conformal_ambiguous": [build_case_dict(i, "conformal_ambiguous") for i in amb_indices],
    }

    with open(_REPORTS_DIR / "edge_cases_v2.json", "w") as f:
        json.dump(_convert_numpy(edge_cases_v2), f, indent=2)
    log.info("Edge cases v2 saved to reports/edge_cases_v2.json")
    log.info("Phase 7 complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_conformal_and_fairness()
