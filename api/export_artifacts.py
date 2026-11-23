"""
export_artifacts.py
===================
Exports lean, self-contained production artifacts into api/artifacts/:
1. Copies models/calibrated_pipeline.joblib
2. Exports api/artifacts/meta.json directly from reports/final_test_report.json and models/metadata.json
3. Exports background quantiles for human-readable local surrogate rules
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.pipeline_builder import clean_raw_df

ROOT_DIR = Path(__file__).resolve().parent.parent
API_ARTIFACTS_DIR = ROOT_DIR / "api" / "artifacts"
API_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def export_production_artifacts():
    print("[1/3] Copying calibrated pipeline...")
    src_pipe = ROOT_DIR / "models" / "calibrated_pipeline.joblib"
    dst_pipe = API_ARTIFACTS_DIR / "calibrated_pipeline.joblib"
    shutil.copy2(src_pipe, dst_pipe)
    print(f"  Copied {src_pipe.name} ({dst_pipe.stat().st_size / 1024:.1f} KB)")

    print("[2/3] Generating api/artifacts/meta.json from reports/final_test_report.json...")
    with open(ROOT_DIR / "reports" / "final_test_report.json") as f:
        final_test = json.load(f)
    with open(ROOT_DIR / "models" / "metadata.json") as f:
        model_meta = json.load(f)

    # Structure multi-threshold metadata
    eval_ops = final_test.get("operating_thresholds_evaluation", {})

    meta_payload = {
        "model_version": "v2.0-calibrated",
        "model_type": "Tuned XGBoost + 5-Fold Isotonic Calibration",
        "default_threshold": float(model_meta.get("chosen_threshold", 0.08259259259259258)),
        "calibration_method": model_meta.get("calibration_method", "isotonic"),
        "operating_points": {
            "cost_based_10": {
                "id": "cost_based_10",
                "label": "Cost-Optimal (FN/FP = 10)",
                "threshold": float(eval_ops.get("cost_based_10", {}).get("threshold", 0.0826)),
                "description": "Balances false negatives 10x higher than false alarms; minimizes clinical misclassification cost.",
                "recall": float(eval_ops.get("cost_based_10", {}).get("metrics", {}).get("recall", 0.62)),
                "recall_ci": [
                    float(eval_ops.get("cost_based_10", {}).get("bootstrap_95_ci", {}).get("recall", {}).get("lo", 0.48)),
                    float(eval_ops.get("cost_based_10", {}).get("bootstrap_95_ci", {}).get("recall", {}).get("hi", 0.76)),
                ],
                "precision": float(eval_ops.get("cost_based_10", {}).get("metrics", {}).get("precision", 0.119)),
                "precision_ci": [
                    float(eval_ops.get("cost_based_10", {}).get("bootstrap_95_ci", {}).get("precision", {}).get("lo", 0.095)),
                    float(eval_ops.get("cost_based_10", {}).get("bootstrap_95_ci", {}).get("precision", {}).get("hi", 0.144)),
                ],
                "false_alarms": int(eval_ops.get("cost_based_10", {}).get("metrics", {}).get("fp", 229)),
                "false_alarms_ci": [
                    round(eval_ops.get("cost_based_10", {}).get("bootstrap_95_ci", {}).get("fp", {}).get("lo", 203)),
                    round(eval_ops.get("cost_based_10", {}).get("bootstrap_95_ci", {}).get("fp", {}).get("hi", 256)),
                ],
                "nns": float(eval_ops.get("cost_based_10", {}).get("metrics", {}).get("nns", 8.4)),
                "nns_ci": [
                    float(eval_ops.get("cost_based_10", {}).get("bootstrap_95_ci", {}).get("nns", {}).get("lo", 6.9)),
                    float(eval_ops.get("cost_based_10", {}).get("bootstrap_95_ci", {}).get("nns", {}).get("hi", 10.6)),
                ],
                "matrix": {
                    "tp": int(eval_ops.get("cost_based_10", {}).get("metrics", {}).get("tp", 31)),
                    "fp": int(eval_ops.get("cost_based_10", {}).get("metrics", {}).get("fp", 229)),
                    "fn": int(eval_ops.get("cost_based_10", {}).get("metrics", {}).get("fn", 19)),
                    "tn": int(eval_ops.get("cost_based_10", {}).get("metrics", {}).get("tn", 743)),
                },
            },
            "min_recall_80pct": {
                "id": "min_recall_80pct",
                "label": "Target 80% OOF Recall",
                "threshold": float(eval_ops.get("min_recall_80pct", {}).get("threshold", 0.0581)),
                "description": "Selected to achieve >= 80% sensitivity out-of-fold during clinical triage.",
                "recall": float(eval_ops.get("min_recall_80pct", {}).get("metrics", {}).get("recall", 0.78)),
                "recall_ci": [
                    float(eval_ops.get("min_recall_80pct", {}).get("bootstrap_95_ci", {}).get("recall", {}).get("lo", 0.66)),
                    float(eval_ops.get("min_recall_80pct", {}).get("bootstrap_95_ci", {}).get("recall", {}).get("hi", 0.88)),
                ],
                "precision": float(eval_ops.get("min_recall_80pct", {}).get("metrics", {}).get("precision", 0.117)),
                "precision_ci": [
                    float(eval_ops.get("min_recall_80pct", {}).get("bootstrap_95_ci", {}).get("precision", {}).get("lo", 0.100)),
                    float(eval_ops.get("min_recall_80pct", {}).get("bootstrap_95_ci", {}).get("precision", {}).get("hi", 0.135)),
                ],
                "false_alarms": int(eval_ops.get("min_recall_80pct", {}).get("metrics", {}).get("fp", 293)),
                "false_alarms_ci": [
                    round(eval_ops.get("min_recall_80pct", {}).get("bootstrap_95_ci", {}).get("fp", {}).get("lo", 266)),
                    round(eval_ops.get("min_recall_80pct", {}).get("bootstrap_95_ci", {}).get("fp", {}).get("hi", 321)),
                ],
                "nns": float(eval_ops.get("min_recall_80pct", {}).get("metrics", {}).get("nns", 8.5)),
                "nns_ci": [
                    float(eval_ops.get("min_recall_80pct", {}).get("bootstrap_95_ci", {}).get("nns", {}).get("lo", 7.4)),
                    float(eval_ops.get("min_recall_80pct", {}).get("bootstrap_95_ci", {}).get("nns", {}).get("hi", 10.0)),
                ],
                "matrix": {
                    "tp": int(eval_ops.get("min_recall_80pct", {}).get("metrics", {}).get("tp", 39)),
                    "fp": int(eval_ops.get("min_recall_80pct", {}).get("metrics", {}).get("fp", 293)),
                    "fn": int(eval_ops.get("min_recall_80pct", {}).get("metrics", {}).get("fn", 11)),
                    "tn": int(eval_ops.get("min_recall_80pct", {}).get("metrics", {}).get("tn", 679)),
                },
            },
            "high_precision": {
                "id": "high_precision",
                "label": "High Precision (Lower False Alarms)",
                "threshold": float(eval_ops.get("high_precision", {}).get("threshold", 0.1365)),
                "description": "Prioritizes positive predictive value; reduces false alarms to 69 while catching 32% of strokes.",
                "recall": float(eval_ops.get("high_precision", {}).get("metrics", {}).get("recall", 0.32)),
                "recall_ci": [
                    float(eval_ops.get("high_precision", {}).get("bootstrap_95_ci", {}).get("recall", {}).get("lo", 0.20)),
                    float(eval_ops.get("high_precision", {}).get("bootstrap_95_ci", {}).get("recall", {}).get("hi", 0.46)),
                ],
                "precision": float(eval_ops.get("high_precision", {}).get("metrics", {}).get("precision", 0.188)),
                "precision_ci": [
                    float(eval_ops.get("high_precision", {}).get("bootstrap_95_ci", {}).get("precision", {}).get("lo", 0.121)),
                    float(eval_ops.get("high_precision", {}).get("bootstrap_95_ci", {}).get("precision", {}).get("hi", 0.263)),
                ],
                "false_alarms": int(eval_ops.get("high_precision", {}).get("metrics", {}).get("fp", 69)),
                "false_alarms_ci": [
                    round(eval_ops.get("high_precision", {}).get("bootstrap_95_ci", {}).get("fp", {}).get("lo", 53)),
                    round(eval_ops.get("high_precision", {}).get("bootstrap_95_ci", {}).get("fp", {}).get("hi", 86)),
                ],
                "nns": float(eval_ops.get("high_precision", {}).get("metrics", {}).get("nns", 5.3)),
                "nns_ci": [
                    float(eval_ops.get("high_precision", {}).get("bootstrap_95_ci", {}).get("nns", {}).get("lo", 3.8)),
                    float(eval_ops.get("high_precision", {}).get("bootstrap_95_ci", {}).get("nns", {}).get("hi", 8.3)),
                ],
                "matrix": {
                    "tp": int(eval_ops.get("high_precision", {}).get("metrics", {}).get("tp", 16)),
                    "fp": int(eval_ops.get("high_precision", {}).get("metrics", {}).get("fp", 69)),
                    "fn": int(eval_ops.get("high_precision", {}).get("metrics", {}).get("fn", 34)),
                    "tn": int(eval_ops.get("high_precision", {}).get("metrics", {}).get("tn", 903)),
                },
            },
        },
        "evaluation_dataset": {
            "n_test": int(final_test["dataset"]["n_test"]),
            "n_positives": int(final_test["dataset"]["n_test_positives"]),
            "n_negatives": int(final_test["dataset"]["n_test_negatives"]),
            "split_label": "Stratified 80/20 held-out evaluation (1,022 patients, 50 strokes, 95% bootstrap CIs)",
        },
        "features": {
            "numerical": ["age", "avg_glucose_level", "bmi"],
            "categorical": ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"],
            "categories": {
                "gender": ["Male", "Female"],
                "hypertension": [0, 1],
                "heart_disease": [0, 1],
                "ever_married": ["Yes", "No"],
                "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                "Residence_type": ["Urban", "Rural"],
                "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
            },
            "bounds": {
                "age": [0, 100],
                "avg_glucose_level": [50.0, 300.0],
                "bmi": [10.0, 100.0],
            },
        },
    }

    with open(API_ARTIFACTS_DIR / "meta.json", "w") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"  Exported meta.json ({(API_ARTIFACTS_DIR / 'meta.json').stat().st_size / 1024:.1f} KB)")

    print("[3/3] Generating background quantiles for human-readable local surrogate rules...")
    df_raw = clean_raw_df(pd.read_csv(ROOT_DIR / "data" / "healthcare-dataset-stroke-data.csv"))
    with open(ROOT_DIR / "reports" / "split_indices.json") as f:
        train_idx = json.load(f)["train_indices"]
    df_train = df_raw.loc[train_idx]

    quantiles = {
        "age": [float(df_train["age"].quantile(q)) for q in [0.25, 0.50, 0.75]],
        "avg_glucose_level": [float(df_train["avg_glucose_level"].quantile(q)) for q in [0.25, 0.50, 0.75]],
        "bmi": [float(df_train["bmi"].dropna().quantile(q)) for q in [0.25, 0.50, 0.75]],
    }
    with open(API_ARTIFACTS_DIR / "background_quantiles.json", "w") as f:
        json.dump(quantiles, f, indent=2)
    print(f"  Exported background_quantiles.json: {quantiles}")
    print("Done! All production artifacts exported to api/artifacts/.")


if __name__ == "__main__":
    export_production_artifacts()
