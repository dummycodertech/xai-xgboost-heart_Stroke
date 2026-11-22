"""
reproduce.py
============
Single master script to reproduce the entire v2 pipeline end-to-end.

Sequence:
1. data_loader.py            -> downloads data, checks cleaning
2. src/save_split_indices.py -> freezes train/test split indices
3. build_v2_pipeline.py      -> builds leak-free sklearn pipeline on train split
4. evaluation.py             -> runs 5x5 repeated stratified CV, bootstrap CIs, DCA
5. model_zoo.py              -> runs 13-model comparison & SMOTE ablation under identical folds
6. hyperparameter_tuning.py  -> Optuna HPO, monotonic constraints, stacking comparison
7. calibrate.py              -> OOF isotonic vs sigmoid calibration, cost threshold selection
8. explainability_v2.py      -> aggregated SHAP, dependence, interactions, LIME evaluation
9. conformal_fairness.py     -> Mondrian conformal prediction, subgroup audit, edge cases v2
10. final_eval.py            -> single one-time held-out test set evaluation with CIs
11. pytest tests/            -> runs complete test suite

Usage:
------
uv run python reproduce.py
"""
import subprocess
import sys
import time

STEPS = [
    ("Freezing Split Indices", [sys.executable, "src/save_split_indices.py"]),
    ("Building v2 Leak-Free Pipeline", [sys.executable, "build_v2_pipeline.py"]),
    ("Robust Evaluation Harness (5x5 CV)", [sys.executable, "evaluation.py"]),
    ("Model Zoo & SMOTE Ablation", [sys.executable, "model_zoo.py"]),
    ("Optuna HPO & Monotone Constraints", [sys.executable, "hyperparameter_tuning.py"]),
    ("Probability Calibration & Cost Threshold", [sys.executable, "calibrate.py"]),
    ("Explainability Upgrades (Aggregated SHAP)", [sys.executable, "explainability_v2.py"]),
    ("Conformal Prediction & Fairness Audit", [sys.executable, "conformal_fairness.py"]),
    ("One-Time Held-Out Test Evaluation", [sys.executable, "final_eval.py"]),
    ("Automated Pytest Suite", [sys.executable, "-m", "pytest", "tests", "-v"]),
]

def main():
    print("=" * 70)
    print("REPRODUCING STROKE XAI PIPELINE (v2 RIGOROUS UPGRADE)")
    print("=" * 70)

    start_total = time.time()
    for title, cmd in STEPS:
        print(f"\n>>> Running: {title}...")
        t0 = time.time()
        res = subprocess.run(cmd)
        elapsed = time.time() - t0
        if res.returncode != 0:
            print(f"❌ FAILED: {title} (exited with {res.returncode})")
            sys.exit(res.returncode)
        print(f"✅ COMPLETED: {title} in {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print(f"ALL STEPS COMPLETED SUCCESSFULLY in {time.time() - start_total:.1f}s!")
    print("Outputs verified in reports/, plots/v2/, and models/v2/")
    print("=" * 70)

if __name__ == "__main__":
    main()
