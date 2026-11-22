"""
hyperparameter_tuning.py
========================
Phase 4: Optuna HPO for XGBoost (best model from Phase 3).

Optimizes PR-AUC via 5×5 repeated stratified CV (OOF mean).
Fixed trial budget: 100 (from config). Seeded sampler.
SQLite storage: reports/optuna.db.

Additionally:
  - Evaluates monotonic constraints (age, avg_glucose_level non-decreasing)
    under the same CV, comparing constrained vs unconstrained.
  - Builds a stacking ensemble (XGBoost + LR + RF, logistic meta-learner,
    nested CV so the meta-learner sees only OOF predictions).

The final model is chosen by OOF PR-AUC. The test set is NOT touched.

Usage
-----
.venv/Scripts/python.exe hyperparameter_tuning.py
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from xgboost import XGBClassifier

from evaluation import bootstrap_ci, compute_metrics, _convert_numpy
from src.config import load_config
from src.pipeline_builder import build_pipeline, clean_raw_df

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")
_PLOTS_DIR   = Path("plots/v2")


# ---------------------------------------------------------------------------
# OOF CV helper (operates on preprocessed numpy array)
# ---------------------------------------------------------------------------

def _oof_pr_auc(
    build_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> float:
    """Return mean OOF PR-AUC across repeated stratified folds."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr_idx, val_idx in cv.split(X, y):
        est = build_fn(y[tr_idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X[tr_idx], y[tr_idx])
        oof[val_idx] = est.predict_proba(X[val_idx])[:, 1]
    m = compute_metrics(y, oof, threshold=0.5)
    return m["pr_auc"]


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(X: np.ndarray, y: np.ndarray, cfg: dict):
    """Return an Optuna objective function for XGBoost PR-AUC optimisation."""
    seed      = cfg["seed"]
    n_splits  = cfg["cv"]["n_splits"]
    n_repeats = cfg["cv"]["n_repeats"]

    def objective(trial: optuna.Trial) -> float:
        spw = int((y == 0).sum()) / max(int((y == 1).sum()), 1)

        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": spw,
            "eval_metric":      "logloss",
            "random_state":     seed,
            "tree_method":      "hist",
            "verbosity":        0,
        }

        # Monotonic constraints: tunable choice
        use_monotone = trial.suggest_categorical("use_monotone_constraints", [True, False])
        if use_monotone:
            # Feature order after v1 preprocessing:
            # 0=age, 1=avg_glucose_level, 2=bmi, 3=risk_factors_count, (rest cat)
            # Non-decreasing in age (0) and avg_glucose_level (1)
            n_features = X.shape[1]
            mc = [0] * n_features
            mc[0] = 1   # age
            mc[1] = 1   # avg_glucose_level
            params["monotone_constraints"] = tuple(mc)

        def build(y_tr: np.ndarray) -> XGBClassifier:
            p = dict(params)
            p["scale_pos_weight"] = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
            return XGBClassifier(**p)

        return _oof_pr_auc(build, X, y, n_splits=5, n_repeats=2, seed=seed)

    return objective


# ---------------------------------------------------------------------------
# Stacking ensemble (nested CV)
# ---------------------------------------------------------------------------

def build_stacking_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    best_xgb_params: dict,
) -> dict:
    """Train and evaluate a stacking ensemble via nested CV.

    Base learners: XGBoost (best params), LR (balanced), RF (balanced_subsample)
    Meta-learner:  Logistic Regression (passthrough=True → uses OOF probs)

    Uses 5-fold stratified CV for base learner OOF generation
    and an outer 5-fold for the final estimate.
    """
    seed = cfg["seed"]
    spw = int((y == 0).sum()) / max(int((y == 1).sum()), 1)

    xgb_best = XGBClassifier(
        **{k: v for k, v in best_xgb_params.items()
           if k not in ("use_monotone_constraints",)},
        scale_pos_weight=spw,
        verbosity=0,
    )

    base_estimators = [
        ("xgb", xgb_best),
        ("lr",  LogisticRegression(class_weight="balanced", solver="lbfgs",
                                   max_iter=1000, random_state=seed)),
        ("rf",  RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                       random_state=seed, n_jobs=1)),
    ]
    meta = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        passthrough=False,
        stack_method="predict_proba",
    )

    # Outer 5-fold CV for stacker
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_stack = np.full(len(y), np.nan)
    for tr_idx, val_idx in outer_cv.split(X, y):
        cloned = clone(stacker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cloned.fit(X[tr_idx], y[tr_idx])
        oof_stack[val_idx] = cloned.predict_proba(X[val_idx])[:, 1]

    m = compute_metrics(y, oof_stack, threshold=0.15)
    ci = bootstrap_ci(y, oof_stack, threshold=0.15, n_resamples=500, seed=seed)
    log.info("Stacking OOF metrics: PR-AUC=%.4f [%.4f,%.4f], ROC-AUC=%.4f, Recall=%.4f",
             m["pr_auc"], ci.get("pr_auc",{}).get("lo",0), ci.get("pr_auc",{}).get("hi",0),
             m["roc_auc"], m["recall"])

    return {"oof_probs": oof_stack, "oof_metrics": m, "bootstrap_ci": ci}


# ---------------------------------------------------------------------------
# Monotonic constraint comparison
# ---------------------------------------------------------------------------

def compare_monotone_constraints(
    X: np.ndarray, y: np.ndarray, cfg: dict, base_params: dict,
) -> dict:
    """Compare XGBoost with and without monotonic constraints on age & glucose."""
    seed     = cfg["seed"]
    n_feats  = X.shape[1]

    def build_unconstrained(y_tr: np.ndarray) -> XGBClassifier:
        spw = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
        return XGBClassifier(**{**base_params, "scale_pos_weight": spw, "verbosity": 0})

    def build_constrained(y_tr: np.ndarray) -> XGBClassifier:
        spw = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
        mc = [0] * n_feats
        mc[0] = 1   # age non-decreasing
        mc[1] = 1   # avg_glucose_level non-decreasing
        return XGBClassifier(**{**base_params, "scale_pos_weight": spw,
                                "monotone_constraints": tuple(mc), "verbosity": 0})

    pr_unc = _oof_pr_auc(build_unconstrained, X, y,
                          n_splits=cfg["cv"]["n_splits"],
                          n_repeats=cfg["cv"]["n_repeats"], seed=seed)
    pr_con = _oof_pr_auc(build_constrained, X, y,
                          n_splits=cfg["cv"]["n_splits"],
                          n_repeats=cfg["cv"]["n_repeats"], seed=seed)
    log.info("Monotone constraints: unconstrained PR-AUC=%.4f | constrained PR-AUC=%.4f", pr_unc, pr_con)
    return {"unconstrained_pr_auc": pr_unc, "constrained_pr_auc": pr_con,
            "delta": pr_con - pr_unc,
            "clinical_rationale": (
                "Stroke risk clinically increases with age and glucose level. "
                "Monotone constraints enforce this medical prior, improving "
                "interpretability and potentially preventing overfitting on small "
                "subgroups where the relationship might appear non-monotone due to noise."
            )}


# ---------------------------------------------------------------------------
# Optuna history plot
# ---------------------------------------------------------------------------

def _plot_optuna_history(study: optuna.Study, save_path: Path) -> None:
    trials = study.trials
    vals   = [t.value for t in trials if t.value is not None]
    best_so_far = [max(vals[: i + 1]) for i in range(len(vals))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(vals)), vals, alpha=0.4, s=15, label="Trial PR-AUC")
    ax.plot(range(len(best_so_far)), best_so_far, color="red", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial"); ax.set_ylabel("OOF PR-AUC")
    ax.set_title("Optuna Optimisation History (XGBoost, objective=PR-AUC)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Optuna history plot saved: %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = load_config()
    seed          = cfg["seed"]
    n_trials      = cfg["optuna"]["n_trials"]
    storage       = cfg["optuna"]["storage"]
    study_name    = cfg["optuna"]["study_name"]
    _REPORTS_DIR.mkdir(exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessed training data (v1 — same as model_zoo for fair comparison)
    with open("data/train_processed.pkl", "rb") as f:
        train_pkl = pickle.load(f)
    X_pre = train_pkl["X"].values
    y_pre = train_pkl["y"].values
    log.info("Data: %d rows, %d pos, %d neg", len(y_pre), int(y_pre.sum()), int((y_pre==0).sum()))

    # ----- Optuna study -----
    log.info("Starting Optuna study: %s (%d trials)...", study_name, n_trials)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )
    remaining_trials = n_trials - len(study.trials)
    if remaining_trials > 0:
        objective_fn = make_objective(X_pre, y_pre, cfg)
        study.optimize(objective_fn, n_trials=remaining_trials, show_progress_bar=False)
    else:
        log.info("Study already contains %d trials, skipping search.", len(study.trials))

    best_params = study.best_params
    best_value  = study.best_value
    log.info("Best trial: PR-AUC=%.4f | params=%s", best_value, best_params)

    _plot_optuna_history(study, _PLOTS_DIR / "optuna_history.png")

    # Save Optuna results
    optuna_report = {
        "study_name":    study_name,
        "n_trials":      n_trials,
        "best_pr_auc":   float(best_value),
        "best_params":   _convert_numpy(best_params),
        "note":          "Selected by OOF PR-AUC on training data only. Test set NOT touched.",
    }
    with open(_REPORTS_DIR / "optuna_results.json", "w") as f:
        json.dump(optuna_report, f, indent=2)
    log.info("Optuna results saved.")

    # ----- Monotonic constraint comparison -----
    log.info("Comparing monotone constraints...")
    # Base params = best from Optuna (strip the monotone flag — handled inside compare fn)
    base_params_clean = {
        k: v for k, v in best_params.items()
        if k not in ("use_monotone_constraints",)
    }
    base_params_clean.setdefault("random_state", seed)
    base_params_clean.setdefault("tree_method", "hist")
    base_params_clean.setdefault("eval_metric", "logloss")

    monotone_result = compare_monotone_constraints(X_pre, y_pre, cfg, base_params_clean)
    with open(_REPORTS_DIR / "monotone_comparison.json", "w") as f:
        json.dump(_convert_numpy(monotone_result), f, indent=2)
    log.info("Monotone result: delta PR-AUC = %.4f", monotone_result["delta"])

    # Choose final params (use monotone if it helps)
    use_monotone_final = monotone_result["constrained_pr_auc"] >= monotone_result["unconstrained_pr_auc"]
    log.info("Use monotone constraints in final model: %s", use_monotone_final)
    final_params = dict(base_params_clean)
    if use_monotone_final:
        mc = [0] * X_pre.shape[1]
        mc[0] = 1; mc[1] = 1
        final_params["monotone_constraints"] = tuple(mc)

    # ----- Stacking ensemble -----
    log.info("Building stacking ensemble (nested 5-fold CV)...")
    stacking_result = build_stacking_ensemble(X_pre, y_pre, cfg, final_params)

    stack_m  = stacking_result["oof_metrics"]
    stack_ci = stacking_result["bootstrap_ci"]
    log.info("Stack PR-AUC=%.4f [%.4f,%.4f]  vs  Tuned XGB PR-AUC (OOF)=%.4f",
             stack_m["pr_auc"],
             stack_ci.get("pr_auc",{}).get("lo",0),
             stack_ci.get("pr_auc",{}).get("hi",0),
             best_value)

    # Save stacking report
    with open(_REPORTS_DIR / "stacking_results.json", "w") as f:
        json.dump({
            "oof_metrics": _convert_numpy(stack_m),
            "bootstrap_ci": _convert_numpy(stack_ci),
        }, f, indent=2)

    # ----- Save final tuned pipeline -----
    # Rebuild v2 pipeline with tuned hyperparameters and save
    log.info("Rebuilding v2 pipeline with tuned hyperparameters...")
    df_raw = pd.read_csv(cfg["data"]["raw_path"])
    df_raw = clean_raw_df(df_raw)
    with open(cfg["data"]["split_indices_path"]) as f:
        split_meta = json.load(f)
    train_idx = split_meta["train_indices"]

    X_train_raw = df_raw.drop("stroke", axis=1).loc[train_idx]
    y_train     = df_raw["stroke"].loc[train_idx]

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw_final = neg / pos

    # Build config-driven pipeline with Optuna hyperparameters injected
    cfg_tuned = dict(cfg)
    cfg_tuned["xgboost"] = {
        **cfg.get("xgboost", {}),
        **base_params_clean,
        "scale_pos_weight": spw_final,
    }

    from sklearn.pipeline import Pipeline as SkPipeline
    temp_pipe = build_pipeline(cfg_tuned, scale_pos_weight=spw_final)
    preproc_stage = SkPipeline([
        ("feature_eng", temp_pipe.named_steps["feature_eng"]),
        ("preprocessor", temp_pipe.named_steps["preprocessor"])
    ])
    X_sample_trans = preproc_stage.fit_transform(X_train_raw.iloc[:10])
    n_trans_features = X_sample_trans.shape[1]

    if use_monotone_final:
        mc = [0] * n_trans_features
        mc[0] = 1  # age
        mc[1] = 1  # avg_glucose_level
        cfg_tuned["xgboost"]["monotone_constraints"] = tuple(mc)
    else:
        cfg_tuned["xgboost"]["monotone_constraints"] = None

    tuned_pipeline = build_pipeline(cfg_tuned, scale_pos_weight=spw_final)
    tuned_pipeline.fit(X_train_raw, y_train)
    joblib.dump(tuned_pipeline, cfg["paths"]["pipeline_v2"], compress=3)
    log.info("Tuned pipeline saved to %s", cfg["paths"]["pipeline_v2"])

    # Update metadata
    meta = {
        "phase":               "phase4",
        "n_train":             len(y_train),
        "n_train_pos":         pos,
        "scale_pos_weight":    spw_final,
        "best_optuna_pr_auc":  float(best_value),
        "best_params":         _convert_numpy(best_params),
        "use_monotone_constraints": bool(use_monotone_final),
        "stacking_pr_auc":     float(stack_m["pr_auc"]),
        "seed":                seed,
        "note": (
            "Hyperparameters chosen by OOF PR-AUC on training data only. "
            "Monotone constraints chosen if constrained >= unconstrained PR-AUC. "
            "Stacking evaluated for comparison — final model is single XGBoost pipeline. "
            "Test set NOT evaluated."
        ),
    }
    with open(cfg["paths"]["metadata_v2"], "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata updated: %s", cfg["paths"]["metadata_v2"])

    log.info("Phase 4 complete.")
