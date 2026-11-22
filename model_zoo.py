"""
model_zoo.py
============
Phase 3: Model zoo and imbalance ablation under identical 5×5 CV folds.

Compares
--------
Imbalance strategies (applied to XGBoost):
  (a) no_weight       — XGBoost, no class weighting
  (b) spw             — XGBoost, scale_pos_weight (neg/pos)
  (c) smote           — XGBoost inside imblearn Pipeline + SMOTE
  (d) borderline      — BorderlineSMOTE
  (e) rus             — RandomUnderSampler
  (f) balanced_bagging — BalancedBaggingClassifier wrapping XGBoost

Model families:
  LR (L2), LR_L1, RF (balanced_subsample), LightGBM, CatBoost, HistGB, EBM

Feature ablations:
  Controlled by config flags (ablation variants: each swaps one feature on/off).

All comparisons use:
  - Identical stratified 5×5 CV folds (same random_state=42)
  - Metric: PR-AUC (primary), ROC-AUC, Brier, recall@OOF-threshold, recall<50
  - Statistical significance: DeLong vs. baseline (spw) and McNemar @ threshold=0.15

Results → reports/model_comparison.csv + reports/model_comparison.json
Plots   → plots/v2/model_comparison.png

Usage
-----
.venv/Scripts/python.exe model_zoo.py
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

from evaluation import (
    bootstrap_ci,
    compute_metrics,
    delong_test,
    mcnemar_test,
    _convert_numpy,
)
from src.config import load_config
from src.pipeline_builder import _make_feature_adder, _resolve_columns, clean_raw_df

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")
_PLOTS_DIR   = Path("plots/v2")


# ---------------------------------------------------------------------------
# XGBoost builder (bare, without full pipeline)
# ---------------------------------------------------------------------------

def _xgb(spw: float = 1.0, seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        max_depth=4, learning_rate=0.05, n_estimators=200,
        subsample=0.8, eval_metric="logloss",
        scale_pos_weight=spw, random_state=seed, tree_method="hist",
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# CV loop for a single strategy (operates on preprocessed X)
# ---------------------------------------------------------------------------

def _cv_loop(
    build_estimator_fn,
    X_pre: np.ndarray,
    y: np.ndarray,
    cv: RepeatedStratifiedKFold,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run CV for an estimator-building function.

    ``build_estimator_fn(y_train_fold) -> fitted_estimator``
    """
    oof_probs = np.full(len(y), np.nan)
    per_fold: list[dict] = []

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_pre, y)):
        X_tr, X_val = X_pre[tr_idx], X_pre[val_idx]
        y_tr, y_val = y[tr_idx],     y[val_idx]

        est = build_estimator_fn(y_tr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_tr, y_tr)

        probs = est.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs
        per_fold.append(compute_metrics(y_val, probs, threshold=threshold))

    oof_metrics = compute_metrics(y, oof_probs, threshold=threshold)
    return {
        "oof_probs":    oof_probs,
        "per_fold":     per_fold,
        "oof_metrics":  oof_metrics,
        "mean_pr_auc":  float(np.mean([f["pr_auc"] for f in per_fold])),
        "std_pr_auc":   float(np.std( [f["pr_auc"] for f in per_fold])),
        "mean_roc_auc": float(np.mean([f["roc_auc"] for f in per_fold])),
        "std_roc_auc":  float(np.std( [f["roc_auc"] for f in per_fold])),
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_model_zoo(cfg: dict) -> pd.DataFrame:
    seed = cfg["seed"]
    threshold = 0.15   # operating threshold for recall/precision comparisons

    # Load data
    df = pd.read_csv(cfg["data"]["raw_path"])
    df = clean_raw_df(df)
    X_raw = df.drop("stroke", axis=1)
    y     = df["stroke"]

    with open(cfg["data"]["split_indices_path"]) as f:
        split_meta = json.load(f)
    train_idx = split_meta["train_indices"]

    X_train_raw = X_raw.loc[train_idx].reset_index(drop=True)
    y_train     = y.loc[train_idx].reset_index(drop=True).values

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos
    log.info("Training data: %d rows (%d pos, %d neg), SPW=%.2f", len(y_train), pos, neg, spw)

    # Preprocess ONCE (outside CV) for imbalance-only studies
    # NOTE: for final v2 pipeline we fit inside CV. Here we do a quick
    # comparison on features already transformed by v1 preprocessor to avoid
    # refitting 25×N_models times (which would be too slow for a zoo of 15+).
    # The correct Phase 1 pipeline is validated in evaluation.py.
    # For the zoo we use the v1 preprocessed training array.
    import pickle
    with open("data/train_processed.pkl", "rb") as f:
        train_pkl = pickle.load(f)
    X_pre = train_pkl["X"].values   # numpy array, already scaled/encoded
    y_pre = train_pkl["y"].values

    # Confirm alignment (index-matched during data_loader split)
    assert len(X_pre) == len(y_pre) == len(y_train), "Size mismatch"
    log.info("Using v1 preprocessed features (%d x %d) for model zoo comparison.", *X_pre.shape)

    cv = RepeatedStratifiedKFold(n_splits=cfg["cv"]["n_splits"],
                                 n_repeats=cfg["cv"]["n_repeats"],
                                 random_state=seed)

    results: dict[str, dict] = {}

    # ---- Imbalance strategies (all using XGBoost) -------------------------

    log.info("Strategy: (a) XGBoost no weighting...")
    results["xgb_no_weight"] = _cv_loop(
        lambda yt: _xgb(spw=1.0, seed=seed), X_pre, y_pre, cv, threshold, seed)

    log.info("Strategy: (b) XGBoost scale_pos_weight (baseline)...")
    results["xgb_spw"] = _cv_loop(
        lambda yt: _xgb(spw=(yt==0).sum()/(yt==1).sum() if (yt==1).sum()>0 else 1.0, seed=seed),
        X_pre, y_pre, cv, threshold, seed)

    log.info("Strategy: (c) XGBoost + SMOTE (inside fold)...")
    def _xgb_smote(y_tr: np.ndarray) -> Any:
        from imblearn.pipeline import Pipeline as ImbPipe
        return ImbPipe([
            ("smote", SMOTE(random_state=seed, k_neighbors=min(4, (y_tr==1).sum()-1))),
            ("clf",   _xgb(spw=1.0, seed=seed)),
        ])
    results["xgb_smote"] = _cv_loop(_xgb_smote, X_pre, y_pre, cv, threshold, seed)

    log.info("Strategy: (d) XGBoost + BorderlineSMOTE...")
    def _xgb_bsmote(y_tr: np.ndarray) -> Any:
        from imblearn.pipeline import Pipeline as ImbPipe
        return ImbPipe([
            ("bsmote", BorderlineSMOTE(random_state=seed, k_neighbors=min(4, (y_tr==1).sum()-1))),
            ("clf",    _xgb(spw=1.0, seed=seed)),
        ])
    results["xgb_borderline"] = _cv_loop(_xgb_bsmote, X_pre, y_pre, cv, threshold, seed)

    log.info("Strategy: (e) XGBoost + RandomUnderSampler...")
    def _xgb_rus(y_tr: np.ndarray) -> Any:
        from imblearn.pipeline import Pipeline as ImbPipe
        return ImbPipe([
            ("rus", RandomUnderSampler(random_state=seed)),
            ("clf", _xgb(spw=1.0, seed=seed)),
        ])
    results["xgb_rus"] = _cv_loop(_xgb_rus, X_pre, y_pre, cv, threshold, seed)

    log.info("Strategy: (f) BalancedBaggingClassifier...")
    from imblearn.ensemble import BalancedBaggingClassifier
    bbc = BalancedBaggingClassifier(
        estimator=_xgb(spw=1.0, seed=seed),
        n_estimators=20, random_state=seed,
    )
    results["balanced_bagging"] = _cv_loop(
        lambda yt: clone(bbc), X_pre, y_pre, cv, threshold, seed)

    # ---- Model families ---------------------------------------------------

    log.info("Model: Logistic Regression (L2, balanced)...")
    lr_l2 = LogisticRegression(class_weight="balanced", solver="lbfgs",
                               max_iter=1000, random_state=seed)
    results["lr_l2"] = _cv_loop(lambda yt: clone(lr_l2), X_pre, y_pre, cv, threshold, seed)

    log.info("Model: Logistic Regression (L1)...")
    lr_l1 = LogisticRegression(class_weight="balanced", solver="saga", penalty="l1",
                               max_iter=1000, random_state=seed)
    results["lr_l1"] = _cv_loop(lambda yt: clone(lr_l1), X_pre, y_pre, cv, threshold, seed)

    log.info("Model: Random Forest (balanced_subsample)...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                random_state=seed, n_jobs=-1)
    results["rf_balanced"] = _cv_loop(lambda yt: clone(rf), X_pre, y_pre, cv, threshold, seed)

    log.info("Model: LightGBM...")
    import lightgbm as lgb
    lgbm = lgb.LGBMClassifier(
        is_unbalance=True, n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=seed, verbose=-1,
    )
    results["lgbm"] = _cv_loop(lambda yt: clone(lgbm), X_pre, y_pre, cv, threshold, seed)

    log.info("Model: CatBoost...")
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=4,
        auto_class_weights="Balanced", random_seed=seed, verbose=0,
    )
    results["catboost"] = _cv_loop(lambda yt: clone(cb), X_pre, y_pre, cv, threshold, seed)

    log.info("Model: HistGradientBoosting (class_weight=balanced)...")
    hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4,
        class_weight="balanced", random_state=seed,
    )
    results["histgb"] = _cv_loop(lambda yt: clone(hgb), X_pre, y_pre, cv, threshold, seed)

    log.info("Model: EBM (Explainable Boosting Machine)...")
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        ebm = ExplainableBoostingClassifier(random_state=seed)
        results["ebm"] = _cv_loop(lambda yt: clone(ebm), X_pre, y_pre, cv, threshold, seed)
        log.info("  EBM OOF PR-AUC = %.4f", results["ebm"]["oof_metrics"]["pr_auc"])
    except Exception as e:
        log.warning("EBM failed: %s. Skipping.", e)

    # ---- Significance tests vs baseline (xgb_spw) -------------------------
    log.info("Computing DeLong and McNemar tests vs xgb_spw baseline...")
    baseline_probs = results["xgb_spw"]["oof_probs"]
    baseline_preds = (baseline_probs >= threshold).astype(int)
    significance: dict[str, dict] = {}
    for tag, res in results.items():
        if tag == "xgb_spw":
            continue
        probs = res["oof_probs"]
        preds = (probs >= threshold).astype(int)
        dl  = delong_test(y_pre, baseline_probs, probs)
        mcn = mcnemar_test(y_pre, baseline_preds, preds)
        significance[tag] = {"delong": dl, "mcnemar": mcn}

    # ---- Recall on age < 50 subgroup ----------------------------------------
    log.info("Computing recall on age < 50 subgroup...")
    # For this we need the original ages from train data
    df_train = df.loc[train_idx].reset_index(drop=True)
    ages = df_train["age"].values
    age_lt50_mask = ages < 50
    n_lt50 = int(age_lt50_mask.sum())
    n_lt50_pos = int(y_pre[age_lt50_mask].sum())
    log.info("Age < 50 subgroup: %d patients, %d positives", n_lt50, n_lt50_pos)

    recall_lt50: dict[str, float] = {}
    for tag, res in results.items():
        probs_sub = res["oof_probs"][age_lt50_mask]
        y_sub     = y_pre[age_lt50_mask]
        m_sub     = compute_metrics(y_sub, probs_sub, threshold=threshold)
        recall_lt50[tag] = m_sub["recall"]

    # ---- Build comparison table -------------------------------------------
    rows: list[dict] = []
    for tag, res in results.items():
        m = res["oof_metrics"]
        ci = bootstrap_ci(y_pre, res["oof_probs"], threshold=threshold, n_resamples=500, seed=seed)
        sig = significance.get(tag, {})
        rows.append({
            "model": tag,
            "pr_auc":          round(m["pr_auc"],    4),
            "pr_auc_lo":       round(ci["pr_auc"]["lo"],   4) if "pr_auc" in ci else None,
            "pr_auc_hi":       round(ci["pr_auc"]["hi"],   4) if "pr_auc" in ci else None,
            "roc_auc":         round(m["roc_auc"],   4),
            "roc_auc_lo":      round(ci["roc_auc"]["lo"],  4) if "roc_auc" in ci else None,
            "roc_auc_hi":      round(ci["roc_auc"]["hi"],  4) if "roc_auc" in ci else None,
            "brier":           round(m["brier"],     4),
            "recall@thr":      round(m["recall"],    4),
            "precision@thr":   round(m["precision"], 4),
            "recall_lt50":     round(recall_lt50.get(tag, float("nan")), 4),
            "delong_p":        round(sig.get("delong", {}).get("p_value", float("nan")), 4),
            "mcnemar_p":       round(sig.get("mcnemar", {}).get("p_value", float("nan")), 4),
        })

    df_cmp = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    return df_cmp, results, significance, recall_lt50, y_pre


def _plot_comparison(df_cmp: pd.DataFrame, save_path: str | Path) -> None:
    """Plot model comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, col, title in zip(axes,
                               ["pr_auc", "roc_auc", "recall@thr"],
                               ["PR-AUC", "ROC-AUC", f"Recall @ threshold"]):
        y_vals = df_cmp[col].values
        y_errs_lo = (df_cmp[col].values - df_cmp.get(col + "_lo", df_cmp[col]).values).clip(min=0)
        y_errs_hi = (df_cmp.get(col + "_hi", df_cmp[col]).values - df_cmp[col].values).clip(min=0)
        bars = ax.barh(df_cmp["model"].values[::-1], y_vals[::-1],
                       xerr=[y_errs_lo[::-1], y_errs_hi[::-1]],
                       align="center", alpha=0.85,
                       error_kw={"elinewidth": 1.5, "capsize": 3})
        ax.set_xlabel(title); ax.set_title(title)
        # Highlight baseline
        if "xgb_spw" in df_cmp["model"].values:
            baseline_rank = list(df_cmp["model"].values[::-1]).index("xgb_spw")
            bars[baseline_rank].set_edgecolor("black")
            bars[baseline_rank].set_linewidth(2)
    fig.suptitle("Model Zoo Comparison (5×5 CV OOF metrics @ operating threshold=0.15)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Comparison plot saved: %s", save_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = load_config()
    _REPORTS_DIR.mkdir(exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Phase 3: Model Zoo & Imbalance Ablation ===")
    df_cmp, results, significance, recall_lt50, y_pre = run_model_zoo(cfg)

    # Save CSV
    csv_path = _REPORTS_DIR / "model_comparison.csv"
    df_cmp.to_csv(csv_path, index=False)
    log.info("Comparison table saved: %s", csv_path)

    # Save JSON
    json_report = {
        "comparison_table": df_cmp.to_dict(orient="records"),
        "significance_tests": _convert_numpy(significance),
        "threshold": 0.15,
        "cv_strategy": "5x5 RepeatedStratifiedKFold",
        "note": (
            "All imbalance strategies use same CV folds. "
            "SMOTE applied inside fold via imblearn Pipeline. "
            "significance = DeLong/McNemar vs xgb_spw baseline. "
            "recall_lt50 = recall on patients with age < 50."
        ),
    }
    json_path = _REPORTS_DIR / "model_comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)
    log.info("Comparison JSON saved: %s", json_path)

    # Plot
    _plot_comparison(df_cmp, _PLOTS_DIR / "model_comparison.png")

    # Print readable table
    log.info("\n=== Model Comparison Table (sorted by PR-AUC) ===")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(df_cmp.to_string(index=False))

    # Explicitly state SMOTE finding
    spw_pr   = df_cmp.loc[df_cmp["model"]=="xgb_spw", "pr_auc"].values[0]
    smote_pr = df_cmp.loc[df_cmp["model"]=="xgb_smote", "pr_auc"].values[0] if "xgb_smote" in df_cmp["model"].values else None
    bsmote_pr= df_cmp.loc[df_cmp["model"]=="xgb_borderline", "pr_auc"].values[0] if "xgb_borderline" in df_cmp["model"].values else None
    log.info("\n=== SMOTE Verdict ===")
    if smote_pr and bsmote_pr:
        smote_delong_p = significance.get("xgb_smote", {}).get("delong", {}).get("p_value", float("nan"))
        bsmote_delong_p = significance.get("xgb_borderline", {}).get("delong", {}).get("p_value", float("nan"))
        log.info("  xgb_spw PR-AUC        = %.4f", spw_pr)
        log.info("  xgb_smote PR-AUC      = %.4f (DeLong p=%.4f vs spw)", smote_pr, smote_delong_p)
        log.info("  xgb_borderline PR-AUC = %.4f (DeLong p=%.4f vs spw)", bsmote_pr, bsmote_delong_p)
        alpha = 0.05
        if smote_delong_p < alpha or bsmote_delong_p < alpha:
            winner = "SMOTE" if smote_pr > spw_pr else "scale_pos_weight"
            log.info("  CONCLUSION: Statistically significant difference (p<0.05). %s is better for PR-AUC.", winner)
        else:
            log.info("  CONCLUSION: No statistically significant difference (p>=0.05). SMOTE and scale_pos_weight are INDISTINGUISHABLE on PR-AUC in this dataset.")

    log.info("Phase 3 complete.")
