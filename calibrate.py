"""
calibrate.py
============
Phase 5: Calibration and cost-based threshold selection.

Steps
-----
1. Load the tuned v2 pipeline from Phase 4.
2. Run 5×5 CV to get OOF calibrated probabilities using:
   - Isotonic regression calibration
   - Platt (sigmoid) scaling
   Compare Brier score, log loss, ECE, and reliability curves before/after.
3. Choose calibration method by OOF Brier score.
4. Threshold selection from OOF calibrated probabilities using:
   - Cost model: minimize FN_cost*FN + FP_cost*FP (sensitivity for ratios 5,10,20,50)
   - F2-optimal threshold
   - Min-recall threshold (target recall >= 0.80)
5. Output table of thresholds × (recall, precision, NNS, flagged/1000).
6. Decision curve analysis (DCA) with the chosen threshold marked.
7. Save chosen threshold + provenance to models/v2/metadata.json.

The test set is NEVER touched in this module.

Usage
-----
.venv/Scripts/python.exe calibrate.py
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from evaluation import bootstrap_ci, compute_metrics, compute_dca, plot_dca, _convert_numpy
from src.config import load_config
from src.pipeline_builder import build_pipeline, clean_raw_df

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")
_PLOTS_DIR   = Path("plots/v2")


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE via equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# OOF calibration
# ---------------------------------------------------------------------------

def oof_calibrate(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: dict,
    method: str = "isotonic",
) -> tuple[np.ndarray, np.ndarray]:
    """Return OOF raw probs and OOF calibrated probs via nested CV.

    For each outer fold:
      - Fit raw pipeline on tr_idx, predict on val_idx -> raw_oof
      - Fit CalibratedClassifierCV(pipeline, method=method, cv=3) on tr_idx,
        predict on val_idx -> cal_oof
    Returns (raw_oof, cal_oof) arrays of shape (n_train,).
    """
    seed     = cfg["seed"]
    n_splits = cfg["cv"]["n_splits"]
    n_repeats = 1  # 5 folds is sufficient and fast for calibration comparison

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X_reset = X.reset_index(drop=True)
    raw_oof = np.full(len(y), np.nan)
    cal_oof = np.full(len(y), np.nan)

    for fold_i, (tr_idx, val_idx) in enumerate(outer_cv.split(X_reset, y)):
        X_tr, X_val = X_reset.iloc[tr_idx], X_reset.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # 1. Raw pipeline
        pipe = clone(pipeline)
        pipe.fit(X_tr, y_tr)
        raw_oof[val_idx] = pipe.predict_proba(X_val)[:, 1]

        # 2. Calibrated pipeline (uses internal 3-fold CV on X_tr, y_tr)
        cal_pipe = CalibratedClassifierCV(estimator=clone(pipeline), method=method, cv=3)
        cal_pipe.fit(X_tr, y_tr)
        cal_oof[val_idx] = cal_pipe.predict_proba(X_val)[:, 1]

        log.info("  Calibration CV fold %d/%d done (%s)", fold_i + 1, n_splits, method)

    return raw_oof, cal_oof


# ---------------------------------------------------------------------------
# Reliability plot
# ---------------------------------------------------------------------------

def plot_reliability(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    method: str,
    save_path: Path,
) -> None:
    """Plot calibration reliability curves before and after calibration."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for probs, label, ls in [
        (raw_probs, "Before calibration", "--"),
        (cal_probs, f"After ({method})", "-"),
    ]:
        try:
            frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
            ax.plot(mean_pred, frac_pos, marker="o", linestyle=ls, label=label)
        except Exception as e:
            log.warning("calibration_curve failed for %s: %s", label, e)

    ax.plot([0, 1], [0, 1], "k:", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability Curve (OOF, {method} calibration)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Reliability plot saved: %s", save_path)


# ---------------------------------------------------------------------------
# Cost-based threshold selection
# ---------------------------------------------------------------------------

def select_threshold_by_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_fp_ratio: float,
    candidate_thresholds: np.ndarray | None = None,
) -> float:
    """Select threshold minimising FN_cost*FN + FP_cost*FP.

    FN_cost = fn_fp_ratio, FP_cost = 1 (normalised).
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.01, 0.99, 500)

    best_thr, best_cost = 0.5, float("inf")
    for thr in candidate_thresholds:
        y_pred = (y_prob >= thr).astype(int)
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        cost = fn_fp_ratio * fn + 1.0 * fp
        if cost < best_cost:
            best_cost, best_thr = cost, thr
    return float(best_thr)


def select_threshold_min_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.80,
    candidate_thresholds: np.ndarray | None = None,
) -> float:
    """Select highest threshold (best precision) achieving >= min_recall."""
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.01, 0.99, 500)
    best_thr = candidate_thresholds[0]
    best_prec = 0.0
    for thr in candidate_thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if recall >= min_recall and prec > best_prec:
            best_prec, best_thr = prec, thr
    return float(best_thr)


def select_threshold_f2(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidate_thresholds: np.ndarray | None = None,
) -> float:
    """Select threshold maximising F2 score."""
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.01, 0.99, 500)
    best_thr, best_f2 = 0.5, 0.0
    for thr in candidate_thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f2 = 5 * prec * recall / (4 * prec + recall) if (4 * prec + recall) > 0 else 0.0
        if f2 > best_f2:
            best_f2, best_thr = f2, thr
    return float(best_thr)


def build_threshold_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float],
    n_per_1000: int = 1000,
) -> pd.DataFrame:
    """Build table of metrics for each named threshold."""
    rows = []
    for rule, thr in thresholds.items():
        m = compute_metrics(y_true, y_prob, threshold=thr)
        rows.append({
            "threshold_rule":  rule,
            "threshold_value": round(thr, 4),
            "recall":          round(m["recall"], 4),
            "precision":       round(m["precision"], 4),
            "nns":             round(m["nns"], 1) if m["nns"] != float("inf") else "∞",
            "flagged_per_1000": round(m["flag_rate"] * n_per_1000, 1),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = load_config()
    seed = cfg["seed"]
    _REPORTS_DIR.mkdir(exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load tuned pipeline
    pipeline = joblib.load(cfg["paths"]["pipeline_v2"])
    log.info("Loaded v2 pipeline: %s", cfg["paths"]["pipeline_v2"])

    # Load training data
    df_raw = pd.read_csv(cfg["data"]["raw_path"])
    df_raw = clean_raw_df(df_raw)
    with open(cfg["data"]["split_indices_path"]) as f:
        split_meta = json.load(f)
    train_idx = split_meta["train_indices"]

    X_train = df_raw.drop("stroke", axis=1).loc[train_idx]
    y_train = df_raw["stroke"].loc[train_idx].values
    log.info("Training set: %d rows (%d pos)", len(y_train), int(y_train.sum()))

    # ----- OOF calibration: isotonic vs sigmoid -----
    log.info("Computing OOF calibration — isotonic...")
    raw_oof, iso_oof = oof_calibrate(pipeline, X_train, y_train, cfg, method="isotonic")
    log.info("Computing OOF calibration — sigmoid...")
    _,       sig_oof = oof_calibrate(pipeline, X_train, y_train, cfg, method="sigmoid")

    # Compare calibration quality
    brier_raw = brier_score_loss(y_train, raw_oof)
    brier_iso = brier_score_loss(y_train, iso_oof)
    brier_sig = brier_score_loss(y_train, sig_oof)
    ll_raw    = log_loss(y_train, raw_oof)
    ll_iso    = log_loss(y_train, iso_oof)
    ll_sig    = log_loss(y_train, sig_oof)
    ece_raw   = expected_calibration_error(y_train, raw_oof)
    ece_iso   = expected_calibration_error(y_train, iso_oof)
    ece_sig   = expected_calibration_error(y_train, sig_oof)

    log.info("Calibration comparison (OOF):")
    log.info("  Brier:  raw=%.4f | isotonic=%.4f | sigmoid=%.4f", brier_raw, brier_iso, brier_sig)
    log.info("  LogLoss:raw=%.4f | isotonic=%.4f | sigmoid=%.4f", ll_raw,    ll_iso,    ll_sig)
    log.info("  ECE:    raw=%.4f | isotonic=%.4f | sigmoid=%.4f", ece_raw,   ece_iso,   ece_sig)

    # Choose best calibration by Brier
    if brier_iso <= brier_sig:
        chosen_method = "isotonic"
        cal_oof = iso_oof
    else:
        chosen_method = "sigmoid"
        cal_oof = sig_oof

    log.info("Chosen calibration: %s (lower OOF Brier)", chosen_method)

    # Reliability plots
    plot_reliability(y_train, raw_oof, iso_oof, "isotonic",
                     _PLOTS_DIR / "reliability_isotonic.png")
    plot_reliability(y_train, raw_oof, sig_oof, "sigmoid",
                     _PLOTS_DIR / "reliability_sigmoid.png")

    # ----- Threshold selection from OOF calibrated probs -----
    candidates = np.linspace(0.01, 0.99, 1000)
    cost_ratios = cfg["cost"]["sensitivity_ratios"]
    min_recall  = cfg["cost"]["min_recall_target"]

    thresholds: dict[str, float] = {}

    # Cost-based thresholds
    for ratio in cost_ratios:
        thr = select_threshold_by_cost(y_train, cal_oof, fn_fp_ratio=float(ratio), candidate_thresholds=candidates)
        thresholds[f"cost_ratio_{ratio}"] = thr
        log.info("Cost ratio %d:  threshold=%.4f", ratio, thr)

    # F2-optimal
    thresholds["f2_optimal"] = select_threshold_f2(y_train, cal_oof, candidates)
    log.info("F2-optimal threshold: %.4f", thresholds["f2_optimal"])

    # Min-recall target
    thresholds[f"min_recall_{int(min_recall*100)}pct"] = select_threshold_min_recall(
        y_train, cal_oof, min_recall=min_recall, candidate_thresholds=candidates
    )
    log.info("Min-recall (>=%.0f%%) threshold: %.4f", min_recall*100,
             thresholds[f"min_recall_{int(min_recall*100)}pct"])

    # Legacy v1 threshold for comparison
    thresholds["legacy_v1"] = 0.15

    # Build threshold table
    thr_table = build_threshold_table(y_train, cal_oof, thresholds)
    log.info("\n=== Threshold Sensitivity Table ===\n%s", thr_table.to_string(index=False))

    thr_table.to_csv(_REPORTS_DIR / "threshold_sensitivity.csv", index=False)
    with open(_REPORTS_DIR / "threshold_sensitivity.json", "w") as f:
        json.dump(_convert_numpy(thr_table.to_dict(orient="records")), f, indent=2)

    # ----- DCA -----
    log.info("Computing DCA...")
    dca = compute_dca(y_train, cal_oof)
    chosen_thr = thresholds[f"cost_ratio_{cfg['cost']['fn_fp_ratio']}"]
    plot_dca(dca, threshold_line=chosen_thr,
             save_path=_PLOTS_DIR / "dca_calibrated.png")

    # ----- Calibration report -----
    cal_report = {
        "chosen_calibration_method": chosen_method,
        "calibration_metrics": {
            "brier_raw":  float(brier_raw),
            "brier_iso":  float(brier_iso),
            "brier_sig":  float(brier_sig),
            "logloss_raw": float(ll_raw),
            "logloss_iso": float(ll_iso),
            "logloss_sig": float(ll_sig),
            "ece_raw":    float(ece_raw),
            "ece_iso":    float(ece_iso),
            "ece_sig":    float(ece_sig),
        },
        "threshold_selection": {k: float(v) for k, v in thresholds.items()},
        "chosen_threshold": float(chosen_thr),
        "chosen_threshold_rule": f"cost_ratio_{cfg['cost']['fn_fp_ratio']}",
        "note": (
            "Threshold and calibration method selected on OOF training-set probabilities only. "
            "Test set NOT touched. Chosen threshold provenance: cost model FN/FP ratio "
            f"= {cfg['cost']['fn_fp_ratio']} applied to OOF calibrated probabilities."
        ),
    }
    with open(_REPORTS_DIR / "calibration_report.json", "w") as f:
        json.dump(cal_report, f, indent=2)
    log.info("Calibration report saved.")

    # ----- Update v2 metadata with threshold + provenance -----
    meta_path = cfg["paths"]["metadata_v2"]
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}

    meta.update({
        "phase":                    "phase5",
        "calibration_method":       chosen_method,
        "chosen_threshold":         float(chosen_thr),
        "chosen_threshold_rule":    f"cost_ratio_{cfg['cost']['fn_fp_ratio']}",
        "threshold_provenance":     (
            "Selected via cost model (FN/FP ratio = "
            f"{cfg['cost']['fn_fp_ratio']}) on OOF calibrated training-set probabilities. "
            "Test set was NOT used for threshold selection."
        ),
        "brier_before_calibration": float(brier_raw),
        "brier_after_calibration":  float(brier_iso if chosen_method == "isotonic" else brier_sig),
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("v2 metadata updated with chosen threshold: %.4f", chosen_thr)

    # Fit final calibrated pipeline on all training data and save
    log.info("Fitting final CalibratedClassifierCV on full training set...")
    final_cal_pipe = CalibratedClassifierCV(estimator=clone(pipeline), method=chosen_method, cv=5)
    final_cal_pipe.fit(X_train, y_train)
    cal_model_path = "models/v2/calibrated_pipeline.joblib"
    joblib.dump(final_cal_pipe, cal_model_path, compress=3)
    log.info("Saved final calibrated pipeline to %s", cal_model_path)

    # ----- Bootstrap CIs on calibrated OOF at chosen threshold -----
    log.info("Bootstrap CIs on calibrated OOF probabilities at chosen threshold...")
    ci = bootstrap_ci(y_train, cal_oof, threshold=chosen_thr,
                      n_resamples=cfg["bootstrap"]["n_resamples"], seed=seed)
    m  = compute_metrics(y_train, cal_oof, threshold=chosen_thr)
    log.info("=== Final Calibrated OOF Metrics @ threshold=%.4f ===", chosen_thr)
    for key in ["recall", "precision", "roc_auc", "pr_auc", "brier", "f2"]:
        if key in ci:
            log.info("  %-15s = %.4f [%.4f, %.4f]",
                     key, ci[key]["mean"], ci[key]["lo"], ci[key]["hi"])

    log.info("Phase 5 complete.")
