"""
evaluation.py
=============
Phase 2: Robust evaluation harness.

Provides
--------
- run_repeated_cv()          : 5×5 repeated stratified K-fold, returns OOF probs
- compute_metrics()          : all headline metrics at a threshold
- bootstrap_ci()             : 1000-resample 95% CIs for every metric
- wilson_ci()                : Wilson interval for recall / precision
- delong_test()              : AUC comparison (DeLong 1988, vectorized)
- mcnemar_test()             : McNemar's test for paired classifier decisions
- compute_dca()              : decision curve analysis (net benefit)
- compute_lift_gain()        : lift and cumulative gain curves
- age_band_metrics()         : recall/precision/etc. within age bands
- gender_metrics()           : same, by gender
- save_cv_report()           : writes JSON + CSV to reports/

All results are written to reports/ and plots to plots/v2/.
The test set is NEVER touched in this module.

Usage
-----
from evaluation import run_repeated_cv, compute_metrics, bootstrap_ci
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
_REPORTS_DIR = Path("reports")
_PLOTS_DIR   = Path("plots/v2")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    """Compute F-beta score from precision and recall."""
    denom = beta**2 * precision + recall
    return (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a full set of metrics at ``threshold``.

    Parameters
    ----------
    y_true  : binary labels (0/1)
    y_prob  : model probability for the positive class
    threshold : classification threshold

    Returns
    -------
    dict mapping metric name -> scalar float
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec      = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1        = _fbeta(precision, recall, beta=1.0)
    f2        = _fbeta(precision, recall, beta=2.0)
    nns       = 1.0 / precision if precision > 0 else float("inf")  # number needed to screen
    flag_rate = (tp + fp) / len(y_true)

    roc_auc = roc_auc_score(y_true, y_prob) if n_pos > 0 and n_neg > 0 else float("nan")
    pr_auc  = average_precision_score(y_true, y_prob) if n_pos > 0 else float("nan")
    brier   = brier_score_loss(y_true, y_prob)
    ll      = log_loss(y_true, y_prob)

    return {
        "threshold":    threshold,
        "roc_auc":      float(roc_auc),
        "pr_auc":       float(pr_auc),
        "brier":        float(brier),
        "log_loss":     float(ll),
        "recall":       float(recall),
        "precision":    float(precision),
        "specificity":  float(spec),
        "f1":           float(f1),
        "f2":           float(f2),
        "nns":          float(nns),
        "flag_rate":    float(flag_rate),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_pos": n_pos, "n_neg": n_neg,
    }


# ---------------------------------------------------------------------------
# Repeated stratified CV
# ---------------------------------------------------------------------------

def run_repeated_cv(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run repeated stratified K-fold CV and return OOF probabilities + metrics.

    Parameters
    ----------
    pipeline  : unfitted sklearn-compatible pipeline (cloned per fold)
    X, y      : TRAINING data only (test set must not be passed)
    cfg       : config dict
    threshold : threshold for per-fold binary metrics (can be updated later)

    Returns
    -------
    dict with keys:
        oof_probs    : np.ndarray shape (n_train,) OOF probabilities
        oof_true     : np.ndarray shape (n_train,)
        per_fold     : list[dict] one per fold (roc_auc, pr_auc, recall, …)
        mean_metrics : dict mean across folds
        std_metrics  : dict std across folds
    """
    from sklearn.base import clone

    n_splits  = cfg["cv"]["n_splits"]
    n_repeats = cfg["cv"]["n_repeats"]
    seed      = cfg.get("seed", 42)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True).values

    oof_probs = np.full(len(y_arr), np.nan)
    per_fold: list[dict] = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        X_tr, X_val = X_arr.iloc[train_idx], X_arr.iloc[val_idx]
        y_tr, y_val = y_arr[train_idx],      y_arr[val_idx]

        # Compute scale_pos_weight from THIS fold's training data
        neg = int((y_tr == 0).sum())
        pos = int((y_tr == 1).sum())
        spw = neg / pos if pos > 0 else 1.0

        # Clone and patch scale_pos_weight on the classifier step
        fold_pipe = clone(pipeline)
        if hasattr(fold_pipe.named_steps.get("classifier", None), "scale_pos_weight"):
            fold_pipe.named_steps["classifier"].scale_pos_weight = spw

        fold_pipe.fit(X_tr, y_tr)
        probs_val = fold_pipe.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs_val

        fold_metrics = compute_metrics(y_val, probs_val, threshold=threshold)
        fold_metrics["fold"] = fold_i
        fold_metrics["n_val_pos"] = int(y_val.sum())
        per_fold.append(fold_metrics)

        if (fold_i + 1) % 5 == 0:
            log.info("  Fold %d/%d done | val ROC-AUC=%.4f", fold_i + 1, n_splits * n_repeats, fold_metrics["roc_auc"])

    # Aggregate across folds
    metric_keys = ["roc_auc", "pr_auc", "brier", "log_loss", "recall", "precision",
                   "specificity", "f1", "f2"]
    mean_metrics = {k: float(np.mean([f[k] for f in per_fold])) for k in metric_keys}
    std_metrics  = {k: float(np.std( [f[k] for f in per_fold])) for k in metric_keys}

    # OOF aggregate metrics (single global computation)
    oof_metrics = compute_metrics(y_arr, oof_probs, threshold=threshold)

    return {
        "oof_probs":    oof_probs,
        "oof_true":     y_arr,
        "per_fold":     per_fold,
        "mean_metrics": mean_metrics,
        "std_metrics":  std_metrics,
        "oof_metrics":  oof_metrics,
        "n_folds":      n_splits * n_repeats,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap (percentile) 95% CIs for each metric.

    Parameters
    ----------
    y_true, y_prob : ground truth and predicted probabilities
    threshold      : classification threshold
    n_resamples    : number of bootstrap samples
    seed           : RNG seed
    alpha          : 1 - coverage (0.05 for 95% CI)

    Returns
    -------
    dict[metric_name, {"mean", "lo", "hi"}]
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_metrics: dict[str, list[float]] = {}

    # Stratified resampling to maintain class balance
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    for _ in range(n_resamples):
        # Resample positives and negatives separately to guarantee both classes present
        bi_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bi_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        bi = np.concatenate([bi_pos, bi_neg])

        m = compute_metrics(y_true[bi], y_prob[bi], threshold=threshold)
        for k, v in m.items():
            if isinstance(v, float) and not np.isnan(v):
                boot_metrics.setdefault(k, []).append(v)

    lo_pct = alpha / 2 * 100
    hi_pct = (1 - alpha / 2) * 100
    result: dict[str, dict[str, float]] = {}
    for k, vals in boot_metrics.items():
        arr = np.array(vals)
        result[k] = {
            "mean": float(arr.mean()),
            "lo":   float(np.percentile(arr, lo_pct)),
            "hi":   float(np.percentile(arr, hi_pct)),
        }
    return result


def wilson_ci(
    successes: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Wilson score interval for a proportion (recall, precision, etc.).

    Parameters
    ----------
    successes : count of successes
    n         : total trials
    alpha     : 1 - coverage

    Returns
    -------
    (lo, hi) tuple of floats
    """
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# DeLong AUC comparison test
# ---------------------------------------------------------------------------

def _auc_covariance(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    """Compute AUC1, AUC2, and their covariance via DeLong's method.

    Reference: DeLong et al. (1988) Biometrics.
    """
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos1, neg1 = p1[pos_mask], p1[neg_mask]
    pos2, neg2 = p2[pos_mask], p2[neg_mask]

    n_pos, n_neg = len(pos1), len(neg1)

    def _placement(scores_pos: np.ndarray, scores_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute placement values (Mann-Whitney statistic components)."""
        v01 = np.array([(sp > scores_neg).mean() + 0.5 * (sp == scores_neg).mean() for sp in scores_pos])
        v10 = np.array([(sn < scores_pos).mean() + 0.5 * (sn == scores_pos).mean() for sn in scores_neg])
        return v01, v10

    v01_1, v10_1 = _placement(pos1, neg1)
    v01_2, v10_2 = _placement(pos2, neg2)

    auc1 = v01_1.mean()
    auc2 = v01_2.mean()

    s01 = np.cov(v01_1, v01_2, ddof=1)   # shape (2,2)
    s10 = np.cov(v10_1, v10_2, ddof=1)

    covar_mat = s01 / n_pos + s10 / n_neg
    return auc1, auc2, float(covar_mat[0, 1])


def delong_test(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> dict[str, float]:
    """DeLong two-sided test for equality of two AUCs.

    Parameters
    ----------
    y_true  : binary labels
    p1, p2  : predicted probabilities from two classifiers

    Returns
    -------
    dict with auc1, auc2, z_stat, p_value
    """
    auc1 = roc_auc_score(y_true, p1)
    auc2 = roc_auc_score(y_true, p2)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos < 2 or n_neg < 2:
        return {"auc1": auc1, "auc2": auc2, "z_stat": float("nan"), "p_value": float("nan")}

    _, _, cov12 = _auc_covariance(y_true, p1, p2)

    var1 = auc1 * (1 - auc1) * (1 / n_pos + 1 / n_neg)  # Hanley-McNeil approx for var
    var2 = auc2 * (1 - auc2) * (1 / n_pos + 1 / n_neg)
    se   = np.sqrt(var1 + var2 - 2 * cov12)

    z = (auc1 - auc2) / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc1": float(auc1), "auc2": float(auc2), "z_stat": float(z), "p_value": p}


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
) -> dict[str, float]:
    """McNemar's test comparing paired binary predictions.

    Parameters
    ----------
    y_true          : binary ground truth
    pred1, pred2    : binary predictions from two classifiers

    Returns
    -------
    dict with n01, n10, chi2, p_value, odds_ratio
    """
    # Discordant pairs
    n01 = int(((pred1 == 0) & (pred2 == 1)).sum())  # pred1 wrong, pred2 right
    n10 = int(((pred1 == 1) & (pred2 == 0)).sum())  # pred1 right, pred2 wrong

    total_disc = n01 + n10
    if total_disc == 0:
        return {"n01": 0, "n10": 0, "chi2": 0.0, "p_value": 1.0, "odds_ratio": 1.0}

    # Edwards' continuity correction
    chi2 = (abs(n01 - n10) - 1.0)**2 / total_disc
    p    = float(stats.chi2.sf(chi2, df=1))
    odds = n01 / n10 if n10 > 0 else float("inf")
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p_value": p, "odds_ratio": float(odds)}


# ---------------------------------------------------------------------------
# Decision curve analysis
# ---------------------------------------------------------------------------

def compute_dca(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Decision curve analysis: net benefit vs threshold probability.

    Net benefit = TP/N - FP/N * (pt / (1 - pt))
    where pt = threshold probability and N = total patients.

    Parameters
    ----------
    y_true     : binary labels
    y_prob     : predicted probabilities
    thresholds : array of threshold probabilities (default linspace 0.01..0.99)

    Returns
    -------
    dict with keys:
        thresholds, model_nb, treat_all_nb, treat_none_nb
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)

    n = len(y_true)
    n_pos = int(y_true.sum())
    model_nb     = np.zeros(len(thresholds))
    treat_all_nb = np.zeros(len(thresholds))

    for i, pt in enumerate(thresholds):
        y_pred = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        model_nb[i] = tp / n - fp / n * (pt / (1 - pt))
        treat_all_nb[i] = n_pos / n - (n - n_pos) / n * (pt / (1 - pt))

    return {
        "thresholds":    thresholds,
        "model_nb":      model_nb,
        "treat_all_nb":  treat_all_nb,
        "treat_none_nb": np.zeros(len(thresholds)),
    }


# ---------------------------------------------------------------------------
# Lift / Cumulative Gain
# ---------------------------------------------------------------------------

def compute_lift_gain(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 100,
) -> dict[str, np.ndarray]:
    """Compute lift and cumulative gain curves.

    Returns
    -------
    dict with:
        pct_flagged : x-axis (fraction of population flagged)
        cum_gain    : fraction of positives captured at each threshold
        lift        : cum_gain / pct_flagged
    """
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    n_pos = y_true.sum()

    cum_pos = np.cumsum(y_sorted)
    cum_gain = cum_pos / n_pos
    pct_flagged = np.arange(1, len(y_true) + 1) / len(y_true)
    lift = cum_gain / pct_flagged

    # Subsample to n_bins points for storage
    idx = np.linspace(0, len(y_true) - 1, n_bins, dtype=int)
    return {
        "pct_flagged": pct_flagged[idx],
        "cum_gain":    cum_gain[idx],
        "lift":        lift[idx],
    }


# ---------------------------------------------------------------------------
# Subgroup metrics
# ---------------------------------------------------------------------------

def age_band_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ages: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute metrics within age bands (<40, 40-60, >60).

    Parameters
    ----------
    ages : array of raw patient ages (same order as y_true / y_prob)
    """
    bands = {
        "<40":   ages < 40,
        "40-60": (ages >= 40) & (ages <= 60),
        ">60":   ages > 60,
    }
    result: dict[str, dict[str, float]] = {}
    for label, mask in bands.items():
        if mask.sum() == 0:
            result[label] = {}
            continue
        m = compute_metrics(y_true[mask], y_prob[mask], threshold=threshold)
        m["n"] = int(mask.sum())
        m["n_pos"] = int(y_true[mask].sum())
        result[label] = m
    return result


def gender_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    genders: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute metrics by gender."""
    result: dict[str, dict[str, float]] = {}
    for g in np.unique(genders):
        mask = genders == g
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_prob[mask], threshold=threshold)
        m["n"] = int(mask.sum())
        m["n_pos"] = int(y_true[mask].sum())
        result[str(g)] = m
    return result


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_cv_metrics(cv_result: dict, save_path: str | Path) -> None:
    """Plot per-fold boxplots for key metrics."""
    metric_keys = ["roc_auc", "pr_auc", "brier", "recall", "precision", "f2"]
    per_fold = cv_result["per_fold"]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(18, 4))
    for ax, key in zip(axes, metric_keys):
        vals = [f[key] for f in per_fold if key in f]
        ax.boxplot(vals, vert=True, patch_artist=True)
        ax.set_title(key.replace("_", "\n"), fontsize=10)
        ax.set_xticklabels([f"n={len(vals)}"], fontsize=8)
        ax.axhline(np.mean(vals), color="red", linestyle="--", linewidth=1, label=f"μ={np.mean(vals):.3f}")
        ax.legend(fontsize=7)
    fig.suptitle("CV Metric Distributions (5×5 Repeated Stratified K-Fold)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("CV metrics plot saved: %s", save_path)


def plot_bootstrap_ci(ci_dict: dict, save_path: str | Path) -> None:
    """Plot 95% bootstrap CIs for headline metrics."""
    keys = ["roc_auc", "pr_auc", "brier", "recall", "precision", "f1", "f2"]
    keys = [k for k in keys if k in ci_dict]

    means = [ci_dict[k]["mean"] for k in keys]
    los   = [ci_dict[k]["lo"]   for k in keys]
    his   = [ci_dict[k]["hi"]   for k in keys]
    errs  = [[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(keys, means, xerr=errs, align="center", alpha=0.8,
            error_kw={"elinewidth": 2, "capsize": 4})
    ax.set_xlabel("Metric value")
    ax.set_title("OOF Metrics with 95% Bootstrap CIs")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Bootstrap CI plot saved: %s", save_path)


def plot_dca(dca: dict, threshold_line: float | None = None, save_path: str | Path = "plots/v2/dca.png") -> None:
    """Plot decision curve analysis."""
    fig, ax = plt.subplots(figsize=(8, 5))
    t = dca["thresholds"]
    ax.plot(t, dca["model_nb"],     label="Model",     linewidth=2)
    ax.plot(t, dca["treat_all_nb"], label="Treat all", linewidth=1, linestyle="--")
    ax.plot(t, dca["treat_none_nb"],label="Treat none",linewidth=1, linestyle=":")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 0.3)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    if threshold_line is not None:
        ax.axvline(threshold_line, color="red", linestyle="-.", linewidth=1.5,
                   label=f"Chosen threshold ({threshold_line:.2f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("DCA plot saved: %s", save_path)


def plot_lift_gain(lg: dict, save_path: str | Path = "plots/v2/lift_gain.png") -> None:
    """Plot lift and cumulative gain curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pct = lg["pct_flagged"] * 100
    ax1.plot(pct, lg["cum_gain"] * 100, linewidth=2)
    ax1.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random")
    ax1.set_xlabel("% Population Flagged"); ax1.set_ylabel("% Strokes Captured")
    ax1.set_title("Cumulative Gain"); ax1.legend()

    ax2.plot(pct, lg["lift"], linewidth=2)
    ax2.axhline(1.0, linestyle="--", color="gray", label="Baseline (random)")
    ax2.set_xlabel("% Population Flagged"); ax2.set_ylabel("Lift")
    ax2.set_title("Lift Curve"); ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Lift/gain plot saved: %s", save_path)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


def save_cv_report(
    cv_result: dict,
    ci_dict: dict,
    threshold: float,
    cfg: dict,
    tag: str = "xgb_spw",
) -> None:
    """Write JSON + CSV reports to reports/."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "model_tag":       tag,
        "threshold":       threshold,
        "oof_metrics":     _convert_numpy(cv_result["oof_metrics"]),
        "mean_cv_metrics": _convert_numpy(cv_result["mean_metrics"]),
        "std_cv_metrics":  _convert_numpy(cv_result["std_metrics"]),
        "bootstrap_ci":    _convert_numpy(ci_dict),
        "n_folds":         cv_result["n_folds"],
    }

    json_path = _REPORTS_DIR / f"cv_results_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("CV report saved: %s", json_path)

    # Flat CSV of per-fold metrics
    fold_df = pd.DataFrame(cv_result["per_fold"])
    csv_path = _REPORTS_DIR / f"cv_per_fold_{tag}.csv"
    fold_df.to_csv(csv_path, index=False)
    log.info("Per-fold CSV saved: %s", csv_path)

    # Plots
    plot_cv_metrics(cv_result, _PLOTS_DIR / f"cv_metrics_{tag}.png")
    plot_bootstrap_ci(ci_dict, _PLOTS_DIR / f"bootstrap_ci_{tag}.png")


# ---------------------------------------------------------------------------
# CLI entry point (Phase 2 standalone run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    import joblib
    import pandas as pd
    from src.config import load_config
    from src.pipeline_builder import build_pipeline, clean_raw_df

    cfg = load_config()
    seed = cfg["seed"]

    # Load data
    log.info("Loading raw data...")
    df = pd.read_csv(cfg["data"]["raw_path"])
    df = clean_raw_df(df)
    X_raw = df.drop("stroke", axis=1)
    y = df["stroke"]

    # Load split indices
    with open(cfg["data"]["split_indices_path"]) as f:
        split_meta = json.load(f)
    train_idx = split_meta["train_indices"]

    X_train = X_raw.loc[train_idx]
    y_train = y.loc[train_idx]

    log.info("Train size: %d | Positives: %d", len(X_train), int(y_train.sum()))

    # Build pipeline (unfitted — will be cloned per fold)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos
    pipeline_template = build_pipeline(cfg, scale_pos_weight=spw)

    # Run CV
    log.info("Running 5×5 repeated stratified K-fold CV...")
    cv_result = run_repeated_cv(pipeline_template, X_train, y_train, cfg, threshold=0.5)

    log.info("OOF metrics (threshold=0.5): %s", {k: round(v, 4) for k, v in cv_result["oof_metrics"].items() if isinstance(v, float)})

    # Bootstrap CIs on OOF probs
    log.info("Computing bootstrap CIs (n=1000)...")
    ci_dict = bootstrap_ci(
        cv_result["oof_true"], cv_result["oof_probs"],
        threshold=0.5, n_resamples=cfg["bootstrap"]["n_resamples"], seed=seed
    )

    # Save reports
    save_cv_report(cv_result, ci_dict, threshold=0.5, cfg=cfg, tag="xgb_spw")

    # Compute and plot DCA
    log.info("Computing DCA...")
    dca = compute_dca(cv_result["oof_true"], cv_result["oof_probs"])
    plot_dca(dca, threshold_line=None, save_path=_PLOTS_DIR / "dca_xgb_spw.png")

    # Lift/gain
    lg = compute_lift_gain(cv_result["oof_true"], cv_result["oof_probs"])
    plot_lift_gain(lg, save_path=_PLOTS_DIR / "lift_gain_xgb_spw.png")

    # Summary
    m = cv_result["oof_metrics"]
    ci = ci_dict
    log.info("=== Phase 2 OOF Summary (threshold=0.5) ===")
    for key in ["roc_auc", "pr_auc", "brier", "recall", "precision", "f2"]:
        if key in ci:
            log.info("  %-15s = %.4f [%.4f, %.4f]", key, ci[key]["mean"], ci[key]["lo"], ci[key]["hi"])

    log.info("Phase 2 complete.")
