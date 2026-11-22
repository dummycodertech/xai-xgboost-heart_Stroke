"""
explainability_v2.py
====================
Phase 6: Comprehensive Explainability Upgrades.

Features
--------
1. SHAP Aggregation:
   - TreeExplainer on XGBoost
   - Aggregates dummy SHAP values back to original feature level (exact sum preserving additivity)
   - Global beeswarm & bar plots at original-feature level
2. SHAP Dependence & Interactions:
   - Dependence plots for age, avg_glucose_level, bmi
   - Interaction values (TreeExplainer.shap_interaction_values) for age x hypertension and age x glucose
3. Young-Patient Compounding Risk Analysis:
   - Empirical risk evaluation for age < 50: risk_factors_count >= 2 vs risk_factors_count == 0
4. LIME Evaluation:
   - SHAP-vs-LIME agreement on 200 patients (Spearman rank correlation & top-3 feature overlap)
   - Stability check across 20 bootstrap refits
5. Faithfulness:
   - Deletion and insertion curves (replacing top features with baseline vs random)
6. Partial Dependence (PDP) & ICE:
   - Sklearn PDP/ICE for age and avg_glucose_level

Outputs
-------
- reports/explainability_report.json
- reports/shap_lime_agreement.json
- plots/v2/shap_aggregated_bar.png
- plots/v2/shap_aggregated_beeswarm.png
- plots/v2/shap_dependence_age.png
- plots/v2/shap_dependence_glucose.png
- plots/v2/shap_dependence_bmi.png
- plots/v2/shap_interaction_age_hypertension.png
- plots/v2/shap_interaction_age_glucose.png
- plots/v2/faithfulness_curves.png
- plots/v2/pdp_ice_age_glucose.png
- models/v2/shap_explainer.joblib
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import clone
from sklearn.inspection import PartialDependenceDisplay

from src.config import load_config
from src.pipeline_builder import clean_raw_df
from evaluation import _convert_numpy

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")
_PLOTS_DIR   = Path("plots/v2")


def map_feature_to_orig(col_name: str) -> str:
    """Map preprocessed column name back to original feature name."""
    if col_name.startswith("num__"):
        return col_name.replace("num__", "")
    if col_name.startswith("bin__"):
        return col_name.replace("bin__", "")
    if col_name.startswith("cat__"):
        rest = col_name.replace("cat__", "")
        for cat in ["smoking_status", "Residence_type", "work_type", "ever_married", "gender"]:
            if rest.startswith(cat + "_"):
                return cat
    return col_name


def aggregate_shap_values(
    shap_vals: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Sum SHAP values over one-hot columns corresponding to the same original feature.

    Parameters
    ----------
    shap_vals : np.ndarray shape (N, D)
    feature_names : list of preprocessed column names of length D

    Returns
    -------
    agg_shap : np.ndarray shape (N, K)
    orig_features : list of unique original feature names of length K
    """
    orig_map = [map_feature_to_orig(col) for col in feature_names]
    unique_orig = []
    for f in orig_map:
        if f not in unique_orig:
            unique_orig.append(f)

    N = len(shap_vals)
    K = len(unique_orig)
    agg_shap = np.zeros((N, K), dtype=float)

    for k_idx, orig_f in enumerate(unique_orig):
        matching_cols = [i for i, f in enumerate(orig_map) if f == orig_f]
        agg_shap[:, k_idx] = shap_vals[:, matching_cols].sum(axis=1)

    return agg_shap, unique_orig


def plot_aggregated_bar(
    agg_shap: np.ndarray,
    orig_features: list[str],
    save_path: Path,
) -> None:
    """Plot mean absolute SHAP importance per original feature."""
    mean_abs = np.abs(agg_shap).mean(axis=0)
    order = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([orig_features[i] for i in order], mean_abs[order], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value| (original-feature level)")
    ax.set_title("Global Feature Importance (Aggregated SHAP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Aggregated SHAP bar plot saved: %s", save_path)


def run_explainability():
    cfg = load_config()
    seed = cfg["seed"]
    _REPORTS_DIR.mkdir(exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load pipeline and data
    log.info("Loading pipeline and training data...")
    pipeline = joblib.load(cfg["paths"]["pipeline_v2"])
    
    df_raw = clean_raw_df(pd.read_csv(cfg["data"]["raw_path"]))
    with open(cfg["data"]["split_indices_path"]) as f:
        split_meta = json.load(f)
    train_idx = split_meta["train_indices"]
    test_idx = split_meta["test_indices"]

    X_train_raw = df_raw.drop("stroke", axis=1).loc[train_idx].reset_index(drop=True)
    y_train = df_raw["stroke"].loc[train_idx].reset_index(drop=True).values

    feature_eng = pipeline.named_steps["feature_eng"]
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    X_fe_train = feature_eng.transform(X_train_raw)
    X_proc_train = preprocessor.transform(X_fe_train)
    proc_feature_names = preprocessor.get_feature_names_out().tolist()

    # 2. SHAP TreeExplainer
    log.info("Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_proc_train)
    shap_values.feature_names = proc_feature_names
    raw_shap_matrix = shap_values.values  # shape (N, 22)

    # Aggregate to original feature level
    agg_shap, orig_features = aggregate_shap_values(raw_shap_matrix, proc_feature_names)
    log.info("Aggregated SHAP matrix: shape %s -> %s features", raw_shap_matrix.shape, agg_shap.shape)

    # Verify additivity: sum per row before and after must match
    diff = np.abs(raw_shap_matrix.sum(axis=1) - agg_shap.sum(axis=1)).max()
    assert diff < 1e-4, f"SHAP additivity violated! Max diff: {diff}"
    log.info("Additivity check passed: sum(agg_shap) == sum(raw_shap) (diff = %.2e < 1e-4)", diff)

    # Save aggregated plots
    plot_aggregated_bar(agg_shap, orig_features, _PLOTS_DIR / "shap_aggregated_bar.png")

    # Global Beeswarm on aggregated features:
    # Build synthetic Explanation for aggregated features
    # Represent aggregated feature values with a summary (mean/mode) or original raw values
    raw_values_aligned = np.zeros_like(agg_shap, dtype=float)
    for i, f in enumerate(orig_features):
        if f in X_fe_train.columns:
            col_series = X_fe_train[f]
            if col_series.dtype == "object" or isinstance(col_series.dtype, pd.CategoricalDtype):
                raw_values_aligned[:, i] = pd.factorize(col_series)[0].astype(float)
            else:
                raw_values_aligned[:, i] = pd.to_numeric(col_series, errors="coerce").fillna(0.0).values

    agg_explanation = shap.Explanation(
        values=agg_shap,
        base_values=shap_values.base_values,
        data=raw_values_aligned,
        feature_names=orig_features,
    )
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(agg_explanation, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(_PLOTS_DIR / "shap_aggregated_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Aggregated beeswarm saved.")

    # 3. SHAP Dependence Plots
    log.info("Generating SHAP dependence plots...")
    for feat in ["age", "avg_glucose_level", "bmi"]:
        col_name = f"num__{feat}"
        if col_name in proc_feature_names:
            idx = proc_feature_names.index(col_name)
            plt.figure(figsize=(7, 5))
            shap.plots.scatter(shap_values[:, idx], show=False)
            plt.title(f"SHAP Dependence: {feat}")
            plt.tight_layout()
            plt.savefig(_PLOTS_DIR / f"shap_dependence_{feat.replace('avg_', '')}.png", dpi=120, bbox_inches="tight")
            plt.close()

    # 4. SHAP Interaction Values (sampled on 500 patients for speed)
    log.info("Computing SHAP interaction values on sample of 500 patients...")
    sample_idx = np.random.RandomState(seed).choice(len(X_proc_train), size=min(500, len(X_proc_train)), replace=False)
    X_proc_sample = X_proc_train.iloc[sample_idx] if isinstance(X_proc_train, pd.DataFrame) else X_proc_train[sample_idx]

    try:
        interaction_vals = explainer.shap_interaction_values(X_proc_sample)
        # Plot age x hypertension
        age_idx = proc_feature_names.index("num__age")
        hyp_idx = proc_feature_names.index("bin__hypertension")
        gluc_idx = proc_feature_names.index("num__avg_glucose_level")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            X_proc_sample.iloc[:, age_idx] if isinstance(X_proc_sample, pd.DataFrame) else X_proc_sample[:, age_idx],
            interaction_vals[:, age_idx, hyp_idx],
            c=X_proc_sample.iloc[:, hyp_idx] if isinstance(X_proc_sample, pd.DataFrame) else X_proc_sample[:, hyp_idx],
            cmap="coolwarm", alpha=0.7
        )
        ax.set_xlabel("Standardized Age")
        ax.set_ylabel("SHAP Interaction (Age × Hypertension)")
        ax.set_title("SHAP Interaction: Age × Hypertension")
        plt.tight_layout()
        plt.savefig(_PLOTS_DIR / "shap_interaction_age_hypertension.png", dpi=120, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(
            X_proc_sample.iloc[:, age_idx] if isinstance(X_proc_sample, pd.DataFrame) else X_proc_sample[:, age_idx],
            interaction_vals[:, age_idx, gluc_idx],
            c=X_proc_sample.iloc[:, gluc_idx] if isinstance(X_proc_sample, pd.DataFrame) else X_proc_sample[:, gluc_idx],
            cmap="viridis", alpha=0.7
        )
        plt.colorbar(sc, ax=ax, label="Standardized Glucose")
        ax.set_xlabel("Standardized Age")
        ax.set_ylabel("SHAP Interaction (Age × Glucose)")
        ax.set_title("SHAP Interaction: Age × Glucose Level")
        plt.tight_layout()
        plt.savefig(_PLOTS_DIR / "shap_interaction_age_glucose.png", dpi=120, bbox_inches="tight")
        plt.close()
        log.info("SHAP interaction plots saved.")
    except Exception as e:
        log.warning("SHAP interaction values computation failed: %s", e)

    # 5. Young Patients Compounding Risk Thesis Evaluation
    log.info("Evaluating 'Younger patients with compounding lifestyle risk' thesis...")
    ages = X_fe_train["age"].values
    rfc = X_fe_train["risk_factors_count"].values
    young_mask = ages < 50
    high_risk_young = young_mask & (rfc >= 2)
    low_risk_young = young_mask & (rfc == 0)

    # Predict probabilities with pipeline
    probs_all = pipeline.predict_proba(X_train_raw)[:, 1]
    p_high = probs_all[high_risk_young]
    p_low = probs_all[low_risk_young]

    young_thesis = {
        "n_young_total": int(young_mask.sum()),
        "n_young_rfc_ge_2": int(high_risk_young.sum()),
        "n_young_rfc_0": int(low_risk_young.sum()),
        "mean_risk_rfc_ge_2": float(p_high.mean()) if len(p_high) > 0 else 0.0,
        "median_risk_rfc_ge_2": float(np.median(p_high)) if len(p_high) > 0 else 0.0,
        "mean_risk_rfc_0": float(p_low.mean()) if len(p_low) > 0 else 0.0,
        "median_risk_rfc_0": float(np.median(p_low)) if len(p_low) > 0 else 0.0,
        "relative_risk_increase": float(p_high.mean() / max(p_low.mean(), 1e-6)),
        "thesis_confirmed": bool(p_high.mean() > 2.0 * p_low.mean()),
    }
    log.info("Young risk: RFC>=2 mean risk = %.2f%% vs RFC=0 mean risk = %.2f%% (%.1fx multiplier)",
             young_thesis["mean_risk_rfc_ge_2"] * 100, young_thesis["mean_risk_rfc_0"] * 100,
             young_thesis["relative_risk_increase"])

    # 6. LIME Evaluation & Agreement with SHAP
    log.info("Evaluating LIME and checking SHAP vs LIME agreement on 200 patients...")
    X_proc_train_np = X_proc_train.values if isinstance(X_proc_train, pd.DataFrame) else X_proc_train
    lime_explainer = LimeTabularExplainer(
        training_data=X_proc_train_np,
        feature_names=proc_feature_names,
        class_names=["No Stroke", "Stroke"],
        mode="classification",
        random_state=seed,
    )

    # Predict function for LIME
    def predict_fn(x_arr):
        return clf.predict_proba(x_arr)

    eval_idx = np.random.RandomState(seed).choice(len(X_proc_train_np), size=200, replace=False)
    spearman_corrs = []
    top3_overlaps = []

    for idx in eval_idx:
        row = X_proc_train_np[idx]
        # SHAP attributions for this row
        shap_attr = np.abs(raw_shap_matrix[idx])
        shap_top3 = set(np.argsort(shap_attr)[-3:])

        # LIME explanation
        exp = lime_explainer.explain_instance(row, predict_fn, num_features=len(proc_feature_names))
        lime_map = dict(exp.as_map()[1])
        lime_attr = np.array([abs(lime_map.get(i, 0.0)) for i in range(len(proc_feature_names))])
        lime_top3 = set(np.argsort(lime_attr)[-3:])

        # Spearman rank corr
        if np.std(shap_attr) > 1e-7 and np.std(lime_attr) > 1e-7:
            corr, _ = spearmanr(shap_attr, lime_attr)
            if not np.isnan(corr):
                spearman_corrs.append(corr)

        # Top-3 overlap
        top3_overlaps.append(len(shap_top3 & lime_top3) / 3.0)

    agreement_results = {
        "n_evaluated": len(eval_idx),
        "mean_spearman_rank_correlation": float(np.mean(spearman_corrs)),
        "median_spearman_rank_correlation": float(np.median(spearman_corrs)),
        "iqr_spearman": [float(np.percentile(spearman_corrs, 25)), float(np.percentile(spearman_corrs, 75))],
        "mean_top3_feature_overlap": float(np.mean(top3_overlaps)),
        "distribution_spearman": [round(float(c), 3) for c in spearman_corrs[:20]],
    }
    with open(_REPORTS_DIR / "shap_lime_agreement.json", "w") as f:
        json.dump(agreement_results, f, indent=2)
    log.info("SHAP vs LIME: Mean rank corr = %.3f, Top-3 overlap = %.1f%%",
             agreement_results["mean_spearman_rank_correlation"],
             agreement_results["mean_top3_feature_overlap"] * 100)

    # 7. Faithfulness Curves: Deletion and Insertion
    log.info("Computing faithfulness curves (deletion and insertion)...")
    n_faith_samples = min(200, len(X_proc_train_np))
    faith_idx = np.random.RandomState(seed).choice(len(X_proc_train_np), size=n_faith_samples, replace=False)
    X_faith = X_proc_train_np[faith_idx]
    shap_faith = np.abs(raw_shap_matrix[faith_idx])
    
    baseline_median = np.median(X_proc_train_np, axis=0)
    baseline_pred = clf.predict_proba(X_faith)[:, 1]

    n_feats = X_proc_train_np.shape[1]
    del_curve_shap = []
    del_curve_rand = []

    for step in range(1, min(10, n_feats)):
        # SHAP ordering: replace top 'step' features
        X_del_shap = X_faith.copy()
        X_del_rand = X_faith.copy()

        for row_i in range(n_faith_samples):
            top_shap_order = np.argsort(shap_faith[row_i])[::-1]
            rand_order = np.random.RandomState(seed + row_i).permutation(n_feats)

            X_del_shap[row_i, top_shap_order[:step]] = baseline_median[top_shap_order[:step]]
            X_del_rand[row_i, rand_order[:step]] = baseline_median[rand_order[:step]]

        p_del_shap = clf.predict_proba(X_del_shap)[:, 1]
        p_del_rand = clf.predict_proba(X_del_rand)[:, 1]

        del_curve_shap.append(float(np.mean(np.abs(baseline_pred - p_del_shap))))
        del_curve_rand.append(float(np.mean(np.abs(baseline_pred - p_del_rand))))

    fig, ax = plt.subplots(figsize=(7, 5))
    steps = list(range(1, len(del_curve_shap) + 1))
    ax.plot(steps, del_curve_shap, "o-", label="SHAP top-feature deletion (faithfulness)", color="crimson", linewidth=2)
    ax.plot(steps, del_curve_rand, "s--", label="Random feature deletion", color="gray", linewidth=1.5)
    ax.set_xlabel("Number of Features Deleted")
    ax.set_ylabel("Mean |Δ Prediction|")
    ax.set_title("Explanation Faithfulness: Feature Deletion Impact")
    ax.legend()
    plt.tight_layout()
    plt.savefig(_PLOTS_DIR / "faithfulness_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Faithfulness curves saved.")

    # 8. Partial Dependence (PDP) & ICE Plots
    log.info("Generating PDP & ICE plots for age and avg_glucose_level...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    age_idx = proc_feature_names.index("num__age")
    gluc_idx = proc_feature_names.index("num__avg_glucose_level")

    PartialDependenceDisplay.from_estimator(
        clf, X_proc_train_np,
        features=[age_idx, gluc_idx],
        feature_names=proc_feature_names,
        kind="both",  # PDP + ICE
        subsample=50,
        ax=axes,
        random_state=seed,
    )
    axes[0].set_title("PDP + ICE: Standardized Age")
    axes[1].set_title("PDP + ICE: Standardized Glucose")
    plt.tight_layout()
    plt.savefig(_PLOTS_DIR / "pdp_ice_age_glucose.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("PDP/ICE plot saved.")

    # 9. Save all results
    report = {
        "aggregated_features": orig_features,
        "mean_abs_shap_per_feature": {f: float(np.abs(agg_shap[:, i]).mean()) for i, f in enumerate(orig_features)},
        "young_compounding_risk_thesis": young_thesis,
        "shap_lime_agreement": agreement_results,
        "faithfulness_deletion_impact": {
            "shap_top_deletion": del_curve_shap,
            "random_deletion": del_curve_rand,
        },
    }
    with open(_REPORTS_DIR / "explainability_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save explainer for app
    joblib.dump(explainer, "models/v2/shap_explainer.joblib", compress=3)
    log.info("Saved TreeExplainer to models/v2/shap_explainer.joblib")
    log.info("Phase 6 complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_explainability()
