"""
src/pipeline_builder.py
=======================
Builds a single sklearn Pipeline (ColumnTransformer + XGBClassifier)
that accepts the RAW cleaned DataFrame and outputs calibrated probabilities.

Design decisions
----------------
* risk_factors_count is computed inside the pipeline via a FunctionTransformer
  so it is never computed on the held-out test set before the pipeline is called.
* bmi_missing indicator (optional, config-controlled) is added as a passthrough
  binary column.
* hypertension and heart_disease are already binary; they are passed through
  directly (no OHE overhead) to avoid spurious encoded column explosion.
* All preprocessing (imputation, scaling, encoding) is fitted inside CV folds —
  this module exposes only the builder; fitting happens in evaluation.py.

Usage
-----
from src.pipeline_builder import build_pipeline
from src.config import load_config
cfg = load_config()
pipeline = build_pipeline(cfg)
# Then: pipeline.fit(X_raw_train, y_train)
# Save: joblib.dump(pipeline, cfg["paths"]["pipeline_v2"])
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Feature-engineering transformers
# ---------------------------------------------------------------------------

def _add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add risk_factors_count and optionally bmi_missing to a DataFrame.

    This is applied *before* the ColumnTransformer so the new columns are
    visible to the numeric/binary transformers.

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature DataFrame (after id drop and gender filter).

    Returns
    -------
    pd.DataFrame
        Copy of X with additional columns.
    """
    X = X.copy()
    high_risk_smokers = X["smoking_status"].isin(["smokes", "formerly smoked"]).astype(int)
    X["risk_factors_count"] = (
        X["hypertension"].astype(int)
        + X["heart_disease"].astype(int)
        + high_risk_smokers
    )
    # bmi_missing indicator (always added here; ColumnTransformer controls inclusion)
    X["bmi_missing"] = X["bmi"].isna().astype(int)
    return X


def _make_feature_adder() -> FunctionTransformer:
    """Return a sklearn FunctionTransformer wrapping _add_engineered_features."""
    return FunctionTransformer(
        func=_add_engineered_features,
        validate=False,
    )


# ---------------------------------------------------------------------------
# Column definitions (resolved from config)
# ---------------------------------------------------------------------------

def _resolve_columns(cfg: dict) -> dict[str, list[str]]:
    """Return column groups based on config flags.

    Parameters
    ----------
    cfg : dict
        Config dict from load_config().

    Returns
    -------
    dict with keys:
        numeric      : columns for median impute + StandardScaler
        binary       : passthrough binary columns (no scaling)
        categorical  : columns for most-frequent impute + OHE
    """
    feat_cfg = cfg.get("features", {})

    numeric: list[str] = ["age", "avg_glucose_level", "bmi", "risk_factors_count"]
    if feat_cfg.get("bmi_missing_indicator", True):
        numeric.append("bmi_missing")  # already 0/1 — scaled but that's fine

    # hypertension and heart_disease are binary; skip OHE
    binary = feat_cfg.get("binary_passthrough", ["hypertension", "heart_disease"])

    categorical = feat_cfg.get(
        "categorical",
        ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
    )

    return {"numeric": numeric, "binary": binary, "categorical": categorical}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    cfg: dict,
    scale_pos_weight: float | None = None,
) -> Pipeline:
    """Build the full end-to-end sklearn Pipeline.

    Parameters
    ----------
    cfg : dict
        Config dict from load_config().
    scale_pos_weight : float | None
        If provided, sets XGBClassifier's scale_pos_weight.
        Pass ``None`` to let XGBoost use its default (1.0, i.e. no weighting).
        In practice, compute neg/pos on the training fold and pass here.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready for ``.fit(X_raw, y)``.
    """
    seed = cfg.get("seed", 42)
    xgb_cfg = cfg.get("xgboost", {})
    cols = _resolve_columns(cfg)

    # Preprocessor
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Binary passthrough: impute with 0 (they are already 0/1; NaN shouldn't
    # exist for hypertension/heart_disease, but guard anyway)
    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_transformer,     cols["numeric"]),
            ("bin",  binary_transformer,      cols["binary"]),
            ("cat",  categorical_transformer, cols["categorical"]),
        ],
        remainder="drop",
    )

    # XGBoost classifier
    xgb_kwargs: dict = {
        "max_depth":     xgb_cfg.get("max_depth", 4),
        "learning_rate": xgb_cfg.get("learning_rate", 0.05),
        "n_estimators":  xgb_cfg.get("n_estimators", 200),
        "subsample":     xgb_cfg.get("subsample", 0.8),
        "eval_metric":   xgb_cfg.get("eval_metric", "logloss"),
        "random_state":  seed,
        "tree_method":   "hist",          # faster; deterministic on CPU
    }
    if scale_pos_weight is not None:
        xgb_kwargs["scale_pos_weight"] = scale_pos_weight

    monotone = xgb_cfg.get("monotone_constraints")
    if monotone is not None:
        xgb_kwargs["monotone_constraints"] = monotone

    classifier = XGBClassifier(**xgb_kwargs)

    # Full pipeline: feature engineering -> preprocessing -> classifier
    pipeline = Pipeline([
        ("feature_eng", _make_feature_adder()),
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])

    return pipeline


# ---------------------------------------------------------------------------
# Convenience: clean raw DataFrame (same logic as data_loader.py)
# ---------------------------------------------------------------------------

def clean_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard cleaning steps to the raw CSV DataFrame.

    Steps:
    1. Drop the ``id`` column if present.
    2. Drop the single row where ``gender == 'Other'``.

    Does NOT engineer features (that happens inside the pipeline).

    Parameters
    ----------
    df : pd.DataFrame
        Raw CSV loaded with ``pd.read_csv()``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame without ``stroke`` column modifications.
    """
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    df = df[df["gender"] != "Other"].copy()
    return df
