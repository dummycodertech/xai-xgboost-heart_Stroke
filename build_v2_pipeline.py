"""
build_v2_pipeline.py
====================
Phase 1 script: trains the leak-free v2 pipeline on the 80% training split
and saves it to models/v2/pipeline.joblib.

The test set is NOT touched here.  Split indices are loaded from
reports/split_indices.json (generated in Phase 0).

Run
---
uv run python build_v2_pipeline.py
# or
.venv/Scripts/python.exe build_v2_pipeline.py
"""
from __future__ import annotations

import json
import logging
import os

import joblib
import pandas as pd

from src.config import load_config
from src.pipeline_builder import build_pipeline, clean_raw_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    raw_path = cfg["data"]["raw_path"]
    split_path = cfg["data"]["split_indices_path"]
    out_path = cfg["paths"]["pipeline_v2"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load raw data and clean
    log.info("Loading raw CSV: %s", raw_path)
    df = pd.read_csv(raw_path)
    df = clean_raw_df(df)

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    # Load pre-saved split indices (guarantees same split as v1)
    log.info("Loading split indices from: %s", split_path)
    with open(split_path) as f:
        split_meta = json.load(f)

    train_idx = split_meta["train_indices"]
    test_idx  = split_meta["test_indices"]

    # Verify no overlap
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test index overlap!"

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test  = X.loc[test_idx]
    y_test  = y.loc[test_idx]

    log.info(
        "Train: %d rows (%d pos) | Test: %d rows (%d pos)",
        len(X_train), int(y_train.sum()), len(X_test), int(y_test.sum()),
    )

    # Compute scale_pos_weight from TRAINING data only
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos
    log.info("scale_pos_weight (from training data): %.4f", spw)

    # Build and fit pipeline on training data only
    log.info("Building v2 pipeline...")
    pipeline = build_pipeline(cfg, scale_pos_weight=spw)

    log.info("Fitting pipeline on training data (test set NOT touched)...")
    pipeline.fit(X_train, y_train)
    log.info("Pipeline fit complete.")

    # Quick sanity check on output shape and probability range
    probs = pipeline.predict_proba(X_train[:5])
    assert probs.shape == (5, 2), f"Unexpected output shape: {probs.shape}"
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilities out of [0,1]"
    log.info("Output shape OK: %s, prob range [%.4f, %.4f]", probs.shape, probs.min(), probs.max())

    # Save pipeline
    log.info("Saving pipeline to: %s", out_path)
    joblib.dump(pipeline, out_path, compress=3)
    log.info("Pipeline saved (%.1f KB).", os.path.getsize(out_path) / 1024)

    # Save basic metadata
    meta_path = cfg["paths"]["metadata_v2"]
    meta = {
        "phase": "phase1",
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "n_train_pos": pos,
        "n_test_pos":  int(y_test.sum()),
        "scale_pos_weight": spw,
        "seed": seed,
        "note": (
            "v2 pipeline — test set NOT evaluated here. "
            "Threshold and calibration determined in Phase 5 from OOF probabilities."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata written to: %s", meta_path)

    log.info("Phase 1 complete.")


if __name__ == "__main__":
    main()
