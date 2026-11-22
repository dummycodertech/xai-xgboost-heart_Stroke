"""Save split indices for reproducibility (Phase 0)."""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
df = df.drop("id", axis=1)
df = df[df["gender"] != "Other"]
high_risk_smokers = df["smoking_status"].isin(["smokes", "formerly smoked"]).astype(int)
df["risk_factors_count"] = df["hypertension"] + df["heart_disease"] + high_risk_smokers

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

split_data = {
    "train_indices": X_train.index.tolist(),
    "test_indices": X_test.index.tolist(),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_train_pos": int(y_train.sum()),
    "n_test_pos": int(y_test.sum()),
    "seed": 42,
    "test_size": 0.20,
    "generated_by": "Phase 0 — split_indices.py",
}
with open("reports/split_indices.json", "w") as f:
    json.dump(split_data, f, indent=2)

print("Split indices saved.")
print(f"Train: {split_data['n_train']} rows ({split_data['n_train_pos']} pos)")
print(f"Test:  {split_data['n_test']} rows ({split_data['n_test_pos']} pos)")
