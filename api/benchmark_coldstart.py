"""
benchmark_coldstart.py
======================
Measures cold start time, inference latency, and peak RAM consumption.
"""
import time
import tracemalloc

import joblib

tracemalloc.start()
t0 = time.perf_counter()

from fastapi.testclient import TestClient

from api.main import ARTIFACTS_DIR, STATE, app

# Lifespan startup simulation
STATE["cal_pipe"] = joblib.load(ARTIFACTS_DIR / "calibrated_pipeline.joblib")
import json

with open(ARTIFACTS_DIR / "meta.json") as f:
    STATE["meta"] = json.load(f)

client = TestClient(app)
t_ready = time.perf_counter()

# Cold start latency
cold_start_ms = (t_ready - t0) * 1000

# Probe /health
r_health = client.get("/health")
assert r_health.status_code == 200

# Benchmark /predict
payload = {
    "age": 62, "avg_glucose_level": 195.0, "bmi": 32.4, "gender": "Male",
    "hypertension": 1, "heart_disease": 0, "ever_married": "Yes",
    "work_type": "Self-employed", "Residence_type": "Urban", "smoking_status": "formerly smoked",
    "operating_point": "cost_based_10"
}

t_pred0 = time.perf_counter()
r_pred = client.post("/predict", json=payload)
t_pred1 = time.perf_counter()

assert r_pred.status_code == 200
pred_ms = (t_pred1 - t_pred0) * 1000

current_mem, peak_mem = tracemalloc.get_traced_memory()

print("BENCHMARK RESULTS:")
print(f"  Cold-start ready time: {cold_start_ms:.1f} ms ({cold_start_ms/1000:.2f} s)")
print(f"  Predict latency (5-fold TreeSHAP + Local Surrogate): {pred_ms:.1f} ms")
print(f"  Current Memory: {current_mem / 1024 / 1024:.1f} MB")
print(f"  Peak Memory: {peak_mem / 1024 / 1024:.1f} MB")
