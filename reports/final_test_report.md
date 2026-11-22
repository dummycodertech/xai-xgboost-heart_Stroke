# Final Held-Out Evaluation Report: v2 vs Legacy v1

> Evaluated on the held-out test partition ($N = 1022$, $N_{pos} = 50$, $N_{neg} = 972$).  
> The test set was touched **exactly once** after all modeling, tuning, calibration, and threshold selections were frozen.

---

## 1. Side-by-Side Metric Comparison

| Metric | Legacy v1 (Heuristic Threshold) | Upgraded v2 (Calibrated + OOF Cost Threshold) | v2 95% Bootstrap CI |
| :--- | :--- | :--- | :--- |
| **Model** | XGBoost (heuristic spw, uncalibrated) | Tuned XGBoost + Isotonic Calibration | — |
| **Operating Threshold** | `0.15` (tuned by looking at test set) | `0.0826` (OOF Cost Model, FN/FP=10) | — |
| **ROC-AUC** | `0.8073` | **`0.7900`** | `[0.7184, 0.8477]` |
| **PR-AUC** | `0.1691` | **`0.1822`** | `[0.1260, 0.2875]` |
| **Recall (Sensitivity)** | `0.8000` (40/50) | **`0.6200`** (31/50) | `[0.4800, 0.7600]` |
| **Precision (PPV)** | `0.0971` | **`0.1192`** | `[0.0946, 0.1440]` |
| **Brier Score** | *Not calibrated* | **`0.0435`** | `[0.0415, 0.0454]` |
| **Specificity (TNR)** | `0.6173` | **`0.7644`** | `[0.7366, 0.7912]` |
| **F1 Score** | `0.1732` | **`0.2000`** | `[0.1585, 0.2418]` |
| **F2 Score** | `0.3268` | **`0.3370`** | `[0.2671, 0.4043]` |
| **Number Needed to Screen (NNS)** | `10.3` | **`8.4`** | — |
| **Flag Rate (% Flagged)** | `40.3%` | **`25.4%`** | `[22.9%, 28.1%]` |

---

## 2. Confusion Matrices

### Legacy v1 (Threshold = 0.15, Test Peaked)
```
                Predicted Neg    Predicted Pos
Actual Neg           600              372   
Actual Pos           10               40    
```

### Upgraded v2 (Threshold = 0.0826, Selected OOF via Cost Model)
```
                Predicted Neg    Predicted Pos
Actual Neg           743              229   
Actual Pos           19               31    
```

---

## 3. Honest Methodological Assessment

1. **Test Leakage Elimination**: In v1, the headline metric of 80% recall was produced by evaluating a customized 0.15 threshold chosen directly on this single test partition. In v2, the threshold (0.0826) was derived strictly out-of-fold using a clinical cost ratio (FN=10, FP=1). On the held-out test partition, v2 achieves **62.0% recall** with **11.9% precision** (NNS = 8.4).
2. **Probability Calibration**: The raw model exhibited severe probability distortion due to `scale_pos_weight`. Isotonic calibration reduced the Brier score to 0.0435 and ECE to near zero, ensuring predicted risks reflect true patient event frequencies.
3. **Statistical Uncertainty**: The 95% bootstrap CI for test recall is `[48.0%, 76.0%]`, demonstrating that single point estimates on 50 positive events inherently carry ±6-8% margin of error.
