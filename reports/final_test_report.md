# Final Held-Out Evaluation Report: Upgraded v2 Pipeline vs Legacy v1

> Evaluated on the held-out test partition ($N = 1022$, $N_{pos} = 50$, $N_{neg} = 972$).  
> The test set was touched **exactly once** after all modeling, tuning, calibration, and threshold selections were frozen.

---

## 1. Multi-Threshold Performance on Held-Out Test Set

All thresholds were fixed **strictly out-of-fold** from the training partition before inspecting the test set.

| Operating Point | Threshold | Recall (Sensitivity) | Precision (PPV) | Specificity (TNR) | False Alarms (FP) | NNS (Alerts/Stroke) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Legacy v1 (Test Peaked)** | `0.1500` | 80.0% (40/50) | 9.7% | 61.7% | 372 | 10.3 |
| **v2: Cost-Optimal (FN/FP=10)** | `0.0826` | **62.0%** `[48.0%, 76.0%]` | **11.9%** `[9.5%, 14.4%]` | **76.4%** `[73.7%, 79.1%]` | **229** `[203, 256]` | **8.4** `[6.9, 10.6]` |
| **v2: Target 80% OOF Recall** | `0.0581` | **78.0%** `[66.0%, 88.0%]` | **11.7%** `[10.0%, 13.5%]` | **69.9%** `[67.0%, 72.6%]` | **293** `[266, 321]` | **8.5** `[7.4, 10.0]` |
| **v2: High Precision** | `0.1365` | **32.0%** `[20.0%, 46.0%]` | **18.8%** `[12.1%, 26.3%]` | **92.9%** `[91.2%, 94.5%]` | **69** `[53, 86]` | **5.3** `[3.8, 8.3]` |

---

## 2. Young Patient Subgroup Audit (Age < 50)

Across the entire dataset ($N=5,109$), stroke events in young patients are extremely rare (**0.69%** overall prevalence). However, stroke risk increases strictly with the count of co-occurring risk factors (`hypertension + heart_disease + smoker`).

### Full Dataset ($N = 2,899$ young patients, 20 strokes):
- **RFC = 0**: $N = 2,099$, Strokes = $11$ ($0.52\%$, 5.2 per 1,000)
- **RFC = 1**: $N = 761$, Strokes = $8$ ($1.05\%$, 10.5 per 1,000)
- **RFC = 2**: $N = 39$, Strokes = $1$ ($2.56\%$, 25.6 per 1,000)
- **RFC = 3**: $N = 0$, Strokes = $0$

### Training Set ($N = 2,316$ young patients, 12 strokes):
- **RFC = 0**: $N = 1,687$, Strokes = $7$ ($0.41\%$)
- **RFC = 1**: $N = 594$, Strokes = $4$ ($0.67\%$)
- **RFC = 2**: $N = 35$, Strokes = $1$ ($2.86\%$)

### Held-Out Test Set ($N = 583$ young patients, 8 strokes):
- **RFC = 0**: $N = 412$, Strokes = $4$ ($0.97\%$)
- **RFC = 1**: $N = 167$, Strokes = $4$ ($2.40\%$)
- **RFC = 2**: $N = 4$, Strokes = $0$ ($0.00\%$)

---

## 3. Honest Methodological Assessment

1. **PR-AUC Change within Sampling Variance**:
   - v1 Test PR-AUC: `0.1691`
   - v2 Test PR-AUC: `0.1822` `[95% CI: 0.1260, 0.2875]`
   - The slight increase in PR-AUC (+0.013) is well within the 95% bootstrap confidence interval. We do not claim superior ranking performance over baseline; rather, v2 provides **calibrated probabilities** and **leak-free threshold selection**.
2. **False Alarm Reduction is a Consequence of the Operating Point**:
   - The drop in false alarms from 372 (v1) to 229 (v2 cost-optimal) or 293 (v2 80% OOF recall) is not "model magic" — it is the direct, expected clinical tradeoff of choosing an out-of-fold calibrated decision threshold.
3. **No Statistically Significant Model Zoo Winner**:
   - Rigorous repeated cross-validation across 13 models demonstrates that tree ensembles, Explainable Boosting Machines (EBM PR-AUC 0.1625, ROC-AUC 0.8339), and penalized Logistic Regression (L1/L2 PR-AUC 0.1635, ROC-AUC 0.8354) have overlapping confidence intervals. DeLong and McNemar tests confirm no statistically significant difference in discriminative ability between tuned XGBoost and simpler glass-box alternatives.
