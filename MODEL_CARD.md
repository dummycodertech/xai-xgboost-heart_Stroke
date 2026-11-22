# Model Card: Stroke Risk Prediction & Triage System (v2 Production Pipeline)

---

## 1. Model Details
- **Model Name**: Stroke Risk Clinical Triage Classifier (v2 Production)
- **Model Architecture**: Tuned XGBoost (`max_depth=8`, `learning_rate=0.206`, `colsample_bytree=0.526`, monotonic constraints on age & glucose) wrapped in an Isotonic `CalibratedClassifierCV` (5-fold).
- **Version**: 2.0.0 (Upgrade over legacy heuristic v1)
- **Primary Developer**: Upgraded for `dummycodertech/xai-xgboost-heart_Stroke`
- **License**: MIT License with Clinical Research Software Disclaimer (see `LICENSE`)
- **Frameworks**: scikit-learn 1.8.0, XGBoost 3.2.0, SHAP 0.51.0, LIME 0.2.0.1, Python 3.11–3.14

---

## 2. Intended Use & Clinical Scope
- **Intended Use**: Clinical triage and risk stratification decision support. Designed to identify adult patients at elevated stroke risk during routine screening, prioritizing high sensitivity (recall) to minimize missed stroke events (False Negatives).
- **Target Setting**: Ambulatory clinics, triage nurse stations, preventive health screenings.
- **Out-of-Scope & Prohibited Uses**:
  - **Diagnostic Confirmation**: This model is NOT a diagnostic device for acute stroke (e.g., ischemic stroke in progress requiring tPA or thrombectomy).
  - **Autonomous Treatment Decisions**: Model predictions must never dictate treatment without independent clinical assessment.
  - **Pediatric Patients**: Not intended for patients under 18 years old.

---

## 3. Training & Evaluation Data
- **Dataset**: Kaggle "Healthcare Stroke Dataset" (5,110 initial records, 12 features).
- **Cleaning**: 
  - Dropped arbitrary identifier column `id`.
  - Dropped single record with `gender == 'Other'` (leaving 5,109 patients).
- **Partitioning**: 
  - Stratified 80/20 train/test split (Random Seed 42).
  - Train: 4,087 patients (199 stroke events, 4.87% prevalence).
  - Test: 1,022 patients (50 stroke events, 4.89% prevalence).
  - Split indices permanently serialized to `reports/split_indices.json`.
  - **Zero Leakage**: All hyperparameter tuning, feature engineering, probability calibration, and threshold selection performed strictly out-of-fold on the 80% training set. Held-out test set evaluated exactly once.

---

## 4. Performance Across Pre-Selected Operating Points (Held-Out Test Set, N=1,022, 50 Positives)

All thresholds were fixed **strictly out-of-fold** from the training partition before evaluating the test set. All metrics include **95% Bootstrap Confidence Intervals** (1,000 resamples):

| Operating Point | Threshold | Recall (Sensitivity) | Precision (PPV) | Specificity (TNR) | False Alarms (FP) | NNS (Screened/Stroke) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Legacy v1 (Test Peaked)** | `0.1500` | 80.0% (40/50) | 9.7% | 61.7% | 372 | 10.3 |
| **v2: Cost-Optimal (FN/FP=10)** | `0.0826` | **62.0%** `[48.0%, 76.0%]` | **11.9%** `[9.5%, 14.4%]` | **76.4%** `[73.7%, 79.1%]` | **229** `[203, 256]` | **8.4** `[6.9, 10.6]` |
| **v2: Target 80% OOF Recall** | `0.0581` | **78.0%** `[66.0%, 88.0%]` | **11.7%** `[10.0%, 13.5%]` | **69.9%** `[67.0%, 72.6%]` | **293** `[266, 321]` | **8.5** `[7.4, 10.0]` |
| **v2: High Precision** | `0.1365` | **32.0%** `[20.0%, 46.0%]` | **18.8%** `[12.1%, 26.3%]` | **92.9%** `[91.2%, 94.5%]` | **69** `[53, 86]` | **5.3** `[3.8, 8.3]` |

### Global Ranking & Calibration Metrics
- **ROC-AUC**: **0.7900** `[0.7184, 0.8477]` (Legacy v1: 0.8073)
- **PR-AUC**: **0.1822** `[0.1260, 0.2875]` (Legacy v1: 0.1691). The change in PR-AUC is within the bootstrap confidence interval.
- **Brier Score**: **0.0435** `[0.0415, 0.0454]` (vs raw uncalibrated 0.0658; ECE reduced to 0.0015).

### Confusion Matrix at Cost-Optimal Threshold ($\tau = 0.0826$)
```
                   Predicted Low Risk    Predicted High Risk (Flagged)
Actual No Stroke          743                         229          [Total: 972]
Actual Stroke              19                          31          [Total:  50]
```

---

## 5. Young Patient Subgroup Audit (Age < 50)

Across the entire dataset ($N=5,109$), stroke events in patients under 50 are extremely rare (**0.69%** overall prevalence, 20 total cases). However, risk rises monotonically with compounding vascular risk factors (`hypertension + heart_disease + smoker`):

- **Full Dataset ($N = 2,899$ young patients, 20 strokes, 0.69% base rate)**:
  - `risk_factors_count = 0`: $N = 2,099$, Strokes = $11$ ($0.52\%$, 5.2 per 1,000)
  - `risk_factors_count = 1`: $N = 761$, Strokes = $8$ ($1.05\%$, 10.5 per 1,000)
  - `risk_factors_count = 2`: $N = 39$, Strokes = $1$ ($2.56\%$, 25.6 per 1,000)
  - `risk_factors_count = 3`: $N = 0$, Strokes = $0$

- **Training Partition ($N = 2,316$ young patients, 12 strokes)**:
  - `RFC = 0`: $N = 1,687$, Strokes = $7$ ($0.41\%$)
  - `RFC = 1`: $N = 594$, Strokes = $4$ ($0.67\%$)
  - `RFC = 2`: $N = 35$, Strokes = $1$ ($2.86\%$)

- **Held-Out Test Partition ($N = 583$ young patients, 8 strokes)**:
  - `RFC = 0`: $N = 412$, Strokes = $4$ ($0.97\%$)
  - `RFC = 1`: $N = 167$, Strokes = $4$ ($2.40\%$)
  - `RFC = 2`: $N = 4$, Strokes = $0$ ($0.00\%$)

---

## 6. Model Zoo Comparison & Glass-Box Models

A controlled 5×5 repeated cross-validation benchmark across 13 model configurations demonstrates **no statistically significant difference** in discriminative ability between gradient boosted trees and interpretable glass-box models:

| Model Architecture | PR-AUC (OOF) | 95% CI PR-AUC | ROC-AUC (OOF) | 95% CI ROC-AUC | DeLong Test vs XGB ($p$-value) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost + scale_pos_weight** | 0.1809 | `[0.1473, 0.2296]` | 0.8204 | `[0.7955, 0.8458]` | Reference |
| **XGBoost (Unweighted)** | 0.1793 | `[0.1485, 0.2228]` | 0.8346 | `[0.8110, 0.8561]` | $p = 0.6887$ |
| **Balanced Bagging Classifier**| 0.1787 | `[0.1521, 0.2170]` | 0.8415 | `[0.8194, 0.8621]` | $p = 0.5489$ |
| **LightGBM** | 0.1740 | `[0.1442, 0.2206]` | 0.8153 | `[0.7909, 0.8390]` | $p = 0.8863$ |
| **HistGradientBoosting** | 0.1729 | `[0.1425, 0.2155]` | 0.8173 | `[0.7913, 0.8412]` | $p = 0.9304$ |
| **Logistic Regression (L1/L2)**| 0.1635 | `[0.1411, 0.1959]` | 0.8354 | `[0.8116, 0.8558]` | $p = 0.6824$ |
| **Explainable Boosting Machine (EBM)**| 0.1625 | `[0.1388, 0.1972]` | 0.8339 | `[0.8111, 0.8560]` | $p = 0.7070$ |
| **CatBoost** | 0.1509 | `[0.1277, 0.1795]` | 0.8177 | `[0.7931, 0.8389]` | $p = 0.9400$ |
| **XGBoost + SMOTE** | 0.1373 | `[0.1161, 0.1699]` | 0.8038 | `[0.7794, 0.8277]` | $p = 0.6540$ |

**Key Takeaways**:
1. Confidence intervals for PR-AUC heavily overlap across all competitive algorithms (XGBoost 0.181, LightGBM 0.174, Logistic Regression 0.164, EBM 0.163).
2. DeLong test $p$-values against XGBoost are all $\gg 0.05$, confirming that no single model is statistically superior.
3. Simpler glass-box models (EBM and Logistic Regression) achieve comparable discriminative performance while offering direct, unapproximated transparency.

---

## 7. Limitations & Honest Methodological Disclosures
1. **Low Positive Sample Size**: With only 249 positive stroke cases in total and 50 in the test set, metrics carry an unavoidable ±6–8% margin of error.
2. **False Alarm Reduction is Threshold-Driven**: The drop in false alarms from 372 to 229 (or 293) reflects the shift in the decision operating point, not inherent algorithmic superiority.
3. **External Validation Constraint (BRFSS Limitation)**: External validation on the CDC Behavioral Risk Factor Surveillance System (BRFSS) was omitted because BRFSS lacks serum glucose measurement (`avg_glucose_level`), exact employment categories (`work_type`), and geographic dwelling designation (`Residence_type`), and relies strictly on telephone self-report rather than EHR clinical records.
