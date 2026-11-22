# Model Card: Stroke Risk Prediction & Triage System (v2 Rigorous Pipeline)

---

## 1. Model Details
- **Model Name**: Stroke Risk Clinical Triage Classifier (v2)
- **Model Architecture**: Tuned XGBoost (`max_depth=8`, `learning_rate=0.206`, `colsample_bytree=0.526`, monotonic constraints on age & glucose) wrapped in an Isotonic `CalibratedClassifierCV` (5-fold).
- **Version**: 2.0.0 (Upgrade over legacy heuristic v1)
- **Primary Developer**: Upgraded by Senior ML Engineer / Applied Statistics Reviewer for `dummycodertech/xai-xgboost-heart_Stroke`
- **License**: Pending user selection (see `LICENSE` file)
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

## 4. Performance Metrics (One-Time Held-Out Test Set, N=1,022, 50 Positives)

All metrics reported with **95% Bootstrap Confidence Intervals** (1,000 resamples):

| Metric | Measured Point Estimate | 95% Bootstrap Confidence Interval |
| :--- | :--- | :--- |
| **Operating Threshold** | `0.0826` | Selected OOF via cost model ($Cost_{FN} / Cost_{FP} = 10$) |
| **ROC-AUC** | **0.7900** | `[0.7184, 0.8477]` |
| **PR-AUC (Average Precision)** | **0.1822** | `[0.1260, 0.2875]` |
| **Recall (Sensitivity)** | **62.00%** (31/50) | `[48.00%, 76.00%]` |
| **Precision (PPV)** | **11.92%** | `[0.0946, 0.1440]` |
| **Specificity (TNR)** | **76.44%** (743/972)| `[0.7366, 0.7912]` |
| **Brier Score** | **0.0435** | `[0.0415, 0.0454]` |
| **F1 Score** | **0.2000** | `[0.1585, 0.2418]` |
| **F2 Score** | **0.3370** | `[0.2671, 0.4043]` |
| **Number Needed to Screen (NNS)** | **8.4 patients** | (Screen 8.4 flagged patients to detect 1 stroke) |
| **Population Flag Rate** | **25.4%** (260/1022) | `[22.89%, 28.08%]` |

### Confusion Matrix (Test Set)
```
                   Predicted Low Risk    Predicted High Risk (Flagged)
Actual No Stroke          743                         229
Actual Stroke              19                          31
```

---

## 5. Probability Calibration
- Uncalibrated models with heavy class-weight penalties (`scale_pos_weight=19.54`) severely distort predicted probabilities, making risk percentages clinically meaningless.
- **Isotonic Calibration** was selected via 5-fold CV based on superior Brier score (0.0427 vs raw 0.0658) and near-zero Expected Calibration Error (**ECE: 0.0015** vs raw 0.0648).
- Calibrated probabilities directly represent empirical event probabilities: a 8% predicted risk indicates an ~8 in 100 chance of stroke in a comparable demographic cohort.

---

## 6. Uncertainty Quantification & Conformal Prediction
- Incorporates **Mondrian Class-Conditional Conformal Prediction** targeting 90% coverage per class.
- **Empirical Coverage**: Overall = **90.1%**, Class 0 = **90.0%**, Class 1 = **93.0%**.
- **Ambiguous Prediction Sets**: 34.3% of patients fall into the ambiguous set `{No Stroke, Stroke}` (risk $p \in [0.0260, 0.1237]$). The application explicitly flags these borderline cases as "Ambiguous - Refer to Clinician" rather than forcing a fragile binary decision.

---

## 7. Subgroup Fairness & Equalized Odds Audit

Evaluated out-of-fold at the operating threshold `0.0826`:

| Subgroup | Sample Size ($N$) | Positives ($N_{pos}$) | Prevalence | Recall (TPR) | 95% CI Recall | False Positive Rate (FPR) | Flag Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age < 40** | 1,466 | 2 | 0.14% | 50.0% | `[0.0%, 100.0%]` | 1.8% | 1.9% |
| **Age 40–60** | 1,698 | 51 | 3.00% | 49.0% | `[35.3%, 62.7%]` | 17.1% | 18.0% |
| **Age > 60** | 923 | 146 | 15.82% | 77.4% | `[70.5%, 83.6%]` | 57.7% | 60.8% |
| **Gender: Female** | 2,410 | 113 | 4.69% | 75.2% | `[67.3%, 82.3%]` | 22.9% | 25.4% |
| **Gender: Male** | 1,677 | 86 | 5.13% | 60.5% | `[50.0%, 70.9%]` | 22.1% | 24.1% |

### Fairness Audit Findings:
- **Gender Equalized Odds**: FPR is well-balanced across genders (Female: 22.9%, Male: 22.1%, gap = **0.008**). However, recall is higher in females (75.2% vs 60.5%, TPR gap = **0.148**).
- **Age Disparity**: Stroke prevalence in patients under 40 is extremely low (0.14%, only 2 positive events in training data). Confidence intervals for young patient recall span the entire range `[0%, 100%]`. Clinicians must be aware that the statistical power for detecting strokes in patients under 40 is severely limited.

---

## 8. Limitations & Known Biases
1. **Low Positive Event Count**: The entire dataset contains only 249 stroke cases. The held-out test partition contains only 50 positive cases, resulting in wide confidence intervals (±6–8% on recall).
2. **Single-Source Synthetic / Survey Risk**: Kaggle healthcare stroke dataset provenance is unverified with unknown sampling frames. True population clinical prevalence differs from hospital cohorts.
3. **Missing BMI Values**: 201 missing BMI values were imputed with training median plus an explicit `bmi_missing` indicator.
4. **Smoking Unknowns**: 1,544 patients have "Unknown" smoking status, retained as a distinct categorical level rather than imputed.
5. **No External Validation**: The pipeline has not been validated on an external hospital EHR dataset. External validation (e.g. BRFSS survey) is detailed in Phase 9.

---

## 9. Ethical & Clinical Considerations
- High recall triage models intentionally generate False Positives (in this model, 229 false positives across 1,022 test patients, NNS = 8.4). 
- In clinical triage, False Positives trigger safe, non-invasive physical follow-ups (e.g., blood pressure check, carotid Doppler ultrasound, dietary review). False Negatives (missed strokes), by contrast, can lead to preventable death or irreversible neurologic deficit.
- The model's threshold was derived using a 10:1 clinical cost ratio to balance these competing harms transparently.
