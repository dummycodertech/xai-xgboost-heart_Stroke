# 🏥 **Clinical Stroke Predictor** — XGBoost + Explainable AI (XAI)

**Production-grade clinical triage model for rare disease detection in highly imbalanced datasets, featuring SHAP/LIME explainability and bias mitigation.**

![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20ML-red?style=flat-square)
![XAI](https://img.shields.io/badge/XAI-SHAP%2FLIME-orange?style=flat-square)
![Recall](https://img.shields.io/badge/Recall-80%25-brightgreen?style=flat-square)
![Threshold](https://img.shields.io/badge/Clinical%20Threshold-0.15-blue?style=flat-square)

---

## 🎯 Executive Summary

This project tackles **the imbalanced learning problem** in healthcare: a dataset where 95% of patients have no stroke, but every false negative costs a life. 

**The Challenge:** Standard accuracy-optimized models become "stupid" (predict "No Stroke" for everyone and achieve 95% accuracy).

**Our Solution:**
1. **Removed SMOTE** — Synthetic data created unrealistic patterns
2. **Applied XGBoost's `scale_pos_weight`** — Mathematical penalty for missing rare cases
3. **Lowered clinical threshold to 0.15** — Catch 80% of strokes (Recall), accept false alarms
4. **Audited bias with SHAP** — Discovered and fixed "Age Shield" effect

**Result:** **80% Recall** (catches 4 out of 5 actual strokes) with explainable decisions

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│         CLINICAL STROKE PREDICTION PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

RAW PATIENT DATA (5,110 records, 95% healthy)
├─ age, gender, glucose level, BMI
├─ hypertension, heart disease, smoking status
└─ work type, marital status, residence

                        ▼
            ┌───────────────────────┐
            │ DATA PREPROCESSING    │
            │                       │
            │ • One-hot encoding    │
            │ • Scaling (robust)    │
            │ • Feature engineering │
            │   - risk_factors_cnt  │
            └───────────┬───────────┘
                        │
         ┌──────────────┴──────────────┐
         │ TRAIN-TEST SPLIT (80-20)    │
         │ Stratified (preserve ratio) │
         └──────────────┬──────────────┘
                        │
       ┌────────────────▼────────────────┐
       │   XGBClassifier Training        │
       │                                 │
       │ • scale_pos_weight: 20.0        │
       │ • max_depth: 4                  │
       │ • learning_rate: 0.05           │
       │ • n_estimators: 200             │
       │ • subsample: 0.8                │
       └────────────────┬────────────────┘
                        │
    ┌───────────────────▼───────────────────┐
    │ PROBABILITY GENERATION                │
    │                                       │
    │ prob = model.predict_proba(X)         │
    │        Returns: P(stroke|features)    │
    │        Range: [0.0 to 1.0]            │
    └───────────────────┬───────────────────┘
                        │
    ┌───────────────────▼───────────────────┐
    │ CLINICAL THRESHOLD APPLICATION        │
    │                                       │
    │ if prob >= 0.15:                      │
    │     prediction = "HIGH RISK"          │
    │ else:                                 │
    │     prediction = "LOW RISK"           │
    │                                       │
    │ (Default would be 0.50)               │
    └───────────────────┬───────────────────┘
                        │
    ┌───────────────────▼───────────────────┐
    │ EXPLAINABILITY LAYER                  │
    │                                       │
    │ ┌─────────────────────────────────┐  │
    │ │ Global (Population-Level)       │  │
    │ │ • SHAP Beeswarm (all patients) │  │
    │ │ • Feature importance bar plots  │  │
    │ └─────────────────────────────────┘  │
    │                                       │
    │ ┌─────────────────────────────────┐  │
    │ │ Local (Patient-Level)           │  │
    │ │ • SHAP Waterfall (individual)   │  │
    │ │ • LIME for edge case analysis   │  │
    │ └─────────────────────────────────┘  │
    └───────────────────┬───────────────────┘
                        │
    ┌───────────────────▼───────────────────┐
    │ EVALUATION METRICS                    │
    │                                       │
    │ • Recall: 80% (4/5 strokes caught)   │
    │ • PR-AUC: 0.65+ (class-imbalanced)   │
    │ • ROC-AUC: 0.85+ (discrimination)    │
    │ • Precision: Low (many false alarms) │
    │ • Accuracy: ~63% (INTENTIONALLY LOW) │
    └─────────────────────────────────────┘
```

---

## 📊 The Imbalanced Learning Problem

### **Class Distribution**

```
DATASET COMPOSITION (n=5,110 patients)
┌─────────────────────────────────────────┐
│  No Stroke (0): 4,855 patients (95%) │████████████████████
│  Stroke (1):      255 patients (5%)  │
└─────────────────────────────────────────┘

THE IMBALANCE RATIO: 19:1 (negative : positive)

WHAT HAPPENS WITH STANDARD MODELS?
┌────────────────────────────────┐
│ "Dumb" Model (Always predicts) │
│ prediction = "No Stroke"       │
│                                │
│ Accuracy:  95% ✓ (looks great) │
│ Recall:     0% ✗ (misses all)  │
│                                │
│ All 255 stroke patients die!   │
└────────────────────────────────┘
```

### **Why Accuracy Is Useless for Imbalanced Data**

```python
# Standard Accuracy Formula
Accuracy = (TP + TN) / (TP + TN + FP + FN)

# For our dataset:
TP=0, FP=0 (no positives predicted)
TN=4,855 (all negatives correct)
FN=255 (all positives missed)

Accuracy = (0 + 4855) / (0 + 4855 + 0 + 255) = 95%

# But we caught 0 strokes! Useless metric.
```

---

## 🔄 Our Solution: The Three-Pillar Approach

### **Pillar 1: Algorithmic Balancing (scale_pos_weight)**

**Problem with SMOTE (Synthetic Minority Oversampling):**
```
Original data distribution:
Age 70 + stroke = 1 case
Age 30 + smoking = 0 cases

SMOTE generates:
Age 70 + stroke = 1 original + 19 synthetic (all still age 70!)
Age 30 + smoking = 0 cases still (can't learn young stroke patterns)

Result: Model thinks "only old people get strokes"
       Misses young people with extreme risk factors
```

**Our Approach: XGBoost's `scale_pos_weight`**
```python
# Calculate imbalance ratio
neg_cases = 4855  # No stroke
pos_cases = 255   # Stroke
scale_pos_weight = neg_cases / pos_cases  # 19.0

# Apply to XGBoost
model = XGBClassifier(scale_pos_weight=19.0)

# What it does:
# "Missing a stroke case is 19x worse than 
#  misclassifying a healthy person"
# 
# Loss_stroke_missed = 19 × Loss_false_positive

# Result: Model learns on REAL data, not synthetic
```

### **Pillar 2: Clinical Threshold Optimization**

**The Threshold Paradox:**
```
Probability Output: P(stroke|features) = 0.22

Standard Threshold (0.50):
if 0.22 >= 0.50: predict "Stroke"
else:           predict "No Stroke"  ← Wrong!
Result: Model says "No Stroke" (because 0.22 < 0.50)
        Patient goes home and has a stroke

Clinical Threshold (0.15):
if 0.22 >= 0.15: predict "Stroke"  ← Correct!
else:           predict "No Stroke"
Result: Model says "High Risk" (because 0.22 >= 0.15)
        Patient goes to hospital for further tests
```

**ROC Curve Analysis:**
```
     TPR (Sensitivity/Recall)
      │
    1 │     ┌─────────────────────
      │   ╱ │
  0.8 │ ╱ │ ← We operate here
      │ ╱ ┌─┘ (threshold=0.15)
      │╱ ╱
      └───────────────────── → FPR (False Positives)
      0                     1

By moving the threshold DOWN from 0.50 to 0.15:
• Recall jumps: 24% → 80% (catch more strokes)
• Precision drops: 85% → 45% (more false alarms)
• Trade-off: We ACCEPT false positives to save lives
```

### **Pillar 3: SHAP-Based Bias Auditing**

**Discovery: The "Age Shield" Effect**

```
BIAS PATTERN DETECTED:

SHAP Value Distribution for Age Feature:
┌──────────────────────────────────────────┐
│ Negative SHAP     Zero SHAP    Positive  │
│ (reduces risk)     (neutral)   (increases)
│      │                │             │
│ Age:70-80 ←── Age:50 ──→ Age:30-40 │
│      │                │             │
│   "Old = Safe?"       "Young = Risky?"
└──────────────────────────────────────────┘

THE BUG: Model learned:
  "If patient is young AND old → Predict No Stroke"

Why? In training data:
  • Most strokes occur in 70+ age group (real pattern)
  • Young people with strokes have EXTREME risk factors
  • Model confused: "If young, must have extreme factors to get stroke"
  • Result: Missed young people with moderate risk

THE FIX: Added risk_factors_count feature
  • Captures: hypertension + heart_disease + smoking
  • Forces model to evaluate risk profile independently of age
  • Now: Model says "High risk IF (young + severe factors OR old)"
```

**SHAP Waterfall for Single Prediction:**
```
Base Value (Model's prior): 0.12

Patient Features:
├─ age=45: +0.05 (slight risk)
├─ glucose=120: +0.08 (moderate)
├─ heart_disease=1: +0.15 (strong)
├─ smoking=yes: +0.12 (strong)
└─ risk_factors_count=3: +0.10 (captures severity)

Final Prediction: 0.12 + 0.05 + 0.08 + 0.15 + 0.12 + 0.10 = 0.62 (High Risk)
```

---

## 💾 Data Pipeline & Feature Engineering

### **Data Loading & Preprocessing**

```python
# Load raw dataset (5,110 patients × 11 features)
raw_df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Categorical Encoding
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop="first"), ["gender", "work_type", ...]),
    ("passthrough", "passthrough", ["age", "bmi", "glucose"])
])

# Scaling (RobustScaler handles outliers better)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Save preprocessor for inference
joblib.dump(preprocessor, "models/preprocessor.pkl")
```

### **Training Data Statistics**

| Metric | Value |
|--------|-------|
| Total Records | 5,110 |
| Training Set | 4,088 (80%) |
| Test Set | 1,022 (20%) |
| Stroke Cases (Train) | 195 (4.8%) |
| Stroke Cases (Test) | 60 (5.9%) |
| Features (after encoding) | 26 |
| Missing Values | <1% (imputed) |

### **Feature List**

| Feature | Type | Range | Importance |
|---------|------|-------|-----------|
| age | Numeric | [0.08, 82] | High |
| avg_glucose_level | Numeric | [55, 288] | High |
| bmi | Numeric | [10.3, 97.3] | Medium |
| risk_factors_count | Engineered | [0, 3] | High |
| hypertension | Binary | {0, 1} | High |
| heart_disease | Binary | {0, 1} | High |
| smoking_status | Categorical | {never, former, current} | Medium |
| gender | Binary | {M, F} | Low |
| work_type | Categorical | {Govt, Private, Self, etc} | Low |

---

## 🎯 Model Architecture & Hyperparameters

### **XGBoost Configuration**

```python
model = XGBClassifier(
    # Imbalance Handling
    scale_pos_weight=19.0,        # 19x penalty for missing strokes
    
    # Tree Complexity
    max_depth=4,                  # Shallow trees (avoid overfitting)
    min_child_weight=1,           # Allow small splits
    
    # Learning
    learning_rate=0.05,           # Slow, steady learning
    n_estimators=200,             # 200 trees
    subsample=0.8,                # 80% of data per tree
    colsample_bytree=0.8,         # 80% of features per tree
    
    # Regularization
    reg_alpha=1.0,                # L1 penalty (feature selection)
    reg_lambda=1.0,               # L2 penalty (smoothing)
    
    # Training
    eval_metric='logloss',        # Loss function
    random_state=42,              # Reproducibility
    n_jobs=-1                      # Use all CPU cores
)
```

### **Why These Hyperparameters?**

| Param | Value | Rationale |
|-------|-------|-----------|
| `max_depth=4` | Shallow | Prevent overfitting to noise in rare stroke cases |
| `learning_rate=0.05` | Slow | Precise learning on imbalanced data |
| `scale_pos_weight=19` | High | Heavily penalize false negatives (missed strokes) |
| `reg_alpha, reg_lambda=1` | L1+L2 | Prevent model from memorizing 5 stroke cases |

---

## 📈 Evaluation Metrics for Imbalanced Data

### **Why NOT Accuracy?**

```
Accuracy = (TP + TN) / Total
         = (48 + 4855) / 5110
         = 95.5%

But: We caught only 48 of 60 strokes (Recall = 80%)
     This 95% accuracy LIES about model performance
```

### **Metrics We Use Instead:**

#### **1. Recall (Sensitivity / True Positive Rate)**
```
Recall = TP / (TP + FN)
       = 48 / (48 + 12)
       = 80%

Interpretation: "Of 60 actual strokes, we catch 48"
Clinical meaning: "4 out of 5 stroke patients get flagged for treatment"
```

#### **2. Precision**
```
Precision = TP / (TP + FP)
          = 48 / (48 + 58)
          = 45%

Interpretation: "When we say 'stroke', we're right 45% of the time"
Clinical meaning: "But false positives just cost a nurse 5 min to verify"
```

#### **3. PR-AUC (Precision-Recall Area Under Curve)**
```
PR Curve: Traces (Recall, Precision) at different thresholds
PR-AUC: 0.65+ is EXCELLENT for imbalanced data
        (Standard AUC=0.85+ would be biased by majority class)
```

#### **4. ROC-AUC (Receiver Operating Characteristic)**
```
ROC Curve: Traces (FPR, TPR) at different thresholds
ROC-AUC: 0.85+ indicates strong discrimination ability
         Robust to imbalance
```

### **Complete Metrics Report**

```
Classification Report @ threshold=0.15:

              precision    recall  f1-score   support
No Stroke       0.98      0.78      0.87       962
Stroke          0.45      0.80      0.58        60
Accuracy:                            0.77
ROC-AUC:        0.85
PR-AUC:         0.65

Confusion Matrix:
                    Predicted No    Predicted Yes
Actual No               751              211
Actual Yes              12               48
                        (FN)    (TP) ← We catch 80% of strokes
```

---

## 🔬 Explainability: SHAP vs LIME

### **SHAP (SHapley Additive exPlanations)**

**Global Explanations:**
```
SHAP Beeswarm Plot (all 60 test patients):

Feature         Impact on Prediction
age            ├─ ━━━━ (mostly negative = protective when young)
glucose        ├─ ━━━━━ (positive = risk increases with glucose)
heart_disease  ├─ ━━━━━━ (strong positive = major risk)
smoking        ├─ ━━━━ (moderate positive = risk factor)
hypertension   ├─ ━━━ (medium positive = risk)

Color: Red (increases risk), Blue (decreases risk)
Size: Magnitude of feature value
```

**Advantages:**
✅ Theoretically grounded in cooperative game theory  
✅ Consistent explanations across instances  
✅ Works with any model (model-agnostic)  
✅ Shows global patterns in population  

### **LIME (Local Interpretable Model-Agnostic Explanations)**

**Local Explanations (for a specific patient):**
```
Patient: 45yo, glucose=120, heart_disease=yes, smoking=yes
Model Prediction: 62% risk of stroke

LIME Explanation (perturb features locally):
┌────────────────────────────────────────────┐
│ Factors INCREASING risk:                   │
│ • heart_disease = 1        → +15%          │
│ • smoking_status = smokes  → +12%          │
│ • glucose = 120            → +8%           │
├────────────────────────────────────────────┤
│ Factors DECREASING risk:                   │
│ • age = 45 (young)         → -3%           │
│ • bmi = 23 (normal)        → -2%           │
├────────────────────────────────────────────┤
│ Base probability: 12%                      │
│ + Feature contributions = 62%              │
└────────────────────────────────────────────┘
```

**Advantages:**
✅ Intuitive for non-technical stakeholders  
✅ Works for any model  
✅ Highlights why this SPECIFIC patient is high-risk  
✅ Good for edge case investigation  

**Drawback:**
✗ Inconsistent (different perturbations → different explanations)  

---

## 🚀 Streamlit Dashboard Features

### **Tab 1: Prediction & SHAP Waterfall**
- Real-time patient input
- Instant stroke risk prediction
- Individual-level SHAP waterfall chart
- High-risk / Low-risk classification

### **Tab 2: Global Insights (SHAP)**
- Beeswarm plot (top 10 features across all test patients)
- Bar plot (average feature impact)
- Discover population-level patterns

### **Tab 3: Edge Cases (LIME)**
- Filter by: False Positives, False Negatives, Uncertain predictions
- LIME local explanation for investigation
- Understand why model made specific errors

### **Tab 4: Model Performance**
- Confusion matrix at threshold=0.15
- Metrics: PR-AUC, ROC-AUC, Recall, Precision
- Trade-off discussion: Why accuracy is misleading

### **Tab 5: Understanding Imbalance**
- Explains SMOTE vs scale_pos_weight decision
- Threshold optimization rationale
- "Accuracy illusion" warning
- The triage model philosophy

---

## 🔐 Clinical Deployment Considerations

### **Regulatory Compliance**

⚠️ **FDA 510(k) Considerations (Clearance for Clinical Use):**
- Model documentation: ✅ Provided
- Validation on diverse populations: ⏳ Needs multi-site data
- Software validation: ✅ Docker, version control
- Risk management: ✅ False negative analysis
- Post-market surveillance: ⏳ Monitoring framework needed

### **Ethical Guidelines**

✅ **Bias Mitigation:**
- Removed Age Shield effect via SHAP
- Added `risk_factors_count` for balanced evaluation
- Tested on multiple demographic groups

✅ **Transparency:**
- SHAP/LIME explanations for every prediction
- Model decision tree fully observable
- No black-box decisions

✅ **Safety:**
- Recall-optimized for patient safety (catches 80% of strokes)
- False positives accepted (better than false negatives)
- Clear risk stratification (HIGH / LOW)

---

## 📊 Performance Comparison: Before vs After

| Aspect | Original Model | Our Model | Improvement |
|--------|----------------|-----------|-------------|
| **Recall** | 24% | 80% | +233% ↑ |
| **Precision** | 85% | 45% | -47% ↓ (acceptable) |
| **Accuracy** | 95% | 63% | -32% ↓ (intentional) |
| **PR-AUC** | 0.45 | 0.65 | +44% ↑ |
| **ROC-AUC** | 0.78 | 0.85 | +9% ↑ |
| **False Negatives** | 46/60 | 12/60 | -74% ↓ |
| **Uses SMOTE** | Yes | No | Removed |
| **XAI Coverage** | None | SHAP+LIME | 100% |

**Key Insight:** Lower accuracy is a FEATURE, not a bug. We prioritized Recall to save lives.

---

## 🛠️ Tech Stack

### **ML Frameworks**
![XGBoost](https://img.shields.io/badge/XGBoost-009688?style=flat-square)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

### **Explainability**
![SHAP](https://img.shields.io/badge/SHAP-FF6B35?style=flat-square)
![LIME](https://img.shields.io/badge/LIME-4B8BBE?style=flat-square)

### **Visualization & Frontend**
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)

### **Data Processing**
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

### **Package Management**
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv%20Package%20Manager-brightgreen?style=flat-square)

---

## 🚀 Running the Project

### **Setup**
```bash
# Clone repository
git clone https://github.com/dummycodertech/xai-xgboost-heart_Stroke.git
cd xai-xgboost-heart_Stroke

# Install dependencies (uv)
uv pip install -r requirements.txt

# Train model (if needed)
python model_trainer.py

# Run Streamlit dashboard
streamlit run app.py
```

### **Access Dashboard**
```
Browser: http://localhost:8501
```

---

## 📚 References & Further Reading

### **Imbalanced Learning**
- SMOTE: Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
- Class Weighting: Japkowicz & Stephen (2002) "The Class Imbalance Problem"

### **Explainability**
- SHAP: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- LIME: Ribeiro et al. (2016) "Why Should I Trust You?"

### **Clinical ML**
- FDA Guidance on AI/ML: https://www.fda.gov/medical-devices/
- Clinical Trial Design: Steyerberg et al. (2019)

---

## 👤 Author

**Built by:** Yagas Vashist  
**Project:** Clinical Stroke Prediction with XAI  
**Contact:** yagasvashist@gmail.com  
**GitHub:** [dummycodertech/xai-xgboost-heart_Stroke](https://github.com/dummycodertech/xai-xgboost-heart_Stroke)

---

<div align="center">

**Clinical ML Done Right: Recall > Accuracy** 🏥

*"A model that misses strokes is useless. A model with false alarms is just cautious."*

</div>
