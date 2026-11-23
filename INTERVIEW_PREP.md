# Interview Preparation & Project Deep-Dive Kit

> **Private Technical Interview Dossier for: Auditable Rare-Event Clinical Decision Support & Triage Engine**  
> *Author: Yagas Vashist ([dummycodertech](https://github.com/dummycodertech))*

---

## 🎯 High-Impact Resume Bullets

- **Engineered an auditable clinical rare-event triage microservice** for imbalanced electronic health records (4.87% prevalence), implementing cost-sensitive XGBoost with out-of-fold isotonic calibration (equal-quantile ECE: 1.90%, Brier: 0.0428) and achieving 62.0% sensitivity at an out-of-fold 10:1 cost-optimal threshold ($\tau = 0.0826$) on a held-out test cohort ($N=1,022$).
- **Architected an in-process dual-xAI interpretability pipeline** combining exact 5-fold TreeSHAP log-odds margin attribution with localized LIME linear surrogates, validated via an empirical 200-patient concordance audit (Spearman $r_s = 0.252$, top-3 overlap $33.2\%$, surrogate $R^2 = 0.842$).
- **Built and deployed a production clinical workstation** combining an asynchronous FastAPI service (sub-250ms p95 latency, anti-spoofed rate limiting, sanitized 422 error handlers) with a responsive Next.js 14 / Three.js 3D anatomical monitor and continuous PSI covariate drift surveillance.

---

## ⏱️ 30-Second Elevator Pitch
> *"I built an auditable clinical decision support system that tackles the 95% accuracy illusion in imbalanced EHR screening. Rather than relying on synthetic oversampling like SMOTE which distorts demographic distributions, I engineered a cost-sensitive XGBoost pipeline with out-of-fold isotonic calibration and pre-registered utility thresholds. The system achieves 62% recall at a 10:1 cost ratio on a held-out test set ($N=1,022$) and couples every score with exact TreeSHAP attributions, LIME bedside surrogate rules, and an interactive 3D Next.js workstation."*

---

## ⏱️ 2-Minute Technical Walkthrough Script
> *"In clinical risk triage, accuracy is a misleading metric. In population stroke screening with a 4.87% base rate, predicting negative for everyone gives 95.1% accuracy while missing every single event.
>
> To solve this without data leakage, I built a three-tier architecture:
> 
> First, on the modeling side, I used cost-sensitive XGBoost with negative-to-positive weighting (`scale_pos_weight = 19.54`) and enforced monotonic constraints on age and glucose. Because boosting margins do not reflect true empirical probabilities, I wrapped the pipeline in a 5-fold out-of-fold isotonic calibrator (`CalibratedClassifierCV`), which reduced Expected Calibration Error down to 1.9% on equal-quantile bins. Decision thresholds were then derived out-of-fold using a 10:1 cost utility matrix, yielding $\tau = 0.0826$.
> 
> Second, on explainability, doctors don't trust single black-box explanations. I built an in-process dual-xAI engine: exact 5-fold TreeSHAP for game-theoretic log-odds margin attribution across all booster trees, alongside a local LIME linear surrogate that translates margins into intuitive probability percentage points. I also published an empirical 200-patient concordance audit showing their correlation and primary feature overlap.
> 
> Finally, I deployed this as a decoupled production system: a high-performance FastAPI microservice on Render running stateless inference with zero patient logging, paired with a Next.js 14 and Three.js 3D anatomical workstation on Vercel with real-time SVG charts and rolling PSI drift surveillance."*

---

## 🧠 The 5 Critical Technical Defenses (Must-Know for Senior ML Interviews)

### 1. The `scale_pos_weight` vs. Calibration Interaction
- **Question**: *"If 5-fold isotonic calibration maps margins back to the 4.87% base rate, what does `scale_pos_weight = 19.54` actually accomplish? And why isn't it 10:1?"*
- **Defense**:
  - `19.54` is the dataset's **class imbalance ratio** ($4,860 / 249 \approx 19.52$), not the cost ratio. It scales the gradient and hessian during tree split finding, forcing trees to prioritize minority-class separation in deeper leaves rather than optimizing purely for 0-1 error.
  - Isotonic calibration then maps those shifted booster margins back to true posterior probabilities $P(Y=1|X)$.
  - The **10:1 clinical cost ratio** is enforced independently at the decision rule stage via the Bayes-optimal threshold $\tau = 0.0826$ ($\approx 1/(1+10) = 0.0909$).
  - Separating loss reweighting from thresholding ensures that output probabilities remain clinically calibrated while decision rules reflect hospital operational costs.

### 2. The Model Zoo Dilemma: Why XGBoost over EBM or Logistic Regression?
- **Question**: *"Your 5×5 CV benchmark shows Balanced Bagging (0.8415), Logistic Regression (0.8354), and EBM (0.8339) all matched or beat XGBoost (0.8204) on ROC-AUC with DeLong p-values > 0.50. Why keep XGBoost in production?"*
- **Defense**:
  - **Empirical Transparency**: Documenting that linear and glass-box models achieve parity on 5,109 tabular rows is an honest finding—not a flaw. On small tabular datasets, inductive biases of linear models often perform comparably to tree ensembles.
  - **Production Justification**: We retained XGBoost for three architectural reasons:
    1. **Monotonic Domain Constraints**: Enforcing `monotone_constraints=(1, 1)` on Age and Glucose prevents non-clinical risk dips between age brackets (which unconstrained GAMs/EBMs can produce in sparse bins).
    2. **Exact Polynomial-Time TreeSHAP**: XGBoost supports exact C++ `pred_contribs` across tree paths in sub-2ms, enabling instant margin decomposition for concurrent web clients without numerical approximation.
    3. **Non-Linear Interactions**: XGBoost models multi-way interactions (e.g. `Age × Glucose × Hypertension`) that Logistic Regression misses without manual interaction engineering.

### 3. Tree Depth & Generalization Gap (`max_depth = 8`)
- **Question**: *"Isn't `max_depth = 8` too deep for 199 positive training cases? Could that explain why OOF ROC-AUC was 0.8204 but held-out test ROC-AUC was 0.7900?"*
- **Defense**:
  - Yes. Optuna's Bayesian search selected `max_depth = 8` combined with aggressive regularization (`colsample_bytree = 0.526`, `subsample = 0.800`). However, on 199 positive training events, trees with depth 8 risk leaf sparsity.
  - The ~0.03 drop from OOF ROC-AUC (0.8204) to held-out test (0.7900) is consistent with finite-sample variance on 50 test positives (95% bootstrap CI: `[0.7184, 0.8477]`), but in a production refactor, constraining `max_depth` to 3–4 is a high-ROI architectural improvement to reduce variance.

### 4. Why TreeSHAP Age Contribution is Negative (-0.57) while LIME is Positive (+6.3%)
- **Question**: *"In your POST /predict example for a 62-year-old, TreeSHAP reports age contribution = -0.573, but LIME reports 'age > 61: +0.063'. Is there a sign bug?"*
- **Defense**:
  - No, this is an exact consequence of loss reweighting in TreeSHAP:
    - **TreeSHAP** operates on the raw margin of the tree booster. Because the model was trained with `scale_pos_weight = 19.54`, the booster's background training expectation is shifted to the re-weighted positive centroid (`base_value = +1.304`). In this dataset, the mean age of stroke patients is **67.7 years**. Relative to the positive stroke centroid, a 62-year-old patient is **5.7 years younger than average**, so age has a negative delta relative to the booster's elevated base margin!
    - **LIME** operates in the **calibrated probability space** relative to the general population. In the general population (mean age 43.2 years), being 62 years old significantly elevates stroke risk (+6.3 percentage points).
  - Highlighting this difference demonstrates mastery of explainability mechanics across dual reference frames.

### 5. Expected Calibration Error (ECE) Audit
- **Question**: *"How did you compute ECE, and why is equal-quantile preferable to equal-width?"*
- **Defense**:
  - In equal-width binning on a rare-event distribution ($4.87\%$ prevalence), 95% of samples fall into the first bin `[0, 0.10)`. Bins above 0.20 are virtually empty, giving a deceptively low equal-width ECE (0.0050 on test, 0.0015 on OOF).
  - To provide a methodologically robust metric, we evaluated **equal-quantile (equal-mass) ECE**: sorting predictions into 10 deciles of equal sample count ($N \approx 102$ per bin), yielding **ECE = 0.0190 (1.90%)** and **Brier score = 0.0428** on the held-out test cohort.

---

## 💼 Likely Interview Questions & Rapid Answers

| Question | Core Takeaway to Deliver |
| :--- | :--- |
| **Why not use SMOTE?** | SMOTE interpolates between minority vectors. Because stroke concentrates in elderly patients, SMOTE synthesizes unrealistic elderly vectors, blinding models to younger patients with metabolic risks. |
| **What is NNS (Number Needed to Screen)?** | $1 / \text{PPV}$. At $\tau=0.0826$, NNS is 8.4, meaning 8.4 patients are flagged for secondary clinical review to detect one true stroke. |
| **How did you prevent test leakage?** | Split indices were fixed before any preprocessing. Median imputation, feature transformers, Optuna tuning, isotonic calibration, and threshold selection were strictly out-of-fold. The test cohort was scored once. |
| **What is the difference between TreeSHAP and LIME?** | TreeSHAP guarantees game-theoretic consistency across all non-linear tree paths in log-odds margins. LIME fits a sparse linear surrogate around the patient's local neighborhood in probability percentage points. |
| **How do you monitor model drift?** | Using rolling 8-week Population Stability Index (PSI) tracking across covariates. A PSI $> 0.10$ triggers an alert; $> 0.25$ triggers retraining. |
