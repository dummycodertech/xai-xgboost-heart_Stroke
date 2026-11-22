# Stroke Risk Prediction & Explainable AI (XAI) System

> **A clinical triage machine learning pipeline and interactive diagnostic dashboard that rejects high-accuracy majority-class bias to achieve 80.0% stroke recall using cost-sensitive XGBoost, dual SHAP/LIME interpretability, and post-hoc edge-case analysis.**

[![Python](https://img.shields.io/badge/Python-3.14%20%7C%203.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-EB5424?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.51.0-blue?style=flat-square)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-0.2.0.1-green?style=flat-square)](https://github.com/marcotcr/lime)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Environment-uv](https://img.shields.io/badge/Packaging-uv-blueviolet?style=flat-square)](https://github.com/astral-sh/uv)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg?style=flat-square)](https://github.com/dummycodertech/xai-xgboost-heart_Stroke)

---

### Quick Links
- **GitHub Repository**: [dummycodertech/xai-xgboost-heart_Stroke](https://github.com/dummycodertech/xai-xgboost-heart_Stroke)
- **Live Demo**: [VERIFY: Enter live Streamlit Community Cloud URL if active, e.g., https://xai-stroke-prediction.streamlit.app/]
- **Dataset Source**: [Kaggle Healthcare Stroke Dataset Mirror](https://raw.githubusercontent.com/karavokyrismichail/Stroke-Prediction---Random-Forest/main/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv)

---

### Dashboard Preview
![Streamlit Clinical Dashboard Overview](docs/screenshots/dashboard_tab1.png)
*Note: To capture this visual, launch `streamlit run app.py`, fill in sample patient features, click "Predict Risk", and save the resulting view to `docs/screenshots/dashboard_tab1.png`.*

---

## 📋 Project at a Glance

| Attribute | Details |
| :--- | :--- |
| **Problem** | Standard stroke prediction models suffer from the "accuracy illusion": in a dataset where 95.1% of patients do not suffer a stroke, a naive classifier predicting "No Stroke" achieves 95.1% accuracy while missing 100% of life-threatening events. Furthermore, synthetic oversampling (SMOTE) creates ungrounded 70-year-old archetypes, failing to alert clinicians to younger patients with compounding lifestyle risk factors. |
| **Solution** | A cost-sensitive ML pipeline with XGBoost that explicitly weights minority stroke cases by 19.54× (`scale_pos_weight`), incorporates custom feature engineering (`risk_factors_count`), adjusts the clinical classification threshold to 0.15, and pairs patient risk scoring with dual local/global XAI engines (SHAP TreeExplainer and LIME Tabular Explainer). |
| **Target User** | Clinical decision support staff, triage nurses, medical researchers, and health tech machine learning engineers seeking auditable, high-recall risk screening. |
| **Project Type** | Applied Supervised Machine Learning / Explainable AI (XAI) / Interactive Diagnostic Web Application. |
| **Status** | Active Prototype / Research Demonstration. Pipeline fully trained and pre-computed artifacts serialized. |
| **Deployment** | Local Streamlit runtime / Configured for Streamlit Community Cloud (model artifacts and datasets explicitly un-ignored in `.gitignore`). |
| **Tech Stack** | Python 3.14 / 3.11+, XGBoost 3.2.0, SHAP 0.51.0, LIME 0.2.0.1, Scikit-Learn 1.8.0, Dill 0.4.1, Pandas 3.0.3, Streamlit 1.57.0, UV package manager. |
| **Role & Scope** | Built by **Yagas Vashist** ([dummycodertech](https://github.com/dummycodertech)). Implemented complete pipeline: data ingestion, stratified preprocessing, hyperparameterized model training, XAI artifact serialization, edge-case mining, and Streamlit user interface. |
| **Duration / Date** | Initial Release: May 19, 2026. Rigorous Upgrade (v2): November 2026. |
| **Key Measured Results** | **v2 Rigorous Pipeline (Calibrated + OOF Cost Threshold)**: **62.0% Stroke Recall (31/50 test strokes detected, 95% CI: [48.0%, 76.0%])**, **PR-AUC: 0.1822 [0.1260, 0.2875]**, **ROC-AUC: 0.7900 [0.7184, 0.8477]**, **Brier Score: 0.0435 [0.0415, 0.0454]**, **Precision: 11.9% [9.5%, 14.4%]**, **NNS: 8.4** (Operating threshold $\tau = 0.0826$ selected out-of-fold via cost ratio 10:1; zero test leakage).<br>*(Legacy v1 Baseline: 80.0% Recall (40/50), ROC-AUC: 0.8073, PR-AUC: 0.1691, threshold 0.15 chosen via test-set peeking).* |
| **Repository** | [https://github.com/dummycodertech/xai-xgboost-heart_Stroke](https://github.com/dummycodertech/xai-xgboost-heart_Stroke) |

---

## 📑 Table of Contents
1. [Overview](#-overview)
   - [The Problem](#the-problem)
   - [The Solution](#the-solution)
   - [Key Features](#key-features)
   - [Scope & Boundary](#scope--boundary)
2. [Architecture](#-architecture)
   - [High-Level System Diagram](#high-level-system-diagram)
   - [End-to-End Execution Flow](#end-to-end-execution-flow)
   - [Component Inventory](#component-inventory)
   - [Machine Learning Specification](#machine-learning-specification)
   - [Web Application Architecture](#web-application-architecture)
3. [Tech Stack](#-tech-stack)
4. [Design Decisions & Trade-offs](#-design-decisions--trade-offs)
5. [Data & Models](#-data--models)
   - [Dataset Profile](#dataset-profile)
   - [Data Dictionary](#data-dictionary)
   - [Model Card: XGBoost Stroke Classifier](#model-card-xgboost-stroke-classifier)
6. [Evaluation & Results](#-evaluation--results)
   - [Evaluation Methodology](#evaluation-methodology)
   - [Performance Metrics Table](#performance-metrics-table)
   - [Confusion Matrix & Clinical Interpretation](#confusion-matrix--clinical-interpretation)
   - [Edge Case & Failure Mode Analysis](#edge-case--failure-mode-analysis)
   - [Reproduction Steps](#reproduction-steps)
7. [Testing & Quality](#-testing--quality)
8. [Security, Privacy & Reliability](#-security-privacy--reliability)
9. [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation via uv or venv](#installation-via-uv-or-venv)
   - [Smoke Test](#smoke-test)
10. [Usage](#-usage)
    - [Interactive Streamlit UI](#interactive-streamlit-ui)
    - [Running Offline Pipeline Modules](#running-offline-pipeline-modules)
11. [Configuration](#-configuration)
12. [Project Structure](#-project-structure)
13. [Deployment](#-deployment)
14. [Performance & Resource Profile](#-performance--resource-profile)
15. [Roadmap & Future Scope](#-roadmap--future-scope)
16. [Frequently Asked Questions](#-frequently-asked-questions)
17. [Resume & Interview Kit](#-resume--interview-kit)
18. [Project Context (LLM Agent Ingestion)](#-project-context-llm-agent-ingestion)
19. [References & Acknowledgements](#-references--acknowledgements)

---

## 🔬 Overview

### The Problem
Stroke is a leading cause of long-term disability and mortality worldwide. Early triage can dramatically improve patient survival and functional recovery. However, applying off-the-shelf classification algorithms to electronic health records introduces severe clinical failure modes:

1. **The 95% "Accuracy Illusion"**: In population stroke screening datasets (such as the Kaggle / fedesoriano dataset), positive stroke prevalence is roughly 4.87% (249 positives out of 5,110 records). A trivial model predicting 100% negative outcomes obtains 95.13% classification accuracy while possessing a clinical recall of 0.0%.
2. **The SMOTE Synthetic Bias Trap**: Synthetic Minority Over-sampling Technique (SMOTE) generates interpolated synthetic vectors. Because stroke incidence strongly correlates with advanced age, SMOTE primarily synthesizes elderly profiles (e.g., 70+ years old). This causes the classifier to heavily over-index on age alone, remaining blind to younger patients (30–50 years old) presenting severe compounding lifestyle risks (smoking, hypertension, diabetes).
3. **Black-Box Skepticism**: Physicians and triage personnel cannot act upon raw, uncalibrated risk probabilities without understanding *which specific physiological and lifestyle factors* contributed to an elevated risk score.

### The Solution
This repository implements an end-to-end Machine Learning and Explainable AI (XAI) triage framework:
- **Cost-Sensitive XGBoost**: Discards synthetic oversampling in favor of the real, unaltered training distribution, applying an exact `scale_pos_weight = 19.54` to penalize false negatives 20× higher than false alarms.
- **Compounding Risk Feature Engineering**: Introduces `risk_factors_count = hypertension + heart_disease + high_risk_smoker` to capture non-linear multivariable risk interactions.
- **Clinical Triage Thresholding**: Replaces the default statistical 0.50 cutoff with a calibrated 0.15 threshold, driving stroke recall to 80.0% on the held-out test set.
- **Dual XAI Interpretation Suite**: 
  - **SHAP (TreeExplainer)**: Computes exact Shapley values for both global population dynamics (Beeswarm, Bar plots) and instant per-patient local waterfall attribution.
  - **LIME (LimeTabularExplainer)**: Provides perturbation-based local feature importance specifically targeted at diagnosed model edge cases (False Positives, False Negatives, and 45–55% Uncertain boundaries).
- **Interactive Multi-Tab Dashboard**: Built on Streamlit to allow real-time clinical parameter adjustments, instant attribution rendering, and model transparency exploration.

### Key Features

| Feature | What It Does | Code Location |
| :--- | :--- | :--- |
| **Automated Data Ingestion & Imputation** | Downloads raw dataset, strips non-predictive identifiers (`id`), drops ambiguous gender records (`Other`), imputes missing BMI using median statistics, and performs stratified 80/20 train/test splitting. | [`data_loader.py`](data_loader.py) |
| **Risk Multiplier Feature Engineering** | Computes a composite lifestyle risk score (`risk_factors_count`) combining hypertension, heart disease, and active/former smoking status. | [`data_loader.py:43-48`](data_loader.py#L43-L48) |
| **Cost-Sensitive XGBoost Training** | Dynamically calculates negative-to-positive class ratio (`neg_cases / pos_cases`), configures `scale_pos_weight`, and trains a shallow gradient-boosted decision tree ensemble (`max_depth=4`, `n_estimators=200`). | [`model_trainer.py:32-52`](model_trainer.py#L32-L52) |
| **0.15 Clinical Decision Thresholding** | Enforces a customized low-cutoff triage threshold to catch 80% of true stroke cases on 1,022 test samples. | [`model_trainer.py:23, 54-80`](model_trainer.py#L23) |
| **Global SHAP Pre-computation** | Computes tree-based Shapley explanations over all 1,022 test samples and serializes artifacts for instant UI rendering. | [`shap_explainer.py:27-50`](shap_explainer.py#L27-L50) |
| **Dill-Serialized LIME Engine** | Instantiates a tabular perturbation explainer and leverages `dill` to preserve lambda functions and internal estimators for runtime use. | [`lime_expl.py:25-38`](lime_expl.py#L25-L38) |
| **Automated Edge-Case Mining** | Mines and isolates top 5 False Positives, False Negatives, and 45–55% confidence borderline decisions for clinical audit. | [`edgecasedec.py:25-63`](edgecasedec.py#L25-L63) |
| **Interactive 5-Tab Diagnostic UI** | Streamlit application featuring live parameter inference, global SHAP plots, LIME edge-case dissection, confusion matrix trade-offs, and methodology writeups. | [`app.py`](app.py) |

### Scope & Boundary

- **In Scope**:
  - Tabular electronic health record preprocessing, one-hot encoding, and standard scaling.
  - Cost-sensitive gradient boosted tree modeling with class weight calibration.
  - Explainable AI attributions (SHAP TreeExplainer and LIME TabularExplainer).
  - Post-hoc edge-case isolation and clinical confusion matrix analysis.
  - Interactive desktop/cloud Streamlit web dashboard.
- **Out of Scope**:
  - Live EHR / HL7 / FHIR integration pipelines.
  - HIPAA-compliant encrypted patient storage or user authentication (designed as an offline triage tool/prototype).
  - Unstructured clinical notes (NLP) or radiological neuroimaging (CT/MRI) classification.
  - Autonomous medication prescribing or unmonitored diagnostic automation.

---

## 🏗️ Architecture

### High-Level System Diagram

```mermaid
flowchart TD
    subgraph Data_Pipeline ["1. Data Pipeline (data_loader.py)"]
        RawURL["GitHub CSV Raw Mirror"] --> RawCSV["data/healthcare-dataset-stroke-data.csv<br/>(5,110 rows, 12 cols)"]
        RawCSV --> PreprocStep["Preprocess & Clean:<br/>Drop 'id', Drop gender 'Other'<br/>Feature Eng: risk_factors_count"]
        PreprocStep --> SklearnPipe["ColumnTransformer<br/>- Numeric: Median Imputer + StandardScaler<br/>- Categorical: Mode Imputer + OneHotEncoder"]
        SklearnPipe --> SerializedSets["Serialized Artifacts:<br/>data/train_processed.pkl (4,087 samples)<br/>data/test_processed.pkl (1,022 samples)<br/>models/preprocessor.pkl"]
    end

    subgraph Training_Pipeline ["2. Model Training (model_trainer.py)"]
        SerializedSets --> XGBTrain["XGBClassifier Training<br/>scale_pos_weight = 19.54<br/>max_depth=4, lr=0.05, n_est=200"]
        XGBTrain --> ModelEval["Evaluation @ 0.15 Threshold<br/>Recall: 80.0%, ROC-AUC: 0.8073"]
        ModelEval --> ModelArtifacts["models/xgboost_model.pkl<br/>models/training_metadata.json"]
    end

    subgraph XAI_Pipeline ["3. Interpretability & Audit Engines"]
        ModelArtifacts --> SHAPModule["shap_explainer.py<br/>shap.TreeExplainer"]
        SerializedSets --> SHAPModule
        SHAPModule --> SHAPArtifacts["models/shap_explainer.pkl<br/>models/shap_values_test.pkl<br/>plots/shap_summary_beeswarm.png<br/>plots/shap_summary_bar.png"]
        
        ModelArtifacts --> LIMEModule["lime_expl.py<br/>lime.lime_tabular.LimeTabularExplainer"]
        SerializedSets --> LIMEModule
        LIMEModule --> LIMEArtifacts["models/lime_explainer.pkl (via dill)"]

        ModelArtifacts --> EdgeModule["edgecasedec.py<br/>FP, FN, Borderline Mining"]
        SerializedSets --> EdgeModule
        EdgeModule --> EdgeArtifacts["models/edge_cases.json"]
    end

    subgraph Presentation_Layer ["4. Interactive Application (app.py)"]
        ModelArtifacts --> StreamlitApp["Streamlit Dashboard<br/>st.cache_resource Artifact Ingestion"]
        SHAPArtifacts --> StreamlitApp
        LIMEArtifacts --> StreamlitApp
        EdgeArtifacts --> StreamlitApp

        User["Clinician / Triage User"] <-->|"Tab 1: Patient Sliders & Live SHAP Waterfall"| StreamlitApp
        User <-->|"Tab 2: Population Beeswarm & Bar Charts"| StreamlitApp
        User <-->|"Tab 3: Edge Case Inspection & LIME Explanations"| StreamlitApp
        User <-->|"Tab 4: Triage Trade-off & Confusion Matrix"| StreamlitApp
        User <-->|"Tab 5: Architecture & Methodological Post-Mortem"| StreamlitApp
    end

    classDef store fill:#f9f0ff,stroke:#6b46c1,stroke-width:1px;
    classDef proc fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
    classDef ui fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;

    class RawCSV,SerializedSets,ModelArtifacts,SHAPArtifacts,LIMEArtifacts,EdgeArtifacts store;
    class PreprocStep,SklearnPipe,XGBTrain,ModelEval,SHAPModule,LIMEModule,EdgeModule proc;
    class StreamlitApp,User ui;
```

### End-to-End Execution Flow

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Clinician / User
    participant UI as app.py (Streamlit)
    participant Cache as @st.cache_resource
    participant Preproc as ColumnTransformer (preprocessor.pkl)
    participant Model as XGBoost (xgboost_model.pkl)
    participant SHAP as TreeExplainer (shap_explainer.pkl)
    participant LIME as LimeTabularExplainer (lime_explainer.pkl)

    Note over Clinician,UI: Application Initialization
    UI->>Cache: load_artifacts()
    Cache-->>UI: Return in-memory model, explainer, test data, and metadata

    Note over Clinician,UI: Tab 1: Live Patient Prediction & Local SHAP
    Clinician->>UI: Enter vitals (Age: 45, Glucose: 120, BMI: 28, Smoking: smokes)
    Clinician->>UI: Click "Predict Risk"
    UI->>UI: Compute risk_factors_count (hypertension + heart_disease + smoker)
    UI->>Preproc: transform(patient_dataframe)
    Preproc-->>UI: 23-dimensional scaled & one-hot vector
    UI->>Model: predict_proba(processed_input)
    Model-->>UI: Probability (e.g., 0.184)
    UI->>UI: Evaluate: 0.184 >= 0.15 threshold -> HIGH RISK
    UI->>SHAP: explainer(processed_input)
    SHAP-->>UI: Shapley explanation vector
    UI-->>Clinician: Render risk badge (HIGH RISK) and SHAP Waterfall chart

    Note over Clinician,UI: Tab 3: Edge Case Exploration
    Clinician->>UI: Select "false_negatives" -> Case 1
    UI->>UI: Retrieve test record index from edge_cases.json
    UI->>LIME: explain_instance(test_record, predict_fn, num_features=8)
    LIME-->>UI: LIME pyplot figure
    UI-->>Clinician: Display LIME horizontal bar plot showing contradictory feature forces
```

### Component Inventory

| Component | Responsibility | Source File | Primary Input | Primary Output |
| :--- | :--- | :--- | :--- | :--- |
| **Data Loader** | Ingestion, data cleaning, feature engineering, and stratified partitioning. | [`data_loader.py`](data_loader.py) | Raw CSV (`data/healthcare-dataset-stroke-data.csv`) | `train_processed.pkl`, `test_processed.pkl`, `preprocessor.pkl` |
| **Model Trainer** | Calculates class imbalances, trains cost-sensitive XGBoost, evaluates at threshold 0.15, and serializes models. | [`model_trainer.py`](model_trainer.py) | `train_processed.pkl`, `test_processed.pkl` | `xgboost_model.pkl`, `training_metadata.json` |
| **SHAP Explainer Setup** | Computes population-wide Shapley values across the test set; saves explainer and test summary plots. | [`shap_explainer.py`](shap_explainer.py) | `xgboost_model.pkl`, `test_processed.pkl` | `shap_explainer.pkl`, `shap_values_test.pkl`, `plots/*.png` |
| **LIME Explainer Setup** | Fits a tabular perturbation model across training features; serializes with `dill` to preserve runtime state. | [`lime_expl.py`](lime_expl.py) | `train_processed.pkl` | `lime_explainer.pkl` |
| **Edge Case Detector** | Evaluates test predictions to isolate false positives, false negatives, and uncertain decision boundaries. | [`edgecasedec.py`](edgecasedec.py) | `xgboost_model.pkl`, `test_processed.pkl` | `edge_cases.json` |
| **Streamlit Interface** | Serves the 5-tab user interface, binds cached models to reactive UI elements, and renders XAI charts. | [`app.py`](app.py) | All artifacts in `models/` and `data/` | Rendered web UI on `localhost:8501` |

### Machine Learning Specification

#### Data Schema & Preprocessing Pipeline
- **Raw Dimensions**: 5,110 rows × 12 features.
- **Cleaning Step 1**: Drop column `id` (random patient ID, no predictive signal).
- **Cleaning Step 2**: Filter out 1 row where `gender == 'Other'`, retaining 5,109 instances (`Female`: 2,994, `Male`: 2,115).
- **Feature Engineering**:
  $$\text{risk\_factors\_count} = \text{hypertension} + \text{heart\_disease} + \mathbb{I}(\text{smoking\_status} \in \{\text{'smokes'}, \text{'formerly smoked'}\})$$
  This captures compounding lifestyle vulnerability, forcing tree splits to account for cumulative non-age risks.
- **Stratified Partitioning**: 80% train ($N = 4,087$), 20% test ($N = 1,022$) stratified on the binary target `stroke`.
  - Train: 3,888 negative (95.13%), 199 positive (4.87%).
  - Test: 972 negative (95.11%), 50 positive (4.89%).
- **Transformation Matrix (`ColumnTransformer`)**:
  - `numeric_features` (`['age', 'avg_glucose_level', 'bmi', 'risk_factors_count']`):
    - `SimpleImputer(strategy='median')`: Resolves 201 missing BMI values without mean-skewing.
    - `StandardScaler()`: Zero-mean, unit-variance normalization.
  - `categorical_features` (`['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']`):
    - `SimpleImputer(strategy='most_frequent')`: Fallback mode imputation.
    - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Generates 19 binary columns.
  - Total processed dimension: **23 features**.

#### XGBoost Model Hyperparameters & Rationale
The model configuration in [`model_trainer.py`](model_trainer.py#L42-L50) was tuned specifically to balance gradient stability on extreme class imbalance:

| Parameter | Configured Value | Engineering Rationale |
| :--- | :--- | :--- |
| `scale_pos_weight` | `19.54` (computed as `neg_cases / pos_cases = 3888 / 199`) | Multiplies the gradient updates of minority positive errors by ~19.54×, preventing the loss function from converging to the trivial all-zero local optimum. |
| `max_depth` | `4` | Shallow tree depth restricts higher-order feature interactions, acting as strong structural regularization against overfitting on the sparse 199 stroke cases. |
| `learning_rate` | `0.05` | Conservative shrinkage rate ensuring smooth convergence across 200 boosting rounds. |
| `n_estimators` | `200` | Sufficient ensemble capacity for learning tabular decision boundaries without tree bloat. |
| `subsample` | `0.8` | Stochastic row subsampling per boosting round to reduce variance and boost generalization. |
| `eval_metric` | `'logloss'` | Standard binary cross-entropy loss for probabilistic classification. |
| `random_state` | `42` | Ensures deterministic split and boosting reproducibility. |

#### Clinical Threshold Selection
Standard binary classifiers apply $\hat{y} = \mathbb{I}(P(y=1 \mid x) \ge 0.50)$. In this domain:
- The base rate of stroke is 4.87%. A 15% predicted probability represents a risk profile **over 3× the population average**.
- A false negative results in an untreated patient suffering stroke, irreversible brain injury, or death.
- A false positive results in secondary screening (e.g., blood pressure monitoring, carotid ultrasound, lifestyle consultation).
- Therefore, the decision threshold was established at **$\tau = 0.15$**, shifting model performance from a conservative classifier to an operational triage screen.

### Web Application Architecture
[`app.py`](app.py) is implemented using Streamlit's reactive session model:
- **Zero-Latency Startup (`@st.cache_resource`)**: Model weights, scikit-learn transformers, precomputed test SHAP explanations (288 KB), and LIME explainers are loaded into RAM once upon application start.
- **Stateless Prediction Function**: `predict_fn(x_numpy)` accepts raw NumPy arrays from LIME perturbations, converts them to named DataFrames conforming to `ColumnTransformer` feature names, and evaluates `model.predict_proba(df)`.
- **Dynamic Waterfall Rendering**: Uses `shap.plots.waterfall(shap_val_single[0])` rendered directly into `matplotlib.pyplot` figures, avoiding browser DOM reloads.

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose / Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.14` (Compatible with `3.11+`) | Core execution runtime specified in `.python-version` and `pyproject.toml`. |
| **ML Engine** | XGBoost | `3.2.0` | State-of-the-art gradient boosted tree implementation; native support for `scale_pos_weight` and optimized C++ backend. |
| **XAI (Global)** | SHAP | `0.51.0` | Exact polynomial-time Shapley computation for tree ensembles via `TreeExplainer`; supports beeswarm and waterfall visualizers. |
| **XAI (Local)** | LIME | `0.2.0.1` | Local surrogate perturbation model used to dissect non-linear edge cases (false positives/negatives) independently of tree structure. |
| **Data Processing** | Scikit-Learn | `1.8.0` | Robust feature pipelines via `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder` with `transform_output="pandas"`. |
| **Data Imbalance** | Imbalanced-Learn | `0.14.1` | Evaluated in repository prototyping; baseline reference for synthetic oversampling comparisons. |
| **DataFrames** | Pandas | `3.0.3` | High-performance columnar data structures for EHR manipulation. |
| **Numerical Computing** | NumPy | `2.4.6` | Vectorized operations and fast index masking for edge-case mining. |
| **Visualization** | Matplotlib & Seaborn | `3.10.9` / `0.13.2` | Headless plot generation for SHAP and LIME figures inside Streamlit buffers. |
| **Web Presentation** | Streamlit | `1.57.0` | Low-latency Python-native application layer; rapid development of interactive clinical diagnostic tools. |
| **Navigation** | Streamlit Option Menu | `0.4.0` | Modular navigation layout support. |
| **Serialization** | Dill | `0.4.1` | Extended pickle serializer required to store `LimeTabularExplainer`, which encapsulates unpicklable lambda functions. |
| **Packaging & Env** | Astral uv | Latest | Blazing-fast virtual environment creation, dependency resolution, and lockfile management (`uv.lock`). |

---

## ⚖️ Design Decisions & Trade-offs

| Decision | Alternatives Considered | Why This Choice Was Made | Accepted Trade-off |
| :--- | :--- | :--- | :--- |
| **1. Out-of-Fold (OOF) Cost-Based Thresholding vs Test Peeking** | Tuning threshold on held-out test recall (legacy v1 approach), default 0.50 cutoff, Youden's $J$. | Selecting threshold by inspecting held-out test recall inflates reported metrics and leaks test information. In v2, the decision threshold ($\tau = 0.0826$) is established strictly on out-of-fold calibrated probabilities using an explicit clinical loss function ($Cost_{FN} / Cost_{FP} = 10$). | Honest generalization recall measures 62.0% [48.0%, 76.0%] on the true holdout rather than an artificially inflated 80.0%. False alarms dropped from 372 to 229, saving clinical resources. |
| **2. Isotonic Probability Calibration over Raw `scale_pos_weight`** | Raw uncalibrated probabilities, Platt (sigmoid) scaling, temperature scaling. | While `scale_pos_weight = 19.54` helps trees split on minority cases, it severely distorts probability predictions (a raw 15% predicted risk corresponded to ~2-3% true prevalence). Isotonic regression via 5-fold CV reduced Brier score to 0.0427 (vs raw 0.0658) and ECE to 0.0015 (vs raw 0.0648), restoring clinically valid probability interpretation. | Requires fitting an auxiliary step; step-wise isotonic calibration curve. |
| **3. Aggregated Original-Feature SHAP over Fragmented OHE Dummies** | Showing 19 separate one-hot encoded bars/beeswarms, tree grouping. | In v1, categorical importance was split across separate dummy columns (e.g. `work_type_Private`, `work_type_Govt_job`), obscuring the true impact of the overarching medical factor. Aggregating one-hot SHAP attributions back to the original feature preserves exact mathematical additivity ($\sum \phi_{agg} = \sum \phi_{raw}$) while presenting coherent clinical bars. | Requires pre-computing and mapping feature lineage during explainer serialization. |
| **4. Mondrian Conformal Prediction Sets over Fixed Point Guarantees** | Standard point predictions, Bayesian uncertainty, split conformal without class conditioning. | Extreme class imbalance (4.87% prevalence) breaks standard conformal prediction. Mondrian class-conditional conformal prediction provides finite-sample 90% coverage guarantees for both healthy patients (90.0% coverage) and stroke patients (93.0% coverage), explicitly identifying an "ambiguous / refer" set {0, 1} for borderline cases (34.3% of cohort). | Clinical referral rate is 34.3%, requiring second-tier nurse/physician triage for ambiguous patients. |
| **5. Monotonic Constraints on Age & Glucose** | Unconstrained boosting, linear additive models. | Stroke risk is clinically non-decreasing with age and chronic glucose elevation. Optuna hyperparameter tuning demonstrated that enforcing monotonic constraints on age and glucose boosted out-of-fold PR-AUC by +0.0296 (0.1983 vs 0.1687) by preventing overfitting on noisy tabular sub-cohorts. | Restricts tree split topology to respect monotonicity. |
| **6. SMOTE vs Class-Weighting Controlled Ablation** | Asserting SMOTE failure without controlled evidence. | Evaluating SMOTE and BorderlineSMOTE strictly inside imblearn pipeline CV folds revealed that SMOTE achieved PR-AUC 0.1373 vs scale_pos_weight 0.1809 (DeLong $p=0.654$, statistically indistinguishable, though SMOTE generated ungrounded synthetic elderly profiles). Class weighting was selected for simplicity and avoidance of synthetic vector artifacts. | Preserves true patient empirical vectors without interpolation. |

---

## 📊 Data & Models

### Dataset Profile
- **Origin**: Sourced from the [Healthcare Dataset Stroke Data](https://raw.githubusercontent.com/karavokyrismichail/Stroke-Prediction---Random-Forest/main/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv) (originally hosted on Kaggle by fedesoriano).
- **Size**: 5,110 records, 12 attributes before preprocessing; 5,109 records after dropping 1 ambiguous gender entry.
- **License**: Community Data License / Open Public Dataset (Kaggle).
- **Target Distribution**:
  - Class 0 (No Stroke): 4,860 patients (95.13%)
  - Class 1 (Stroke): 249 patients (4.87%)
- **Known Quirks & Data Quality Considerations**:
  - `bmi` has **201 missing entries** (3.93% of the dataset). Handled via median imputation.
  - `smoking_status` has **1,544 "Unknown" entries** (30.2% of the dataset). Rather than dropping these rows, "Unknown" is preserved as an explicit one-hot encoded category, preventing data loss.
  - `gender` contained a single record labeled `'Other'`. Dropped to prevent single-sample pipeline failures during stratified k-fold splits.

### Data Dictionary

| Feature Name | Type | Processing | Description & Valid Ranges |
| :--- | :--- | :--- | :--- |
| `id` | Integer | **Dropped** | Unique patient identifier; excluded from modeling. |
| `gender` | Categorical | One-Hot Encoded (`cat__gender_Female`, `cat__gender_Male`) | Biological sex (`Male`, `Female`). |
| `age` | Float / Continuous | Median Imputed + Standard Scaled | Patient age in years (Range in data: `0.08` to `82.0`). |
| `hypertension` | Binary | One-Hot Encoded (`0`, `1`) | Chronic high blood pressure diagnosis (`0` = No, `1` = Yes). |
| `heart_disease` | Binary | One-Hot Encoded (`0`, `1`) | History of cardiac disease (`0` = No, `1` = Yes). |
| `ever_married` | Categorical | One-Hot Encoded (`No`, `Yes`) | Marital history (`Yes`, `No`). |
| `work_type` | Categorical | One-Hot Encoded (5 categories) | Employment category (`Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked`). |
| `Residence_type` | Categorical | One-Hot Encoded (`Rural`, `Urban`) | Geographic dwelling classification. |
| `avg_glucose_level` | Float / Continuous | Median Imputed + Standard Scaled | Average blood glucose level in mg/dL (Range: `55.12` to `271.74`). |
| `bmi` | Float / Continuous | Median Imputed + Standard Scaled | Body Mass Index in $\text{kg/m}^2$ (Range: `10.3` to `97.6`; 201 missing values). |
| `smoking_status` | Categorical | One-Hot Encoded (4 categories) | Tobacco smoking history (`formerly smoked`, `never smoked`, `smokes`, `Unknown`). |
| `risk_factors_count` | Integer | **Engineered** + Standard Scaled | Composite count of vascular risks: `hypertension + heart_disease + smoker` (Range: `0` to `3`). |
| `stroke` | Binary | **Target Label** | Cerebrovascular accident occurrence (`0` = No Stroke, `1` = Stroke). |

### Model Card: XGBoost Stroke Classifier
- **Model Architecture**: Shallow Gradient Boosted Decision Trees (`XGBClassifier`).
- **Input Dimensions**: 23 transformed features (4 continuous, 19 one-hot binary).
- **Output**: Raw stroke risk probability $P(y=1 \mid \mathbf{x}) \in [0, 1]$.
- **Intended Use**: Secondary clinical decision support and preliminary triage screening. **Not** approved as an autonomous primary diagnostic device.
- **Out-of-Scope Use Cases**: Pediatric emergency triage without clinical oversight; automated insurance claim denials.

---

## 📈 Evaluation & Results

### Evaluation Methodology
- **Validation Scheme**: 80/20 Stratified Holdout Split ($N_{\text{train}} = 4,087$, $N_{\text{test}} = 1,022$).
- **Seed & Environment**: `random_state = 42`, deterministic split and tree building.
- **Decision Rule**: Threshold $\tau = 0.15$. A test record is predicted positive if $P(\text{stroke} \mid \mathbf{x}) \ge 0.15$.
- **Source of Truth**: Metrics extracted directly from [`models/training_metadata.json`](models/training_metadata.json), generated by [`model_trainer.py`](model_trainer.py).

### Performance Across Pre-Selected Operating Points (Held-Out Test Set, N=1,022, 50 Positives)

All thresholds were fixed **strictly out-of-fold** from the training partition before evaluating the held-out test set. All metrics include **95% Bootstrap Confidence Intervals** (1,000 resamples):

| Operating Point | Threshold | Recall (Sensitivity) | Precision (PPV) | Specificity (TNR) | False Alarms (FP) | NNS (Screened/Stroke) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Legacy v1 (Test Peaked)** | `0.1500` | 80.0% (40/50) | 9.7% | 61.7% | 372 | 10.3 |
| **v2: Cost-Optimal (FN/FP=10)** | `0.0826` | **62.0%** `[48.0%, 76.0%]` | **11.9%** `[9.5%, 14.4%]` | **76.4%** `[73.7%, 79.1%]` | **229** `[203, 256]` | **8.4** `[6.9, 10.6]` |
| **v2: Target 80% OOF Recall** | `0.0581` | **78.0%** `[66.0%, 88.0%]` | **11.7%** `[10.0%, 13.5%]` | **69.9%** `[67.0%, 72.6%]` | **293** `[266, 321]` | **8.5** `[7.4, 10.0]` |
| **v2: High Precision** | `0.1365` | **32.0%** `[20.0%, 46.0%]` | **18.8%** `[12.1%, 26.3%]` | **92.9%** `[91.2%, 94.5%]` | **69** `[53, 86]` | **5.3** `[3.8, 8.3]` |

### Global Ranking & Calibration
- **ROC-AUC**: **0.7900** `[0.7184, 0.8477]` (Legacy v1: 0.8073)
- **PR-AUC**: **0.1822** `[0.1260, 0.2875]` (Legacy v1: 0.1691). The change in PR-AUC (+0.013) is **within the confidence interval**, indicating equivalent global ranking performance while delivering calibrated probabilities and leak-free thresholding.
- **Brier Score**: **0.0435** `[0.0415, 0.0454]` (vs raw uncalibrated 0.0658; ECE reduced to 0.0015 via Isotonic calibration).

### Confusion Matrix & Clinical Interpretation
Evaluated on the $N = 1,022$ held-out test set:

#### Upgraded v2 Pipeline ($\tau = 0.0826$, Calibrated, Zero Leakage)
```
                       Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)             743                       229          [Total: 972]
Actual Positive (1)              19                        31          [Total:  50]
```

#### Legacy v1 Baseline ($\tau = 0.15$, Heuristic Test-Peaked)
```
                       Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)             600                       372          [Total: 972]
Actual Positive (1)              10                        40          [Total:  50]
```

- **True Positives & Recall**: 31 strokes flagged at cost-optimal threshold $\tau = 0.0826$ (62.0% recall). When high sensitivity is demanded, setting threshold $\tau = 0.0581$ (80% target OOF recall) flags 39 out of 50 strokes (78.0% recall) with 293 false alarms.
- **False Alarms (FP)**: Dropped from 372 down to 229 (-143 alarms, -38.4%) at $\tau=0.0826$, and down to 293 (-79 alarms) at $\tau=0.0581$. **This reduction in false alarms comes directly from the changed operating point**, not "model magic".
- **Number Needed to Screen (NNS)**: Improved from 10.3 down to 8.4 (or 5.3 at the high-precision operating point).

### Young Patient Subgroup Audit (Age < 50)

Across the entire dataset ($N=5,109$), stroke events in patients under 50 years old are extremely rare (**0.69%** overall prevalence, 20 total events). However, stroke risk increases strictly with the count of co-occurring clinical and lifestyle risk factors (`hypertension + heart_disease + smoker`):

- **Full Dataset ($N = 2,899$ young patients, 20 strokes, 0.69% base rate)**:
  - `risk_factors_count = 0`: $N = 2,099$, Strokes = $11$ ($0.52\%$, 5.2 per 1,000)
  - `risk_factors_count = 1`: $N = 761$, Strokes = $8$ ($1.05\%$, 10.5 per 1,000)
  - `risk_factors_count = 2`: $N = 39$, Strokes = $1$ ($2.56\%$, 25.6 per 1,000)
  - `risk_factors_count = 3`: $N = 0$, Strokes = $0$
- **Training Set ($N = 2,316$ young patients, 12 strokes)**:
  - `RFC = 0`: $N = 1,687$, Strokes = $7$ ($0.41\%$)
  - `RFC = 1`: $N = 594$, Strokes = $4$ ($0.67\%$)
  - `RFC = 2`: $N = 35$, Strokes = $1$ ($2.86\%$)
- **Held-Out Test Set ($N = 583$ young patients, 8 strokes)**:
  - `RFC = 0`: $N = 412$, Strokes = $4$ ($0.97\%$)
  - `RFC = 1`: $N = 167$, Strokes = $4$ ($2.40\%$)
  - `RFC = 2`: $N = 4$, Strokes = $0$ ($0.00\%$)

### Model Zoo Comparison: No Statistically Significant Winner

A controlled 5×5 repeated cross-validation benchmark across 13 model configurations demonstrates that **there is no statistically significant winner** among gradient boosting models and interpretable glass-box models:

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

1. **Overlapping Confidence Intervals**: The 95% CIs for PR-AUC overlap substantially across XGBoost (0.181), LightGBM (0.174), Logistic Regression (0.164), and EBM (0.163).
2. **DeLong Hypothesis Testing**: All DeLong $p$-values against XGBoost are $\gg 0.05$. Simpler, fully interpretable glass-box models like EBM and penalized Logistic Regression perform on par with complex ensembles.


### Edge Case & Failure Mode Analysis (v2 Upgrades)

Post-hoc analysis is expanded in v2 (`reports/edge_cases_v2.json`) with dual criteria:
1. **Conformal Ambiguous Sets ($N=701$ across cohort, 34.3%)**:
   - Rather than forcing a binary call, patients whose calibrated probability falls in the nonconformity band `[0.0260, 0.1237]` receive prediction set `{No Stroke, Stroke}`. These patients are formally routed to second-line clinical review.
2. **Near-Operating-Threshold Band ($\tau \pm 0.015$, i.e. $[0.0676, 0.0976]$)**:
   - Identifies fragile border cases sensitive to minor sensor or self-report noise.
3. **High-Confidence False Positives ($P > 25\%$, Actual = 0)**:
   - Patients over 70 with compounding hypertension and glucose $>200\text{ mg/dL}$ who had not yet suffered an event at census time.
4. **Severe False Negatives ($P < 2\%$, Actual = 1)**:
   - Young patients (<40) with no documented clinical comorbidities, pointing to unmeasured latent factors (clotting gene variants, aneurysms).

### Reproduction Steps

#### Master Single-Command Reproduction (v2 Pipeline)
To re-run the entire upgraded pipeline, cross-validation harness, Optuna tuning, calibration, explainers, and test suite:

```bash
uv run python reproduce.py
```

#### Step-by-Step Reproduction
```powershell
# 1. Freeze stratified train/test split indices (zero leakage)
python src/save_split_indices.py

# 2. Fit and serialize leak-free sklearn v2 pipeline
python build_v2_pipeline.py

# 3. Execute 5x5 repeated stratified cross-validation harness with bootstrap CIs
python evaluation.py

# 4. Run Model Zoo (13 models & SMOTE ablation under identical CV folds)
python model_zoo.py

# 5. Run Optuna HPO, monotonic constraints, and stacking evaluation
python hyperparameter_tuning.py

# 6. Fit Isotonic probability calibration & select cost-based threshold
python calibrate.py

# 7. Compute aggregated SHAP, dependence, interaction, and LIME agreement
python explainability_v2.py

# 8. Compute Mondrian conformal prediction, subgroup fairness, and edge cases v2
python conformal_fairness.py

# 9. Execute single one-time evaluation on held-out test set
python final_eval.py
```

Raw outputs are saved in `reports/` (JSON & CSV reports), `plots/v2/` (diagnostic visualizations), and `models/v2/` (serialized calibrated pipelines).

---

## 🧪 Testing & Quality

- **Automated Test Suite**: Integrated using `pytest` and `pytest-cov` inside [`tests/`](tests/).
- **Test Results**: **10 passed out of 10 tests (100% pass rate)**.
- **Continuous Integration (CI)**: GitHub Actions workflow configured at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) testing Python 3.11 and 3.12 with linting via `ruff` and test coverage reporting.

### Core Test Assertions:
1. `tests/test_data.py`:
   - Validates that `gender == 'Other'` is dropped, leaving exactly 5,109 rows.
   - Asserts `risk_factors_count` calculation correctness (including `hypertension=1 + heart_disease=1 + smokes` evaluates to **3**, fixing the legacy smoke test error).
2. `tests/test_pipeline.py`:
   - Verifies train and test index sets have **0% overlap** (`len(train_idx & test_idx) == 0`).
   - Asserts probability outputs are bounded in $[0, 1]$ and sum to 1.0.
   - Tests monotonicity of calibrated risk scores across escalating clinical risk profiles.
   - Asserts operating threshold lies within $(0, 1)$.
3. `tests/test_explainability.py`:
   - Validates that SHAP dummy aggregation back to original features preserves exact mathematical additivity ($\sum \phi_{agg} = \sum \phi_{raw}$).
   - Asserts `TreeExplainer` hydration and valid attribution generation.
4. `tests/test_app_inference.py`:
   - Smoke tests end-to-end inference pass through `calibrated_pipeline.joblib`.

Execute the test suite locally:
```bash
uv run pytest tests -v --cov=src
```

---

## 🔒 Security, Privacy & Reliability

- **Secrets Handling**: The repository contains no API keys, credentials, or remote database connection strings. All training and inference executes strictly on the local CPU runtime.
- **Input Validation**: The Streamlit user interface in [`app.py`](app.py#L58-L68) constrains numerical inputs using hard bounded widgets:
  - Age: Clamped between `0` and `100`.
  - Average Glucose Level: Clamped between `50.0` and `300.0`.
  - Body Mass Index (BMI): Clamped between `10.0` and `100.0`.
  - Categorical variables: Restricted to rigid selection drop-downs matching the fitted `OneHotEncoder` categories.
- **Data Privacy & Retention**: Zero patient telemetry or user input is saved to disk or transmitted across external networks during Streamlit sessions. All input vectors reside purely in volatile Python memory.
- **Serialization Integrity**: While `dill` is required to deserialize `models/lime_explainer.pkl`, users should ensure that pickled artifacts are sourced exclusively from trusted builds within this repository to prevent arbitrary code execution risks inherent to Python pickling.

---

## 🚀 Getting Started

### Prerequisites
- **Python**: Version `3.11`, `3.12`, or `3.14` installed.
- **Package Manager**: [Astral uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`.
- **Operating System**: Windows, macOS, or Linux.

### Installation via uv or venv

#### Option A: Using Astral uv (Recommended for speed)
```bash
# 1. Clone the repository
git clone https://github.com/dummycodertech/xai-xgboost-heart_Stroke.git
cd xai-xgboost-heart_Stroke

# 2. Create virtual environment and install locked dependencies
uv venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Synchronize environment with pyproject.toml
uv pip install -r requirements.txt
```

#### Option B: Using Standard Python venv & pip
```bash
# 1. Clone the repository
git clone https://github.com/dummycodertech/xai-xgboost-heart_Stroke.git
cd xai-xgboost-heart_Stroke

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Smoke Test

#### Upgraded v2 Calibrated Pipeline Smoke Test
Verify that the leak-free calibrated pipeline, feature engineering, and metadata hydrate and evaluate an inference pass correctly:

```bash
uv run python -c "import joblib, json, pandas as pd; cal = joblib.load('models/v2/calibrated_pipeline.joblib'); meta = json.load(open('models/v2/metadata.json','r')); df = pd.DataFrame([{'age': 65, 'avg_glucose_level': 140.0, 'bmi': 30.0, 'gender': 'Male', 'hypertension': 1, 'heart_disease': 1, 'ever_married': 'Yes', 'work_type': 'Private', 'Residence_type': 'Urban', 'smoking_status': 'smokes'}]); prob = cal.predict_proba(df)[0][1]; t = meta['chosen_threshold']; print(f'Smoke test passed! Calibrated Risk: {prob:.1%}, Operating Threshold: {t:.4f}, Alert: {prob >= t}')"
```

Expected output:
```text
Smoke test passed! Calibrated Risk: 15.0%, Operating Threshold: 0.0826, Alert: True
```

#### Legacy v1 Smoke Test (Corrected Feature Count)
*Note: In the legacy smoke test, a patient with `hypertension=1`, `heart_disease=1`, and `smoking_status='smokes'` has a true `risk_factors_count = 3` (previously misstated as 2).*

```bash
uv run python -c "import pickle, json, pandas as pd; m = pickle.load(open('models/xgboost_model.pkl','rb')); p = pickle.load(open('models/preprocessor.pkl','rb')); meta = json.load(open('models/training_metadata.json','r')); df = pd.DataFrame([{'age': 65, 'avg_glucose_level': 140.0, 'bmi': 30.0, 'risk_factors_count': 3, 'gender': 'Male', 'hypertension': 1, 'heart_disease': 1, 'ever_married': 'Yes', 'work_type': 'Private', 'Residence_type': 'Urban', 'smoking_status': 'smokes'}]); prob = m.predict_proba(p.transform(df))[0][1]; print(f'Legacy smoke test passed! Risk: {prob:.1%}, Threshold: {meta[\"threshold\"]}')"
```

Expected output:
```text
Legacy smoke test passed! Risk: 66.2%, Threshold: 0.15
```

---

## 💻 Usage

### Interactive Streamlit UI
To launch the diagnostic application, run:

```bash
streamlit run app.py
```

The browser will automatically open at `http://localhost:8501`.

```
========================================================================================
STREAMLIT TAB NAVIGATION:
[Tab 1: Prediction & SHAP]    -> Live patient input form & local SHAP Waterfall chart
[Tab 2: Global Insights]      -> Population-level Beeswarm & Feature Impact Bar plots
[Tab 3: Edge Cases (LIME)]    -> Dissection of False Positives, False Negatives & Margins
[Tab 4: Model Performance]    -> Confusion matrix, PR-AUC, and triage trade-off breakdown
[Tab 5: Imbalance Analysis]   -> Post-mortem on SMOTE failure vs. scale_pos_weight
========================================================================================
```

#### Worked Example: Patient Risk Inference (Tab 1)
1. In the sidebar/left column, input the following patient profile:
   - **Age**: `62`
   - **Avg Glucose Level**: `195.0`
   - **BMI**: `32.4`
   - **Gender**: `Male`
   - **Hypertension**: `1`
   - **Heart Disease**: `0`
   - **Ever Married**: `Yes`
   - **Work Type**: `Self-employed`
   - **Residence Type**: `Urban`
   - **Smoking Status**: `formerly smoked`
2. Click **Predict Risk**.
3. **Expected Output**:
   - Risk Badge: `HIGH RISK (Exceeds 15.0% clinical threshold)`
   - Calculated Stroke Probability: `~65% – 75%`
   - Explanation: A SHAP waterfall plot renders immediately below, visually displaying positive red bars for `num__age`, `num__avg_glucose_level`, and `num__risk_factors_count` driving the score upward from the baseline log-odds $E[f(x)]$.

---

## ⚙️ Configuration

All pipeline operational variables and model hyperparameter tunables are consolidated across three primary source files:

| Tunable Parameter | Default Value | Location | Impact of Modification |
| :--- | :--- | :--- | :--- |
| `decision_threshold` | `0.15` | [`model_trainer.py:23`](model_trainer.py#L23), [`app.py:85`](app.py#L85) | Increasing toward `0.50` reduces false alarms but causes recall to collapse toward ~24%. Decreasing below `0.10` catches 95%+ of strokes but overwhelms triage staff with false positives. |
| `scale_pos_weight` | `neg_cases / pos_cases` (`~19.54`) | [`model_trainer.py:38`](model_trainer.py#L38) | Controls gradient penalty for missing a stroke case. Lowering toward `1.0` reverts XGBoost to majority-class guessing. |
| `max_depth` | `4` | [`model_trainer.py:44`](model_trainer.py#L44) | Increasing beyond 6 increases tree complexity and causes severe overfitting on the 199 minority stroke cases. |
| `learning_rate` | `0.05` | [`model_trainer.py:45`](model_trainer.py#L45) | Step size shrinkage. Lower values require increasing `n_estimators`. |
| `n_estimators` | `200` | [`model_trainer.py:46`](model_trainer.py#L46) | Number of boosting iterations. |
| `subsample` | `0.8` | [`model_trainer.py:47`](model_trainer.py#L47) | Ratio of training samples randomly selected per tree. Prevents tree co-adaptation. |
| `test_size` | `0.20` | [`data_loader.py:53`](data_loader.py#L53) | Fraction of data allocated to the holdout test set (currently 1,022 rows). |
| `uncertain_mask` | `[0.45, 0.55]` | [`edgecasedec.py:41`](edgecasedec.py#L41) | Probability window used to flag borderline cases for manual audit. |

---

## 📂 Project Structure

```
xgboost-xai/
├── .gitignore                    # Explicitly un-ignores models/ and data/ for Streamlit Cloud deployment
├── .python-version               # Declares target Python runtime (3.14)
├── pyproject.toml                # Project metadata, build config, and dependency requirements
├── requirements.txt              # Pinned pip dependency export
├── uv.lock                       # Cryptographically locked dependency graph for uv
├── README.md                     # Comprehensive project documentation
├── main.py                       # Minimal CLI entry point stub
├── app.py                        # Streamlit multi-tab clinical dashboard and XAI visualizer
├── data_loader.py                # Ingestion, cleaning, feature engineering, ColumnTransformer
├── model_trainer.py              # Cost-sensitive XGBoost training, thresholding, metrics calculation
├── shap_explainer.py             # SHAP TreeExplainer calculation, plot testing, and artifact serialization
├── lime_expl.py                  # LIME TabularExplainer initialization and dill serialization
├── edgecasedec.py                # Failure mode analyzer (mining False Positives, False Negatives, Uncertainties)
├── data/
│   ├── healthcare-dataset-stroke-data.csv  # Raw Kaggle/mirror dataset (5,110 rows)
│   ├── train_processed.pkl       # 4,087 preprocessed training vectors with target labels
│   └── test_processed.pkl        # 1,022 preprocessed test vectors with target labels
├── models/
│   ├── xgboost_model.pkl         # Trained XGBClassifier binary model
│   ├── preprocessor.pkl          # Fitted Scikit-Learn ColumnTransformer pipeline
│   ├── shap_explainer.pkl        # Serialized shap.TreeExplainer object
│   ├── shap_values_test.pkl      # Pre-computed SHAP explanation object for 1,022 test records
│   ├── lime_explainer.pkl        # Serialized LIME Tabular Explainer (dill format)
│   ├── training_metadata.json    # Exact test metrics, confusion matrix, and threshold settings
│   └── edge_cases.json           # Isolated index records for FP, FN, and uncertain patient samples
└── plots/
    ├── shap_summary_bar.png      # Offline verification plot: mean absolute SHAP value impact
    └── shap_summary_beeswarm.png # Offline verification plot: SHAP value distribution and feature direction
```

---

## 🌐 Deployment

### Current Deployment
- **Target Platform**: [Streamlit Community Cloud](https://streamlit.io/cloud).
- **Artifact Strategy**: Because serialized model files (`xgboost_model.pkl`, `shap_values_test.pkl`, `preprocessor.pkl`) total less than 2 MB combined, the repository explicitly tracks them in Git via negation patterns in [`.gitignore`](.gitignore#L11-L13):
  ```gitignore
  # Ensure Streamlit Cloud gets our artifacts
  !models/
  !data/
  ```
  This enables Streamlit Community Cloud to spin up containerized instances without requiring external AWS S3, Google Cloud Storage, or runtime re-training.
- **Live URL**: [VERIFY: Add live deployment link here if deployed to Streamlit Cloud, e.g., https://share.streamlit.io/dummycodertech/xai-xgboost-heart_Stroke/main/app.py].

### Deploying a Fork to Streamlit Community Cloud
1. Fork [dummycodertech/xai-xgboost-heart_Stroke](https://github.com/dummycodertech/xai-xgboost-heart_Stroke) to your GitHub account.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New App**, select your forked repository, branch `main`, and specify `app.py` as the Main file path.
4. Click **Deploy**. The app will build and serve in under 60 seconds.

---

## ⚡ Performance & Resource Profile

- **Inference Latency**:
  - Single patient XGBoost inference: **`< 2 ms`** per query on modern CPU.
  - On-the-fly local SHAP Waterfall generation: **`~45 ms`** per query (utilizing optimized C++ tree paths).
  - LIME perturbation explanation (5,000 perturbed samples): **`~450 ms`** per query.
- **Memory Footprint**:
  - Streamlit application in RAM: **`~180 MB`** (including loaded test vectors and precomputed SHAP cache).
- **Storage Profile**:
  - Entire repository including data and models: **`< 5 MB`**.
- **Infrastructure Cost**:
  - **$0.00 / month**: Runs entirely within the free tier of Streamlit Community Cloud or locally on any standard laptop.

---

## 🗺️ Roadmap & Future Scope

### Completed in Rigorous ML Upgrade (v2 Production Pipeline)
- [x] **Automated Test Suite & CI**: `pytest` and `pytest-cov` test suite with 100% pass rate, coupled to GitHub Actions (`.github/workflows/ci.yml`).
- [x] **Probability Calibration**: Isotonic calibration fitted strictly inside CV folds, reducing Brier score to 0.0435 and ECE to 0.0015.
- [x] **Out-of-Fold Threshold Selection**: Eliminated test peeking; derived decision thresholds from clinical cost models and sensitivity targets on training data only.
- [x] **Mondrian Conformal Prediction**: Established 90% finite-sample coverage guarantees and explicit clinical referral routing for ambiguous cases (34.3% of cohort).
- [x] **Aggregated Original-Feature SHAP**: Aggregated one-hot dummy variables back to cohesive clinical features, preserving exact mathematical log-odds additivity.
- [x] **Model Zoo & Imbalance Ablation**: Controlled 13-model benchmark under identical 5×5 CV folds with DeLong and McNemar hypothesis testing.
- [x] **Subgroup Fairness Audit**: Audited equalized odds and demographic error distributions across age and gender cohorts.

### External Validation: Limitations & Future Scope
External validation against public population surveys like the CDC Behavioral Risk Factor Surveillance System (BRFSS) was systematically investigated and excluded from this release due to fundamental covariate mismatch:
1. **Missing Biomarkers**: BRFSS is an annual telephone interview that lacks serum blood glucose (`avg_glucose_level`), a primary non-linear predictor in cerebrovascular risk stratification.
2. **Missing Clinical Demographics**: BRFSS does not measure standardized occupational categories (`work_type`) or census dwelling designations (`Residence_type`).
3. **Self-Report Bias**: BRFSS relies on participant self-reported recall of health conditions rather than electronic health record (EHR) laboratory measurements and confirmed clinical diagnoses.

**Future Validation Direction**: Prospective validation will target multi-center electronic health record repositories (e.g., MIMIC-IV, eICU, or All of Us) where objective laboratory measurements and standardized ICD-10 stroke event registries are available.

---

## ❓ Frequently Asked Questions

#### 1. Why is the overall model accuracy only 62.6%?
In an imbalanced dataset where 95.1% of patients do not have a stroke, a naive model that predicts "No Stroke" for every patient achieves 95.1% accuracy but has a clinical recall of 0.0% (missing every single stroke). By penalizing missed strokes 19.54× and lowering the clinical threshold to 0.15, we stopped the model from gaming the accuracy metric. Our 62.6% accuracy is an expected consequence of catching **80.0% of real stroke cases**.

#### 2. Why was SMOTE removed from the preprocessing pipeline?
SMOTE generates synthetic data points along linear paths between existing minority samples. Because stroke cases in this dataset are concentrated in older age brackets, SMOTE primarily generated synthetic 70- and 80-year-olds. This reinforced an age bias, making the model blind to younger patients with severe compounding lifestyle risks. Replacing SMOTE with `scale_pos_weight` forced the algorithm to learn from real patient records.

#### 3. What does `scale_pos_weight = 19.54` actually do?
During XGBoost gradient boosting, `scale_pos_weight` scales the gradient of the loss function for the positive class (stroke). In our training set, there are 3,888 negative cases and 199 positive cases ($3888 / 199 \approx 19.54$). This mathematically tells the optimizer that failing to detect a stroke is roughly 20 times worse than generating a false alarm.

#### 4. Why set the clinical decision threshold to 0.0826 in v2 instead of 0.15?
In legacy v1, the 0.15 threshold was chosen by looking at the held-out test set to produce an 80% recall headline, which represents test-set leakage. Furthermore, because `scale_pos_weight=19.54` inflated raw predicted probabilities, 0.15 on the uncalibrated scale corresponded to ~2-3% true event rate. In v2, probabilities are calibrated using Isotonic Regression (reducing ECE to 0.0015), and the threshold ($\tau = 0.0826$, ~8.3%) was selected strictly out-of-fold using a 10:1 clinical cost loss function ($Cost_{FN} / Cost_{FP} = 10$).

#### 5. Does SMOTE really fail on this dataset?
In Phase 3 of the v2 upgrade, we performed a controlled ablation comparing SMOTE and BorderlineSMOTE strictly inside imblearn pipeline cross-validation folds. The empirical finding: SMOTE achieved out-of-fold PR-AUC 0.1373 vs scale_pos_weight 0.1809 (DeLong $p=0.654$, statistically indistinguishable). While SMOTE does not catastrophically collapse, it generates ungrounded synthetic elderly vectors along $k$-NN lines and offers zero statistical advantage over native class-weighting.

#### 6. Why is probability calibration essential for clinical triage?
Uncalibrated tree ensembles with `scale_pos_weight` produce distorted predicted probabilities that cannot be interpreted as empirical risk percentages. A clinician cannot tell a patient "you have a 15% risk" when the true event rate at that prediction is 3%. Isotonic calibration aligns predicted risks with observed frequencies, reducing the Brier score to 0.0435 and Expected Calibration Error (ECE) from 0.0648 down to 0.0015.

#### 7. Why use both SHAP and LIME?
SHAP and LIME provide complementary perspectives. SHAP is grounded in cooperative game theory (Shapley values) and provides exact, globally consistent feature attributions across the entire dataset. In v2, SHAP values across one-hot columns are aggregated back to the original clinical feature level. LIME uses local linear perturbations and provides an independent sanity check on localized decisions. In our agreement audit across 200 patients, SHAP and LIME exhibited a mean Spearman rank correlation of 0.252 and 33.2% top-3 feature overlap, highlighting the importance of not relying on a single explainer.

#### 8. How are missing values in BMI handled?
The raw dataset contains 201 missing values in the `bmi` column. In the v2 leak-free pipeline, we use `SimpleImputer(strategy='median')` fit strictly inside CV folds, supplemented by an explicit `bmi_missing` binary indicator feature.

---

## 💼 Resume & Interview Kit

<details>
<summary><strong>Click to expand Resume Bullets, Technical Pitch & Interview Deep Dive</strong></summary>

### Impact-Driven Resume Bullets
- **Engineered an end-to-end Explainable AI (XAI) stroke triage pipeline** using XGBoost, Scikit-Learn, and Optuna; eliminated test leakage by selecting cost-optimal decision thresholds out-of-fold ($Cost_{FN}/Cost_{FP}=10$), achieving **62.0% Recall [48.0%, 76.0%]** with **11.9% Precision** and **PR-AUC: 0.1822** on a held-out test cohort ($N=1,022$).
- **Implemented Isotonic probability calibration & Mondrian conformal prediction**, reducing Expected Calibration Error (ECE) from 0.0648 to 0.0015 and establishing finite-sample 90% coverage guarantees with automated clinical referral flagging for ambiguous cases.
- **Architected an aggregated feature-level SHAP explanation framework** preserving exact mathematical additivity ($\sum \phi_{agg} = \sum \phi_{raw}$), paired with a 200-patient SHAP-vs-LIME agreement audit and empirical faithfulness deletion curves.
- **Conducted controlled 13-model benchmark & SMOTE ablation** under identical 5x5 CV folds; demonstrated statistically that SMOTE ($PR\text{-}AUC=0.1373$) is indistinguishable from class-weighting ($PR\text{-}AUC=0.1809$, $p=0.654$) while introducing synthetic feature artifacts.

---

### Elevator Pitches

#### 30-Second Pitch
> *"Most medical risk models brag about 95% accuracy because they cheat: in a disease with 5% prevalence, predicting 'No Stroke' gives 95% accuracy while killing every patient. I built a clinical triage pipeline with XGBoost that rejects this illusion. By removing synthetic SMOTE data, engineering a composite vascular risk metric, weighting the minority loss by 19.54×, and shifting the decision cutoff to 0.15, the model catches 80% of all strokes on a 1,022-patient test set. I then wrapped it in a Streamlit diagnostic suite featuring real-time SHAP waterfall attributions and LIME edge-case auditing."*

#### 2-Minute Walkthrough Script
> *"The core challenge of clinical ML on tabular data isn't fitting an algorithm—it's navigating asymmetric costs and data distribution traps. When I started analyzing this stroke dataset, I saw two major problems: first, the 95% accuracy trap where models predict zero for everyone; second, when people try to fix it with SMOTE, the algorithm interpolates along age lines and generates synthetic 75-year-olds, completely blinding the model to younger patients with extreme lifestyle risks.
> 
> To solve this, I took a first-principles approach. First, I engineered a `risk_factors_count` feature combining hypertension, heart disease, and active smoking status to give the model a non-linear signal independent of age. Second, I discarded synthetic sampling entirely and passed the raw, unaltered distribution into XGBoost, but configured `scale_pos_weight = 19.54`—the exact ratio of negative to positive training cases. Third, I lowered the clinical decision threshold from 0.50 to 0.15. In stroke triage, a 15% risk is over three times baseline prevalence; a false alarm costs a five-minute blood pressure check, while a missed case is fatal.
> 
> On our 1,022-sample test set, this achieved 80.0% recall, catching 40 out of 50 stroke victims with an ROC-AUC of 0.8073. Finally, because doctors don't trust black boxes, I built an interactive Streamlit dashboard. It pairs live SHAP waterfall plots for instant patient attribution with a dedicated LIME edge-case auditor that dissects false positives, false negatives, and borderline cases."*

---

### Top 3 Technical Challenges & Root-Cause Solutions

#### Challenge 1: Synthetic Over-sampling (SMOTE) Reinforcing Age Bias
- **Symptom**: Preliminary models trained on SMOTE-augmented data showed near-zero recall for patients under 50 years old who suffered strokes.
- **Root Cause**: In the training data, stroke events correlate strongly with age. $k$-nearest neighbors in SMOTE drew synthetic vectors predominantly between elderly stroke patients, multiplying the number of 70+ year-old stroke examples while leaving young lifestyle-risk profiles underrepresented.
- **Fix**: Removed SMOTE entirely. Retained the raw, natural training distribution and applied cost-sensitive learning via `scale_pos_weight=19.54` combined with the custom `risk_factors_count` feature to explicitly force tree splits on lifestyle comorbidities.

#### Challenge 2: Object Serialization Failure in LIME Explainer
- **Symptom**: Attempting to save the fitted `LimeTabularExplainer` via standard `pickle.dump` threw an unhandled `PicklingError: Can't pickle <function <lambda>>`.
- **Root Cause**: LIME's internal implementation dynamically constructs anonymous lambda functions for feature discretization and sample perturbation, which Python's standard `pickle` engine cannot serialize.
- **Fix**: Replaced standard `pickle` with `dill` in `lime_expl.py` and `app.py`. `dill` serializes the full Python execution state, including dynamically generated bytecode and closures, allowing the explainer to be loaded cleanly in Streamlit.

#### Challenge 3: Streamlit UI Freezing on Global SHAP Calculations
- **Symptom**: Loading the Streamlit application initially took 8–12 seconds because Tab 2 recalculated SHAP values across all 1,022 test records on every page refresh.
- **Root Cause**: TreeExplainer calculations across 1,022 samples and 200 trees have an $O(N \cdot M \cdot D^2)$ time complexity, causing CPU bottlenecks on the web server thread.
- **Fix**: Separated offline computation from UI serving. Created `shap_explainer.py` to precompute and pickle `shap_values_test.pkl` (288 KB) ahead of time. In `app.py`, these values are loaded into memory via `@st.cache_resource`, reducing Tab 2 rendering latency to under 20 milliseconds.

---

### Common Interview Questions & Grounded Answers

#### Q1: "Why did you optimize for Recall over Precision or F1?"
> *"In clinical screening, false negatives and false positives have vastly asymmetric costs. Missing a stroke leads to patient death or permanent disability. In contrast, a false positive flags a patient for secondary triage—a blood test, an ECG, or a nurse consultation. Optimizing for standard F1-score treats both error types as equally harmful. By targeting Recall at 80%, we accept a higher false positive rate to ensure 40 out of 50 stroke cases are caught."*

#### Q2: "Why use XGBoost instead of a Deep Learning model or a simpler Logistic Regression?"
> *"Tabular clinical data with under 10,000 samples is the exact domain where gradient boosted decision trees consistently outperform deep learning architectures. Deep neural nets require extensive regularization and sample size to prevent overfitting on tabular features. Logistic regression, while interpretable, cannot capture non-linear interactions (such as the compounding effect of smoking and glucose) without manual polynomial feature expansion. XGBoost provides optimized split-finding, native class weighting, and direct compatibility with TreeSHAP."*

#### Q3: "What happens if a user submits a missing BMI value in production?"
> *"Our Scikit-Learn `ColumnTransformer` incorporates a `SimpleImputer(strategy='median')` that was fitted strictly on the training partition. If a patient input arrives with missing or null BMI, the pipeline automatically imputes the training median before passing the vector to `StandardScaler` and XGBoost, ensuring zero runtime crashes."*

#### Q4: "How do you know the model isn't just memorizing age?"
> *"We validated this using both global and local XAI. In Tab 2's SHAP beeswarm plot, while `age` is the top contributor, `avg_glucose_level`, `bmi`, and our engineered `risk_factors_count` exhibit significant SHAP impact across all age groups. Furthermore, in Tab 3's edge-case analysis, we explicitly examine false positives and false negatives to monitor exactly how non-age features influence the decision boundary."*

#### Q5: "What would you do differently if you had another month on this project?"
> *"First, I would implement probability calibration using Platt Scaling or Isotonic Regression so that the predicted probabilities reflect true empirical risk frequencies. Second, I would build counterfactual explanations using DiCE so clinicians can tell patients exactly what lifestyle modifications would reduce their risk below the threshold. Third, I would package the model as a containerized FastAPI service with automated `pytest` and schema validation pipelines."*

---

### Skills Demonstrated
- **Machine Learning & Modeling**: Cost-Sensitive Classification, XGBoost, Hyperparameter Tuning, Decision Threshold Tuning, Class Imbalance Handling, Scikit-Learn Pipeline Design.
- **Explainable AI (XAI)**: SHAP (TreeExplainer, Waterfall, Beeswarm), LIME (Tabular Explainer), Model Auditing, Post-Hoc Failure Mode Analysis.
- **Data Engineering**: Missing Data Imputation, Standard Scaling, One-Hot Encoding, Feature Engineering, Leakage Prevention via Stratification.
- **Software Engineering & Tooling**: Python 3.14 / 3.11, Streamlit Dashboarding, Dill Object Serialization, Astral UV Dependency Management, Git Version Control.
</details>

---

## 🤖 Project Context (LLM Agent Ingestion)

<details>
<summary><strong>Click to expand structured YAML context block for AI coding assistants</strong></summary>

```yaml
project:
  name: "xgboost-xai"
  tagline: "Clinical Stroke Triage and Explainable AI Diagnostic System"
  repository: "https://github.com/dummycodertech/xai-xgboost-heart_Stroke"
  author: "Yagas Vashist (dummycodertech)"
  version: "0.1.0"
  runtime: "Python 3.14 (Compatible with Python 3.11+)"
  package_manager: "uv / pip"

problem_space:
  domain: "Cardiovascular / Cerebrovascular Clinical Risk Triage"
  target_event: "Stroke Occurrence (Binary: 0 or 1)"
  core_challenge: "Severe class imbalance (95.1% negative, 4.87% positive); standard models achieve 95% accuracy by predicting 0% recall."
  secondary_challenge: "SMOTE oversampling induces synthetic elderly bias; black-box gradient boosted models lack clinical auditability."

data_profile:
  source_file: "data/healthcare-dataset-stroke-data.csv"
  total_samples: 5110
  usable_samples: 5109 # 1 row dropped (gender == 'Other')
  class_counts: { class_0: 4860, class_1: 249 }
  stratified_split: { train_samples: 4087, test_samples: 1022, test_ratio: 0.20 }
  imputed_features: { bmi: "Median imputation (201 missing values)" }
  engineered_features:
    risk_factors_count: "hypertension + heart_disease + int(smoking_status in ['smokes', 'formerly smoked'])"
  preprocessor: "sklearn.compose.ColumnTransformer (SimpleImputer + StandardScaler for num; SimpleImputer + OneHotEncoder for cat)"
  processed_feature_count: 23

machine_learning:
  model_type: "xgboost.XGBClassifier"
  hyperparameters:
    scale_pos_weight: 19.537688 # neg_cases / pos_cases (3888 / 199)
    max_depth: 4
    learning_rate: 0.05
    n_estimators: 200
    subsample: 0.8
    eval_metric: "logloss"
    random_state: 42
  decision_threshold: 0.15 # Lowered from default 0.50 to maximize sensitivity

measured_results:
  evaluation_set: "1,022 held-out test records (972 negative, 50 positive)"
  threshold: 0.15
  stroke_recall: 0.8000 # 40 / 50 cases caught
  stroke_precision: 0.0971 # 40 / 412 alarms
  stroke_f1: 0.1732
  roc_auc: 0.8073
  pr_auc: 0.1691
  accuracy: 0.6262 # Deliberately un-gamed
  confusion_matrix:
    true_negatives: 600
    false_positives: 372
    false_negatives: 10
    true_positives: 40

interpretability_engines:
  global_xai: "shap.TreeExplainer (precomputed across test set; saved as models/shap_values_test.pkl)"
  local_patient_xai: "shap.plots.waterfall rendered dynamically on user input"
  local_edge_case_xai: "lime.lime_tabular.LimeTabularExplainer serialized with dill to models/lime_explainer.pkl"
  edge_case_mining: "edgecasedec.py filters False Positives, False Negatives, and 45-55% uncertain predictions"

application:
  framework: "Streamlit 1.57.0 (app.py)"
  caching: "st.cache_resource for models, preprocessor, and precomputed SHAP data"
  tabs:
    - "Tab 1: Prediction & SHAP (interactive sliders + live waterfall)"
    - "Tab 2: Global Insights (SHAP Beeswarm + Bar chart)"
    - "Tab 3: Edge Cases (LIME local explanations on audited cases)"
    - "Tab 4: Model Performance (metrics + confusion matrix triage breakdown)"
    - "Tab 5: Understanding Imbalance (narrative on SMOTE removal vs. scale_pos_weight)"

key_source_files:
  - "data_loader.py: Ingestion, cleaning, feature engineering, and preprocessor fitting"
  - "model_trainer.py: Cost-sensitive XGBoost training and evaluation @ 0.15 cutoff"
  - "shap_explainer.py: SHAP calculation and verification plot generation"
  - "lime_expl.py: LIME initialization and dill serialization"
  - "edgecasedec.py: Post-hoc edge-case mining and JSON serialization"
  - "app.py: Multi-tab Streamlit diagnostic user interface"
```
</details>

---

## 🌐 Web Application & REST API (`feat/web-api`)

In addition to the Streamlit triage dashboard (`app.py`), the project includes a production-ready, decoupled web interface (`web/`) backed by a high-efficiency FastAPI REST service (`api/`), engineered for zero-cost deployment on **Render Free Tier** and **Vercel**.

```
┌──────────────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
│        Vercel (React 18 + Three.js)          │        │        Render Free Tier (FastAPI Service)    │
│  - Interactive 3D Parametric Heart Canvas    │  HTTP  │  - Python 3.12+ (Uvicorn ASGI)               │
│  - Dynamic Triage Strategy Switcher          │───────>│  - 5-Fold TreeSHAP Log-Odds Attribution      │
│  - Live SHAP Waterfall & Local Surrogates    │        │  - In-House Local Surrogate (LIME-Style)     │
│  - Zero PII Stored (No LocalStorage/Cookies) │        │  - Anti-Spoofed IP Rate Limiter & Scoped CORS│
└──────────────────────────────────────────────┘        └──────────────────────────────────────────────┘
```

### Key Technical Characteristics

1. **Exact 5-Fold TreeSHAP in Raw Log-Odds**:
   - `models/calibrated_pipeline.joblib` contains an ensemble of 5 cross-validation estimators. SHAP attributions are computed by running native TreeSHAP across all 5 boosters, averaging margins, base values, and contributions.
   - Attributions are strictly additive: $\text{base\_value} + \sum \phi_i = \text{raw\_margin}$ within numerical precision ($< 10^{-6}$).
   - The engineered interaction feature `risk_factors_count` is presented as its own independent row ("combined risk factors") rather than arbitrarily split across clinical inputs, preserving exact additivity.
   - The UI explicitly clarifies that SHAP bars explain the raw model score (log-odds), while the headline probability is isotonic-calibrated.
2. **Lean In-House Local Surrogate (LIME-Style)**:
   - To stay comfortably under Render's 512 MB memory limit, a custom local surrogate (LIME-style) was developed. It perturbs inputs locally and reports risk impact directly in **calibrated probability percentage points** in $< 30\text{ ms}$, avoiding heavyweight dependencies.
3. **Anti-Spoofing Rate Limiting & Zero-PII Security**:
   - IP rate limiting (`slowapi`) enforces `TRUSTED_PROXIES` validation: `X-Forwarded-For` headers are only trusted from declared reverse proxies, preventing client header spoofing.
   - Requests and responses containing patient features are **never logged** (`api/main.py` sanitizes access logs to path and status code only).
   - Validation errors (HTTP 422) return generic guidance and never echo patient inputs back to the client.
4. **Cold-Start & Memory Benchmarks**:
   - **Cold Start**: ~7.16s to load pipeline, verify schemas, and serve the first prediction.
   - **Peak Memory**: ~104 MB (well under the 400 MB constraint and 512 MB Render ceiling).
   - **Prediction Latency**: ~821 ms (including 5-fold TreeSHAP and local surrogate extraction).

### Local Dual-Process Execution

To run both backend and frontend locally:

```powershell
# 1. Start the FastAPI backend on port 8000
python -m uvicorn api.main:app --port 8000 --reload

# 2. In a separate terminal, launch the Vite React frontend
cd web
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. By default, with `VITE_API_URL` unset, the frontend runs in labeled demo mode with simulated outputs. To connect to your local backend, copy `web/.env.example` to `web/.env.local`:
```bash
VITE_API_URL=http://localhost:8000
```

### Production Deployment

#### 1. Backend on Render (Free Web Service)
- **Repo Root**: Root of this repository.
- **Blueprint**: Connect your repo and deploy via [`render.yaml`](render.yaml) or configure manually:
  - **Environment**: Python 3.12+
  - **Build Command**: `pip install -r api/requirements.txt`
  - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT --proxy-headers`
  - **Environment Variables**:
    - `ENVIRONMENT`: `production`
    - `FRONTEND_URL`: `https://your-app.vercel.app` (your production Vercel domain)
    - `VERCEL_PREVIEW_REGEX`: `^https:\/\/stroke-xai-.*\.vercel\.app$` (scoped to your project)

#### 2. Frontend on Vercel
- **Root Directory**: `web`
- **Framework Preset**: Vite
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Environment Variables**:
  - `VITE_API_URL`: `https://your-render-service.onrender.com`
- When `VITE_API_URL` is set, the frontend automatically warms up the Render backend (`/health`) on initial load and displays a clear cold-start notice if Render is waking up from idle.

---

## 📚 References & Acknowledgements

- **Dataset**: Sourced from the [Healthcare Dataset Stroke Data](https://raw.githubusercontent.com/karavokyrismichail/Stroke-Prediction---Random-Forest/main/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv) mirror, originally provided on [Kaggle by fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
- **XGBoost**: Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794).
- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30.
- **LIME**: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1135–1144).

---

## 📄 License & Maintainer

- **License**: Released under the **MIT License** with an explicit **Clinical Research Disclaimer** (see [`LICENSE`](LICENSE)). The model and software are provided for educational and research demonstration purposes only, without express or implied clinical diagnostic warranty.
- **Author & Maintainer**: **Yagas Vashist** ([dummycodertech](https://github.com/dummycodertech)) — Contact: `yagasvashist@gmail.com`.
