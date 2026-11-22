import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import joblib
import shap
import dill
import matplotlib.pyplot as plt
from sklearn import set_config

set_config(transform_output="pandas")
st.set_page_config(page_title="Stroke Risk Triage System (XAI v2)", layout="wide")

# Research prototype disclaimer banner
st.warning(
    "⚠️ **RESEARCH PROTOTYPE ONLY**: This triage system is an experimental clinical decision-support research tool. "
    "It is NOT a certified medical diagnostic device and must never be used as the sole basis for clinical treatment."
)

@st.cache_resource
def load_v2_artifacts():
    cal_pipe = joblib.load("models/v2/calibrated_pipeline.joblib")
    raw_pipe = joblib.load("models/v2/pipeline.joblib")
    with open("models/v2/metadata.json", "r") as f:
        metadata = json.load(f)
    with open("reports/threshold_sensitivity.json", "r") as f:
        threshold_table = json.load(f)
    with open("reports/edge_cases_v2.json", "r") as f:
        edge_cases = json.load(f)
    with open("reports/conformal_results.json", "r") as f:
        conformal_meta = json.load(f)
    with open("reports/model_comparison.json", "r") as f:
        model_cmp = json.load(f)
    with open("reports/final_test_report.json", "r") as f:
        final_test = json.load(f)
    shap_explainer = joblib.load("models/v2/shap_explainer.joblib")
    
    return cal_pipe, raw_pipe, metadata, threshold_table, edge_cases, conformal_meta, model_cmp, final_test, shap_explainer

@st.cache_resource
def load_legacy_artifacts():
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("models/shap_explainer.pkl", "rb") as f:
        shap_explainer = pickle.load(f)
    with open("models/shap_values_test.pkl", "rb") as f:
        shap_values = pickle.load(f)
    with open("models/lime_explainer.pkl", "rb") as f:
        lime_explainer = dill.load(f)
    with open("models/training_metadata.json", "r") as f:
        metadata = json.load(f)
    with open("models/edge_cases.json", "r") as f:
        edge_cases = json.load(f)
    with open("data/test_processed.pkl", "rb") as f:
        test_data = pickle.load(f)
    return model, preprocessor, shap_explainer, shap_values, lime_explainer, metadata, edge_cases, test_data

# Sidebar version switch
st.sidebar.title("Pipeline Settings")
version = st.sidebar.radio(
    "Select Pipeline Version",
    ["v2 (Rigorous, Calibrated & Leak-Free)", "v1 (Legacy Baseline)"],
    index=0
)

raw_feature_names = [
    'age', 'avg_glucose_level', 'bmi', 'gender', 'hypertension',
    'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
]

if version.startswith("v2"):
    cal_pipe, raw_pipe, metadata, threshold_table, edge_cases, conformal_meta, model_cmp, final_test, shap_explainer = load_v2_artifacts()
    threshold = float(metadata.get("chosen_threshold", 0.0826))
    q0 = conformal_meta.get("p_threshold_for_label_0", 0.1237)
    q1 = conformal_meta.get("p_threshold_for_label_1", 0.0260)
    
    st.title("Stroke Risk Triage System (v2 Rigorous Pipeline)")
    st.markdown(
        f"**Methodological Upgrades**: Isotonic probability calibration, out-of-fold cost-based threshold selection "
        f"(`{threshold:.3f}`), 90% Mondrian conformal referral sets, and aggregated feature-level SHAP."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Patient Prediction & SHAP", "Global Insights (SHAP)", "Edge Cases & Referrals", "Model Performance & Triage", "Model Zoo & SMOTE Ablation"
    ])

    with tab1:
        st.header("Patient Risk Assessment (Calibrated Triage)")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            age = st.slider("Age (years)", 0, 100, 45)
            glucose = st.number_input("Average Glucose Level (mg/dL)", 50.0, 300.0, 105.0)
            bmi = st.number_input("BMI (kg/m²)", 10.0, 100.0, 26.5)
            gender = st.selectbox("Gender", ["Male", "Female"])
            hypertension = st.selectbox("Hypertension Diagnosis", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            heart_disease = st.selectbox("Heart Disease History", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence = st.selectbox("Residence Type", ["Urban", "Rural"])
            smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
            predict_btn = st.button("Evaluate Patient Risk", type="primary")

        with col2:
            if predict_btn:
                input_df = pd.DataFrame([{
                    "age": age, "avg_glucose_level": glucose, "bmi": bmi,
                    "gender": gender, "hypertension": hypertension, "heart_disease": heart_disease,
                    "ever_married": ever_married, "work_type": work_type, "Residence_type": residence,
                    "smoking_status": smoking
                }])
                
                # Calibrated probability
                prob = cal_pipe.predict_proba(input_df)[0][1]
                
                # Display risk
                st.subheader(f"Calibrated 5-Year Stroke Risk: **{prob:.1%}**")
                
                # Conformal assessment
                is_ambiguous = (prob >= q1) and (prob <= q0)
                if is_ambiguous:
                    st.warning(f"⚠️ **Conformal Prediction: Ambiguous Set {{No Stroke, Stroke}}**. Model recommends second-line clinical review.")
                
                if prob >= threshold:
                    st.error(f"🚨 **HIGH RISK**: Exceeds clinical triage threshold ({threshold:.1%}). Flagged for physical screening.")
                else:
                    st.success(f"✅ **LOW RISK**: Below clinical triage threshold ({threshold:.1%}). Routine monitoring recommended.")

                st.markdown("---")
                st.markdown("### Aggregated Feature Attributions (SHAP)")
                
                # Compute SHAP on raw pipeline's tree classifier
                fe_step = raw_pipe.named_steps["feature_eng"]
                prep_step = raw_pipe.named_steps["preprocessor"]
                clf_step = raw_pipe.named_steps["classifier"]
                
                X_trans = prep_step.transform(fe_step.transform(input_df))
                shap_val = shap_explainer(X_trans)
                
                proc_cols = prep_step.get_feature_names_out().tolist()
                
                from explainability_v2 import aggregate_shap_values
                agg_shap, orig_names = aggregate_shap_values(shap_val.values, proc_cols)
                
                # Plot horizontal bar chart of aggregated SHAP
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sorted_idx = np.argsort(np.abs(agg_shap[0]))
                colors = ["#d62728" if agg_shap[0][i] > 0 else "#2ca02c" for i in sorted_idx]
                ax.barh([orig_names[i] for i in sorted_idx], agg_shap[0][sorted_idx], color=colors)
                ax.set_xlabel("SHAP Impact on Log-Odds (Red = Increases Stroke Risk, Green = Decreases)")
                ax.set_title("Aggregated Patient Feature Importance")
                ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
                st.pyplot(fig)
                plt.clf()

    with tab2:
        st.header("Global Model Interpretability")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Aggregated SHAP Beeswarm")
            st.image("plots/v2/shap_aggregated_beeswarm.png", caption="Combined impact of original clinical features.")
        with col_g2:
            st.subheader("Aggregated Feature Importance (Bar)")
            st.image("plots/v2/shap_aggregated_bar.png", caption="Mean absolute SHAP value per feature.")
        
        st.markdown("---")
        st.subheader("Nonlinear Risk Dependence & Interaction Analysis")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.image("plots/v2/shap_dependence_age.png", caption="SHAP Dependence: Age")
        with col_d2:
            st.image("plots/v2/shap_dependence_glucose_level.png", caption="SHAP Dependence: Glucose")
        with col_d3:
            st.image("plots/v2/shap_dependence_bmi.png", caption="SHAP Dependence: BMI")

    with tab3:
        st.header("Auditing Edge Cases & Ambiguous Referrals")
        case_category = st.selectbox(
            "Select Case Category",
            ["conformal_ambiguous", "near_threshold", "false_positives", "false_negatives"],
            format_func=lambda x: {
                "conformal_ambiguous": "Ambiguous Conformal Sets {0, 1} (Clinical Referrals)",
                "near_threshold": "Near Operating Threshold (Borderline Risk)",
                "false_positives": "High-Confidence False Positives (False Alarms)",
                "false_negatives": "False Negatives (Missed Strokes)",
            }[x]
        )
        cases_list = edge_cases.get(case_category, [])
        if not cases_list:
            st.info("No cases found in this category.")
        else:
            selected_case = st.selectbox(
                "Select Patient Record",
                options=cases_list,
                format_func=lambda c: f"Patient #{c['index']} | Prob: {c['calibrated_prob']:.1%} | Actual: {'Stroke' if c['actual_stroke']==1 else 'No Stroke'}"
            )
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.json(selected_case["patient_features"])
            with col_c2:
                st.write(f"**Calibrated Probability**: {selected_case['calibrated_prob']:.1%}")
                st.write(f"**Actual Outcome**: {'Stroke' if selected_case['actual_stroke'] == 1 else 'No Stroke'}")
                st.write(f"**Conformal Set**: {selected_case['conformal_set']}")
                st.write("**Top Attributed SHAP Factors**:")
                for item in selected_case["top_shap_factors"]:
                    st.write(f"- `{item['feature']}`: {item['shap_value']:+.4f}")

    with tab4:
        st.header("Out-of-Fold Triage Performance & Interactive Threshold Simulation")
        st.caption("All simulation numbers are computed on stored out-of-fold (OOF) CV predictions to prevent test leakage.")
        
        sim_threshold = st.slider(
            "Interactive Decision Threshold",
            min_value=0.01, max_value=0.30, value=threshold, step=0.005,
            format="%.3f"
        )
        
        # Pull from sensitivity table or interpolate
        df_sens = pd.DataFrame(threshold_table)
        st.subheader(f"Performance at Selected Threshold ({sim_threshold:.3f})")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        # Approximate matching row
        closest_row = df_sens.iloc[(df_sens['threshold_value'] - sim_threshold).abs().argsort()[:1]].iloc[0]
        col_m1.metric("Recall (Sensitivity)", f"{closest_row['recall']:.1%}")
        col_m2.metric("Precision (PPV)", f"{closest_row['precision']:.1%}")
        col_m3.metric("Number Needed to Screen (NNS)", f"{closest_row['nns']}")
        col_m4.metric("Flagged per 1,000 Patients", f"{closest_row['flagged_per_1000']:.0f}")

        st.markdown("---")
        st.subheader("Decision Curve Analysis (Net Clinical Benefit)")
        st.image("plots/v2/dca_calibrated.png", caption="Net benefit across decision thresholds.")

    with tab5:
        st.header("Model Zoo & SMOTE Ablation Study")
        st.markdown(
            "Controlled comparison across 13 model configurations evaluated under identical 5x5 CV folds. "
            "SMOTE was fitted strictly inside folds via `imblearn.pipeline`."
        )
        st.dataframe(pd.DataFrame(model_cmp["comparison_table"]))
        st.image("plots/v2/model_comparison.png", caption="Model Zoo Comparison on PR-AUC, ROC-AUC, and Recall.")

else:
    # Legacy v1 Mode
    model, preprocessor, shap_explainer, shap_values, lime_explainer, metadata, edge_cases, test_data = load_legacy_artifacts()
    processed_feature_names = test_data['X'].columns.tolist()

    def predict_fn(x_numpy):
        df = pd.DataFrame(x_numpy, columns=processed_feature_names)
        return model.predict_proba(df)

    st.title("Stroke Risk Prediction (Legacy v1 Baseline)")
    st.markdown("⚠️ Running original uncalibrated model with heuristic threshold (0.15).")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Prediction & SHAP", "Global Insights (SHAP)", "Edge Cases (LIME)", "Model Performance", "Understanding Imbalance"
    ])

    with tab1:
        st.header("Patient Prediction & Explanation")
        col1, col2 = st.columns([1, 2])
        with col1:
            age = st.slider("Age", 0, 100, 45)
            glucose = st.number_input("Avg Glucose Level", 50.0, 300.0, 100.0)
            bmi = st.number_input("BMI", 10.0, 100.0, 25.0)
            gender = st.selectbox("Gender", ["Male", "Female"])
            hypertension = st.selectbox("Hypertension", [0, 1])
            heart_disease = st.selectbox("Heart Disease", [0, 1])
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], index=0)
            residence = st.selectbox("Residence Type", ["Urban", "Rural"])
            smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
            predict_btn = st.button("Predict Risk", type="primary")

        with col2:
            if predict_btn:
                risk_count = hypertension + heart_disease + (1 if smoking in ['smokes', 'formerly smoked'] else 0)
                input_dict = {
                    "age": age, "avg_glucose_level": glucose, "bmi": bmi, "risk_factors_count": risk_count,
                    "gender": gender, "hypertension": hypertension, "heart_disease": heart_disease,
                    "ever_married": ever_married, "work_type": work_type, "Residence_type": residence,
                    "smoking_status": smoking
                }
                raw_names = ['age', 'avg_glucose_level', 'bmi', 'risk_factors_count', 'gender',
                             'hypertension', 'heart_disease', 'ever_married', 'work_type',
                             'Residence_type', 'smoking_status']
                input_df = pd.DataFrame([input_dict], columns=raw_names)
                processed_input = preprocessor.transform(input_df)
                prob = model.predict_proba(processed_input)[0][1]
                threshold = metadata.get('threshold', 0.15)
                st.subheader(f"Stroke Risk: {prob:.1%}")
                if prob >= threshold:
                    st.error(f"HIGH RISK (Exceeds {threshold:.1%} clinical threshold)")
                else:
                    st.success(f"LOW RISK (Below {threshold:.1%} clinical threshold)")
                shap_val_single = shap_explainer(processed_input)
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(shap_val_single[0], max_display=10, show=False)
                st.pyplot(fig)
                plt.clf()

    with tab2:
        st.header("Global Feature Importance (v1)")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.beeswarm(shap_values, max_display=10, show=False)
            st.pyplot(fig)
            plt.clf()
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.bar(shap_values, max_display=10, show=False)
            st.pyplot(fig)
            plt.clf()

    with tab3:
        st.header("Investigating Model Edge Cases (LIME)")
        case_type = st.selectbox("Select Case Category", ["false_positives", "false_negatives", "uncertain"])
        selected_cases = edge_cases.get(case_type, [])
        if selected_cases:
            case_idx_options = {i: f"Case {i+1} (Prob: {c['prob']:.2f}, Actual: {c['actual']})" for i, c in enumerate(selected_cases)}
            selected_i = st.selectbox("Select specific patient record", options=list(case_idx_options.keys()), format_func=lambda x: case_idx_options[x])
            case_data = selected_cases[selected_i]
            real_idx = case_data["index"]
            patient_row = test_data['X'].iloc[real_idx]
            st.write(f"**Model Probability:** {case_data['prob']:.1%}")
            st.write(f"**Actual Outcome:** {'Stroke' if case_data['actual'] == 1 else 'No Stroke'}")
            exp = lime_explainer.explain_instance(data_row=patient_row.values, predict_fn=predict_fn, num_features=8)
            st.pyplot(exp.as_pyplot_figure())
            plt.clf()

    with tab4:
        st.header("Metrics for Imbalanced Data (v1)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PR-AUC", f"{metadata['pr_auc']:.3f}")
        col2.metric("ROC-AUC", f"{metadata['roc_auc']:.3f}")
        col3.metric(f"Recall (@ {metadata['threshold']} thresh)", f"{metadata['classification_report']['1']['recall']:.3f}")
        col4.metric("Accuracy (Misleading)", f"{metadata['accuracy']:.3f}")

    with tab5:
        st.header("Understanding Imbalance (v1 Context)")
        st.info("Legacy documentation on scale_pos_weight vs SMOTE.")
