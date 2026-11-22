import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import shap
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
    # Try models/ first, then models/v2/
    model_dir = "models"
    cal_pipe = joblib.load(f"{model_dir}/calibrated_pipeline.joblib")
    raw_pipe = joblib.load(f"{model_dir}/pipeline.joblib")
    with open(f"{model_dir}/metadata.json", "r") as f:
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
    shap_explainer = joblib.load(f"{model_dir}/shap_explainer.joblib")
    
    return cal_pipe, raw_pipe, metadata, threshold_table, edge_cases, conformal_meta, model_cmp, final_test, shap_explainer

raw_feature_names = [
    'age', 'avg_glucose_level', 'bmi', 'gender', 'hypertension',
    'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
]

cal_pipe, raw_pipe, metadata, threshold_table, edge_cases, conformal_meta, model_cmp, final_test, shap_explainer = load_v2_artifacts()

# Thresholds available
primary_eval = final_test["primary_model_evaluation"]
threshold = float(metadata.get("chosen_threshold", 0.0826))
q0 = conformal_meta.get("p_threshold_for_label_0", 0.1237)
q1 = conformal_meta.get("p_threshold_for_label_1", 0.0260)

st.title("Stroke Risk Triage System (Production v2 Pipeline)")
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
            proc_names = prep_step.get_feature_names_out().tolist()
            
            import xgboost as xgb
            dm = xgb.DMatrix(X_trans.values, feature_names=proc_names)
            phi = clf_step.get_booster().predict(dm, pred_contribs=True)[0]
            base_value = phi[-1]
            contribs = phi[:-1]
            
            # Aggregate one-hot back to raw features
            agg_dict = {f: 0.0 for f in raw_feature_names}
            for col_name, val in zip(proc_names, contribs):
                matched = False
                for rf in raw_feature_names:
                    if col_name.startswith(f"cat__{rf}_") or col_name == f"num__{rf}" or col_name == f"bin__{rf}" or col_name == rf:
                        agg_dict[rf] += val
                        matched = True
                        break
                if not matched and "risk_factors_count" in col_name:
                    agg_dict["risk_factors_count"] = agg_dict.get("risk_factors_count", 0.0) + val
            
            agg_series = pd.Series(agg_dict).sort_values(key=abs, ascending=True)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in agg_series.values]
            ax.barh(agg_series.index, agg_series.values, color=colors)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
            ax.set_xlabel("Attribution to Log-Odds (SHAP Contribution)")
            ax.set_title("Aggregated Clinical Feature Impact (Red = Risk Factor, Blue = Protective)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()

with tab2:
    st.header("Global Feature Insights & Interactions (SHAP)")
    st.markdown("Aggregated feature-level attributions and learned non-linear clinical risk surfaces.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Aggregated Feature Importance")
        st.image("plots/v2/shap_feature_importance.png", caption="Aggregated mean absolute SHAP values per clinical feature.")
    with col2:
        st.subheader("Non-linear Age Dependency")
        st.image("plots/v2/shap_dependence_age.png", caption="SHAP dependence plot for patient age.")
        
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Age × Hypertension Interaction")
        st.image("plots/v2/shap_interaction_age_hypertension.png", caption="Excess risk from combined age and hypertensive status.")
    with col4:
        st.subheader("Age × Glucose Interaction")
        st.image("plots/v2/shap_interaction_age_glucose.png", caption="Synergistic interaction between hyperglycaemia and advanced age.")

with tab3:
    st.header("Edge Cases, Borderline Patients & Referrals")
    st.markdown(
        f"Auditing failure modes around the operating threshold (`{threshold:.3f}`) and within "
        f"the conformal ambiguous region."
    )
    
    case_cat = st.selectbox(
        "Select Edge-Case Category",
        ["borderline_operating_threshold", "conformal_ambiguous_referrals", "confident_false_positives", "missed_strokes_false_negatives"]
    )
    
    case_list = edge_cases.get(case_cat, [])
    if case_list:
        st.markdown(f"Found **{len(case_list)}** patients in this category.")
        sel_idx = st.selectbox("Select Patient Case", range(len(case_list)), format_func=lambda i: f"Patient #{case_list[i]['dataset_index']} (Risk: {case_list[i]['calibrated_probability']:.1%}, Actual: {'Stroke' if case_list[i]['actual_stroke']==1 else 'No Stroke'})")
        case = case_list[sel_idx]
        
        c1, c2 = st.columns(2)
        with c1:
            st.json(case["clinical_profile"])
        with c2:
            st.metric("Calibrated Stroke Risk", f"{case['calibrated_probability']:.1%}")
            st.metric("Actual Clinical Outcome", "STROKE (1)" if case["actual_stroke"] == 1 else "NO STROKE (0)")
            st.info(f"Clinical notes: {case['description']}")

with tab4:
    st.header("Clinical Triage & Operating Thresholds")
    st.markdown(
        "Performance measured on held-out test data across three pre-selected out-of-fold operating points:"
    )
    
    # Display table of operating points from final_test_report
    ops = final_test.get("operating_thresholds_evaluation", {})
    op_rows = []
    for k, v in ops.items():
        m = v["metrics"]
        ci = v["bootstrap_95_ci"]
        op_rows.append({
            "Operating Point": v["name"],
            "Threshold": f"{v['threshold']:.4f}",
            "Recall (Sensitivity)": f"{m['recall']:.1%} [{ci['recall']['lo']:.1%}, {ci['recall']['hi']:.1%}]",
            "Precision (PPV)": f"{m['precision']:.1%} [{ci['precision']['lo']:.1%}, {ci['precision']['hi']:.1%}]",
            "Specificity": f"{m['specificity']:.1%} [{ci['specificity']['lo']:.1%}, {ci['specificity']['hi']:.1%}]",
            "False Alarms (FP)": f"{m['fp']} [{ci['fp']['lo']:.0f}, {ci['fp']['hi']:.0f}]",
            "NNS": f"{m['nns']:.1f} [{ci['nns']['lo']:.1f}, {ci['nns']['hi']:.1f}]",
        })
    st.table(pd.DataFrame(op_rows))

    st.markdown("---")
    st.subheader("Simulate Alternative Operating Thresholds")
    sim_threshold = st.slider("Adjust Decision Threshold", 0.01, 0.50, threshold, 0.005)
    
    df_sens = pd.DataFrame(threshold_table)
    st.subheader(f"Performance at Selected Threshold ({sim_threshold:.3f})")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    closest_row = df_sens.iloc[(df_sens['threshold_value'] - sim_threshold).abs().argsort()[:1]].iloc[0]
    col_m1.metric("Recall (Sensitivity)", f"{closest_row['recall']:.1%}")
    col_m2.metric("Precision (PPV)", f"{closest_row['precision']:.1%}")
    col_m3.metric("Number Needed to Screen (NNS)", f"{closest_row['nns']}")
    col_m4.metric("Flagged per 1,000 Patients", f"{closest_row['flagged_per_1000']:.0f}")

    st.markdown("---")
    st.subheader("Decision Curve Analysis (Net Clinical Benefit)")
    st.image("plots/v2/dca_calibrated.png", caption="Net benefit across decision thresholds.")

with tab5:
    st.header("Model Zoo & Imbalance Ablation Study")
    st.markdown(
        "Controlled comparison across 13 model configurations evaluated under identical 5x5 CV folds. "
        "SMOTE was fitted strictly inside folds via `imblearn.pipeline`."
    )
    st.dataframe(pd.DataFrame(model_cmp["comparison_table"]))
    st.image("plots/v2/model_comparison.png", caption="Model Zoo Comparison on PR-AUC, ROC-AUC, and Recall.")
