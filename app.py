import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import dill
import matplotlib.pyplot as plt
from sklearn import set_config

set_config(transform_output="pandas")
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
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
        
    raw_feature_names = ['age', 'avg_glucose_level', 'bmi', 'risk_factors_count', 'gender', 
                         'hypertension', 'heart_disease', 'ever_married', 'work_type', 
                         'Residence_type', 'smoking_status']
        
    return model, preprocessor, shap_explainer, shap_values, lime_explainer, metadata, edge_cases, test_data, raw_feature_names

model, preprocessor, shap_explainer, shap_values, lime_explainer, metadata, edge_cases, test_data, raw_feature_names = load_artifacts()
processed_feature_names = test_data['X'].columns.tolist()

def predict_fn(x_numpy):
    df = pd.DataFrame(x_numpy, columns=processed_feature_names)
    return model.predict_proba(df)

st.title("Stroke Risk Prediction: Overcoming Age Bias")
st.markdown("Exploring stroke prediction using XGBoost, Feature Engineering, and Custom Thresholds.")

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
            
            input_df = pd.DataFrame([input_dict], columns=raw_feature_names)
            processed_input = preprocessor.transform(input_df)
            prob = model.predict_proba(processed_input)[0][1]
            
            threshold = metadata.get('threshold', 0.15)
            
            st.subheader(f"Stroke Risk: {prob:.1%}")
            if prob >= threshold:
                st.error(f"HIGH RISK (Exceeds {threshold:.1%} clinical threshold)")
            else:
                st.success(f"LOW RISK (Below {threshold:.1%} clinical threshold)")
                
            st.markdown("### Why this prediction? (SHAP Waterfall)")
            shap_val_single = shap_explainer(processed_input)
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_val_single[0], max_display=10, show=False)
            st.pyplot(fig)
            plt.clf()

with tab2:
    st.header("Global Feature Importance")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance (Beeswarm)")
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.beeswarm(shap_values, max_display=10, show=False)
        st.pyplot(fig)
        plt.clf()
    with col2:
        st.subheader("Average Impact (Bar)")
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig)
        plt.clf()

with tab3:
    st.header("Investigating Model Edge Cases")
    case_type = st.selectbox("Select Case Category", ["false_positives", "false_negatives", "uncertain"], format_func=lambda x: x.replace("_", " ").title())
    selected_cases = edge_cases[case_type]
    
    if not selected_cases:
        st.info("No cases found for this category.")
    else:
        case_idx_options = {i: f"Case {i+1} (Prob: {c['prob']:.2f}, Actual: {c['actual']})" for i, c in enumerate(selected_cases)}
        selected_i = st.selectbox("Select specific patient record", options=list(case_idx_options.keys()), format_func=lambda x: case_idx_options[x])
        case_data = selected_cases[selected_i]
        real_idx = case_data["index"]
        patient_row = test_data['X'].iloc[real_idx]
        
        st.write(f"**Model Probability:** {case_data['prob']:.1%}")
        st.write(f"**Actual Outcome:** {'Stroke' if case_data['actual'] == 1 else 'No Stroke'}")
        
        st.markdown("### Local Explanation (LIME)")
        exp = lime_explainer.explain_instance(data_row=patient_row.values, predict_fn=predict_fn, num_features=8)
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
        plt.clf()

with tab4:
    st.header("Metrics for Imbalanced Data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PR-AUC", f"{metadata['pr_auc']:.3f}")
    col2.metric("ROC-AUC", f"{metadata['roc_auc']:.3f}")
    col3.metric(f"Recall (@ {metadata['threshold']} thresh)", f"{metadata['classification_report']['1']['recall']:.3f}")
    col4.metric("Accuracy (Misleading)", f"{metadata['accuracy']:.3f}")
    
    st.markdown("---")
    
    col_cm, col_text = st.columns([1, 1.5])
    
    with col_cm:
        st.subheader(f"Confusion Matrix")
        st.caption(f"Evaluated at {metadata['threshold']} probability threshold")
        cm = metadata['confusion_matrix']
        st.code(f"""
                      Predicted Neg   Predicted Pos
        Actual Neg         {cm[0][0]}             {cm[0][1]}
        Actual Pos         {cm[1][0]}              {cm[1][1]}
        """)

    with col_text:
        st.subheader("The Triage Trade-off")
        st.info("""
        **The Recall Win:** By lowering the threshold to 0.15 and adding class weights, we forced the model to stop being conservative. We successfully caught **80% of actual strokes** (True Positives), up from just 24% in our initial unweighted model.
        
        **The Precision Cost:** To cast this wider safety net, the model generated many False Positives (alarms for healthy people). The precision is low, meaning when it rings the alarm, it's often just a warning.
        
        **The Accuracy Drop:** Our accuracy tanked to ~63%. This is a badge of honor! It means we stopped letting the algorithm "cheat" by blindly guessing "No Stroke" for the 95% majority class.
        """)

with tab5:
    st.header("Why did we pivot from our original approach?")
    st.markdown("""
    ### 1. The Trap of Synthetic Data (SMOTE)
    We originally used SMOTE to balance the dataset. However, because strokes are overwhelmingly tied to old age, SMOTE primarily created synthetic 70-year-olds. It failed to teach the model that young people with extreme lifestyle risks (smoking, heart disease) can also have strokes. We ripped out SMOTE to prevent the model from hallucinating.
    
    ### 2. Feature Engineering & Weighting
    Instead of faking data, we passed the raw, imbalanced data to XGBoost, but used `scale_pos_weight`. This mathematical penalty tells the algorithm that missing a stroke case is 20x worse than misclassifying a healthy person. We also engineered a `risk_factors_count` feature to force the model to look at lifestyle severity, not just age.
    
    ### 3. Lowering the Clinical Threshold
    By default, algorithms predict "True" only if probability > 50%. But in clinical screening, a 15% probability of a stroke is incredibly dangerous and warrants medical intervention. By lowering our decision threshold to **0.15**, we accepted more False Positives (alarms) in exchange for a massive boost in **Recall** (catching actual strokes).
    """)
    
    st.warning("""
    ** The Accuracy Illusion:** If a hospital uses a "dumb" model that just guesses "No Stroke" for everyone, it achieves **95% accuracy** but sends 100% of stroke victims home to die. 
    
    By shifting our objective from Accuracy to **Recall**, we built a **Triage Model**. It rings the alarm more often, but it successfully catches 80% of actual strokes. False positives just cost a nurse 5 minutes to run a physical check; False negatives cost lives.
    """)