# Step 3: Save your Streamlit app (you can modify this block with your app code)
%%writefile app.py
import streamlit as st
import joblib
import numpy as np

st.title("Heart Transplantation Prediction System")

st.sidebar.header("Select Module")
module = st.sidebar.selectbox("Choose a module", ["Module 1: Donor-Recipient Matching",
                                                   "Module 2: Post-Transplant Monitoring",
                                                   "Module 3: Long-Term Survival Prediction",
                                                   "Module 4: Risk Stratification"])

base_path = "/content/drive/MyDrive/dataset/"

if module == "Module 1: Donor-Recipient Matching":
    st.subheader("Predict Match Score")
    model = joblib.load(base_path + "xgb_module1_donor_recipient_matching.pkl")
    input_data = st.text_area("Enter feature values (comma-separated)", "1,4,11,3,1,2013,...")
    if st.button("Predict"):
        features = np.array([list(map(float, input_data.split(",")))]).reshape(1, -1)
        score = model.predict_proba(features)[0][1]
        st.success(f"Predicted Match Score: {score:.3f}")

elif module == "Module 2: Post-Transplant Monitoring":
    st.subheader("Predict 1-Year Survival Probability")
    rsf = joblib.load(base_path + "rsf_module2_post_transplant_monitoring.pkl")
    input_data = st.text_area("Enter patient features (comma-separated)", "value1,value2,...")
    if st.button("Predict"):
        features = np.array([list(map(float, input_data.split(",")))]).reshape(1, -1)
        surv_probs = rsf.predict_survival_function(features, return_array=True)[0]
        st.success(f"Day 365 Survival Probability: {np.interp(365, rsf.unique_times_, surv_probs):.3f}")

elif module == "Module 3: Long-Term Survival Prediction":
    st.subheader("Survival Prediction at 3, 5, 10 Years")
    rsf = joblib.load(base_path + "rsf_module3_long_term_survival.pkl")
    input_data = st.text_area("Enter patient features (comma-separated)", "value1,value2,...")
    if st.button("Predict"):
        features = np.array([list(map(float, input_data.split(",")))]).reshape(1, -1)
        surv_probs = rsf.predict_survival_function(features, return_array=True)[0]
        for day in [1095, 1825, 3650]:
            prob = np.interp(day, rsf.unique_times_, surv_probs)
            st.write(f"Survival Probability at {day//365} years: {prob:.3f}")

elif module == "Module 4: Risk Stratification":
    st.subheader("Predict Risk Level")
    model = joblib.load(base_path + "xgb_module4_risk_stratification.pkl")
    input_data = st.text_area("Enter patient features (comma-separated)", "value1,value2,...")
    if st.button("Predict"):
        features = np.array([list(map(float, input_data.split(",")))]).reshape(1, -1)
        prediction = model.predict(features)[0]
        st.success(f"Predicted Risk Level: {int(prediction)}")
