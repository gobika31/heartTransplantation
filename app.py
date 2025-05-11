# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Heart Transplant Decision Support", layout="wide")

st.title("🫀 AI-Powered Heart Transplant Decision Support System")
st.sidebar.title("Modules")

# Load models
@st.cache_resource
def load_models():
    base_path = "models/"  # Update if stored in a different GitHub folder
    models = {
        "Module 1": joblib.load(base_path + "xgb_module1_donor_recipient_matching.pkl"),
        "Module 2": joblib.load(base_path + "rsf_module2_post_transplant_monitoring.pkl"),
        "Module 3": joblib.load(base_path + "rsf_module3_long_term_survival.pkl"),
        "Module 4": joblib.load(base_path + "xgb_module4_risk_stratification.pkl"),
    }
    return models

models = load_models()

# Module Selection
module = st.sidebar.radio("Select a Module", ["Module 1", "Module 2", "Module 3", "Module 4"])

# Common input method
def get_user_input(features):
    input_data = {}
    for col in features:
        input_data[col] = st.number_input(f"{col}", value=0.0)
    return pd.DataFrame([input_data])

# Module 1
if module == "Module 1":
    st.header("Donor-Recipient Matching")
    df = pd.read_csv("datasets/reduced_module1.csv")
    input_df = get_user_input(df.drop(columns=['CRSMATCH_DONE']).columns)
    match_score = models["Module 1"].predict_proba(input_df)[:, 1][0]
    st.success(f"Match Score: {match_score:.2f}")

# Module 2
elif module == "Module 2":
    st.header("Post-Transplantation Monitoring")
    df = pd.read_csv("datasets/reduced_module2.csv")
    input_df = get_user_input(df.drop(columns=['365DaySurvival', 'TX_YEAR', 'event']).columns)
    rsf = models["Module 2"]
    surv_fn = rsf.predict_survival_function(input_df.to_numpy(), return_array=True)[0]

    st.subheader("Survival Curve (Next 3 Years)")
    times = rsf.unique_times_
    plt.step(times, surv_fn, where="post")
    plt.xlabel("Days since Transplant")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    st.pyplot(plt)

# Module 3
elif module == "Module 3":
    st.header("Long-Term Survival Prediction")
    df = pd.read_csv("datasets/reduced_module3.csv")
    input_df = get_user_input(df.drop(columns=['365DaySurvival', 'TX_YEAR', 'event']).columns)
    rsf = models["Module 3"]
    surv_fn = rsf.predict_survival_function(input_df.to_numpy(), return_array=True)[0]
    times = [1095, 1825, 3650]
    probs = [np.interp(t, rsf.unique_times_, surv_fn) for t in times]
    st.success(f"3-Year Survival: {probs[0]:.2f}, 5-Year: {probs[1]:.2f}, 10-Year: {probs[2]:.2f}")

# Module 4
elif module == "Module 4":
    st.header("Risk Stratification")
    df = pd.read_csv("datasets/reduced_module4.csv")
    input_df = get_user_input(df.drop(columns=['risk_level']).columns)
    risk = models["Module 4"].predict(input_df)[0]
    risk_map = {0: "Low", 1: "Moderate", 2: "High"}
    st.success(f"Predicted Risk Level: {risk_map[risk]}")

