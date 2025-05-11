import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest

# Load data and models from GitHub raw links
@st.cache_data
def load_data_and_models():
    base_url = "https://github.com/gobika31/heartTransplantation.git"

    df1 = pd.read_csv(base_url + "reduced_module1.csv")
    df2 = pd.read_csv(base_url + "reduced_module2.csv")
    df3 = pd.read_csv(base_url + "reduced_module3.csv")
    df4 = pd.read_csv(base_url + "reduced_module4.csv")

    xgb1 = joblib.load("xgb_module1_donor_recipient_matching.pkl")
    rsf2 = joblib.load("rsf_module2_post_transplant_monitoring.pkl")
    rsf3 = joblib.load("rsf_module3_long_term_survival.pkl")
    xgb4 = joblib.load("xgb_module4_risk_stratification.pkl")

    return df1, df2, df3, df4, xgb1, rsf2, rsf3, xgb4

df1, df2, df3, df4, xgb1, rsf2, rsf3, xgb4 = load_data_and_models()

st.title("Heart Transplantation ML System")

module = st.sidebar.selectbox("Choose Module", [
    "Module 1: Donor-Recipient Matching",
    "Module 2: Post-Transplant Monitoring",
    "Module 3: Long-Term Survival Prediction",
    "Module 4: Risk Stratification"
])

if module == "Module 1: Donor-Recipient Matching":
    st.header("Module 1: Donor-Recipient Matching")
    st.write("Predicts compatibility between donor and recipient.")
    sample = df1.drop(columns=["CRSMATCH_DONE"]).iloc[[0]]
    prob = xgb1.predict_proba(sample)[:, 1][0]
    st.success(f"Match Score: {prob:.3f}")

elif module == "Module 2: Post-Transplant Monitoring":
    st.header("Module 2: Survival over Time")
    st.write("Survival probability over time post transplant.")
    sample = df2.drop(columns=["365DaySurvival", "TX_YEAR"]).iloc[[0]]
    event = df2["365DaySurvival"].iloc[0] == 1
    tx_year = df2["TX_YEAR"].iloc[0]

    y = Surv.from_arrays([event], [tx_year])
    probs = rsf2.predict_survival_function(sample.to_numpy(), return_array=True)[0]

    fig, ax = plt.subplots()
    ax.step(rsf2.unique_times_, probs, where="post")
    ax.set_title("Survival Probability over Time")
    ax.set_xlabel("Days since transplant")
    ax.set_ylabel("Survival Probability")
    st.pyplot(fig)

elif module == "Module 3: Long-Term Survival Prediction":
    st.header("Module 3: Long-Term Survival Prediction")
    sample = df3.drop(columns=["365DaySurvival", "TX_YEAR"]).iloc[[0]]
    probs = rsf3.predict_survival_function(sample.to_numpy(), return_array=True)[0]
    times = [1095, 1825, 3650]
    survs = [np.interp(t, rsf3.unique_times_, probs) for t in times]
    for y, p in zip(["3", "5", "10"], survs):
        st.write(f"Predicted {y}-year survival: {p:.2f}")

    fig, ax = plt.subplots()
    ax.step(rsf3.unique_times_, probs, where="post")
    ax.set_title("Long-Term Survival Curve")
    st.pyplot(fig)

elif module == "Module 4: Risk Stratification":
    st.header("Module 4: Risk Stratification")
    sample = df4.drop(columns=["risk_level"]).iloc[[0]]
    pred = xgb4.predict(sample)[0]
    st.success(f"Predicted Risk Level: {pred}")




