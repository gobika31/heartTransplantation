import streamlit as st
import pandas as pd
import joblib

# Cache model loading to prevent reloading on every run
@st.cache_resource
def load_models():
    models = {
        "Module 1": joblib.load("xgb_module1_donor_recipient_matching.pkl"),
        "Module 2": joblib.load("rsf_module2_post_transplant_monitoring.pkl"),
        "Module 3": joblib.load("rsf_module3_long_term_survival.pkl"),
        "Module 4": joblib.load("xgb_module4_risk_stratification.pkl")
    }
    return models

# Load models once
models = load_models()

# App UI
st.title("Heart Transplantation ML System")
module = st.selectbox("Select a Module", ["Module 1", "Module 2", "Module 3", "Module 4"])

st.write(f"You selected: {module}")
model = models[module]

# Load example dataset
data_file = f"reduced_module{module[-1]}.csv"
try:
    df = pd.read_csv(data_file)
    st.write("Sample input data:")
    st.dataframe(df.head())
except FileNotFoundError:
    st.warning(f"{data_file} not found.")

# Placeholder: Prediction button and output
if st.button("Run Prediction"):
    st.info("Running prediction...")

    # Simulate predictions on full dataset
    try:
        if "xgb" in data_file:
            preds = model.predict(df)
        else:
            preds = model.predict(df)  # or model.predict_survival_function(df) for RSF
        st.success("Prediction complete.")
        st.write(preds)
    except Exception as e:
        st.error(f"Prediction failed: {e}")


