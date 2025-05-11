import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    return {
        "Module 1": joblib.load("xgb_module1_donor_recipient_matching.pkl"),
        "Module 2": joblib.load("rsf_module2_post_transplant_monitoring (3).pkl"),
        "Module 3": joblib.load("rsf_module3_long_term_survival.pkl"),
        "Module 4": joblib.load("xgb_module4_risk_stratification.pkl"),
    }


models = load_models()

# App layout
st.title("Heart Transplantation ML System")
module = st.selectbox("Select a Module", list(models.keys()))

# Load corresponding data sample (optional)
csv_file = f"reduced_module{module[-1]}.csv"
try:
    df = pd.read_csv(csv_file)
    st.subheader("Sample Input Data")
    st.dataframe(df.head())
except FileNotFoundError:
    st.warning(f"{csv_file} not found. Please upload input data.")

# Run prediction
if st.button("Run Prediction"):
    try:
        model = models[module]
        prediction = model.predict(df)
        st.success("Prediction complete.")
        st.write(prediction)
    except Exception as e:
        st.error(f"Prediction failed: {e}")



