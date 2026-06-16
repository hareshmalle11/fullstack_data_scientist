"""Streamlit app to predict house price using a saved SVR pipeline."""
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "svr_real_estate_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def build_input_df(house_age, mrt_distance, convenience_stores):
    return pd.DataFrame(
        {
            "X2 house age": [house_age],
            "X3 distance to the nearest MRT station": [mrt_distance],
            "X4 number of convenience stores": [convenience_stores],
        }
    )


st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")
st.title("Real Estate Price Predictor")
st.write("Predict house price using age, MRT distance, and nearby convenience stores.")

model = load_model()
if model is None:
    st.error("Model file not found. Run svr_regression.ipynb to train and save svr_real_estate_model.pkl.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    house_age = st.number_input("House Age (years)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
with col2:
    mrt_distance = st.number_input(
        "Distance to MRT (meters)", min_value=0.0, value=500.0, step=10.0
    )
with col3:
    convenience_stores = st.number_input(
        "Convenience Stores", min_value=0, max_value=50, value=5, step=1
    )

if st.button("Predict"):
    input_df = build_input_df(house_age, mrt_distance, convenience_stores)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price per Unit Area: {prediction:.2f}")
