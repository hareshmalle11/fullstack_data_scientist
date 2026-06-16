"""Streamlit app to predict Titanic fare using a saved model pipeline."""
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = os.path.join(os.path.dirname(__file__), "ticket_model_small.pkl")


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def build_input_df(sex, age, pclass):
    return pd.DataFrame(
        {
            "Sex": [sex],
            "Age": [age],
            "Pclass": [pclass],
        }
    )


st.set_page_config(page_title="Titanic Fare Predictor", layout="centered")
st.title("Titanic Fare Predictor")
st.write("Predict fare based on Sex, Age, and Pclass.")

artifacts = load_artifacts()
if artifacts is None:
    st.error("Model file not found. Run rf_ticket_classification.ipynb to train and save ticket_model_small.pkl.")
    st.stop()

model = artifacts["model"]

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["female", "male"])
    pclass = st.selectbox("Pclass", [1, 2, 3])
with col2:
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

if st.button("Predict"):
    input_df = build_input_df(sex, age, pclass)
    predicted_fare = float(model.predict(input_df)[0])
    st.success(f"Predicted Fare: {predicted_fare:.2f}")
