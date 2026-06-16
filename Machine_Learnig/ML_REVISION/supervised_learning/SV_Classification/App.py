"""Streamlit app to predict Titanic survival using a saved SVM pipeline."""
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "svm_titanic_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def build_input_df(sex, age, ticket, fare):
    return pd.DataFrame(
        {
            "Sex": [sex],
            "Age": [age],
            "Ticket": [ticket],
            "Fare": [fare],
        }
    )


st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("Titanic Survival Predictor")
st.write("Predict survival based on Sex, Age, Ticket, and Fare.")

model = load_model()
if model is None:
    st.error("Model file not found. Run svm_classification.ipynb to train and save svm_titanic_model.pkl.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["female", "male"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

with col2:
    ticket = st.text_input("Ticket", value="A/5 21171")
    fare = st.number_input("Fare", min_value=0.0, value=32.0, step=1.0)

if st.button("Predict"):
    input_df = build_input_df(sex, age, ticket, fare)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    label = "Survived" if prediction == 1 else "Not Survived"
    st.success(f"Prediction: {label}")
    st.write(f"Survival Probability: {proba:.2%}")
