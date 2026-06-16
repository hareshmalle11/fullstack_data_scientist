"""Streamlit app to predict loan approval using a saved logistic regression pipeline."""
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "logistic_regression_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def build_input_df(
    person_age,
    person_gender,
    person_education,
    person_income,
    person_emp_exp,
    person_home_ownership,
    loan_amnt,
    loan_intent,
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,
    credit_score,
    previous_loan_defaults_on_file,
):
    return pd.DataFrame(
        {
            "person_age": [person_age],
            "person_gender": [person_gender],
            "person_education": [person_education],
            "person_income": [person_income],
            "person_emp_exp": [person_emp_exp],
            "person_home_ownership": [person_home_ownership],
            "loan_amnt": [loan_amnt],
            "loan_intent": [loan_intent],
            "loan_int_rate": [loan_int_rate],
            "loan_percent_income": [loan_percent_income],
            "cb_person_cred_hist_length": [cb_person_cred_hist_length],
            "credit_score": [credit_score],
            "previous_loan_defaults_on_file": [previous_loan_defaults_on_file],
        }
    )


st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval.")

model = load_model()
if model is None:
    st.error(
        "Model file not found. Run logistic_regression.ipynb to train and save logistic_regression_model.pkl."
    )
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    person_gender = st.selectbox("Gender", ["female", "male"])
    person_education = st.selectbox(
        "Education",
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    )
    person_income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)

with col2:
    person_emp_exp = st.number_input(
        "Employment Experience (years)", min_value=0, max_value=50, value=2, step=1
    )
    person_home_ownership = st.selectbox(
        "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )
    loan_amnt = st.number_input("Loan Amount", min_value=500.0, value=15000.0, step=500.0)
    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        ],
    )

with col3:
    loan_int_rate = st.number_input(
        "Interest Rate", min_value=0.0, max_value=40.0, value=12.0, step=0.1
    )
    loan_percent_income = st.number_input(
        "Loan % of Income", min_value=0.0, max_value=1.0, value=0.25, step=0.01
    )
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length", min_value=0.0, max_value=50.0, value=3.0, step=0.5
    )
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
    previous_loan_defaults_on_file = st.selectbox(
        "Previous Defaults", ["No", "Yes"]
    )

if st.button("Predict"):
    input_df = build_input_df(
        person_age,
        person_gender,
        person_education,
        person_income,
        person_emp_exp,
        person_home_ownership,
        loan_amnt,
        loan_intent,
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        previous_loan_defaults_on_file,
    )
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    label = "Approved" if prediction == 1 else "Rejected"
    st.success(f"Prediction: {label}")
    st.write(f"Approval Probability: {proba:.2%}")
