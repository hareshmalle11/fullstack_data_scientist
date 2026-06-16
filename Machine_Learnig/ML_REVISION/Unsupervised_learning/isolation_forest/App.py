import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "isolation_forest_model.pkl"

st.set_page_config(page_title="Anomaly Detector (Isolation Forest)", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #0F172A; }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-title { color: #94A3B8; font-size: 14px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { font-size: 32px; font-weight: 700; }
    .stButton>button {
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        color: white; border: none; border-radius: 8px; padding: 12px 24px; font-weight: 700; width: 100%; transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4); }
</style>
""", unsafe_allow_html=True)

st.title("🚨 Anomaly and Fraud Detector - Isolation Forest")
st.write("Identify outlier behavior profiles from credit card transactions.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found! Please run the 'isolation_forest.ipynb' notebook first to train and save the model.")
    st.stop()

# User Inputs organized in tabs
st.sidebar.header("🔧 Customer Profile Inputs")
tab1, tab2, tab3, tab4 = st.tabs(["💳 Balance & Credit", "🛍️ Purchases", "💸 Cash Advance", "📊 Payments"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        balance = st.number_input("Balance ($)", min_value=0.0, value=873.38, step=50.0)
        balance_frequency = st.slider("Balance Frequency", 0.0, 1.0, 1.0, step=0.05)
    with col2:
        credit_limit = st.number_input("Credit Limit ($)", min_value=50.0, value=3000.0, step=100.0)
        tenure = st.slider("Tenure (months)", 6, 12, 12)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        purchases = st.number_input("Total Purchases ($)", min_value=0.0, value=361.28, step=50.0)
        oneoff_purchases = st.number_input("One-off Purchases ($)", min_value=0.0, value=38.0, step=10.0)
        installments_purchases = st.number_input("Installments Purchases ($)", min_value=0.0, value=89.0, step=10.0)
    with col2:
        purchases_frequency = st.slider("Purchases Frequency", 0.0, 1.0, 0.5, step=0.05)
        oneoff_purchases_frequency = st.slider("One-off Purchases Frequency", 0.0, 1.0, 0.08, step=0.05)
        purchases_installments_frequency = st.slider("Installments Purchases Frequency", 0.0, 1.0, 0.16, step=0.05)
        purchases_trx = st.number_input("Purchases Transactions", min_value=0, value=7, step=1)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        cash_advance = st.number_input("Cash Advance ($)", min_value=0.0, value=0.0, step=50.0)
        cash_advance_frequency = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.0, step=0.05)
    with col2:
        cash_advance_trx = st.number_input("Cash Advance Transactions", min_value=0, value=0, step=1)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        payments = st.number_input("Total Payments ($)", min_value=0.0, value=856.90, step=50.0)
        minimum_payments = st.number_input("Minimum Payments ($)", min_value=0.0, value=312.34, step=50.0)
    with col2:
        prc_full_payment = st.slider("PRC Full Payment", 0.0, 1.0, 0.0, step=0.05)

# Prediction
input_dict = {
    'BALANCE': balance,
    'BALANCE_FREQUENCY': balance_frequency,
    'PURCHASES': purchases,
    'ONEOFF_PURCHASES': oneoff_purchases,
    'INSTALLMENTS_PURCHASES': installments_purchases,
    'CASH_ADVANCE': cash_advance,
    'PURCHASES_FREQUENCY': purchases_frequency,
    'ONEOFF_PURCHASES_FREQUENCY': oneoff_purchases_frequency,
    'PURCHASES_INSTALLMENTS_FREQUENCY': purchases_installments_frequency,
    'CASH_ADVANCE_FREQUENCY': cash_advance_frequency,
    'CASH_ADVANCE_TRX': cash_advance_trx,
    'PURCHASES_TRX': purchases_trx,
    'CREDIT_LIMIT': credit_limit,
    'PAYMENTS': payments,
    'MINIMUM_PAYMENTS': minimum_payments,
    'PRC_FULL_PAYMENT': prc_full_payment,
    'TENURE': tenure
}
input_df = pd.DataFrame([input_dict])

if st.button("🔮 Evaluate Profile"):
    # IsolationForest.predict returns 1 for inlier, -1 for outlier
    prediction = model.predict(input_df)[0]
    # decision_function gives anomaly score (lower is more anomalous)
    anomaly_score = model.decision_function(input_df)[0]
    
    status_text = "🚨 Outlier / Anomaly Detected" if prediction == -1 else "✅ Normal / Typical Profile"
    card_color = "#EF4444" if prediction == -1 else "#10B981"
    
    st.markdown(f"""
    <div class="metric-card" style="border-left: 6px solid {card_color};">
        <div class="metric-title">Anomaly Status</div>
        <div class="metric-value" style="color: {card_color};">{status_text}</div>
        <p style="margin-top: 10px; color: #94A3B8;">
            The Isolation Forest model classified this profile with an anomaly score of <b>{anomaly_score:.4f}</b>.
            (Negative scores indicate anomalies).
        </p>
    </div>
    """, unsafe_allow_html=True)
