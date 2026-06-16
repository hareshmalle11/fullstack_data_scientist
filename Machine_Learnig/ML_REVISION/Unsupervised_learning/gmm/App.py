import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

st.set_page_config(
    page_title="Customer Segmentation (GMM)",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "gmm_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "CC GENERAL.csv")

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
    .metric-value { color: #EC4899; font-size: 32px; font-weight: 700; }
    .stButton>button {
        background: linear-gradient(135deg, #EC4899 0%, #D946EF 100%);
        color: white; border: none; border-radius: 8px; padding: 12px 24px; font-weight: 700; width: 100%; transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(236, 72, 153, 0.4); }
</style>
""", unsafe_allow_html=True)

st.title("🎭 Soft Customer Clustering - Gaussian Mixture Models (GMM)")
st.write("Calculate soft cluster membership probabilities using a trained GMM.")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Failed to load model")
    st.exception(e)
    st.stop()

# ----------------------------
# Dataset Stats
# ----------------------------
@st.cache_data
def get_data_stats():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        
        X = df.drop(columns=["CUST_ID"])
        
        numeric_cols = X.select_dtypes(include=np.number).columns
        X[numeric_cols] = X[numeric_cols].fillna(
            X[numeric_cols].median()
        )
        
        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]
        
        labels = model.predict(X)
        X['Cluster'] = labels
        cluster_means = X.groupby('Cluster').mean()
        overall_means = X.drop(columns=['Cluster']).mean()
        return X, cluster_means, overall_means
    return None, None, None

df_full, cluster_means, overall_means = get_data_stats()

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

if hasattr(model, "feature_names_in_"):
    input_df = input_df[model.feature_names_in_]

if st.button("🔮 Calculate Probabilities"):
    try:
        # GMM has predict and predict_proba methods
        predicted_cluster = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
    except Exception as e:
        st.exception(e)
        st.stop()
   
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Primary Cluster Assignment</div>
        <div class="metric-value">Segment / Cluster {predicted_cluster}</div>
        <p style="margin-top: 10px; color: #94A3B8;">
            The model suggests this customer most strongly aligns with <b>Cluster {predicted_cluster}</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
   
    st.subheader("📊 Membership Probability Distribution")
    prob_df = pd.DataFrame({
        'Cluster': [f'Cluster {i}' for i in range(len(probabilities))],
        'Probability': probabilities
    })
   
    fig = px.bar(
        prob_df,
        x='Cluster',
        y='Probability',
        text='Probability',
        color='Probability',
        color_continuous_scale='RdPu',
        labels={'Probability': 'Membership Probability'}
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8FAFC'),
        yaxis=dict(range=[0, 1.1])
    )
    st.plotly_chart(fig, use_container_width=True)