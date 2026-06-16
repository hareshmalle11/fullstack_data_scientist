import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Segmentation (K-Means)",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "kmeans_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "CC GENERAL.csv")

# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>
.metric-card {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}

.metric-title {
    color: #94A3B8;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #38BDF8;
    font-size: 32px;
    font-weight: 700;
}

.stButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("🎯 Customer Segmentation Hub - K-Means Clustering")
st.write(
    "Determine customer clusters using a trained K-Means model "
    "on credit card usage features."
)

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

    if not os.path.exists(DATA_PATH):
        return None, None, None

    df = pd.read_csv(DATA_PATH)

    if "CUST_ID" in df.columns:
        X = df.drop(columns=["CUST_ID"])
    else:
        X = df.copy()

    numeric_cols = X.select_dtypes(include=np.number).columns

    X[numeric_cols] = X[numeric_cols].fillna(
        X[numeric_cols].median()
    )

    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    labels = model.predict(X)

    X["Cluster"] = labels

    cluster_means = X.groupby("Cluster").mean()

    overall_means = (
        X.drop(columns=["Cluster"])
        .mean()
    )

    return X, cluster_means, overall_means

try:
    df_full, cluster_means, overall_means = get_data_stats()
except Exception as e:
    st.exception(e)
    st.stop()

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🔧 Customer Profile Inputs")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💳 Balance & Credit",
        "🛍️ Purchases",
        "💸 Cash Advance",
        "📊 Payments"
    ]
)

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        balance = st.number_input(
            "Balance ($)",
            min_value=0.0,
            value=873.38
        )

        balance_frequency = st.slider(
            "Balance Frequency",
            0.0,
            1.0,
            1.0
        )

    with col2:
        credit_limit = st.number_input(
            "Credit Limit ($)",
            min_value=50.0,
            value=3000.0
        )

        tenure = st.slider(
            "Tenure (months)",
            6,
            12,
            12
        )

with tab2:

    col1, col2 = st.columns(2)

    with col1:

        purchases = st.number_input(
            "Total Purchases ($)",
            min_value=0.0,
            value=361.28
        )

        oneoff_purchases = st.number_input(
            "One-off Purchases ($)",
            min_value=0.0,
            value=38.0
        )

        installments_purchases = st.number_input(
            "Installments Purchases ($)",
            min_value=0.0,
            value=89.0
        )

    with col2:

        purchases_frequency = st.slider(
            "Purchases Frequency",
            0.0,
            1.0,
            0.5
        )

        oneoff_purchases_frequency = st.slider(
            "One-off Purchases Frequency",
            0.0,
            1.0,
            0.08
        )

        purchases_installments_frequency = st.slider(
            "Installments Purchases Frequency",
            0.0,
            1.0,
            0.16
        )

        purchases_trx = st.number_input(
            "Purchases Transactions",
            min_value=0,
            value=7
        )

with tab3:

    col1, col2 = st.columns(2)

    with col1:

        cash_advance = st.number_input(
            "Cash Advance ($)",
            min_value=0.0,
            value=0.0
        )

        cash_advance_frequency = st.slider(
            "Cash Advance Frequency",
            0.0,
            1.0,
            0.0
        )

    with col2:

        cash_advance_trx = st.number_input(
            "Cash Advance Transactions",
            min_value=0,
            value=0
        )

with tab4:

    col1, col2 = st.columns(2)

    with col1:

        payments = st.number_input(
            "Total Payments ($)",
            min_value=0.0,
            value=856.90
        )

        minimum_payments = st.number_input(
            "Minimum Payments ($)",
            min_value=0.0,
            value=312.34
        )

    with col2:

        prc_full_payment = st.slider(
            "PRC Full Payment",
            0.0,
            1.0,
            0.0
        )

# ----------------------------
# Input Data
# ----------------------------
input_dict = {
    "BALANCE": balance,
    "BALANCE_FREQUENCY": balance_frequency,
    "PURCHASES": purchases,
    "ONEOFF_PURCHASES": oneoff_purchases,
    "INSTALLMENTS_PURCHASES": installments_purchases,
    "CASH_ADVANCE": cash_advance,
    "PURCHASES_FREQUENCY": purchases_frequency,
    "ONEOFF_PURCHASES_FREQUENCY": oneoff_purchases_frequency,
    "PURCHASES_INSTALLMENTS_FREQUENCY": purchases_installments_frequency,
    "CASH_ADVANCE_FREQUENCY": cash_advance_frequency,
    "CASH_ADVANCE_TRX": cash_advance_trx,
    "PURCHASES_TRX": purchases_trx,
    "CREDIT_LIMIT": credit_limit,
    "PAYMENTS": payments,
    "MINIMUM_PAYMENTS": minimum_payments,
    "PRC_FULL_PAYMENT": prc_full_payment,
    "TENURE": tenure
}

input_df = pd.DataFrame([input_dict])

if hasattr(model, "feature_names_in_"):
    input_df = input_df[model.feature_names_in_]

# ----------------------------
# Predict
# ----------------------------
if st.button("🔮 Analyze & Segment Customer"):

    try:
        predicted_cluster = model.predict(input_df)[0]

    except Exception as e:
        st.exception(e)
        st.stop()

    st.success(
        f"Customer belongs to Cluster {predicted_cluster}"
    )

    if cluster_means is not None:

        key_features = [
            "BALANCE",
            "PURCHASES",
            "CREDIT_LIMIT",
            "PAYMENTS"
        ]

        user_vals = [
            balance,
            purchases,
            credit_limit,
            payments
        ]

        cluster_vals = (
            cluster_means
            .loc[predicted_cluster, key_features]
            .values
        )

        avg_vals = (
            overall_means[key_features]
            .values
        )

        fig = go.Figure()

        fig.add_bar(
            name="Input Customer",
            x=key_features,
            y=user_vals
        )

        fig.add_bar(
            name=f"Cluster {predicted_cluster}",
            x=key_features,
            y=cluster_vals
        )

        fig.add_bar(
            name="Overall Average",
            x=key_features,
            y=avg_vals
        )

        fig.update_layout(
            barmode="group",
            title="Key Metrics Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Cluster Characteristics")

        for feature in key_features:
            value = cluster_means.loc[
                predicted_cluster,
                feature
            ]

            st.write(
                f"**{feature}** : ${value:,.2f}"
            )