import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

MODEL_PATH = "tsne_data.pkl"

st.set_page_config(page_title="Customer Projection (t-SNE)", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #0F172A; }
    .profile-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .profile-header { color: #A855F7; font-size: 20px; font-weight: 700; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🌀 Manifold Visualization Hub - t-SNE")
st.write("Explore high-dimensional customer behaviors mapped onto 2D space using t-SNE.")

@st.cache_resource
def load_data():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

df_tsne = load_data()
if df_tsne is None:
    st.error(f"⚠️ Precomputed t-SNE file '{MODEL_PATH}' not found! Please run the 't-sne.ipynb' notebook first to calculate and save the dataset coordinates.")
    st.stop()

st.sidebar.header("🔍 Explorer Controls")
color_feature = st.sidebar.selectbox(
    "Color Points By:",
    ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS', 'TENURE']
)

customer_id = st.sidebar.text_input("Search Customer ID (e.g. C10001, C10002):", "")

col_map, col_details = st.columns([2, 1])

with col_map:
    # Highlight customer if searched
    df_plot = df_tsne.copy()
    if customer_id:
        df_plot['Is_Searched'] = (df_plot['CUST_ID'].astype(str) == customer_id).map({True: 'Searched', False: 'Others'})
        fig = px.scatter(
            df_plot,
            x='TSNE1',
            y='TSNE2',
            color='Is_Searched',
            hover_data=['CUST_ID', 'BALANCE', 'PURCHASES'],
            color_discrete_map={'Searched': '#EF4444', 'Others': '#A855F7'},
            opacity=0.6,
            title=f"t-SNE Projection: Highlighting Customer {customer_id}"
        )
    else:
        fig = px.scatter(
            df_plot,
            x='TSNE1',
            y='TSNE2',
            color=color_feature,
            hover_data=['CUST_ID', 'BALANCE', 'PURCHASES'],
            color_continuous_scale='Plasma',
            title=f"t-SNE Projection Color-coded by {color_feature}"
        )
        
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8FAFC')
    )
    st.plotly_chart(fig, use_container_width=True)

with col_details:
    st.subheader("📋 Customer Details Explorer")
    if customer_id:
        record = df_tsne[df_tsne['CUST_ID'].astype(str) == customer_id]
        if not record.empty:
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-header">Customer {customer_id} Profile</div>
                <p><b>Balance:</b> ${record['BALANCE'].values[0]:,.2f}</p>
                <p><b>Total Purchases:</b> ${record['PURCHASES'].values[0]:,.2f}</p>
                <p><b>Credit Limit:</b> ${record['CREDIT_LIMIT'].values[0]:,.2f}</p>
                <p><b>Total Payments:</b> ${record['PAYMENTS'].values[0]:,.2f}</p>
                <p><b>Tenure:</b> {record['TENURE'].values[0]} months</p>
                <p><b>t-SNE Coordinate 1:</b> {record['TSNE1'].values[0]:.4f}</p>
                <p><b>t-SNE Coordinate 2:</b> {record['TSNE2'].values[0]:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Customer ID '{customer_id}' not found in dataset.")
    else:
        st.info("Search for a Customer ID in the sidebar to view detailed profiles.")
