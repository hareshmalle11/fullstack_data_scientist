import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Interactive ML Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f. read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state. page = 'home'

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# ============= HOME PAGE =============
def home_page():
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">🌐 Interactive Machine Learning Platform</h1>
            <p class="hero-subtitle">Build, analyze, and understand machine learning models without writing code</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
                <div class="feature-card">
                    <div class="card-icon">🤖</div>
                    <h2>Machine Learning Models</h2>
                    <p>Train and evaluate ML models with your data</p>
                    <ul class="feature-list">
                        <li>Linear Regression</li>
                        <li>Model Evaluation</li>
                        <li>Visualizations</li>
                        <li>Predictions</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start ML Modeling", key="ml_btn", use_container_width=True):
                navigate_to('ml')
        
        with col_b:
            st.markdown("""
                <div class="feature-card">
                    <div class="card-icon">📊</div>
                    <h2>Data Analysis</h2>
                    <p>Explore and understand your datasets</p>
                    <ul class="feature-list">
                        <li>Statistical Summary</li>
                        <li>Missing Values</li>
                        <li>Correlations</li>
                        <li>Visualizations</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("📈 Analyze Data", key="analysis_btn", use_container_width=True):
                navigate_to('analysis')

# ============= ML PAGE =============
def ml_page():
    from models.linear_regression import linear_regression_app
    
    st.markdown('<div class="page-header">🤖 Machine Learning Models</div>', unsafe_allow_html=True)
    
    if st.button("⬅️ Back to Home", key="back_ml"):
        navigate_to('home')
    
    st.markdown("---")
    linear_regression_app()

# ============= ANALYSIS PAGE =============
def analysis_page():
    from analysis.data_analysis import data_analysis_app
    
    st.markdown('<div class="page-header">📊 Data Analysis</div>', unsafe_allow_html=True)
    
    if st.button("⬅️ Back to Home", key="back_analysis"):
        navigate_to('home')
    
    st.markdown("---")
    data_analysis_app()

# ============= ROUTING =============
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'ml': 
    ml_page()
elif st.session_state.page == 'analysis':
    analysis_page()