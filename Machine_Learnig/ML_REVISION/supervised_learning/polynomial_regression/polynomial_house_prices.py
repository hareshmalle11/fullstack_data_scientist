import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Polynomial Regression - House Price Prediction", layout="wide")
st.title("🏠 Polynomial Regression: House Price Prediction")
st.markdown("---")

# Sidebar Navigation
with st.sidebar:
    st.header("Navigation")
    section = st.radio("Select Section", [
        "📊 Overview",
        "🔍 Explore Data",
        "🤖 Model Training",
        "💰 Price Prediction",
        "📈 Visualizations",
        "📋 Model Performance"
    ])

# Load or Train Model
@st.cache_resource
def load_or_train_model():
    try:
        with open('polynomial_house_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('house_prices.csv')
    return df

# Section 1: Overview
if section == "📊 Overview":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What is Polynomial Regression?")
        st.write("""
        Polynomial Regression extends Linear Regression by modeling non-linear relationships 
        between features and target. Instead of fitting a straight line, it fits a curve.
        
        **Formula:**
        $$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + ... + \beta_n x^n$$
        
        **Key Differences from Linear:**
        - Linear: Straight line (degree 1)
        - Polynomial: Curved line (degree 2+)
        - Better captures non-linear patterns
        - Risk of overfitting with high degrees
        """)
    
    with col2:
        st.subheader("Use Case: House Prices")
        st.write("""
        **Problem:** Predict house prices based on square footage
        
        **Why Polynomial?**
        - Price doesn't increase linearly with size
        - Large houses have diminishing price increase
        - Quadratic or cubic relationships common
        
        **Dataset Features:**
        - Square Footage (input)
        - Price (target, in $1000s)
        - Polynomial Degree: 2 (quadratic)
        """)
    
    st.markdown("---")
    
    with st.expander("🔬 Mathematical Details"):
        st.write("""
        ### Quadratic Model (Degree 2)
        
        Original features: [x]
        
        Polynomial features: [1, x, x²]
        
        Model: y = β₀ + β₁x + β₂x²
        
        ### Why This Helps
        - x²term captures curvature
        - Negative coefficient means price increases slower for larger houses
        - Example: β₁=20, β₂=-0.01
          - At 1000 sqft: y = 20(1000) - 0.01(1000²) = 20,000 - 10,000 = 10,000
          - At 2000 sqft: y = 20(2000) - 0.01(2000²) = 40,000 - 40,000 = 0 (overshoot!)
        """)

# Section 2: Explore Data
elif section == "🔍 Explore Data":
    st.subheader("Dataset Overview")
    
    df = load_data()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Houses", len(df))
    with col2:
        st.metric("Avg Price ($K)", f"{df['price'].mean():.1f}")
    with col3:
        st.metric("Avg Sqft", f"{df['sqft'].mean():.0f}")
    
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Raw Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['price'], bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel("Price ($K)")
        ax.set_ylabel("Count")
        ax.set_title("House Price Distribution")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Square Footage Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['sqft'], bins=20, color='lightcoral', edgecolor='black')
        ax.set_xlabel("Square Footage")
        ax.set_ylabel("Count")
        ax.set_title("Square Footage Distribution")
        st.pyplot(fig)

# Section 3: Model Training
elif section == "🤖 Model Training":
    st.subheader("Train Polynomial Regression Model")
    
    df = load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        degree = st.slider("Polynomial Degree", 1, 5, 2, help="Higher = more complex, risk of overfitting")
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
    
    if st.button("Train Model", key="train"):
        st.write("Training in progress...")
        
        # Prepare data
        X = df[['sqft']].values
        y = df['price'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Scaler
        scaler = StandardScaler()
        X_train_poly_scaled = scaler.fit_transform(X_train_poly)
        X_test_poly_scaled = scaler.transform(X_test_poly)
        
        # Model
        model = LinearRegression()
        model.fit(X_train_poly_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_poly_scaled)
        y_pred_test = model.predict(X_test_poly_scaled)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Save model
        model_data = {
            'model': model,
            'poly': poly,
            'scaler': scaler,
            'degree': degree
        }
        
        with open('polynomial_house_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        st.success("✅ Model trained and saved!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Metrics")
            st.metric("Train MAE", f"${train_mae:.2f}K")
            st.metric("Train R²", f"{train_r2:.3f}")
        
        with col2:
            st.subheader("Test Metrics")
            st.metric("Test MAE", f"${test_mae:.2f}K")
            st.metric("Test R²", f"{test_r2:.3f}")
        
        # Coefficients
        st.subheader("Model Coefficients")
        coef_names = poly.get_feature_names_out(['sqft'])
        coef_df = pd.DataFrame({
            'Feature': coef_names,
            'Coefficient': model.coef_
        })
        st.dataframe(coef_df, use_container_width=True)

# Section 4: Price Prediction
elif section == "💰 Price Prediction":
    st.subheader("Predict House Price")
    
    model_data = load_or_train_model()
    
    if model_data is None:
        st.warning("⚠️ Train model first in 'Model Training' section")
    else:
        model = model_data['model']
        poly = model_data['poly']
        scaler = model_data['scaler']
        
        sqft_input = st.number_input("Enter Square Footage", min_value=500, max_value=5000, value=2000)
        
        if st.button("Predict Price"):
            X_input = np.array([[sqft_input]])
            X_input_poly = poly.transform(X_input)
            X_input_scaled = scaler.transform(X_input_poly)
            predicted_price = model.predict(X_input_scaled)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🏠 Square Footage: {sqft_input:,}")
            with col2:
                st.success(f"💵 Predicted Price: ${predicted_price*1000:,.2f}")
            
            # Explanation
            st.subheader("Prediction Breakdown")
            features = poly.get_feature_names_out(['sqft'])
            X_input_poly_flat = X_input_poly[0]
            
            breakdown = pd.DataFrame({
                'Feature': features,
                'Value': X_input_poly_flat,
                'Coefficient': model.coef_,
                'Contribution': X_input_poly_flat * model.coef_
            })
            
            st.dataframe(breakdown, use_container_width=True)
            st.write(f"**Intercept:** {model.intercept_:.2f}")
            st.write(f"**Total Prediction:** ${predicted_price*1000:,.2f}")

# Section 5: Visualizations
elif section == "📈 Visualizations":
    st.subheader("Model Visualization")
    
    df = load_data()
    model_data = load_or_train_model()
    
    if model_data is None:
        st.warning("⚠️ Train model first")
    else:
        model = model_data['model']
        poly = model_data['poly']
        scaler = model_data['scaler']
        
        # Create smooth curve
        X_range = np.linspace(df['sqft'].min(), df['sqft'].max(), 100).reshape(-1, 1)
        X_range_poly = poly.transform(X_range)
        X_range_scaled = scaler.transform(X_range_poly)
        y_pred_range = model.predict(X_range_scaled)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.scatter(df['sqft'], df['price'], alpha=0.6, s=50, label='Actual Prices', color='blue')
        ax.plot(X_range, y_pred_range, 'r-', linewidth=2, label=f'Polynomial Fit (Degree {model_data["degree"]})')
        
        ax.set_xlabel("Square Footage", fontsize=12)
        ax.set_ylabel("Price ($K)", fontsize=12)
        ax.set_title("Polynomial Regression: Square Footage vs Price", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Residual Plot
        X_all = df[['sqft']].values
        X_all_poly = poly.transform(X_all)
        X_all_scaled = scaler.transform(X_all_poly)
        y_pred_all = model.predict(X_all_scaled)
        residuals = df['price'] - y_pred_all
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.scatter(y_pred_all, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel("Predicted Price ($K)", fontsize=12)
        ax2.set_ylabel("Residuals ($K)", fontsize=12)
        ax2.set_title("Residual Plot - Polynomial Regression", fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)

# Section 6: Model Performance
elif section == "📋 Model Performance":
    st.subheader("Detailed Model Performance")
    
    df = load_data()
    model_data = load_or_train_model()
    
    if model_data is None:
        st.warning("⚠️ Train model first")
    else:
        model = model_data['model']
        poly = model_data['poly']
        scaler = model_data['scaler']
        
        # Get predictions
        X_all = df[['sqft']].values
        X_all_poly = poly.transform(X_all)
        X_all_scaled = scaler.transform(X_all_poly)
        y_pred_all = model.predict(X_all_scaled)
        
        # Metrics
        mae = mean_absolute_error(df['price'], y_pred_all)
        mse = mean_squared_error(df['price'], y_pred_all)
        rmse = np.sqrt(mse)
        r2 = r2_score(df['price'], y_pred_all)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"${mae:.2f}K")
        with col2:
            st.metric("RMSE", f"${rmse:.2f}K")
        with col3:
            st.metric("R² Score", f"{r2:.3f}")
        with col4:
            st.metric("Degree", model_data['degree'])
        
        # Top Predictions vs Actuals
        st.subheader("Sample Predictions vs Actual")
        pred_df = pd.DataFrame({
            'Square Footage': df['sqft'],
            'Actual Price ($K)': df['price'],
            'Predicted Price ($K)': y_pred_all,
            'Error ($K)': df['price'] - y_pred_all,
            'Error %': (abs(df['price'] - y_pred_all) / df['price'] * 100).round(2)
        }).sort_values('Error %').head(10)
        
        st.dataframe(pred_df, use_container_width=True)

st.markdown("---")
st.write("*Polynomial Regression Model - House Price Prediction | Last Updated: June 2026*")
