import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils.metrics import regression_metrics, explain_metrics
from utils.visualizations import (
    actual_vs_predicted_plot,
    residual_plot,
    error_distribution_plot,
    feature_importance_plot
)

def linear_regression_app():
    st.markdown('<h2 style="color: #6366f1;">📈 Linear Regression Model</h2>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'show_metrics' not in st.session_state:
        st.session_state.show_metrics = False
    if 'show_viz' not in st.session_state:
        st.session_state.show_viz = False
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    
    # STEP 1: Upload Dataset
    st.markdown("### 📁 Step 1: Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file containing your dataset"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to get started")
        return
    
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded successfully! Shape: {df. shape}")
        
        with st.expander("🔍 Preview Dataset", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
    
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return
    
    st.markdown("---")
    
    # STEP 2: Feature Selection
    st.markdown("### 🎯 Step 2: Select Features and Target")
    
    col1, col2 = st. columns(2)
    
    with col1:
        features = st.multiselect(
            "Select Input Features (X)",
            options=df.columns.tolist(),
            help="Choose one or more columns as input features"
        )
    
    with col2:
        target = st.selectbox(
            "Select Target Variable (y)",
            options=df.columns. tolist(),
            help="Choose the column you want to predict"
        )
    
    if not features:
        st.warning("⚠️ Please select at least one input feature")
        return
    
    if not target:
        st.warning("⚠️ Please select a target variable")
        return
    
    if target in features:
        st.error("❌ Target variable cannot be in input features")
        return
    
    # Validate numeric columns
    try:
        X = df[features]. select_dtypes(include=[np.number])
        y = df[target]
        
        if X.shape[1] == 0:
            st.error("❌ Selected features must be numeric")
            return
        
        if not pd.api.types.is_numeric_dtype(y):
            st.error("❌ Target variable must be numeric")
            return
            
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        return
    
    st.markdown("---")
    
    # STEP 3: Action Buttons Section
    st.markdown("### ⚡ Step 3: Actions")
    st.markdown('<div class="action-section">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training model..."):
                try:
                    # Split data
                    test_size = st.session_state.get('test_size', 0.2)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Train model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.features = features
                    st.session_state.model_trained = True
                    
                    st.success("✅ Model trained successfully!")
                    
                except Exception as e: 
                    st.error(f"❌ Error training model: {e}")
    
    with col2:
        if st.button("📊 Show Metrics", use_container_width=True, disabled=not st.session_state. model_trained):
            st.session_state.show_metrics = not st.session_state.show_metrics
    
    with col3:
        if st.button("📈 Visualizations", use_container_width=True, disabled=not st.session_state.model_trained):
            st.session_state.show_viz = not st.session_state.show_viz
    
    with col4:
        if st.button("🔮 Predict", use_container_width=True, disabled=not st.session_state.model_trained):
            st.session_state.show_prediction = not st.session_state.show_prediction
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Coefficients (if trained)
    if st.session_state.model_trained:
        with st.expander("📐 Model Coefficients", expanded=False):
            model = st.session_state.model
            coef_df = pd.DataFrame({
                'Feature': st.session_state.features,
                'Coefficient': model.coef_
            })
            st.dataframe(coef_df, use_container_width=True)
            st.write(f"**Intercept:** {model.intercept_:.4f}")
    
    # METRICS SECTION
    if st.session_state.show_metrics and st.session_state.model_trained:
        st.markdown("---")
        st.markdown("### 📊 Model Performance Metrics")
        
        y_test = st.session_state.y_test
        y_pred = st. session_state.y_pred
        
        metrics = regression_metrics(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"{metrics['MAE']:.4f}")
        with col2:
            st. metric("MSE", f"{metrics['MSE']:.4f}")
        with col3:
            st.metric("RMSE", f"{metrics['RMSE']:.4f}")
        with col4:
            st.metric("R² Score", f"{metrics['R2']:.4f}")
        
        with st.expander("ℹ️ What do these metrics mean?"):
            st.markdown(explain_metrics())
    
    # VISUALIZATION SECTION
    if st.session_state.show_viz and st.session_state.model_trained:
        st.markdown("---")
        st.markdown("### 📈 Visualizations")
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Actual vs Predicted", "Residual Plot", "Error Distribution", "Feature Importance"]
        )
        
        y_test = st.session_state.y_test
        y_pred = st.session_state. y_pred
        model = st.session_state.model
        features = st.session_state.features
        
        if viz_type == "Actual vs Predicted": 
            fig = actual_vs_predicted_plot(y_test, y_pred)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Residual Plot":
            fig = residual_plot(y_test, y_pred)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Error Distribution":
            fig = error_distribution_plot(y_test, y_pred)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Feature Importance":
            fig = feature_importance_plot(model, features)
            st.pyplot(fig)
    
    # PREDICTION SECTION
    if st.session_state.show_prediction and st.session_state.model_trained:
        st.markdown("---")
        st.markdown("### 🔮 Make Predictions on New Data")
        
        model = st.session_state.model
        features = st.session_state.features
        
        st.write("Enter values for prediction:")
        
        input_data = {}
        cols = st.columns(len(features))
        
        for idx, feature in enumerate(features):
            with cols[idx]:
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f"
                )
        
        if st.button("🎯 Predict", key="predict_btn"):
            try:
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                
                st.success(f"### Predicted Value: **{prediction:.4f}**")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")