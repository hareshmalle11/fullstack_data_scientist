import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib. pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

def data_analysis_app():
    st.markdown('<h2 style="color: #6366f1;">📊 Data Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'show_stats' not in st.session_state:
        st.session_state.show_stats = False
    if 'show_missing' not in st.session_state:
        st.session_state.show_missing = False
    if 'show_types' not in st.session_state:
        st.session_state.show_types = False
    if 'show_viz' not in st.session_state:
        st.session_state.show_viz = False
    
    # Upload Dataset
    st.markdown("### 📁 Upload Dataset for Analysis")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your dataset for comprehensive analysis"
    )
    
    if uploaded_file is None: 
        st.info("👆 Upload a dataset to begin analysis")
        return
    
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded!  Shape: {df.shape}")
        
        with st.expander("🔍 Quick Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
    
    except Exception as e: 
        st.error(f"❌ Error loading file: {e}")
        return
    
    st.markdown("---")
    
    # Dataset Overview
    st.markdown("### 📌 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np. number]).columns))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    # Action Buttons
    st.markdown("### ⚡ Analysis Actions")
    st.markdown('<div class="action-section">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Statistical Summary", use_container_width=True):
            st.session_state.show_stats = not st.session_state.show_stats
    
    with col2:
        if st.button("❗ Missing Values", use_container_width=True):
            st.session_state.show_missing = not st.session_state.show_missing
    
    with col3:
        if st.button("🔤 Data Types", use_container_width=True):
            st.session_state.show_types = not st.session_state.show_types
    
    with col4:
        if st.button("📈 Visualizations", use_container_width=True):
            st.session_state. show_viz = not st.session_state.show_viz
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # STATISTICAL SUMMARY
    if st.session_state.show_stats:
        st.markdown("---")
        st.markdown("### 📊 Statistical Summary")
        
        tab1, tab2 = st.tabs(["Numeric Features", "Categorical Features"])
        
        with tab1:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns found")
        
        with tab2:
            categorical_df = df.select_dtypes(include=['object'])
            if not categorical_df.empty:
                st. dataframe(categorical_df.describe(), use_container_width=True)
            else:
                st.info("No categorical columns found")
    
    # MISSING VALUES
    if st.session_state.show_missing:
        st.markdown("---")
        st.markdown("### ❗ Missing Values Analysis")
        
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        
        missing_data = missing_data[missing_data['Missing Count'] > 0]. sort_values(
            'Missing Count', ascending=False
        )
        
        if not missing_data.empty:
            st.dataframe(missing_data, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                missing_data,
                x='Column',
                y='Missing Percentage',
                title='Missing Values by Column',
                color='Missing Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    # DATA TYPES
    if st.session_state.show_types:
        st.markdown("---")
        st.markdown("### 🔤 Data Types Information")
        
        dtypes_df = pd.DataFrame({
            'Column':  df.columns,
            'Data Type': df.dtypes. values,
            'Unique Values':  [df[col].nunique() for col in df.columns],
            'Sample Value':  [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
        })
        
        st.dataframe(dtypes_df, use_container_width=True)
    
    # VISUALIZATIONS
    if st.session_state.show_viz:
        st.markdown("---")
        st.markdown("### 📈 Data Visualizations")
        
        viz_option = st.selectbox(
            "Select Visualization Type",
            ["Correlation Heatmap", "Distribution Plots", "Box Plots", "Pair Plot"]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st. warning("⚠️ No numeric columns available for visualization")
            return
        
        if viz_option == "Correlation Heatmap": 
            st.markdown("#### 🔗 Correlation Heatmap")
            
            corr = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Feature Correlation Matrix"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Distribution Plots": 
            st.markdown("#### 📊 Distribution Plots")
            
            selected_col = st.selectbox("Select column", numeric_cols)
            
            fig = px.histogram(
                df,
                x=selected_col,
                marginal="box",
                title=f"Distribution of {selected_col}",
                color_discrete_sequence=['#6366f1']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Box Plots":
            st.markdown("#### 📦 Box Plots")
            
            selected_cols = st.multiselect(
                "Select columns",
                numeric_cols,
                default=numeric_cols[: 3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_cols:
                fig = px.box(
                    df[selected_cols],
                    title="Box Plot Comparison",
                    color_discrete_sequence=['#6366f1', '#8b5cf6', '#ec4899']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Pair Plot":
            st.markdown("#### 🔗 Pair Plot")
            
            selected_cols = st.multiselect(
                "Select columns (max 5)",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_cols and len(selected_cols) <= 5:
                fig = px.scatter_matrix(
                    df,
                    dimensions=selected_cols,
                    title="Pair Plot Matrix"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9',
                    height=800
                )
                st.plotly_chart(fig, use_container_width=True)
            elif len(selected_cols) > 5:
                st.warning("⚠️ Please select maximum 5 columns for pair plot")