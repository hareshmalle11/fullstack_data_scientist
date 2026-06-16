import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from src.component.model_trainer import ClusteringTrainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix
from src.component.data_analyzer import DataAnalyzer, ClassificationAnalyzer
from src.component.data_ingestion import DataIngestion
from src.component.data_transform import DataTransform, ClassificationTransform
from src.component.model_trainer import ModelTrainer, ClassificationTrainer
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.utils import save_object

CLEANED_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "cleaned_data.csv")
BEST_MODEL_TRAINING_PATH = os.path.join(os.getcwd(), "artifacts", "best_model_training.pkl")
BEST_MODEL_GRID_PATH = os.path.join(os.getcwd(), "artifacts", "best_model_grid.pkl")

# ── Page init ──────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "transformed_df" not in st.session_state:
    st.session_state.transformed_df = None
if "transformed_path" not in st.session_state:
    st.session_state.transformed_path = None
if "correlated_to_drop" not in st.session_state:
    st.session_state.correlated_to_drop = []
if "correlated_pairs" not in st.session_state:
    st.session_state.correlated_pairs = []
if "training_results_df" not in st.session_state:
    st.session_state.training_results_df = None
if "best_training_result" not in st.session_state:
    st.session_state.best_training_result = None
if "grid_search_output" not in st.session_state:
    st.session_state.grid_search_output = None
if "saved_model_path" not in st.session_state:
    st.session_state.saved_model_path = None
if "last_transform_option" not in st.session_state:
    st.session_state.last_transform_option = "Train Without Scaling"
if "ml_mode" not in st.session_state:
    st.session_state.ml_mode = "supervised"


def persist_cleaned_df(df: pd.DataFrame) -> str:
    """
    Keep a single progressive cleaned DataFrame in session and on disk.
    """
    st.session_state.cleaned_df = df.copy()
    return DataTransform().save_final_dataset(st.session_state.cleaned_df, filename="cleaned_data.csv")


def get_default_model_grid(model_name: str):
    default_grids = {
        "RandomForestRegressor": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
        "XGBRegressor": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        },
        "LinearRegression": {
            "fit_intercept": [True, False],
        },
        "SVR": {
            "kernel": ["rbf", "linear"],
            "C": [0.5, 1.0, 5.0],
            "epsilon": [0.1, 0.2],
            "gamma": ["scale", "auto"],
        },
        "KNeighborsRegressor": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        },
        "DecisionTreeRegressor": {
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
        "Ridge": {
            "alpha": [0.1, 1.0, 10.0],
        },
        "Lasso": {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "max_iter": [1000, 3000, 5000],
        },
        "RandomForestClassifier": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
        "LogisticRegression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        },
        "SVC": {
            "kernel": ["rbf", "linear"],
            "C": [0.5, 1.0, 5.0],
            "gamma": ["scale", "auto"],
        },
        "KNeighborsClassifier": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        },
        "DecisionTreeClassifier": {
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
        "XGBClassifier": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        },
    }
    return default_grids.get(model_name, {})


def collect_model_params_ui(model_name: str):
    if model_name == "RandomForestRegressor":
        return {
            "n_estimators": st.number_input("n_estimators", min_value=50, max_value=1000, value=200, step=50, key=f"{model_name}_n_estimators"),
            "max_depth": st.number_input("max_depth", min_value=1, max_value=50, value=10, step=1, key=f"{model_name}_max_depth"),
            "min_samples_split": st.number_input("min_samples_split", min_value=2, max_value=20, value=2, step=1, key=f"{model_name}_min_samples_split"),
        }
    if model_name == "XGBRegressor":
        return {
            "n_estimators": st.number_input("n_estimators", min_value=50, max_value=1000, value=200, step=50, key=f"{model_name}_n_estimators"),
            "max_depth": st.number_input("max_depth", min_value=1, max_value=20, value=6, step=1, key=f"{model_name}_max_depth"),
            "learning_rate": st.slider("learning_rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01, key=f"{model_name}_learning_rate"),
            "subsample": st.slider("subsample", min_value=0.5, max_value=1.0, value=1.0, step=0.05, key=f"{model_name}_subsample"),
            "colsample_bytree": st.slider("colsample_bytree", min_value=0.5, max_value=1.0, value=1.0, step=0.05, key=f"{model_name}_colsample"),
        }
    if model_name == "LinearRegression":
        return {
            "fit_intercept": st.checkbox("fit_intercept", value=True, key=f"{model_name}_fit_intercept"),
        }
    if model_name == "SVR":
        return {
            "kernel": st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0, key=f"{model_name}_kernel"),
            "C": st.slider("C (regularization)", min_value=0.1, max_value=20.0, value=1.0, step=0.1, key=f"{model_name}_C"),
            "epsilon": st.slider("epsilon", min_value=0.01, max_value=2.0, value=0.1, step=0.01, key=f"{model_name}_epsilon"),
            "gamma": st.selectbox("gamma", ["scale", "auto"], index=0, key=f"{model_name}_gamma"),
        }
    if model_name == "KNeighborsRegressor":
        return {
            "n_neighbors": st.number_input("n_neighbors", min_value=1, max_value=50, value=5, step=1, key=f"{model_name}_n_neighbors"),
            "weights": st.selectbox("weights", ["uniform", "distance"], index=0, key=f"{model_name}_weights"),
            "metric": st.selectbox("metric", ["minkowski", "euclidean", "manhattan"], index=0, key=f"{model_name}_metric"),
        }
    if model_name == "DecisionTreeRegressor":
        return {
            "max_depth": st.number_input("max_depth", min_value=1, max_value=50, value=8, step=1, key=f"{model_name}_max_depth"),
            "min_samples_split": st.number_input("min_samples_split", min_value=2, max_value=20, value=2, step=1, key=f"{model_name}_min_samples_split"),
        }
    if model_name == "Ridge":
        return {
            "alpha": st.slider("alpha", min_value=0.001, max_value=20.0, value=1.0, step=0.001, key=f"{model_name}_alpha"),
            "max_iter": st.number_input("max_iter", min_value=100, max_value=20000, value=1000, step=100, key=f"{model_name}_max_iter"),
        }
    if model_name == "Lasso":
        return {
            "alpha": st.slider("alpha", min_value=0.0001, max_value=2.0, value=0.01, step=0.0001, key=f"{model_name}_alpha"),
            "max_iter": st.number_input("max_iter", min_value=100, max_value=20000, value=2000, step=100, key=f"{model_name}_max_iter"),
        }
    if model_name == "RandomForestClassifier":
        return {
            "n_estimators": st.number_input("n_estimators", min_value=50, max_value=1000, value=200, step=50, key=f"{model_name}_n_estimators"),
            "max_depth": st.number_input("max_depth", min_value=1, max_value=50, value=10, step=1, key=f"{model_name}_max_depth"),
            "min_samples_split": st.number_input("min_samples_split", min_value=2, max_value=20, value=2, step=1, key=f"{model_name}_min_samples_split"),
        }
    if model_name == "LogisticRegression":
        return {
            "C": st.slider("C (Inverse Regularization)", min_value=0.01, max_value=20.0, value=1.0, step=0.01, key=f"{model_name}_C"),
            "solver": st.selectbox("solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"], index=0, key=f"{model_name}_solver"),
            "max_iter": st.number_input("max_iter", min_value=100, max_value=5000, value=100, step=100, key=f"{model_name}_max_iter")
        }
    if model_name == "SVC":
        return {
            "kernel": st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0, key=f"{model_name}_kernel"),
            "C": st.slider("C (Regularization)", min_value=0.1, max_value=20.0, value=1.0, step=0.1, key=f"{model_name}_C"),
            "gamma": st.selectbox("gamma", ["scale", "auto"], index=0, key=f"{model_name}_gamma"),
            "probability": st.checkbox("probability", value=True, key=f"{model_name}_probability")
        }
    if model_name == "KNeighborsClassifier":
        return {
            "n_neighbors": st.number_input("n_neighbors", min_value=1, max_value=50, value=5, step=1, key=f"{model_name}_n_neighbors"),
            "weights": st.selectbox("weights", ["uniform", "distance"], index=0, key=f"{model_name}_weights"),
            "metric": st.selectbox("metric", ["minkowski", "euclidean", "manhattan"], index=0, key=f"{model_name}_metric"),
        }
    if model_name == "DecisionTreeClassifier":
        return {
            "max_depth": st.number_input("max_depth", min_value=1, max_value=50, value=8, step=1, key=f"{model_name}_max_depth"),
            "min_samples_split": st.number_input("min_samples_split", min_value=2, max_value=20, value=2, step=1, key=f"{model_name}_min_samples_split"),
        }
    if model_name == "XGBClassifier":
        return {
            "n_estimators": st.number_input("n_estimators", min_value=50, max_value=1000, value=200, step=50, key=f"{model_name}_n_estimators"),
            "max_depth": st.number_input("max_depth", min_value=1, max_value=20, value=6, step=1, key=f"{model_name}_max_depth"),
            "learning_rate": st.slider("learning_rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01, key=f"{model_name}_learning_rate"),
        }
    return {}

# ══════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════
if st.session_state.page == "home":
    st.title("🤖 Intelligent ML System")

    tab1, tab2, tab3 = st.tabs([
        "🎯 Supervised Learning",
        "🔍 Unsupervised Learning",
        "📊 Data Exploration"
    ])

    with tab1:
        st.subheader("Supervised Learning")

        uploaded_file = st.file_uploader("Upload CSV or Excel for Supervised Learning", type=["csv", "xlsx"], key="supervised_upload")

        if uploaded_file is not None:
            df = DataIngestion().ingest(uploaded_file)
            st.session_state.uploaded_df = df
            st.session_state.cleaned_df = df.copy()
            st.session_state.correlated_to_drop = []
            st.session_state.correlated_pairs = []
            persist_cleaned_df(st.session_state.cleaned_df)

            if st.button("▶ Proceed to Analysis (Supervised)"):
                st.session_state.ml_mode = "supervised"
                st.session_state.page = "analysis"
                st.rerun()

    with tab2:
        st.subheader("Unsupervised Learning")
        
        uploaded_file_unsup = st.file_uploader("Upload CSV or Excel for Unsupervised Learning (Clustering)", type=["csv", "xlsx"], key="unsupervised_upload")

        if uploaded_file_unsup is not None:
            df = DataIngestion().ingest(uploaded_file_unsup)
            st.session_state.uploaded_df = df
            st.session_state.cleaned_df = df.copy()
            st.session_state.correlated_to_drop = []
            st.session_state.correlated_pairs = []
            persist_cleaned_df(st.session_state.cleaned_df)

            if st.button("▶ Proceed to Analysis (Unsupervised)"):
                st.session_state.ml_mode = "unsupervised"
                st.session_state.page = "analysis"
                st.rerun()

    with tab3:
        st.subheader("Data Exploration")
        st.info("Coming soon.")

# ══════════════════════════════════════
# ANALYSIS PAGE (placeholder)
# ══════════════════════════════════════
elif st.session_state.page == "analysis":
    st.title("📊 Dataset Analysis")

    # Always work on the progressively cleaned version if it exists
    df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.uploaded_df

    analyzer = DataAnalyzer()

    # ── 4 button bar ──────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔢 Shape", use_container_width=True):
            st.session_state.analysis_view = "shape"
    with col2:
        if st.button("👀 Head", use_container_width=True):
            st.session_state.analysis_view = "head"
    with col3:
        if st.button("ℹ️ Info", use_container_width=True):
            st.session_state.analysis_view = "info"
    with col4:
        if st.button("📈 Describe", use_container_width=True):
            st.session_state.analysis_view = "describe"

    st.divider()

    # ── Display ───────────────────────────────
    view = st.session_state.get("analysis_view", "shape")

    if view == "shape":
        result = analyzer.get_shape(df)
        st.subheader("🔢 Dataset Shape")
        st.metric("Rows", result["rows"])
        st.metric("Columns", result["columns"])

    elif view == "head":
        result = analyzer.get_head(df)
        st.subheader("👀 First 10 Rows")
        st.dataframe(result, use_container_width=True)

    elif view == "info":
        result = analyzer.get_info(df)
        st.subheader("ℹ️ Dataset Info")
        st.dataframe(result, use_container_width=True)

    elif view == "describe":
        result = analyzer.get_describe(df)
        st.subheader("📈 Statistical Summary")
        st.dataframe(result, use_container_width=True)





    st.divider()
    # Visualization buttons
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if st.button("📊 Histogram", use_container_width=True, key="btn_hist"):
            st.session_state.viz_view = "histogram"

    with col6:
        if st.button("📦 Box Plot", use_container_width=True, key="btn_box"):
            st.session_state.viz_view = "boxplot"

    with col7:
        if st.button("🔵 Scatter", use_container_width=True, key="btn_scatter"):
            st.session_state.viz_view = "scatter"

    with col8:
        if st.button("📈 Regression", use_container_width=True, key="btn_reg"):
            st.session_state.viz_view = "regression"

    st.divider()

    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    viz_view = st.session_state.get("viz_view", None)

    # ── show only what was clicked ─────────────────
    if viz_view == "histogram":
        selected = st.selectbox("Select column", num_cols, key="hist_sel")
        #if st.button("Show Histogram"):
        fig = analyzer.plot_histogram(df, selected)
        st.pyplot(fig)

    elif viz_view == "boxplot":
        selected = st.selectbox("Select column", num_cols, key="box_sel")
        #if st.button("Show Boxplot"):
        fig = analyzer.plot_boxplot(df, selected)
        st.pyplot(fig)

    elif viz_view == "scatter":
        col_a, col_b = st.columns(2)
        with col_a:
            col_x = st.selectbox("X-axis", num_cols, key="scat_x")
        with col_b:
            col_y = st.selectbox("Y-axis", num_cols, key="scat_y")
        #if st.button("Show Scatter"):
            fig = analyzer.plot_scatter(df, col_x, col_y)
            st.pyplot(fig)

    elif viz_view == "regression":
        col_a, col_b = st.columns(2)
        with col_a:
            col_x = st.selectbox("X (independent)", num_cols, key="reg_x")
        with col_b:
            col_y = st.selectbox("Y (dependent)", num_cols, key="reg_y")
        #if st.button("Show Regression"):
            fig = analyzer.plot_regression(df, col_x, col_y)
            st.pyplot(fig)





    # # get heat map from data_analseu with col 9

    st.divider()

    col9, = st.columns(1)

    with col9:
        if st.button("🌡️ Heatmap", use_container_width=True):
            st.session_state.col_view = "heatmap"
    col_view = st.session_state.get("col_view", None)
    # ── outside col9 ──────────────────────
    if col_view == "heatmap":
        num_cols = analyzer.get_numerical_cols(df)
        
        col_a, col_b = st.columns(2)
        with col_a:
            feature1 = st.selectbox("Select Feature 1", num_cols, key="heatmap_f1")
        with col_b:
            feature2 = st.selectbox("Select Feature 2", num_cols, key="heatmap_f2")

        #if st.button("Show Heatmap"):
            fig = analyzer.get_heatmap(df, feature1, feature2)
            st.pyplot(fig)

    # ── Modify Data ──────────────────────
    st.divider()
    st.subheader("🛠️ Data Cleaning")
    col10, col11, col12 ,col13= st.columns(4)
    with col10: 
        if st.button("🧹 Handle Nulls", use_container_width=True):
            st.session_state.clean_view = "nulls"
    with col11:
        if st.button("🚫 Outliers", use_container_width=True):
            st.session_state.clean_view = "outliers"
    with col12:
        if st.button("🔤 Encoding", use_container_width=True):
            st.session_state.clean_view = "encoding"
    with col13:
        if st.button("🔗 Remove Correlated", use_container_width=True):
            st.session_state.clean_view = "correlation"
    clean_view = st.session_state.get("clean_view", None)
    if clean_view == "nulls":
        null_counts = df.isnull().sum()
        if null_counts.sum() == 0:
            st.success("No null values found.")
        else:
            st.warning("Null values detected. Choose a method to handle them.")
            method = st.selectbox("Select Method", ["Fill with Mean", "Fill with Mode", "Fill with Median"])
            if st.button("Apply"):
                cleaned_df = analyzer.handle_nulls(df, method)
                saved_path = persist_cleaned_df(cleaned_df)
                st.success(f"Null values handled using {method}.")
                st.caption(f"Saved progressive cleaned data: {saved_path}")
                st.rerun()

    elif clean_view == "outliers":
        outliers = analyzer.detect_outliers(df)
        if outliers.empty:
            st.success("No outliers detected.")
        else:
            st.warning(f"Outliers detected in columns: {', '.join(outliers.columns)}")
            if st.button("Remove Outliers"):
                cleaned_df = analyzer.remove_outliers(df)
                saved_path = persist_cleaned_df(cleaned_df)
                st.success("Outliers removed.")
                st.success(f"New dataset shape: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns.")
                st.caption(f"Saved progressive cleaned data: {saved_path}")
                st.rerun()


    elif clean_view == "encoding":
        cat_cols = analyzer.get_categorical_cols(df)
        if not cat_cols:
            st.success("No categorical columns found.")
        else:
            st.warning(f"Categorical columns detected: {', '.join(cat_cols)}")
            method = st.selectbox("Select Encoding Method", ["Remove Category", "Convert All into Numerical"])
            if st.button("Apply Encoding"):
                cleaned_df = analyzer.encode_categorical(df, method)
                saved_path = persist_cleaned_df(cleaned_df)
                st.success(f"Categorical columns handled using {method}.")
                st.caption(f"Saved progressive cleaned data: {saved_path}")
                st.rerun()
    elif clean_view == "correlation":
        st.subheader("🔗 Remove Highly Correlated Features")

        num_cols = analyzer.get_numerical_cols(df)
        
        if len(num_cols) < 2:
            st.info("Not enough numerical columns to check correlation.")
        else:
            ml_mode = st.session_state.get("ml_mode", "supervised")
            target_column = None
            if ml_mode == "supervised":
                target_column = st.selectbox(
                    "Select target column (this will never be removed)",
                    df.columns
                )
            else:
                st.info("Unsupervised Mode: All features are checked for correlation.")

            threshold = st.slider(
                "Correlation Threshold",
                min_value=0.5,
                max_value=0.99,
                value=0.85,
                step=0.01
            )

            if st.button("Find Correlated Features"):
                to_drop, pairs = analyzer.remove_highly_correlated_features(df, target_column, threshold)
                st.session_state.correlated_to_drop = to_drop
                st.session_state.correlated_pairs = pairs

                if not to_drop:
                    st.success("No highly correlated features found.")
                else:
                    st.warning(f"Suggested features to remove: {', '.join(to_drop)}")

            if st.session_state.correlated_pairs:
                st.write("Correlated feature pairs:")
                for a, b, c in st.session_state.correlated_pairs:
                    st.write(f"{a} ↔ {b} = {c:.3f}")

            if st.session_state.correlated_to_drop and st.button("Remove Suggested Features"):
                cleaned_df = df.drop(columns=st.session_state.correlated_to_drop)
                saved_path = persist_cleaned_df(cleaned_df)
                st.success(f"Removed features: {', '.join(st.session_state.correlated_to_drop)}")
                st.success(f"New dataset shape: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns.")
                st.caption(f"Saved progressive cleaned data: {saved_path}")
                st.session_state.correlated_to_drop = []
                st.session_state.correlated_pairs = []
                st.rerun()
    # ── Save Final Dataset ────────────────────────────────────────
    st.divider()
    st.subheader("💾 Save Final Dataset")

    working_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.uploaded_df

    st.info(
        f"Current working dataset: **{working_df.shape[0]} rows × {working_df.shape[1]} columns**"
    )

    if st.button("💾 Save to artifacts/cleaned_data.csv", use_container_width=True):
        saved_path = persist_cleaned_df(working_df)
        st.success(f"✅ Final dataset saved → `{saved_path}`")
        st.dataframe(working_df.head(5), use_container_width=True)

    st.divider()
    if st.button("▶ Proceed to Transform Data", use_container_width=True):
        st.session_state.page = "transform"
        st.rerun()

elif st.session_state.page == "transform":
    st.title("⚙️ Data Transform")

    base_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.uploaded_df
    transformer = DataTransform()

    if base_df is None:
        st.warning("Please upload and analyze data first.")
        if st.button("⬅ Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    else:
        st.info(f"Input dataset shape: {base_df.shape[0]} rows × {base_df.shape[1]} columns")

        transform_option = st.radio(
            "Choose transformation type",
            [
                "Standard Scaler",
                "Min Max Scaler",
                "Normalization of All Feature Values",
                "Train Without Scaling"
            ],
            index=0
        )

        if st.button("Apply Transform", use_container_width=True):
            if transform_option == "Train Without Scaling":
                transformed_df = base_df.copy()
                file_name = "transformed_without_scaling.csv"
            else:
                method_map = {
                    "Standard Scaler": "Standard",
                    "Min Max Scaler": "MinMax",
                    "Normalization of All Feature Values": "Normalize"
                }
                transformed_df = transformer.scale_features(base_df, method=method_map[transform_option])
                file_name = "transformed_data.csv"

            saved_path = transformer.save_transformed_dataset(transformed_df, filename=file_name)
            st.session_state.transformed_df = transformed_df
            st.session_state.transformed_path = saved_path
            st.session_state.last_transform_option = transform_option

            st.success(f"Transformation completed and saved to: {saved_path}")
            st.subheader("Head After Transformation")
            st.dataframe(transformed_df.head(10), use_container_width=True)

        if st.session_state.transformed_df is not None:
            st.divider()
            st.caption(f"Saved transformed file: {st.session_state.transformed_path}")
            st.subheader("Current Transformed Preview")
            st.dataframe(st.session_state.transformed_df.head(10), use_container_width=True)

            if st.button("▶ Proceed to Training", use_container_width=True):
                st.session_state.page = "training"
                st.rerun()

        if st.button("⬅ Back to Analysis"):
            st.session_state.page = "analysis"
            st.rerun()

elif st.session_state.page == "training":
    st.title("🏋️ Training Setup")

    train_df = (
        st.session_state.transformed_df
        if st.session_state.transformed_df is not None
        else st.session_state.cleaned_df
    )
    training_scale_used = (
        st.session_state.last_transform_option
        if st.session_state.transformed_df is not None
        else "Train Without Scaling"
    )
    
    ml_mode = st.session_state.get("ml_mode", "supervised")

    if train_df is None:
        st.warning("No cleaned data available. Please finish analysis/cleaning first.")
        if st.button("⬅ Back to Analysis"):
            st.session_state.page = "analysis"
            st.rerun()
    else:
        st.info(f"Training dataset shape: {train_df.shape[0]} rows × {train_df.shape[1]} columns | Mode: {ml_mode.capitalize()}")

        tab_train, tab_grid = st.tabs(["Model setup / Results", "Grid search config / Results"])

        with tab_train:
            st.subheader("Model Setup")
            
            target_col = None
            problem_type = "Clustering" if ml_mode == "unsupervised" else "Regression"
            cv_folds = 5
            scoring = None
            
            if ml_mode == "supervised":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    target_col = st.selectbox("Target column", train_df.columns, key="train_target_col")
                with col_b:
                    problem_type = st.selectbox("Problem type", ["Regression", "Classification"], key="train_problem_type")
                with col_c:
                    cv_folds = st.number_input("CV folds", min_value=0, max_value=20, value=5, step=1, key="train_cv_folds")

                col_d, col_e = st.columns(2)
                with col_d:
                    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="train_test_size")
                with col_e:
                    if problem_type == "Regression":
                        scoring = st.selectbox("Scoring metric", ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"], index=0, key="train_scoring")
                    else:
                        scoring = st.selectbox("Scoring metric", ["accuracy", "f1", "precision", "recall"], index=0, key="train_scoring")

                if problem_type == "Regression":
                    models_registry = ModelTrainer.get_regression_models()
                else:
                    models_registry = ClassificationTrainer.get_classification_models()
            else:
                # Unsupervised Setup
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info("No Target Column needed for Clustering.")
                with col_b:
                    test_size = st.slider("Test size (for out-of-sample silhouette test)", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="train_test_size")
                
                models_registry = ClusteringTrainer.get_clustering_models()

            model_names = list(models_registry.keys())

            selected_model_names = st.multiselect(
                "Select models",
                model_names,
                default=model_names,
                key="selected_model_names",
            )

            st.subheader("Parameter Adjustment")
            selected_model_params = {}
            if selected_model_names:
                parameter_tabs = st.tabs(selected_model_names)
                for i, model_name in enumerate(selected_model_names):
                    with parameter_tabs[i]:
                        selected_model_params[model_name] = collect_model_params_ui(model_name)

            if st.button("Train selected models", use_container_width=True):
                if not selected_model_names:
                    st.warning("Please select at least one model.")
                else:
                    target_encoder = None
                    working_df = train_df.copy()
                    
                    if ml_mode == "supervised":
                        if problem_type == "Classification":
                            clf_transform = ClassificationTransform()
                            working_df = clf_transform.encode_target(working_df, target_col)
                            target_encoder = clf_transform.label_encoder

                        X = working_df.drop(columns=[target_col])
                        y = working_df[target_col]

                        # Convert categorical features to numeric
                        X = pd.get_dummies(X, drop_first=False)

                        if problem_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
                            st.error("Selected target is not numeric. Please encode target or choose a numeric target.")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                            if problem_type == "Regression":
                                trainer = ModelTrainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, problem_type="regression")
                            else:
                                trainer = ClassificationTrainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

                            models_to_train = {name: {"model": models_registry[name], "parameters": selected_model_params.get(name, {})} for name in selected_model_names}

                            results = trainer.train_selected_models(selected_models=models_to_train, cv_folds=(int(cv_folds) if cv_folds > 1 else None), scoring=scoring)

                            if problem_type == "Regression":
                                results_df = ModelTrainer.results_to_dataframe(results)
                                best_metric = "Test Score"
                            else:
                                results_df = ClassificationTrainer.results_to_dataframe(results)
                                best_metric = "Test Accuracy"
                    else:
                        # Unsupervised
                        X = working_df.copy()
                        X = pd.get_dummies(X, drop_first=False)
                        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
                        
                        trainer = ClusteringTrainer(X_train=X_train, X_test=X_test)
                        models_to_train = {name: {"model": models_registry[name], "parameters": selected_model_params.get(name, {})} for name in selected_model_names}
                        
                        results = trainer.train_selected_models(selected_models=models_to_train)
                        results_df = ClusteringTrainer.results_to_dataframe(results)
                        best_metric = "Test Silhouette"

                    st.session_state.training_results_df = results_df

                    if not results_df.empty:
                        best_idx = results_df[best_metric].astype(float).idxmax()
                        best_row = results_df.loc[best_idx].to_dict()
                        best_row["_target_encoder"] = target_encoder
                        
                        # Pack the wrapper if it was unsupervised
                        if ml_mode == "unsupervised":
                            # find the exact model obj and save it
                            for res in results:
                                if res["model_name"] == best_row["Model"]:
                                    best_row["_model_obj"] = res["model_obj"]
                                    break
                                    
                        st.session_state.best_training_result = best_row
                    else:
                        st.session_state.best_training_result = None

            if st.session_state.best_training_result is not None:
                best = st.session_state.best_training_result
                if ml_mode == "supervised":
                    if problem_type == "Regression":
                        st.success(f"Best Model: {best['Model']} | Test Score: {best.get('Test Score', 0):.4f} | Parameters: {best.get('Parameters', {})}")
                    else:
                        st.success(f"Best Model: {best['Model']} | Test Accuracy: {best.get('Test Accuracy', 0):.4f} | Test F1: {best.get('Test F1', 0):.4f} | Parameters: {best.get('Parameters', {})}")
                else:
                    st.success(f"Best Model: {best['Model']} | Test Silhouette: {best.get('Test Silhouette', 0):.4f} | Train DB-Index: {best.get('Train DB-Index', 0):.4f} | Parameters: {best.get('Parameters', {})}")

                if st.button("💾 Save Best Model (Training)", use_container_width=True):
                    try:
                        working_df = train_df.copy()
                        if ml_mode == "supervised":
                            if best.get("_target_encoder") is not None:
                                working_df[target_col] = best["_target_encoder"].transform(working_df[target_col].astype(str))

                            X = working_df.drop(columns=[target_col])
                            y = working_df[target_col]
                            X = pd.get_dummies(X, drop_first=False)
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                            best_model_name = best["Model"]
                            best_params = best["Parameters"] if isinstance(best["Parameters"], dict) else {}
                            best_model = models_registry[best_model_name](**best_params)
                            best_model.fit(X_train, y_train)

                            raw_input_df = train_df.drop(columns=[target_col]).copy()
                        else:
                            X = working_df.copy()
                            X = pd.get_dummies(X, drop_first=False)
                            
                            best_model = best.get("_model_obj")
                            best_model_name = best["Model"]
                            best_params = best["Parameters"] if isinstance(best["Parameters"], dict) else {}
                            # Unsupervised wrapper fit for predictions
                            if not hasattr(best_model, "predict"):
                                from sklearn.neighbors import KNeighborsClassifier
                                knn = KNeighborsClassifier(n_neighbors=1)
                                knn.fit(X, best_model.fit_predict(X))
                                best_model.predict_wrapper = knn

                            raw_input_df = train_df.copy()

                        cat_cols = raw_input_df.select_dtypes(include=["object", "category"]).columns.tolist()
                        num_cols = raw_input_df.select_dtypes(include="number").columns.tolist()
                        cat_options = {
                            col: [str(v) for v in raw_input_df[col].dropna().astype(str).unique().tolist()[:100]]
                            for col in cat_cols
                        }

                        model_bundle = {
                            "model": best_model,
                            "model_name": best_model_name,
                            "parameters": best_params,
                            "feature_columns": X.columns.tolist(),
                            "input_columns": raw_input_df.columns.tolist(),
                            "numeric_input_columns": num_cols,
                            "categorical_input_options": cat_options,
                            "target_column": target_col if ml_mode == "supervised" else None,
                            "problem_type": problem_type.lower(),
                            "scale_used": training_scale_used,
                            "label_encoder": best.get("_target_encoder"),
                        }
                        save_object(BEST_MODEL_TRAINING_PATH, model_bundle)
                        st.session_state.saved_model_path = BEST_MODEL_TRAINING_PATH
                        st.success(f"Saved best training model to {BEST_MODEL_TRAINING_PATH}")
                    except Exception as e:
                        st.error(f"Saving training best model failed: {e}")

            if st.session_state.training_results_df is not None:
                st.dataframe(st.session_state.training_results_df, use_container_width=True)

        with tab_grid:
            st.subheader("Grid Search Config")

            grid_target_col = None
            grid_problem_type = "Clustering" if ml_mode == "unsupervised" else "Regression"
            grid_cv_folds = 5
            grid_scoring = None
            
            if ml_mode == "supervised":
                col_g1, col_g2, col_g3, col_g4 = st.columns(4)
                with col_g1:
                    grid_target_col = st.selectbox("Target column", train_df.columns, key="grid_target_col")
                with col_g2:
                    grid_problem_type = st.selectbox("Problem type", ["Regression", "Classification"], key="grid_problem_type")
                with col_g3:
                    grid_cv_folds = st.number_input("CV folds", min_value=2, max_value=20, value=5, step=1, key="grid_cv_folds")
                with col_g4:
                    if grid_problem_type == "Regression":
                        grid_scoring = st.selectbox("Scoring metric", ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"], index=0, key="grid_scoring")
                    else:
                        grid_scoring = st.selectbox("Scoring metric", ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"], index=0, key="grid_scoring")

                if grid_problem_type == "Regression":
                    grid_models_registry = ModelTrainer.get_regression_models()
                else:
                    grid_models_registry = ClassificationTrainer.get_classification_models()
            else:
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.info("Unsupervised parameters: Grids evaluated via Silhouette Score")
                with col_g2:
                    pass
                grid_models_registry = ClusteringTrainer.get_clustering_models()

            grid_model_names = list(grid_models_registry.keys())

            grid_selected_models = st.multiselect(
                "Select models for grid search",
                grid_model_names,
                default=[grid_model_names[0]] if grid_model_names else [],
                key="grid_selected_models",
            )

            model_grid_inputs = {}
            if grid_selected_models:
                grid_tabs = st.tabs(grid_selected_models)
                for i, model_name in enumerate(grid_selected_models):
                    with grid_tabs[i]:
                        default_grid = get_default_model_grid(model_name)
                        model_grid_inputs[model_name] = st.text_area(
                            f"Parameter Grid for {model_name} (JSON format)",
                            value=json.dumps(default_grid, indent=2),
                            height=220,
                            key=f"grid_json_{model_name}",
                        )

            if st.button("Run Grid Search", use_container_width=True):
                if not grid_selected_models:
                    st.warning("Please select at least one model for grid search.")
                else:
                    grid_target_encoder = None
                    working_df = train_df.copy()
                    
                    if ml_mode == "supervised":
                        if grid_problem_type == "Classification":
                             clf_transform = ClassificationTransform()
                             working_df = clf_transform.encode_target(working_df, grid_target_col)
                             grid_target_encoder = clf_transform.label_encoder

                        X = working_df.drop(columns=[grid_target_col])
                        y = working_df[grid_target_col]
                        X = pd.get_dummies(X, drop_first=False)

                        if grid_problem_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
                            st.error("Selected target is not numeric. Please encode target or choose a numeric target.")
                        else:
                            try:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                                if grid_problem_type == "Regression":
                                    trainer = ModelTrainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, problem_type="regression")
                                else:
                                    trainer = ClassificationTrainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

                                all_rows = []
                                best_model_name = None
                                best_score = None
                                best_params = None

                                for model_name in grid_selected_models:
                                    model_grid_text = model_grid_inputs.get(model_name, "{}")
                                    parameter_grid = json.loads(model_grid_text)
                                    if not isinstance(parameter_grid, dict):
                                        raise ValueError(f"Grid for {model_name} must be a JSON object.")

                                    grid_result = trainer.run_grid_search(
                                        model_class=grid_models_registry[model_name],
                                        parameter_grid=parameter_grid,
                                        cv_folds=int(grid_cv_folds),
                                        scoring=grid_scoring,
                                        model_name=model_name,
                                    )

                                    if best_score is None or grid_result["best_score"] > best_score:
                                        best_score = grid_result["best_score"]
                                        best_model_name = grid_result["model_name"]
                                        best_params = grid_result["best_parameters"]

                                    model_table = grid_result["full_results_table"].copy()
                                    model_table["Model"] = model_name
                                    all_rows.append(model_table)

                                combined_results = pd.concat(all_rows, ignore_index=True)
                                if "Test Score" in combined_results.columns:
                                    combined_results = combined_results.sort_values("Test Score", ascending=False)
                                top_5 = combined_results.head(5).copy()

                                st.session_state.grid_search_output = {
                                    "model_name": best_model_name,
                                    "best_score": float(best_score) if best_score is not None else None,
                                    "best_parameters": best_params,
                                    "full_results_table": combined_results,
                                    "top_5_results": top_5,
                                    "problem_type": grid_problem_type,
                                    "target_encoder": grid_target_encoder,
                                }

                            except Exception as e:
                                st.error(f"Grid search failed: {e}")
                    else:
                        # Unsupervised Grid Search
                        X = working_df.copy()
                        X = pd.get_dummies(X, drop_first=False)
                        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
                        
                        try:
                            trainer = ClusteringTrainer(X_train=X_train, X_test=X_test)

                            all_rows = []
                            best_model_name = None
                            best_score = None
                            best_params = None

                            for model_name in grid_selected_models:
                                model_grid_text = model_grid_inputs.get(model_name, "{}")
                                parameter_grid = json.loads(model_grid_text)
                                if not isinstance(parameter_grid, dict):
                                    raise ValueError(f"Grid for {model_name} must be a JSON object.")

                                grid_result = trainer.run_grid_search(
                                    model_class=grid_models_registry[model_name],
                                    parameter_grid=parameter_grid,
                                    model_name=model_name,
                                )

                                if best_score is None or grid_result["best_score"] > best_score:
                                    best_score = grid_result["best_score"]
                                    best_model_name = grid_result["model_name"]
                                    best_params = grid_result["best_parameters"]

                                model_table = grid_result["full_results_table"].copy()
                                model_table["Model"] = model_name
                                all_rows.append(model_table)

                            combined_results = pd.concat(all_rows, ignore_index=True)
                            if "Test Silhouette" in combined_results.columns:
                                combined_results = combined_results.sort_values("Test Silhouette", ascending=False)
                            top_5 = combined_results.head(5).copy()

                            st.session_state.grid_search_output = {
                                "model_name": best_model_name,
                                "best_score": float(best_score) if best_score is not None else None,
                                "best_parameters": best_params,
                                "full_results_table": combined_results,
                                "top_5_results": top_5,
                                "problem_type": "clustering",
                                "target_encoder": None,
                            }
                        except Exception as e:
                            st.error(f"Unsupervised Grid search failed: {e}")

            if st.session_state.grid_search_output is not None:
                grid_out = st.session_state.grid_search_output
                st.success(
                    f"Best Model: {grid_out['model_name']} | Best Score: {grid_out['best_score']:.4f} | Best Parameters: {grid_out['best_parameters']}"
                )

                if st.button("💾 Save Best Model (Grid Search)", use_container_width=True):
                    try:
                        working_df = train_df.copy()
                        if ml_mode == "supervised":
                            if grid_out.get("target_encoder") is not None:
                                working_df[grid_target_col] = grid_out["target_encoder"].transform(working_df[grid_target_col].astype(str))

                            X = working_df.drop(columns=[grid_target_col])
                            y = working_df[grid_target_col]
                            X = pd.get_dummies(X, drop_first=False)
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            best_model_name = grid_out["model_name"]
                            best_params = grid_out["best_parameters"]
                            grid_models = ModelTrainer.get_regression_models() if grid_out.get("problem_type") == "Regression" else ClassificationTrainer.get_classification_models()
                            best_model = grid_models[best_model_name](**best_params)
                            best_model.fit(X_train, y_train)
                            raw_input_df = train_df.drop(columns=[grid_target_col]).copy()
                        else:
                            X = working_df.copy()
                            X = pd.get_dummies(X, drop_first=False)
                            
                            best_model_name = grid_out["model_name"]
                            best_params = grid_out["best_parameters"]
                            grid_models = ClusteringTrainer.get_clustering_models()
                            best_model = grid_models[best_model_name](**best_params)
                            
                            labels = best_model.fit_predict(X)
                            if not hasattr(best_model, "predict"):
                                from sklearn.neighbors import KNeighborsClassifier
                                knn = KNeighborsClassifier(n_neighbors=1)
                                knn.fit(X, labels)
                                best_model.predict_wrapper = knn
                            
                            raw_input_df = train_df.copy()

                        cat_cols = raw_input_df.select_dtypes(include=["object", "category"]).columns.tolist()
                        num_cols = raw_input_df.select_dtypes(include="number").columns.tolist()
                        cat_options = {
                            col: [str(v) for v in raw_input_df[col].dropna().astype(str).unique().tolist()[:100]]
                            for col in cat_cols
                        }

                        model_bundle = {
                            "model": best_model,
                            "model_name": best_model_name,
                            "parameters": best_params,
                            "feature_columns": X.columns.tolist(),
                            "input_columns": raw_input_df.columns.tolist(),
                            "numeric_input_columns": num_cols,
                            "categorical_input_options": cat_options,
                            "target_column": grid_target_col if ml_mode == "supervised" else None,
                            "problem_type": grid_out.get("problem_type", "clustering").lower(),
                            "scale_used": training_scale_used,
                            "label_encoder": grid_out.get("target_encoder"),
                        }
                        save_object(BEST_MODEL_GRID_PATH, model_bundle)
                        st.session_state.saved_model_path = BEST_MODEL_GRID_PATH
                        st.success(f"Saved best grid-search model to {BEST_MODEL_GRID_PATH}")
                    except Exception as e:
                        st.error(f"Saving grid-search best model failed: {e}")

                st.subheader("Top 5 Results Across Selected Models")
                st.dataframe(grid_out["top_5_results"], use_container_width=True)
                st.subheader("All Grid Search Results")
                st.dataframe(grid_out["full_results_table"], use_container_width=True)

        st.divider()
        st.subheader("🔎 Explore Your Model Prediction")

        available_model_paths = []
        if os.path.exists(BEST_MODEL_TRAINING_PATH):
            available_model_paths.append(BEST_MODEL_TRAINING_PATH)
        if os.path.exists(BEST_MODEL_GRID_PATH):
            available_model_paths.append(BEST_MODEL_GRID_PATH)

        if available_model_paths:
            selected_model_path = st.selectbox("Select saved model", available_model_paths)

            pipeline = PredictionPipeline(selected_model_path)
            st.caption(f"Model: {pipeline.model_name} | Scale used: {pipeline.bundle.get('scale_used', 'Unknown')}")

            predict_mode = st.radio(
                "Prediction mode",
                ["Compare Actual vs Predicted", "Custom Input Prediction"],
                index=0,
            )

            if predict_mode == "Compare Actual vs Predicted" and st.button("Run Prediction Exploration", use_container_width=True):
                try:
                    prediction_df = train_df.copy()

                    if pipeline.bundle.get("problem_type", "regression") == "clustering":
                        y_pred = pipeline.predict(prediction_df)
                        from src.component.data_analyzer import ClusteringAnalyzer
                        c_analyzer = ClusteringAnalyzer()
                        fig = c_analyzer.plot_clusters_pca(prediction_df, y_pred, title=f"KMeans Clusters via PCA ({pipeline.model_name})")
                        if fig is not None:
                            st.pyplot(fig)
                        else:
                            st.error("Could not construct PCA cluster visualization.")
                    else:
                        if pipeline.target_column not in prediction_df.columns:
                            st.error("Target column from saved model not found in current dataframe.")
                        else:
                            y_true = prediction_df[pipeline.target_column]
                            y_pred = pipeline.predict(prediction_df)

                            compare_df = pd.DataFrame(
                                {
                                    "Actual": y_true.values,
                                    "Predicted": y_pred,
                                }
                            )

                            st.caption(f"Model used: {pipeline.model_name}")
                            st.dataframe(compare_df.head(50), use_container_width=True)

                            if pipeline.bundle.get("problem_type", "regression") == "classification":
                                acc = accuracy_score(y_true, y_pred)
                                
                                is_binary = len(np.unique(y_true)) == 2
                                avg_method = "binary" if is_binary else "weighted"

                                f1 = f1_score(y_true, y_pred, average=avg_method)

                                m1, m2 = st.columns(2)
                                with m1:
                                    st.metric("Accuracy", f"{acc:.4f}")
                                with m2:
                                    st.metric("F1 Score", f"{f1:.4f}")

                                st.subheader("Confusion Matrix")
                                cm = confusion_matrix(y_true, y_pred)
                                fig, ax = plt.subplots(figsize=(6, 5))
                                
                                labels = None
                                if pipeline.label_encoder is not None:
                                    labels = pipeline.label_encoder.classes_
                                    
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                            xticklabels=labels if labels is not None else "auto",
                                            yticklabels=labels if labels is not None else "auto")
                                ax.set_xlabel("Predicted")
                                ax.set_ylabel("Actual")
                                ax.set_title("Confusion Matrix")
                                st.pyplot(fig)
                                
                            else:
                                mse = mean_squared_error(y_true, y_pred)
                                rmse = mse ** 0.5
                                mae = mean_absolute_error(y_true, y_pred)
                                r2 = r2_score(y_true, y_pred)

                                m1, m2, m3 = st.columns(3)
                                with m1:
                                    st.metric("RMSE", f"{rmse:.4f}")
                                with m2:
                                    st.metric("MAE", f"{mae:.4f}")
                                with m3:
                                    st.metric("R2", f"{r2:.4f}")

                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.scatter(y_true, y_pred, alpha=0.6)
                                min_val = min(float(y_true.min()), float(pd.Series(y_pred).min()))
                                max_val = max(float(y_true.max()), float(pd.Series(y_pred).max()))
                                ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
                                ax.set_xlabel("Actual")
                                ax.set_ylabel("Predicted")
                                ax.set_title("Actual vs Predicted")
                                ax.grid(alpha=0.3)
                                st.pyplot(fig)

                except Exception as e:
                    st.error(f"Prediction exploration failed: {e}")

            if predict_mode == "Custom Input Prediction":
                try:
                    input_columns = pipeline.bundle.get("input_columns", [])
                    numeric_columns = pipeline.bundle.get("numeric_input_columns", [])
                    categorical_options = pipeline.bundle.get("categorical_input_options", {})

                    if not input_columns:
                        st.info("This saved model does not include custom input schema metadata.")
                    else:
                        st.write("Enter custom values to get one prediction:")
                        custom_row = {}
                        for col in input_columns:
                            if col in numeric_columns:
                                custom_row[col] = st.number_input(
                                    f"{col}",
                                    value=0.0,
                                    key=f"custom_input_{col}",
                                )
                            else:
                                options = categorical_options.get(col, [""])
                                custom_row[col] = st.selectbox(
                                    f"{col}",
                                    options,
                                    key=f"custom_input_{col}",
                                )

                        if st.button("Predict Custom Input", use_container_width=True):
                            custom_df = pd.DataFrame([custom_row])
                            pred = pipeline.predict(custom_df)
                            if pipeline.bundle.get("problem_type", "regression") == "classification":
                                st.success(f"Predicted Class: {pred[0]}")
                            elif pipeline.bundle.get("problem_type", "regression") == "clustering":
                                st.success(f"Predicted Cluster: {pred[0]}")
                            else:
                                st.success(f"Predicted output: {float(pred[0]):.6f}")
                except Exception as e:
                    st.error(f"Custom input prediction failed: {e}")
        else:
            st.info("No saved best model found yet. Save best model from training or grid search first.")

        if st.button("⬅ Back to Transform"):
            st.session_state.page = "transform"
            st.rerun()


