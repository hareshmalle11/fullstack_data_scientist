import os
import requests
from datetime import datetime
import pandas as pd
import streamlit as st
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# logger
def log(message):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# session state initialization
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

# folder set up
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(base_dir, 'data', 'raw')
cleaned_dir = os.path.join(base_dir, 'data', 'cleaned')

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(cleaned_dir, exist_ok=True)

log("Application Started")
log(f"raw_dir: {raw_dir}")
log(f"cleaned_dir: {cleaned_dir}")

# page configuration
st.set_page_config(page_title="End to End SVM", layout="wide")
st.title("End to End SVM Platform")

# Sidebar navigation: model settings
st.sidebar.title("SVM settings")
kernal = st.sidebar.selectbox("Select Kernel", ("linear", "poly", "rbf", "sigmoid"))
C = st.sidebar.slider("Select C (Regularization parameter)", 0.01, 10.0, 1.0, 0.01)
gamma = st.sidebar.selectbox("Select Gamma", ("scale", "auto"))

log(f"Model settings - Kernel: {kernal}, C: {C}, Gamma: {gamma}")

# Step 1 : Data Ingestion
st.header("Step 1: Data Ingestion")
log("Data Ingestion Step Started")

option = st.radio("Choose data source", ["Upload CSV", "Download Dataset"])
df = None
raw_path = None

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        raw_path = os.path.join(raw_dir, uploaded_file.name)
        with open(raw_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(raw_path)
        st.success("File uploaded successfully")
        log(f"File uploaded to {raw_path}")

elif option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)
        raw_path = os.path.join(raw_dir, "iris.csv")
        with open(raw_path, 'wb') as f:
            f.write(response.content)
        df = pd.read_csv(raw_path)
        st.success("Iris dataset downloaded successfully")
        log(f"Iris dataset downloaded to {raw_path}")

# Step 2: EDA
if df is not None:
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    log("EDA Step Started")

    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.write("Shape of the data:", df.shape)
    st.write("Missing values:", df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    log("EDA Step Completed")

# Step 3: Data Cleaning
if df is not None:
    st.header("Step 3: Data Cleaning")
    log("Data Cleaning Step Started")

    strategy = st.selectbox("Missing value strategy", ["mean", "median", "mode", "drop"])
    df_cleaned = df.copy()

    if strategy == "drop":
        df_cleaned.dropna(inplace=True)
    else:
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            if strategy == "mean":
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            elif strategy == "median":
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            elif strategy == "mode":
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    st.session_state.cleaned_data = df_cleaned
    st.success("Data Cleaning Completed")
    log("Data Cleaning Step Completed")
else:
    st.info("Please complete Data Ingestion step first.")

# Step 4: Save cleaned data
if st.button("Save Cleaned Data"):
    if st.session_state.cleaned_data is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_file_name = f"cleaned_data_{timestamp}.csv"
        clean_path = os.path.join(cleaned_dir, clean_file_name)
        st.session_state.cleaned_data.to_csv(clean_path, index=False)
        st.success(f"Cleaned data saved to {clean_path}")
        log(f"Cleaned data saved to {clean_path}")
    else:
        st.info("No cleaned data to save.")

# Step 5: Load cleaned data
st.header("Step 5: Load Cleaned Data")
clean_files = os.listdir(cleaned_dir)

if clean_files:
    selected_file = st.selectbox("Select cleaned data file", clean_files)
    if st.button("Load Cleaned Data"):
        clean_path = os.path.join(cleaned_dir, selected_file)
        df_cleaned = pd.read_csv(clean_path)
        st.session_state.cleaned_data = df_cleaned
        st.success(f"Cleaned data loaded from {clean_path}")
        log(f"Cleaned data loaded from {clean_path}")
        st.dataframe(df_cleaned.head())
else:
    st.info("No cleaned data files found.")

# Step 6: Model Training
if st.session_state.cleaned_data is not None:
    st.header("Step 6: Model Training")

    df_cleaned = st.session_state.cleaned_data
    target = st.selectbox("Select target variable", df_cleaned.columns)

    y = df_cleaned[target]
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    X = df_cleaned.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')

    if X.empty:
        st.error("No numeric features available for training.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = SVC(kernel=kernal, C=C, gamma=gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("Model training completed")
        log("Model Training Step Completed")

        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model Accuracy: {acc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
else:
    st.info("Please load cleaned data before training the model.")
