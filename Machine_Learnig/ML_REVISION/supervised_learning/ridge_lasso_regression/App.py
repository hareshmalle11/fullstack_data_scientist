"""Streamlit app to predict car prices with Ridge/Lasso regression."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ridge_lasso_model.pkl"


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


def number_input(label, defaults, key, min_value=None, step=1.0, fmt="%.2f"):
    value = float(defaults.get(key, 0.0))
    return st.number_input(label, min_value=min_value, value=value, step=step, format=fmt)


st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("Car Price Predictor")
st.write("Enter car details to predict price using the best Ridge or Lasso model.")

try:
    bundle = load_model_bundle()
except FileNotFoundError:
    st.error("Model file not found. Run `python ridge_lasso_regression.py` first.")
    st.stop()

model = bundle["model"]
defaults = bundle["input_defaults"]
options = bundle["category_options"]

with st.sidebar:
    st.subheader("Model")
    st.write(bundle["best_model_name"])
    for name, metrics in bundle["metrics"].items():
        st.metric(name, f"R2 {metrics['test_r2']:.3f}", f"RMSE {metrics['rmse']:.0f}")

tab_basic, tab_engine, tab_body = st.tabs(["Basic", "Engine", "Body"])

with tab_basic:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        symboling = st.slider("Symboling", min_value=-3, max_value=3, value=int(defaults.get("symboling", 0)))
        fueltype = st.selectbox("Fuel Type", options["fueltype"])
    with col2:
        aspiration = st.selectbox("Aspiration", options["aspiration"])
        doornumber = st.selectbox("Door Number", options["doornumber"])
    with col3:
        carbody = st.selectbox("Car Body", options["carbody"])
        drivewheel = st.selectbox("Drive Wheel", options["drivewheel"])
    with col4:
        enginelocation = st.selectbox("Engine Location", options["enginelocation"])
        fuelsystem = st.selectbox("Fuel System", options["fuelsystem"])

with tab_engine:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        enginetype = st.selectbox("Engine Type", options["enginetype"])
        cylindernumber = st.selectbox("Cylinder Number", options["cylindernumber"])
    with col2:
        enginesize = number_input("Engine Size", defaults, "enginesize", min_value=0.0)
        horsepower = number_input("Horsepower", defaults, "horsepower", min_value=0.0)
    with col3:
        peakrpm = number_input("Peak RPM", defaults, "peakrpm", min_value=0.0)
        compressionratio = number_input("Compression Ratio", defaults, "compressionratio", min_value=0.0)
    with col4:
        boreratio = number_input("Bore Ratio", defaults, "boreratio", min_value=0.0, step=0.1)
        stroke = number_input("Stroke", defaults, "stroke", min_value=0.0, step=0.1)

with tab_body:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wheelbase = number_input("Wheelbase", defaults, "wheelbase", min_value=0.0, step=0.1)
        carlength = number_input("Car Length", defaults, "carlength", min_value=0.0, step=0.1)
    with col2:
        carwidth = number_input("Car Width", defaults, "carwidth", min_value=0.0, step=0.1)
        carheight = number_input("Car Height", defaults, "carheight", min_value=0.0, step=0.1)
    with col3:
        curbweight = number_input("Curb Weight", defaults, "curbweight", min_value=0.0)
        citympg = number_input("City MPG", defaults, "citympg", min_value=0.0)
    with col4:
        highwaympg = number_input("Highway MPG", defaults, "highwaympg", min_value=0.0)

features = pd.DataFrame(
    [
        {
            "symboling": symboling,
            "fueltype": fueltype,
            "aspiration": aspiration,
            "doornumber": doornumber,
            "carbody": carbody,
            "drivewheel": drivewheel,
            "enginelocation": enginelocation,
            "wheelbase": wheelbase,
            "carlength": carlength,
            "carwidth": carwidth,
            "carheight": carheight,
            "curbweight": curbweight,
            "enginetype": enginetype,
            "cylindernumber": cylindernumber,
            "enginesize": enginesize,
            "fuelsystem": fuelsystem,
            "boreratio": boreratio,
            "stroke": stroke,
            "compressionratio": compressionratio,
            "horsepower": horsepower,
            "peakrpm": peakrpm,
            "citympg": citympg,
            "highwaympg": highwaympg,
        }
    ],
    columns=bundle["feature_columns"],
)

if st.button("Predict Price"):
    prediction = model.predict(features)[0]
    st.success(f"Predicted Car Price: ${prediction:,.2f}")
