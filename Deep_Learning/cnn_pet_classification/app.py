import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

os.environ["KERAS_BACKEND"] = "numpy"

import keras


st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

st.title("Cat vs Dog Classifier")
st.write("Upload an image and the trained model will predict whether it is a cat or a dog.")

MODEL_PATH = Path(__file__).with_name("cat_dog_classifier.keras")
IMAGE_SIZE = (128, 128)
CLASS_NAMES = {0: "Cat", 1: "Dog"}


@st.cache_resource
def load_model() -> keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. Put the .keras file in the same folder as app.py."
        )
    return keras.saving.load_model(MODEL_PATH, compile=False)


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array


uploaded_file = st.file_uploader("Choose a cat or dog image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    processed_image = prepare_image(image)
    prediction = model.predict(processed_image, verbose=0)[0]
    probability = float(prediction[0])

    predicted_class = CLASS_NAMES[1] if probability >= 0.5 else CLASS_NAMES[0]
    confidence = probability if predicted_class == CLASS_NAMES[1] else 1.0 - probability

    st.subheader("Prediction")
    st.success(f"{predicted_class} ({confidence:.2%} confidence)")

    st.write("Model output")
    st.progress(probability)
    st.caption(f"Probability of Dog: {probability:.4f}")
    st.caption(f"Probability of Cat: {1.0 - probability:.4f}")
else:
    st.info("Upload an image to see a prediction.")
