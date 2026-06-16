import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "imdb_rnn.keras")
WORD_INDEX_PATH = os.path.join(ARTIFACT_DIR, "word_index.json")
META_PATH = os.path.join(ARTIFACT_DIR, "meta.json")


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(WORD_INDEX_PATH, "r", encoding="utf-8") as f:
        word_index = json.load(f)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, word_index, meta


def encode_review(text, word_index, num_words):
    words = text.lower().split()
    encoded = [word_index["<START>"]]
    oov_idx = word_index["<OOV>"]
    for word in words:
        idx = word_index.get(word, oov_idx)
        if idx >= num_words:
            idx = oov_idx
        encoded.append(idx)
    return encoded


def predict_sentiment(text, model, word_index, max_len, num_words):
    encoded = encode_review(text, word_index, num_words)
    padded = pad_sequences([encoded], maxlen=max_len, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "good" if score >= 0.5 else "bad"
    return score, label


st.set_page_config(page_title="RNN Movie Review", page_icon="🎬")

st.title("RNN Movie Review Sentiment")
st.write("Enter a movie review to predict whether it is good or bad.")

review = st.text_area("Review text", height=200)

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        if not os.path.exists(MODEL_PATH):
            st.error("Model not found. Run the training notebook first.")
        else:
            model, word_index, meta = load_artifacts()
            score, label = predict_sentiment(review, model, word_index, meta["max_len"], meta["num_words"])
            st.subheader(f"Prediction: {label.upper()}")
            st.write(f"Score: {score:.4f}")
            st.progress(min(max(score, 0.0), 1.0))
