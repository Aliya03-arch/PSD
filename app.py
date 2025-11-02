import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib

# === Load Model dan Encoder ===
scaler = joblib.load("scaler.pkl")
model = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🎙️ Klasifikasi Suara: Buka vs Tutup")

uploaded_file = st.file_uploader("Unggah file audio (.wav)", type=["wav"])

def extract_features(x):
    return np.array([
        np.mean(x), np.std(x), np.max(x), np.min(x),
        np.sqrt(np.mean(x**2)),
        np.mean(librosa.feature.zero_crossing_rate(x)),
        pd.Series(x).skew(),
        pd.Series(x).kurt()
    ]).reshape(1, -1)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Load dan ekstraksi fitur
    x, sr = librosa.load(uploaded_file, sr=None)
    feat = extract_features(x)
    feat_scaled = scaler.transform(feat)

    pred_num = model.predict(feat_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]

    st.success(f"🔊 Hasil Prediksi: **{pred_label.upper()}**")