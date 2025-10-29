# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

PROJECT_ROOT = Path(r"D:\projects\crop recommendation")
DATA_PATH = PROJECT_ROOT / "data" / "crop_recommendation.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.joblib"

st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("Crop Recommendation System")
st.write("Provide soil & weather parameters and get top crop suggestions.")

# Load dataset (or uploaded CSV)
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    st.error(f"No dataset found. Put CSV at: {DATA_PATH} or upload one.")
    st.stop()

# Infer features
from src.utils import infer_features_and_target
features, target = infer_features_and_target(df)
st.sidebar.write("Detected features:", features)
st.sidebar.write("Target:", target)

st.header("Enter feature values")
user_input = {}
col1, col2, col3 = st.columns(3)
for i, feat in enumerate(features):
    default = float(df[feat].median()) if feat in df.columns else 0.0
    with (col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3)):
        user_input[feat] = st.number_input(f"{feat}", value=default, format="%.3f")

k = st.slider("How many suggestions (top-k)?", min_value=1, max_value=5, value=3)

# Load model artifact
if not MODEL_PATH.exists():
    st.warning(f"Model not found at {MODEL_PATH}. Run: python src\\train.py to train and save the model.")
else:
    artifact = joblib.load(str(MODEL_PATH))
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    trained_features = artifact.get("features", features)

    if st.button("Predict"):
        # ensure feature order matches trained features
        try:
            X = np.array([[float(user_input[f]) for f in trained_features]], dtype=float)
        except KeyError as e:
            st.error(f"Input missing feature: {e}. Trained features: {trained_features}")
        else:
            probs = pipeline.predict_proba(X)[0]
            topk_idxs = np.argsort(-probs)[:k]
            labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
            st.subheader("Top crop suggestions")
            for idx in topk_idxs:
                st.write(f"**{labels[idx]}** — probability: {probs[idx]:.3f}")

st.markdown("---")
st.header("Dataset preview")
st.dataframe(df.head(200))
st.caption(f"Dataset used: {DATA_PATH.name} (you can upload a different CSV via the sidebar).")
