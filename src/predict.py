# src/predict.py
from pathlib import Path
import joblib
import numpy as np

PROJECT_ROOT = Path(r"D:\projects\crop recommendation")
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.joblib"

def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python src\\train.py")
    artifact = joblib.load(str(MODEL_PATH))
    return artifact

def predict_topk_from_dict(input_dict: dict, top_k: int = 3):
    """
    input_dict: {feature_name: float_value} - must match trained features order
    Returns list of (label, probability) tuples (top_k).
    """
    artifact = load_model_artifact()
    pipeline = artifact["pipeline"]
    le = artifact["label_encoder"]
    features = artifact.get("features")
    if features is None:
        raise RuntimeError("Model artifact missing feature names.")

    # make sure we map input_dict to same feature order
    X = np.array([[float(input_dict[f]) for f in features]], dtype=float)
    probs = pipeline.predict_proba(X)[0]
    topk_idxs = np.argsort(-probs)[:top_k]
    labels = le.inverse_transform(np.arange(len(le.classes_)))
    results = [(labels[i], float(probs[i])) for i in topk_idxs]
    return results

# quick test helper (not executed unless explicitly run)
if __name__ == "__main__":
    sample = { }  # fill with proper keys if testing
    try:
        print(predict_topk_from_dict(sample, top_k=3))
    except Exception as e:
        print("Test failed (expected until you train the model):", e)
