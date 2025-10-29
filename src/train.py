# src/train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import load_dataset, infer_features_and_target

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# absolute project root (update if you move the project)
PROJECT_ROOT = Path(r"D:\projects\crop recommendation")
DATA_PATH = PROJECT_ROOT / "data" / "crop_recommendation.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_model.joblib"

# helper functions (importing src/utils is fine, but keep local import to avoid sys.path issues)
from src.utils import load_dataset, infer_features_and_target

def main():
    print("Project root:", PROJECT_ROOT)
    print("Loading dataset from:", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download and place the CSV there.")

    df = load_dataset(str(DATA_PATH))
    X_cols, y_col = infer_features_and_target(df)
    print("Using features:", X_cols)
    print("Target column:", y_col)

    X = df[X_cols].copy()
    y = df[y_col].copy().astype(str)

    # Minimal automatic preprocessing
    X = X.fillna(X.median())

    # encode target labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # train-test split (preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    # pipeline: scaler + RF
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    print("Training RandomForest...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # Save plain dict (pipeline + label encoder) to avoid pickle class import issues later
    artifact = {"pipeline": pipeline, "label_encoder": le, "features": X_cols}
    joblib.dump(artifact, str(MODEL_PATH))
    print("Saved model artifact to:", MODEL_PATH)

if __name__ == "__main__":
    main()
