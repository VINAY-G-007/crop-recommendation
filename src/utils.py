# src/utils.py
import pandas as pd

DEFAULT_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"

def load_dataset(path: str):
    """Load dataset from CSV and return dataframe."""
    df = pd.read_csv(path)
    return df

def infer_features_and_target(df):
    """
    Return (feature_columns, target_column).
    Prefer DEFAULT_FEATURES + TARGET_COL if present; otherwise infer.
    """
    cols = df.columns.tolist()
    if TARGET_COL in cols and all(f in cols for f in DEFAULT_FEATURES):
        return DEFAULT_FEATURES, TARGET_COL

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric = [c for c in cols if c not in numeric_cols]

    if non_numeric:
        # assume last non-numeric is label
        return numeric_cols, non_numeric[-1]

    # fallback: all numeric except last column treated as features
    if len(numeric_cols) >= 2:
        return numeric_cols[:-1], numeric_cols[-1]

    # ultimate fallback: first 7 columns as features, last as target
    if len(cols) >= 8:
        return cols[:7], cols[-1]

    raise ValueError("Unable to infer features & target automatically. Please check the CSV columns.")
