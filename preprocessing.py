"""
preprocessing.py
================
Shared data loading and preprocessing for all approaches.

Key responsibilities:
  1. Load raw CSV
  2. Identify column types (categorical, binary, integer, continuous)
  3. Create a local train/eval split (stratified by label) for offline AUPRC scoring
  4. Provide one-hot encoding utilities matching the Kaggle evaluation pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiment_config import RANDOM_SEED, LOCAL_EVAL_SPLIT


# ---------------------------------------------------------------------------
# Column-type identification
# ---------------------------------------------------------------------------
def identify_column_types(df, categorical_cols, label_col):
    """
    Classify every feature column into one of:
      - categorical:  string / object dtype
      - binary:       numeric with only {0, 1}
      - integer:      numeric with all-integer values (but more than 2 unique)
      - continuous:   everything else (float-valued)

    Returns a dict with keys: categorical, binary, integer, continuous, all_feature.
    """
    feature_cols = [c for c in df.columns if c not in [label_col, "id"]]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    binary, integer, continuous = [], [], []
    for col in numeric_cols:
        vals = df[col].dropna().to_numpy()
        unique_vals = set(np.unique(vals).tolist())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            binary.append(col)
        elif np.all(np.isclose(vals, np.round(vals))):
            integer.append(col)
        else:
            continuous.append(col)

    return {
        "categorical": list(categorical_cols),
        "binary": binary,
        "integer": integer,
        "continuous": continuous,
        "numeric": numeric_cols,
        "all_feature": feature_cols,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(cfg):
    """
    Load a dataset from config, returning the full DataFrame and column info.
    """
    df = pd.read_csv(cfg["path"])
    col_types = identify_column_types(df, cfg["categorical_cols"], cfg["label_col"])
    return df, col_types


def get_normal_data(df, label_col):
    """Return only the normal-class rows (label == 0), dropping the label column."""
    feature_cols = [c for c in df.columns if c not in [label_col, "id"]]
    return df.loc[df[label_col] == 0, feature_cols].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Local train / eval split
# ---------------------------------------------------------------------------
def local_train_eval_split(df, label_col):
    """
    Stratified split of the full labeled training set.
    Returns:
      - train_df: 80% of data (use normal rows from this for generator training)
      - eval_df:  20% of data (use ALL rows for local AUPRC scoring)
    """
    train_df, eval_df = train_test_split(
        df,
        test_size=LOCAL_EVAL_SPLIT,
        stratify=df[label_col],
        random_state=RANDOM_SEED,
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# One-hot encoding (mirrors Kaggle evaluation)
# ---------------------------------------------------------------------------
def one_hot_encode(df, categorical_cols, fit_categories=None):
    """
    One-hot encode categorical columns, matching Kaggle evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Data to encode.
    categorical_cols : list[str]
        Columns to one-hot encode.
    fit_categories : dict or None
        If provided, maps col_name -> list of categories to use.
        Ensures train and eval have consistent columns.

    Returns
    -------
    encoded_df : pd.DataFrame
        DataFrame with categoricals replaced by one-hot columns.
    categories : dict
        col_name -> sorted list of categories (for re-use on other DataFrames).
    """
    result = df.drop(columns=categorical_cols, errors="ignore").copy()
    categories = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        if fit_categories and col in fit_categories:
            cats = fit_categories[col]
        else:
            cats = sorted(df[col].dropna().unique().tolist())
        categories[col] = cats

        for cat in cats:
            result[f"{col}_{cat}"] = (df[col] == cat).astype(int)

    return result, categories


# ---------------------------------------------------------------------------
# ID column helper & Validation helper
# ---------------------------------------------------------------------------
def add_id_column(synthetic_df):
    """
    Ensure 'id' column exists as the very first column (0, 1, ..., N-1).
    """
    df = synthetic_df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df.insert(0, "id", np.arange(len(df)))
    return df


def validate_synthetic(synthetic_df, col_types, target_n):
    """
    Sanity-check that generated data is valid for submission.
    Raises ValueError on problems; returns True if all checks pass.
    """
    errors = []
    if len(synthetic_df) != target_n:
        errors.append(f"Expected {target_n} rows, got {len(synthetic_df)}")

    if "id" not in synthetic_df.columns:
        errors.append("Missing required 'id' column as first column")
    elif synthetic_df.columns[0] != "id":
        errors.append("'id' column exists but is not the first column")

    expected_cols = col_types["all_feature"]
    feature_cols = [c for c in synthetic_df.columns if c != "id"]
    missing = set(expected_cols) - set(feature_cols)
    extra = set(feature_cols) - set(expected_cols)
    if missing:
        errors.append(f"Missing columns: {missing}")
    if extra:
        errors.append(f"Extra columns: {extra}")

    if synthetic_df.isna().any().any():
        nan_cols = synthetic_df.columns[synthetic_df.isna().any()].tolist()
        errors.append(f"NaN values in columns: {nan_cols}")

    numeric_vals = synthetic_df[col_types["numeric"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_vals).all():
        errors.append("Infinite values detected in numeric columns")

    if errors:
        raise ValueError("Validation failed:\n  " + "\n  ".join(errors))
    return True


if __name__ == "__main__":
    # Quick self-test
    from experiment_config import DATASETS
    for key, cfg in DATASETS.items():
        df, col_types = load_dataset(cfg)
        print(f"\n=== {cfg['name']} ===")
        print(f"  Shape: {df.shape}")
        print(f"  Categorical: {col_types['categorical']}")
        print(f"  Binary:      {col_types['binary']}")
        print(f"  Integer:     {col_types['integer']}")
        print(f"  Continuous:  {col_types['continuous']}")
        print(f"  Normal rows: {(df[cfg['label_col']] == 0).sum()}")

        train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
        print(f"  Train split: {len(train_df)} | Eval split: {len(eval_df)}")
        print(f"  Eval anomaly rate: {eval_df[cfg['label_col']].mean():.2%}")
