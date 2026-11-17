"""
Data loading and preprocessing
  1. Load raw CSV
  2. Identify column types (categorical, binary, integer, continuous)
  3. Create a local train/eval split (stratified by label) for offline AUPRC scoring
  4. Provide one-hot encoding utilities
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from revised_experiment_config import (
    RANDOM_SEED,
    LOCAL_EVAL_SPLIT,
)

#   Column-type identification
def identify_column_types(df, categorical_cols, label_col):
    feature_cols = [c for c in df.columns if c not in [label_col, "id"]]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    binary, integer, continuous = [], [], []
    for col in numeric_cols:
        vals = df[col].dropna().to_numpy()
        unique_vals = set(np.unique(vals).tolist())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            binary.append(col)
        elif np.all(np.abs(vals - np.round(vals)) < 1e-9):
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

#   Data loading
def load_dataset(cfg):
    path = cfg["path"]

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset does not exist: {path}"
        )

    df = pd.read_csv(path)

    print(f"  Loaded dataset: {os.path.abspath(path)}")
    print(f"  Dataset shape: {df.shape}")

    required_cols = (
        cfg["categorical_cols"] + [cfg["label_col"]]
    )
    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}"
        )

    col_types = identify_column_types(
        df,
        cfg["categorical_cols"],
        cfg["label_col"],
    )

    return df, col_types

#   Return only the normal-class rows, dropping the label column  
def get_normal_data(df, label_col):
    feature_cols = [c for c in df.columns if c not in [label_col, "id"]]
    return df.loc[df[label_col] == 0, feature_cols].copy().reset_index(drop=True)

#   Local train/eval split
def local_train_eval_split(df, label_col, random_state=RANDOM_SEED):
    train_df, eval_df = train_test_split(
        df,
        test_size=LOCAL_EVAL_SPLIT,
        stratify=df[label_col],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


#   One-hot encoding categorical columns
def one_hot_encode(df, categorical_cols, fit_categories=None):
    result = df.drop(columns=categorical_cols, errors="ignore").copy()
    categories = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        if fit_categories and col in fit_categories:
            cats = fit_categories[col]
        else:
            cats = sorted(df[col].dropna().unique().tolist(), key=lambda value: str(value))
        categories[col] = cats

        for cat in cats:
            result[f"{col}_{cat}"] = (df[col] == cat).astype(int)

    return result, categories


def add_id_column(synthetic_df):
    df = synthetic_df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df.insert(0, "id", np.arange(len(df)))
    df.attrs = synthetic_df.attrs.copy()
    return df

#   Sanity check for synthetic data
def validate_synthetic(synthetic_df, col_types, target_n):
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

    for col in col_types["binary"]:
        if not synthetic_df[col].isin([0, 1]).all():
            errors.append(f"Non-binary values in {col}")
    for col in col_types["integer"]:
        values = synthetic_df[col].to_numpy(float)
        if not np.all(np.abs(values - np.rint(values)) < 1e-9):
            errors.append(f"Fractional values in integer feature {col}")
    if errors:
        raise ValueError("Validation failed:\n  " + "\n  ".join(errors))
    return True


if __name__ == "__main__":
    from revised_experiment_config import DATASETS
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
