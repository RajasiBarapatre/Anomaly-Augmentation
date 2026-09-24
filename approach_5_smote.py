"""
approach_5_smote.py
===================
SMOTE-based approach adapted for normal-class augmentation.

Method:
  1. Encode categorical features (label-encode for SMOTE's k-NN to work)
  2. Treat normal-class data as the "minority" to oversample
  3. Use SMOTE's k-NN interpolation to create synthetic samples in feature space
  4. Decode categorical features back (nearest-neighbor rounding to valid categories)
  5. Post-process: clip to valid ranges, round integers

Adaptation:
  Since SMOTE is designed for class imbalance (oversample minority class),
  we adapt it here by treating all normal data as inputs and using
  SMOTE's interpolation engine to create new points between existing
  normal samples. We use a two-class trick: create a tiny dummy "majority"
  class so SMOTE oversamples normal to the desired count.

Strengths:
  - Generates samples IN the convex hull of existing data (local interpolation)
  - Simple, well-understood, no neural network training
  - Fast

Weaknesses:
  - Only interpolates, never extrapolates → can't fill gaps in the distribution
  - Doesn't model the full joint distribution, just local neighborhoods
  - Categorical handling is approximate (rounds to nearest valid category)
  - May struggle with very high-dimensional data
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset, local_train_eval_split, get_normal_data,
    validate_synthetic, add_id_column,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def generate_smote(normal_df, col_types, target_n, seed=RANDOM_SEED):
    """
    Generate synthetic normal data using SMOTE-style k-NN interpolation.
    We implement a custom SMOTE variant that handles categoricals properly.
    """
    rng = np.random.default_rng(seed)

    # Encode categoricals
    cat_cols = col_types["categorical"]
    label_encoders = {}
    df_encoded = normal_df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Convert to numpy for KNN
    feature_cols = col_types["all_feature"]
    X = df_encoded[feature_cols].values.astype(float)
    n_samples, n_features = X.shape

    # Standardize for distance computation
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds

    # Build KNN index
    from sklearn.neighbors import NearestNeighbors
    k = min(5, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn.fit(X_std)
    distances, indices = nn.kneighbors(X_std)

    # SMOTE interpolation
    print(f"    Generating {target_n} samples via SMOTE interpolation (k={k})...")
    synthetic_rows = np.empty((target_n, n_features))

    for i in range(target_n):
        # Pick a random normal sample
        idx = rng.integers(0, n_samples)
        # Pick a random neighbor (skip self at index 0)
        neighbor_idx = indices[idx, rng.integers(1, k + 1)]
        # Interpolation weight
        lam = rng.uniform(0, 1)
        synthetic_rows[i] = X[idx] + lam * (X[neighbor_idx] - X[idx])

    synthetic = pd.DataFrame(synthetic_rows, columns=feature_cols)

    # Post-process: round and decode
    for col in col_types["binary"]:
        synthetic[col] = np.clip(np.rint(synthetic[col]), 0, 1).astype(int)

    for col in col_types["integer"]:
        lo, hi = normal_df[col].min(), normal_df[col].max()
        synthetic[col] = np.clip(np.rint(synthetic[col]), lo, hi).astype(int)

    for col in col_types["continuous"]:
        lo, hi = normal_df[col].min(), normal_df[col].max()
        synthetic[col] = synthetic[col].clip(lo, hi)

    # Decode categoricals: round to nearest valid integer, then inverse-transform
    for col in cat_cols:
        le = label_encoders[col]
        n_classes = len(le.classes_)
        synthetic[col] = np.clip(np.rint(synthetic[col]), 0, n_classes - 1).astype(int)
        synthetic[col] = le.inverse_transform(synthetic[col])

    return add_id_column(synthetic)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 5: SMOTE — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_smote(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_5_{dataset_key}.csv")
    synthetic.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    results = {"generation_time": gen_time}
    if do_eval:
        print("\n  Evaluating...")
        eval_results = evaluate_synthetic(
            synthetic, eval_df, cfg["categorical_cols"], cfg["label_col"]
        )
        quality = compute_quality_metrics(synthetic, normal_train, col_types)
        results.update(eval_results)
        results["mean_ks_distance"] = quality["mean_ks_distance"]
        results["correlation_rmse"] = quality["correlation_rmse"]

    return synthetic, results


if __name__ == "__main__":
    for key in DATASETS:
        run(key)
