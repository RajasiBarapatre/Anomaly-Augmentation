"""
approach_6_kde.py
=================
KDE (Kernel Density Estimation) sampling approach.

Method:
  1. Handle categorical features separately (sample from empirical distribution)
  2. Transform skewed numeric features via log1p to reduce skew before fitting KDE
  3. Fit a multivariate Gaussian KDE on the (transformed) normal-class data
  4. Sample from the KDE, then inverse-transform (expm1) to original scale
  5. Post-process: clip, round integers, sample categoricals from normal-class freqs

Strengths:
  - Non-parametric: adapts to any distribution shape
  - Interpretable (bandwidth = smoothing parameter)
  - Fast for moderate dimensions
  - Handles skew via pre-transforms

Weaknesses:
  - Curse of dimensionality: bandwidth selection degrades in high-D
  - Cannot handle categorical features natively (separate empirical sampling)
  - May over-smooth or under-smooth depending on bandwidth
  - Independent handling of categoricals loses cat-numeric correlations
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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
def generate_kde(normal_df, col_types, target_n, seed=RANDOM_SEED):
    """
    KDE-based synthetic data generation with skew-reducing transforms.
    """
    rng = np.random.default_rng(seed)
    cat_cols = col_types["categorical"]
    num_cols = col_types["numeric"]

    # --- Step 1: Sample categoricals from normal-class empirical distribution ---
    cat_data = {}
    for col in cat_cols:
        val_counts = normal_df[col].value_counts(normalize=True)
        categories = val_counts.index.tolist()
        probs = val_counts.values
        cat_data[col] = rng.choice(categories, size=target_n, p=probs)

    # --- Step 2: Transform skewed numerics ---
    X_orig = normal_df[num_cols].values.astype(float)
    n_features = X_orig.shape[1]

    # Identify which columns to log-transform (non-negative + skew > 3)
    skews = np.array([pd.Series(X_orig[:, j]).skew() for j in range(n_features)])
    is_nonneg = np.array([(X_orig[:, j] >= 0).all() for j in range(n_features)])
    do_log = is_nonneg & (skews > 3)

    X_transformed = X_orig.copy()
    for j in range(n_features):
        if do_log[j]:
            X_transformed[:, j] = np.log1p(X_orig[:, j])

    # --- Step 3: Standardize (helps KDE bandwidth selection) ---
    means = X_transformed.mean(axis=0)
    stds = X_transformed.std(axis=0)
    stds[stds == 0] = 1.0
    X_std = (X_transformed - means) / stds

    # --- Step 4: Fit KDE ---
    # Use Scott's rule for bandwidth, but we need to handle high dimensionality
    # by fitting on PCA-reduced space if d > 15
    from sklearn.decomposition import PCA

    d = X_std.shape[1]
    use_pca = d > 15
    n_components = min(15, d)

    if use_pca:
        print(f"    Using PCA ({d} -> {n_components} dims) for KDE fitting...")
        pca = PCA(n_components=n_components, random_state=seed)
        X_pca = pca.fit_transform(X_std)
    else:
        X_pca = X_std
        pca = None

    print(f"    Fitting KDE on {X_pca.shape[1]}-dimensional data...")
    kde = gaussian_kde(X_pca.T, bw_method="scott")

    # --- Step 5: Sample ---
    print(f"    Sampling {target_n} points from KDE...")
    samples_pca = kde.resample(target_n, seed=seed).T  # (target_n, n_components)

    if use_pca:
        samples_std = pca.inverse_transform(samples_pca)
    else:
        samples_std = samples_pca

    # --- Step 6: Inverse transforms ---
    X_synth_transformed = samples_std * stds + means

    X_synth = X_synth_transformed.copy()
    for j in range(n_features):
        if do_log[j]:
            X_synth[:, j] = np.expm1(X_synth_transformed[:, j])

    # --- Step 7: Post-process ---
    synthetic = pd.DataFrame(X_synth, columns=num_cols)

    # Clip to observed ranges
    for col in num_cols:
        lo, hi = normal_df[col].min(), normal_df[col].max()
        synthetic[col] = synthetic[col].clip(lo, hi)

    # Round binary/integer features
    for col in col_types["binary"]:
        synthetic[col] = np.clip(np.rint(synthetic[col]), 0, 1).astype(int)

    for col in col_types["integer"]:
        lo, hi = normal_df[col].min(), normal_df[col].max()
        synthetic[col] = np.clip(np.rint(synthetic[col]), lo, hi).astype(int)

    # Add categorical columns
    for col in cat_cols:
        synthetic[col] = cat_data[col]

    # Reorder columns to match original
    synthetic = synthetic[col_types["all_feature"]]

    return add_id_column(synthetic)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 6: KDE — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_kde(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_6_{dataset_key}.csv")
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
