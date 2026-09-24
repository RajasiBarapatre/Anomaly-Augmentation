"""
approach_14_density_adaptive_jitter.py
======================================
Approach 14: Density-Adaptive Local Jitter

Hypothesis & Motivation:
  Standard Baseline Jitter (Approach 1) applies uniform Gaussian noise across all rows,
  which over-perturbs tightly clustered modes and under-explores sparse gaps.
  
  Approach 14 implements density-adaptive local jitter:
    1. Measures local sample density using k-nearest neighbor distance in standardized space.
    2. Modulates the noise scale dynamically:
         - In high-density clusters (small k-NN distance): applies minimal noise (0.05-0.15 std)
           to preserve tight modal structure and avoid blurring.
         - In sparse peripheral regions (larger k-NN distance): applies broader noise (0.3-0.5 std)
           to smoothly bridge sparse support gaps.
    3. Guarantees non-negative bounds, integer rounding, and categorical consistency.

  *CRITICAL REQUIREMENT*: Universal pipeline for both datasets.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset, local_train_eval_split, get_normal_data,
    validate_synthetic, add_id_column,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics

warnings.filterwarnings("ignore")

BASE_JITTER_SCALE = 0.35


def generate_density_adaptive_jitter(
    normal_df,
    col_types,
    target_n,
    base_scale=BASE_JITTER_SCALE,
    seed=RANDOM_SEED,
):
    """
    Generate synthetic data via density-adaptive local jitter.
    """
    rng = np.random.default_rng(seed)
    feature_cols = col_types["all_feature"]
    num_cols = col_types["numeric"]
    n_samples = len(normal_df)

    # 1. Measure local density using k-NN on standardized continuous/numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(normal_df[num_cols].to_numpy(dtype=float))
    
    k_density = min(10, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=k_density, algorithm="auto", n_jobs=-1)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    # Average distance to 10 nearest neighbors (higher distance = sparser region)
    avg_knn_dist = np.mean(distances, axis=1)
    median_knn_dist = np.median(avg_knn_dist) + 1e-10

    # Normalized density scale factor per sample in [0.2, 1.2]
    density_scale = np.clip(avg_knn_dist / median_knn_dist, 0.2, 1.2)

    # 2. Bootstrap rows
    sampled_indices = rng.choice(n_samples, size=target_n, replace=True)
    synthetic = normal_df.iloc[sampled_indices][feature_cols].copy().reset_index(drop=True)
    sample_density_factors = density_scale[sampled_indices]

    # 3. Apply adaptive per-sample jitter to numeric columns
    jitter_cols = col_types["integer"] + col_types["continuous"]
    for col in jitter_cols:
        std = normal_df[col].std(ddof=0)
        if not np.isfinite(std) or std == 0:
            continue

        # Per-sample noise magnitude
        noise_std = base_scale * std * sample_density_factors
        noise = rng.normal(0, noise_std, size=target_n)
        synthetic[col] = synthetic[col].astype(float) + noise

        # Clip to empirical bounds
        lo, hi = normal_df[col].min(), normal_df[col].max()
        synthetic[col] = synthetic[col].clip(lo, hi)

        # Round integers
        if col in col_types["integer"]:
            synthetic[col] = np.rint(synthetic[col]).astype(int)

    return add_id_column(synthetic)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 14: Density-Adaptive Local Jitter — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_density_adaptive_jitter(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_14_{dataset_key}.csv")
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
