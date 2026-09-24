"""
approach_9_filtered_hybrid.py
==============================
Approach 9: Density-Filtered Generative Hybrid (TVAE + SMOTE + Density Quality Gate)

Hypothesis & Motivation:
  TVAE learns excellent continuous latent representations, achieving #1 on UNSW-NB15 (0.44849).
  However, unconstrained variational sampling can occasionally produce extreme tail points
  or low-density artifacts that distort downstream ECOD tail estimation and IForest splits.
  
  Approach 9 addresses this by:
    1. Over-generating candidate synthetic samples from TVAE (1.5x of target allocation).
    2. Passing candidates through an Empirical Density Filter trained on real normal data.
    3. Retaining the top highest-confidence TVAE samples (pruning out-of-distribution outliers).
    4. Blending the filtered TVAE samples with SMOTE manifold interpolations (50/50 split).
    5. Adding the mandatory 'id' column as the first column.

  *CRITICAL REQUIREMENT*: The exact same pipeline and filter criteria are applied to
  both NSL-KDD and UNSW-NB15 without dataset-specific branching.
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
    validate_synthetic, add_id_column, one_hot_encode,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics
from approach_4_tvae import generate_tvae
from approach_5_smote import generate_smote

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration (Identical across all datasets)
# ---------------------------------------------------------------------------
OVER_SAMPLE_FACTOR = 1.5   # Generate 1.5x TVAE candidates for filtering
BLEND_RATIO_TVAE = 0.50    # 50% Filtered TVAE
BLEND_RATIO_SMOTE = 0.50   # 50% SMOTE


def filter_synthetic_candidates(tvae_candidates, real_normal_df, categorical_cols, keep_n):
    """
    Score synthetic candidates against the empirical density of real normal data.
    Retains the top keep_n points that are closest to the real normal manifold.
    """
    # One-hot encode both datasets with identical feature representation
    real_encoded, categories = one_hot_encode(real_normal_df, categorical_cols)
    cand_encoded, _ = one_hot_encode(tvae_candidates, categorical_cols, fit_categories=categories)

    # Align columns
    all_cols = sorted(set(real_encoded.columns) | set(cand_encoded.columns))
    for c in all_cols:
        if c not in real_encoded.columns:
            real_encoded[c] = 0
        if c not in cand_encoded.columns:
            cand_encoded[c] = 0

    X_real = real_encoded[all_cols].to_numpy(dtype=float)
    X_cand = cand_encoded[all_cols].to_numpy(dtype=float)

    # Standardize based on real data
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_cand_scaled = scaler.transform(X_cand)

    # Measure distance to k-nearest neighbors in real normal data (density metric)
    k_eval = min(10, len(X_real_scaled) - 1)
    nn = NearestNeighbors(n_neighbors=k_eval, algorithm="auto", n_jobs=-1)
    nn.fit(X_real_scaled)
    distances, _ = nn.kneighbors(X_cand_scaled)
    
    # Average distance to 10 nearest real normal neighbors (lower = higher density / more realistic)
    density_scores = np.mean(distances, axis=1)

    # Keep the top keep_n samples with lowest distance (highest normal likelihood)
    top_indices = np.argsort(density_scores)[:keep_n]
    return tvae_candidates.iloc[top_indices].copy().reset_index(drop=True)


def generate_filtered_hybrid(
    normal_df,
    col_types,
    target_n,
    over_factor=OVER_SAMPLE_FACTOR,
    blend_tvae=BLEND_RATIO_TVAE,
    blend_smote=BLEND_RATIO_SMOTE,
    seed=RANDOM_SEED,
):
    """
    Generate synthetic data via Density-Filtered TVAE + SMOTE Hybrid.
    """
    n_tvae_target = int(round(target_n * blend_tvae))
    n_smote_target = target_n - n_tvae_target
    n_tvae_raw = int(round(n_tvae_target * over_factor))

    print(f"    Filtered Hybrid Allocation (Total: {target_n}):")
    print(f"      - SMOTE component (50%): {n_smote_target} samples")
    print(f"      - TVAE target (50%): {n_tvae_target} samples (over-sampling {n_tvae_raw} candidates for filtering)")

    # 1. Generate candidate TVAE points
    print(f"    [1/3] Training & sampling {n_tvae_raw} TVAE candidate points...")
    synth_tvae_raw = generate_tvae(normal_df, col_types, n_tvae_raw, seed=seed)
    if "id" in synth_tvae_raw.columns:
        synth_tvae_raw = synth_tvae_raw.drop(columns=["id"])

    # 2. Quality Gate: Density filtering against real normal manifold
    print(f"    [2/3] Applying density quality filter (filtering out {n_tvae_raw - n_tvae_target} low-density outliers)...")
    synth_tvae_filtered = filter_synthetic_candidates(
        synth_tvae_raw,
        normal_df,
        col_types["categorical"],
        keep_n=n_tvae_target,
    )

    # 3. Generate SMOTE component
    print(f"    [3/3] Generating {n_smote_target} SMOTE manifold samples...")
    synth_smote = generate_smote(normal_df, col_types, n_smote_target, seed=seed)
    if "id" in synth_smote.columns:
        synth_smote = synth_smote.drop(columns=["id"])

    # Ensure column order matches exactly
    feature_cols = col_types["all_feature"]
    synth_tvae_filtered = synth_tvae_filtered[feature_cols]
    synth_smote = synth_smote[feature_cols]

    # Combine both components
    combined = pd.concat([synth_tvae_filtered, synth_smote], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Prepend id column as first column
    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 9: Density-Filtered TVAE+SMOTE Hybrid — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_filtered_hybrid(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_9_{dataset_key}.csv")
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
