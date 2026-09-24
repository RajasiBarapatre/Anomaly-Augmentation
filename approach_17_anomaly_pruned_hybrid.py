"""
approach_17_anomaly_pruned_hybrid.py
====================================
# IMPROVEMENT OF APPROACH 9

Hypothesis & Motivation:
  Approach 9 filtered synthetic TVAE candidates by real-manifold density, but this occasionally
  pruned legitimate low-density normal tails.
  
  Approach 17 inverts and enhances the quality filter into an **Anomaly-Repulsion Gate**:
    1. Lowered oversampling factor to 1.3x (preventing dilution of high-confidence modes).
    2. Measures distance of each synthetic candidate to known training attacks (A_seen).
    3. Prunes candidates that fall too close to the attack distribution (preventing synthetic
       normals from overlapping with anomalous traffic and confusing downstream detectors).
    4. Blends the anomaly-repelled TVAE points (50%) with Cluster-Aware SMOTE (50%).
    5. Universal pipeline for both datasets.

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
    validate_synthetic, add_id_column, one_hot_encode,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics
from approach_4_tvae import generate_tvae
from approach_15_improved_cluster_smote import generate_improved_cluster_smote

warnings.filterwarnings("ignore")

OVER_SAMPLE_FACTOR = 1.3  # 1.3x candidates
BLEND_TVAE = 0.50
BLEND_SMOTE = 0.50


def filter_by_anomaly_repulsion(tvae_candidates, anomaly_df, categorical_cols, keep_n):
    """
    Score synthetic candidates by distance to known training anomalies (A_seen).
    Retains the top keep_n candidates furthest from the anomaly space.
    """
    cand_encoded, fit_cats = one_hot_encode(tvae_candidates, categorical_cols)
    anom_encoded, _ = one_hot_encode(anomaly_df, categorical_cols, fit_categories=fit_cats)

    all_cols = sorted(set(cand_encoded.columns) | set(anom_encoded.columns))
    for c in all_cols:
        if c not in cand_encoded.columns:
            cand_encoded[c] = 0
        if c not in anom_encoded.columns:
            anom_encoded[c] = 0

    X_cand = cand_encoded[all_cols].to_numpy(dtype=float)
    X_anom = anom_encoded[all_cols].to_numpy(dtype=float)

    # Standardize
    scaler = StandardScaler()
    X_anom_scaled = scaler.fit_transform(X_anom)
    X_cand_scaled = scaler.transform(X_cand)

    # Distance to nearest 3 anomaly samples
    k_eval = min(3, len(X_anom_scaled))
    nn = NearestNeighbors(n_neighbors=k_eval, algorithm="auto", n_jobs=-1)
    nn.fit(X_anom_scaled)
    distances, _ = nn.kneighbors(X_cand_scaled)
    mean_anom_dist = np.mean(distances, axis=1)

    # Keep candidates with LARGEST distance to anomalies (safest from attack overlap)
    top_indices = np.argsort(-mean_anom_dist)[:keep_n]
    return tvae_candidates.iloc[top_indices].copy().reset_index(drop=True)


def generate_anomaly_pruned_hybrid(
    train_df,
    label_col,
    col_types,
    target_n,
    over_factor=OVER_SAMPLE_FACTOR,
    seed=RANDOM_SEED,
):
    """
    Generate synthetic data via Anomaly-Repulsion TVAE + Cluster-SMOTE Hybrid.
    """
    feature_cols = col_types["all_feature"]
    normal_df = train_df.loc[train_df[label_col] == 0, feature_cols].copy().reset_index(drop=True)
    anomaly_df = train_df.loc[train_df[label_col] == 1, feature_cols].copy().reset_index(drop=True)

    n_tvae_target = int(round(target_n * BLEND_TVAE))
    n_smote_target = target_n - n_tvae_target
    n_tvae_raw = int(round(n_tvae_target * over_factor))

    print(f"    [1/3] Sampling {n_tvae_raw} TVAE candidate points ({over_factor}x factor)...")
    synth_tvae_raw = generate_tvae(normal_df, col_types, n_tvae_raw, seed=seed).drop(columns=["id"], errors="ignore")

    print(f"    [2/3] Applying Anomaly-Repulsion Gate (pruning {n_tvae_raw - n_tvae_target} candidates closest to attacks)...")
    synth_tvae_filtered = filter_by_anomaly_repulsion(
        synth_tvae_raw,
        anomaly_df,
        col_types["categorical"],
        keep_n=n_tvae_target,
    )

    print(f"    [3/3] Generating {n_smote_target} Cluster-Aware SMOTE anchor samples...")
    synth_smote = generate_improved_cluster_smote(normal_df, col_types, n_smote_target, seed=seed).drop(columns=["id"], errors="ignore")

    combined = pd.concat([synth_tvae_filtered[feature_cols], synth_smote[feature_cols]], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 17: Anomaly-Pruned TVAE+SMOTE Hybrid — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_anomaly_pruned_hybrid(train_df, cfg["label_col"], col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_17_{dataset_key}.csv")
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
