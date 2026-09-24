"""
approach_11_per_detector_specialized_blend.py
=============================================
Approach 11: Per-Detector Specialized Dual Blend

Hypothesis & Motivation:
  ECOD and Isolation Forest operate on fundamentally different mathematical mechanics:
    1. ECOD evaluates marginal tail probabilities independently across each feature
       using empirical cumulative distribution functions (ECDFs).
    2. Isolation Forest constructs recursive axis-aligned random partitions, isolating
       points quickly if they fall in sparse peripheral regions.
  
  Approach 11 constructs two complementary synthetic sub-populations:
    - Sub-Population A (ECOD-Optimized): Exact empirical quantile-matching generator
      that samples from empirical marginal inverse-CDFs, preserving extreme tail
      boundaries and percentiles.
    - Sub-Population B (IForest-Optimized): Cluster-aware boundary densifier that
      thickens the outer normal envelope, making anomalous attacks faster to isolate.
    - Final Set: 50% Sub-Pop A + 50% Sub-Pop B.

  *CRITICAL REQUIREMENT*: The exact same pipeline is applied to both datasets.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset, local_train_eval_split, get_normal_data,
    validate_synthetic, add_id_column,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics
from approach_8_cluster_smote import generate_cluster_smote

warnings.filterwarnings("ignore")


def generate_ecod_tail_matched_subset(normal_df, col_types, target_n, seed=RANDOM_SEED):
    """
    Generate synthetic data strictly matching empirical marginal tail quantiles.
    """
    rng = np.random.default_rng(seed)
    feature_cols = col_types["all_feature"]
    cat_cols = col_types["categorical"]
    num_cols = col_types["numeric"]

    # Start by bootstrapping categorical rows to preserve categorical co-occurrence
    cat_df = normal_df[cat_cols].sample(n=target_n, replace=True, random_state=seed).reset_index(drop=True)
    synth_dict = {c: cat_df[c].values for c in cat_cols}

    # For each numeric feature, sample from continuous empirical quantile distribution
    for col in num_cols:
        vals = np.sort(normal_df[col].dropna().to_numpy(dtype=float))
        n_vals = len(vals)
        # Uniform percentiles
        u = rng.uniform(0.001, 0.999, size=target_n)
        # Linear interpolation across empirical quantiles
        synth_vals = np.interp(u, np.linspace(0.001, 0.999, n_vals), vals)
        
        # Add micro-jitter to prevent duplicate discrete spikes
        std_local = np.std(vals)
        if std_local > 0:
            synth_vals += rng.normal(0, 0.01 * std_local, size=target_n)

        # Clip and round as needed
        lo, hi = vals[0], vals[-1]
        synth_vals = np.clip(synth_vals, lo, hi)
        if col in col_types["integer"] or col in col_types["binary"]:
            synth_vals = np.rint(synth_vals).astype(int)

        synth_dict[col] = synth_vals

    return pd.DataFrame(synth_dict)[feature_cols]


def generate_per_detector_blend(normal_df, col_types, target_n, seed=RANDOM_SEED):
    """
    Generate 50% ECOD-specialized tail data + 50% IForest-specialized boundary data.
    """
    n_ecod = target_n // 2
    n_iforest = target_n - n_ecod

    print(f"    [1/2] Generating {n_ecod} ECOD-specialized marginal quantile samples...")
    synth_ecod = generate_ecod_tail_matched_subset(normal_df, col_types, n_ecod, seed=seed)

    print(f"    [2/2] Generating {n_iforest} IForest-specialized cluster boundary samples...")
    synth_iforest = generate_cluster_smote(normal_df, col_types, n_iforest, seed=seed).drop(columns=["id"], errors="ignore")

    feature_cols = col_types["all_feature"]
    combined = pd.concat([synth_ecod[feature_cols], synth_iforest[feature_cols]], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 11: Per-Detector Specialized Dual Blend — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_per_detector_blend(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_11_{dataset_key}.csv")
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
