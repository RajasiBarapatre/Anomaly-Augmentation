"""
approach_16_improved_ensemble_mixture.py
========================================
# IMPROVEMENT OF APPROACH 7

Hypothesis & Motivation:
  Approach 7 (Unified Mixture Ensemble) achieved the highest score on UNSW (0.44770) and 0.92896 on NSL-KDD.
  Approach 16 upgrades the mixture components:
    1. Upgrades the SMOTE component from naive global SMOTE to Approach 8/15's Cluster-Aware SMOTE,
       eliminating cross-cluster interpolation artifacts.
    2. Integrates 4 complementary generation mechanisms in a unified ratio:
         - 45% Cluster-Aware SMOTE (high-precision local cluster anchors)
         - 30% TVAE (smooth global continuous latent space)
         - 15% CTGAN (discrete multimodal conditional coverage)
         - 10% Density-Adaptive Jitter (micro-dispersion around empirical modes)
    3. Guarantees universal execution for both datasets.

  *CRITICAL REQUIREMENT*: Universal pipeline for both datasets.
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
from approach_15_improved_cluster_smote import generate_improved_cluster_smote
from approach_4_tvae import generate_tvae
from approach_3_ctgan import generate_ctgan
from approach_14_density_adaptive_jitter import generate_density_adaptive_jitter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 4-Way Unified Mixture Weights
# ---------------------------------------------------------------------------
WEIGHT_CLUSTER_SMOTE = 0.45
WEIGHT_TVAE = 0.30
WEIGHT_CTGAN = 0.15
WEIGHT_JITTER = 0.10


def generate_improved_ensemble_mixture(
    normal_df,
    col_types,
    target_n,
    w_smote=WEIGHT_CLUSTER_SMOTE,
    w_tvae=WEIGHT_TVAE,
    w_ctgan=WEIGHT_CTGAN,
    w_jitter=WEIGHT_JITTER,
    seed=RANDOM_SEED,
):
    """
    Generate synthetic data via upgraded 4-way unified mixture ensemble.
    """
    n_smote = int(round(target_n * w_smote))
    n_tvae = int(round(target_n * w_tvae))
    n_ctgan = int(round(target_n * w_ctgan))
    n_jitter = target_n - n_smote - n_tvae - n_ctgan

    print(f"    4-Way Upgraded Mixture Allocation (Total: {target_n}):")
    print(f"      - Cluster-SMOTE ({w_smote*100:.0f}%): {n_smote} samples")
    print(f"      - TVAE          ({w_tvae*100:.0f}%): {n_tvae} samples")
    print(f"      - CTGAN         ({w_ctgan*100:.0f}%): {n_ctgan} samples")
    print(f"      - Adaptive Jitter ({w_jitter*100:.0f}%): {n_jitter} samples")

    # 1. Cluster-Aware SMOTE
    print(f"    [1/4] Generating {n_smote} samples via Cluster-Aware SMOTE...")
    synth_smote = generate_improved_cluster_smote(normal_df, col_types, n_smote, seed=seed).drop(columns=["id"], errors="ignore")

    # 2. TVAE
    print(f"    [2/4] Generating {n_tvae} samples via TVAE...")
    synth_tvae = generate_tvae(normal_df, col_types, n_tvae, seed=seed).drop(columns=["id"], errors="ignore")

    # 3. CTGAN
    print(f"    [3/4] Generating {n_ctgan} samples via CTGAN...")
    synth_ctgan = generate_ctgan(normal_df, col_types, n_ctgan, seed=seed).drop(columns=["id"], errors="ignore")

    # 4. Adaptive Jitter
    print(f"    [4/4] Generating {n_jitter} samples via Density-Adaptive Jitter...")
    synth_jitter = generate_density_adaptive_jitter(normal_df, col_types, n_jitter, seed=seed).drop(columns=["id"], errors="ignore")

    feature_cols = col_types["all_feature"]
    combined = pd.concat([
        synth_smote[feature_cols],
        synth_tvae[feature_cols],
        synth_ctgan[feature_cols],
        synth_jitter[feature_cols],
    ], ignore_index=True)

    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 16: Upgraded 4-Way Mixture Ensemble — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_improved_ensemble_mixture(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_16_{dataset_key}.csv")
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
