"""
approach_7_ensemble_mixture.py
================================
Approach 7: Multi-Generator Unified Mixture Ensemble

Hypothesis & Motivation:
  Kaggle leaderboard feedback established that:
    - SMOTE (Approach 5) achieves the highest score on NSL-KDD (0.93039) due to sharp
      boundary preservation and lack of out-of-distribution hallucinations.
    - TVAE (Approach 4) achieves the highest score on UNSW-NB15 (0.44849) due to its
      continuous latent manifold modeling across complex, heavy-tailed traffic.
    - CTGAN (Approach 3) provides strong multimodal categorical conditioning.

  By forming a unified mixture ensemble:
    - 50% SMOTE (sharp local manifold interpolation)
    - 35% TVAE  (smooth global latent space coverage)
    - 15% CTGAN (multimodal categorical mode coverage)
  the downstream detectors (ECOD & Isolation Forest) receive both high-precision local
  density anchors AND smooth continuous generative support.

  *CRITICAL REQUIREMENT*: The exact same pipeline and mixture ratio is applied to
  both NSL-KDD and UNSW-NB15 without dataset-specific branching.
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

# Import generator functions from individual approaches
from approach_3_ctgan import generate_ctgan
from approach_4_tvae import generate_tvae
from approach_5_smote import generate_smote

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Unified Mixture Weights (Identical across all datasets)
# ---------------------------------------------------------------------------
WEIGHT_SMOTE = 0.50
WEIGHT_TVAE  = 0.35
WEIGHT_CTGAN = 0.15


def generate_ensemble_mixture(
    normal_df,
    col_types,
    target_n,
    w_smote=WEIGHT_SMOTE,
    w_tvae=WEIGHT_TVAE,
    w_ctgan=WEIGHT_CTGAN,
    seed=RANDOM_SEED,
):
    """
    Generate synthetic data using a unified multi-generator mixture ensemble.
    """
    # Calculate exact counts
    n_smote = int(round(target_n * w_smote))
    n_tvae = int(round(target_n * w_tvae))
    n_ctgan = target_n - n_smote - n_tvae  # ensure exact sum to target_n

    print(f"    Mixture allocation (Total: {target_n}):")
    print(f"      - SMOTE ({w_smote*100:.0f}%): {n_smote} samples")
    print(f"      - TVAE  ({w_tvae*100:.0f}%): {n_tvae} samples")
    print(f"      - CTGAN ({w_ctgan*100:.0f}%): {n_ctgan} samples")

    # 1. Generate SMOTE component
    print(f"    [1/3] Generating {n_smote} samples via SMOTE...")
    synth_smote = generate_smote(normal_df, col_types, n_smote, seed=seed)
    if "id" in synth_smote.columns:
        synth_smote = synth_smote.drop(columns=["id"])

    # 2. Generate TVAE component
    print(f"    [2/3] Generating {n_tvae} samples via TVAE...")
    synth_tvae = generate_tvae(normal_df, col_types, n_tvae, seed=seed)
    if "id" in synth_tvae.columns:
        synth_tvae = synth_tvae.drop(columns=["id"])

    # 3. Generate CTGAN component
    print(f"    [3/3] Generating {n_ctgan} samples via CTGAN...")
    synth_ctgan = generate_ctgan(normal_df, col_types, n_ctgan, seed=seed)
    if "id" in synth_ctgan.columns:
        synth_ctgan = synth_ctgan.drop(columns=["id"])

    # Ensure column order matches exactly
    feature_cols = col_types["all_feature"]
    synth_smote = synth_smote[feature_cols]
    synth_tvae = synth_tvae[feature_cols]
    synth_ctgan = synth_ctgan[feature_cols]

    # Combine all parts
    combined = pd.concat([synth_smote, synth_tvae, synth_ctgan], ignore_index=True)

    # Shuffle to interleave the mixture components
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Prepend id column as first column
    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 7: Multi-Generator Mixture Ensemble — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_ensemble_mixture(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_7_{dataset_key}.csv")
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
