"""
approach_1_baseline_jitter.py
=============================
Baseline approach: Bootstrap + Gaussian jitter (adapted from the starter kit).

Method:
  1. Sample N rows from normal-class training data (with replacement)
  2. Add small Gaussian noise to continuous and integer features
  3. Clip values to observed min/max range
  4. Round integer features back to integers
  5. Keep categorical and binary features unchanged

This is the floor performance we aim to beat.
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset, local_train_eval_split, get_normal_data,
    identify_column_types, validate_synthetic, add_id_column,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
JITTER_SCALE = 0.5  # noise std = JITTER_SCALE * feature std


def generate_baseline_jitter(normal_df, col_types, target_n, seed=RANDOM_SEED):
    """
    Bootstrap + Jitter generator.
    """
    rng = np.random.default_rng(seed)

    # Bootstrap sample
    synthetic = normal_df.sample(n=target_n, replace=True,
                                  random_state=seed).reset_index(drop=True)

    # Add jitter to integer and continuous features
    jitter_cols = col_types["integer"] + col_types["continuous"]
    for col in jitter_cols:
        std = normal_df[col].std(ddof=0)
        if not np.isfinite(std) or std == 0:
            continue

        noise = rng.normal(0, JITTER_SCALE * std, target_n)
        synthetic[col] = synthetic[col].astype(float) + noise

        # Clip to observed range
        lo, hi = normal_df[col].min(), normal_df[col].max()
        synthetic[col] = synthetic[col].clip(lo, hi)

        # Round integers
        if col in col_types["integer"]:
            synthetic[col] = np.rint(synthetic[col]).astype(int)

    return add_id_column(synthetic)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 1: Baseline Jitter — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_baseline_jitter(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    # Validate
    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"approach_1_{dataset_key}.csv")
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
