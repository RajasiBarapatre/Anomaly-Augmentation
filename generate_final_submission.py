"""
generate_final_submission.py
============================
Official Submission Generator for Kaggle.

This script trains the selected approach on 100% of the provided training data
(using all labeled normal and anomaly samples) to maximize generalization,
and outputs exactly:
  - 40,000 synthetic normal samples for NSL-KDD
  - 70,000 synthetic normal samples for UNSW-NB15

Default Approach:
  - Approach 17 (Anomaly-Pruned TVAE + Cluster-SMOTE Hybrid) [WINNING APPROACH]

Usage:
  python3 generate_final_submission.py                 # Generate Approach 17 submissions
  python3 generate_final_submission.py --approach 16   # Generate Approach 16 submissions
  python3 generate_final_submission.py --approach 15   # Generate Approach 15 submissions
  python3 generate_final_submission.py --dataset nslkdd # Generate for NSL-KDD only
"""

import sys
import os
import time
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED
from preprocessing import load_dataset, validate_synthetic, add_id_column

SUBMISSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions")


def generate_submission_for_dataset(dataset_key, approach_id=17):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*70}")
    print(f"GENERATING SUBMISSION: {cfg['name']} (Target: {cfg['target_n']} samples)")
    print(f"Approach: {approach_id}")
    print(f"{'='*70}")

    # Load 100% of the training dataset
    df, col_types = load_dataset(cfg)
    feature_cols = col_types["all_feature"]
    normal_df = df.loc[df[cfg["label_col"]] == 0, feature_cols].copy().reset_index(drop=True)
    anomaly_df = df.loc[df[cfg["label_col"]] == 1, feature_cols].copy().reset_index(drop=True)

    print(f"  Total Training Samples: {len(df)}")
    print(f"  Normal Training Samples: {len(normal_df)}")
    print(f"  Anomaly Training Samples: {len(anomaly_df)}")
    print(f"  Target Synthetic Output: {cfg['target_n']}")

    t0 = time.time()

    # Dispatch to approach generator
    str_app = str(approach_id).lower().replace("_", "").replace("-", "")
    if str_app == "19":
        from approach_19_refined_a18 import generate_manifold_balanced_hybrid
        synthetic = generate_manifold_balanced_hybrid(
            train_df=df,
            label_col=cfg["label_col"],
            col_types=col_types,
            target_n=cfg["target_n"],
            seed=RANDOM_SEED,
        )
    elif str_app == "18":
        from approach_18_manifold_balanced_hybrid import generate_manifold_balanced_hybrid
        synthetic = generate_manifold_balanced_hybrid(
            train_df=df,
            label_col=cfg["label_col"],
            col_types=col_types,
            target_n=cfg["target_n"],
            seed=RANDOM_SEED,
        )
    elif str_app == "17v2":
        from approach_17_anomaly_pruned_hybrid_v2 import generate_anomaly_pruned_hybrid
        synthetic = generate_anomaly_pruned_hybrid(
            train_df=df,
            label_col=cfg["label_col"],
            col_types=col_types,
            target_n=cfg["target_n"],
            seed=RANDOM_SEED,
        )
    elif str_app == "17":
        from approach_17_anomaly_pruned_hybrid import generate_anomaly_pruned_hybrid
        synthetic = generate_anomaly_pruned_hybrid(
            train_df=df,
            label_col=cfg["label_col"],
            col_types=col_types,
            target_n=cfg["target_n"],
            seed=RANDOM_SEED,
        )
    elif str_app == "16":
        from approach_16_improved_ensemble_mixture import generate_improved_ensemble_mixture
        synthetic = generate_improved_ensemble_mixture(
            normal_df=normal_df,
            col_types=col_types,
            target_n=cfg["target_n"],
            seed=RANDOM_SEED,
        )
    elif str_app == "15":
        from approach_15_improved_cluster_smote import generate_improved_cluster_smote
        synthetic = generate_improved_cluster_smote(
            normal_df=normal_df,
            col_types=col_types,
            target_n=cfg["target_n"],
            seed=RANDOM_SEED,
        )
    elif str_app == "7":
        from approach_7_ensemble_mixture import generate_ensemble_mixture
        synthetic = generate_ensemble_mixture(
            normal_df=normal_df,
            col_types=col_types,
            target_n=cfg["target_n"],
            seed=RANDOM_SEED,
        )
    elif str_app == "8":
        from approach_8_cluster_smote import generate_cluster_smote
        synthetic = generate_cluster_smote(
            normal_df=normal_df,
            col_types=col_types,
            target_n=cfg["target_n"],
            seed=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unsupported approach ID for submission generation: {approach_id}")

    gen_time = time.time() - t0
    print(f"  Generation Time: {gen_time:.1f}s")

    # Validate schema, target row count, id column, and numeric values
    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Schema & Data Integrity Verification Passed")

    # Ensure submissions directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Save final submission file
    out_filename = f"{dataset_key}_submission_approach{approach_id}.csv"
    out_path = os.path.join(SUBMISSION_DIR, out_filename)
    synthetic.to_csv(out_path, index=False)
    print(f"  ✓ Saved submission file to: {out_path}")
    print(f"  ✓ File shape: {synthetic.shape} (Rows: {len(synthetic)}, Columns: {len(synthetic.columns)})")
    print(f"  ✓ First column: '{synthetic.columns[0]}' (Values: {synthetic['id'].iloc[0]} ... {synthetic['id'].iloc[-1]})")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate final Kaggle submission files")
    parser.add_argument("--approach", type=str, default="18",
                        help="Approach ID to use for generation (default: 18, or 17v2, 17, 16, 15, etc.)")
    parser.add_argument("--dataset", type=str, choices=["nslkdd", "unsw"], default=None,
                        help="Generate for specific dataset only (default: both)")
    args = parser.parse_args()

    datasets_to_generate = [args.dataset] if args.dataset else list(DATASETS.keys())

    print("=" * 70)
    print("CS5344 FINAL KAGGLE SUBMISSION GENERATOR")
    print("=" * 70)
    print(f"Selected Approach: Approach {args.approach}")
    print(f"Datasets: {datasets_to_generate}")
    print("=" * 70)

    generated_files = []
    for ds_key in datasets_to_generate:
        path = generate_submission_for_dataset(ds_key, approach_id=args.approach)
        generated_files.append(path)

    print("\n" + "=" * 70)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 70)
    for p in generated_files:
        print(f"  -> {p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
