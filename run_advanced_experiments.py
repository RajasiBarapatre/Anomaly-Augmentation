"""
run_advanced_experiments.py
===========================
Master Orchestrator for Advanced & Improved Approaches (10–17):
  - 10: Detector-in-the-Loop Selection
  - 11: Per-Detector Specialized Dual Blend (Tail quantiles + Boundary SMOTE)
  - 12: Boundary-Adversarial SMOTE (Densifying near A_seen)
  - 13: Copula-Guided SMOTE (PIT marginals + k-NN interpolation)
  - 14: Density-Adaptive Local Jitter
  - 15: Improved Cluster-Aware SMOTE (# IMPROVEMENT OF APPROACH 8)
  - 16: Upgraded 4-Way Mixture Ensemble (# IMPROVEMENT OF APPROACH 7)
  - 17: Anomaly-Pruned TVAE+SMOTE Hybrid (# IMPROVEMENT OF APPROACH 9)

Usage:
  python3 run_advanced_experiments.py              # Run all 10-17 on both datasets
  python3 run_advanced_experiments.py --approach 15 # Run only Approach 15
  python3 run_advanced_experiments.py --fast       # Run fast approaches (11, 12, 13, 14, 15)
  python3 run_advanced_experiments.py --dataset nslkdd # Run on NSL-KDD only
"""

import sys
import os
import time
import argparse
import traceback
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, OUTPUT_DIR

ADVANCED_APPROACH_NAMES = {
    10: "Detector-in-the-Loop Candidate Selection",
    11: "Per-Detector Specialized Dual Blend",
    12: "Boundary-Adversarial SMOTE",
    13: "Copula-Guided SMOTE",
    14: "Density-Adaptive Local Jitter",
    15: "Improved Cluster-Aware SMOTE [IMP OF 8]",
    16: "Upgraded 4-Way Mixture Ensemble [IMP OF 7]",
    17: "Anomaly-Pruned TVAE+SMOTE Hybrid (v1) [IMP OF 9]",
    "17v2": "Conservative Anomaly-Repelled TVAE+Cluster-SMOTE Hybrid (v2)",
    18: "Manifold-Balanced Anomaly-Repulsion Hybrid [IMP OF 17]",
    19: "Refined Manifold-Balanced Hybrid (with Stray-Outlier Repair) [IMP OF 18]",
}


def get_approach_runner(approach_id):
    str_id = str(approach_id).lower().replace("_", "").replace("-", "")
    if str_id == "10":
        from approach_10_detector_in_the_loop import run
    elif str_id == "11":
        from approach_11_per_detector_specialized_blend import run
    elif str_id == "12":
        from approach_12_boundary_adversarial_smote import run
    elif str_id == "13":
        from approach_13_copula_guided_smote import run
    elif str_id == "14":
        from approach_14_density_adaptive_jitter import run
    elif str_id == "15":
        from approach_15_improved_cluster_smote import run
    elif str_id == "16":
        from approach_16_improved_ensemble_mixture import run
    elif str_id == "17":
        from approach_17_anomaly_pruned_hybrid import run
    elif str_id == "17v2":
        from approach_17_anomaly_pruned_hybrid_v2 import run
    elif str_id == "18":
        from approach_18_manifold_balanced_hybrid import run
    elif str_id == "19":
        from approach_19_refined_a18 import run
    else:
        raise ValueError(f"Unknown advanced approach ID: {approach_id}")
    return run


def format_results_table(all_results):
    rows = []
    for (approach_id, dataset_key), res in sorted(all_results.items(), key=lambda x: str(x[0])):
        if "score_S" in res:
            name = ADVANCED_APPROACH_NAMES.get(approach_id, f"Approach {approach_id}")
            rows.append({
                "Approach": f"{approach_id}. {name}",
                "Dataset": dataset_key.upper(),
                "AUPRC (ECOD)": f"{res.get('auprc_ecod', 0):.4f}",
                "AUPRC (IForest)": f"{res.get('auprc_iforest_mean', 0):.4f}",
                "Score S": f"{res.get('score_S', 0):.4f}",
                "Mean KS Dist": f"{res.get('mean_ks_distance', 0):.4f}",
                "Corr RMSE": f"{res.get('correlation_rmse', 0):.4f}",
                "Gen Time (s)": f"{res.get('generation_time', 0):.1f}",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run advanced and improved experiments")
    parser.add_argument("--approach", type=str, default=None,
                        help="Run only this approach ID (e.g., 10-19, 17v2)")
    parser.add_argument("--fast", action="store_true",
                        help="Run only fast approaches (11, 12, 13, 14, 15)")
    parser.add_argument("--dataset", type=str, choices=["nslkdd", "unsw"], default=None,
                        help="Run only on this dataset")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation, only generate synthetic data")
    args = parser.parse_args()

    if args.approach:
        # Normalize approach argument
        app_arg = args.approach.lower().replace("_", "").replace("-", "")
        if app_arg in ("17v2",):
            approaches_to_run = ["17v2"]
        elif app_arg in ("18", "19"):
            approaches_to_run = [int(app_arg)]
        else:
            try:
                approaches_to_run = [int(app_arg)]
            except ValueError:
                approaches_to_run = [args.approach]
    elif args.fast:
        approaches_to_run = [11, 12, 13, 14, 15]
    else:
        approaches_to_run = [10, 11, 12, 13, 14, 15, 16, 17, "17v2", 18, 19]

    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())

    print("=" * 70)
    print("ADVANCED & IMPROVED EXPERIMENTS RUNNER (Approaches 10–17)")
    print("=" * 70)
    print(f"Approaches: {[ADVANCED_APPROACH_NAMES[a] for a in approaches_to_run]}")
    print(f"Datasets:   {datasets_to_run}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    total_start = time.time()

    for app_id in approaches_to_run:
        runner = get_approach_runner(app_id)
        for ds_key in datasets_to_run:
            print(f"\n{'#'*70}")
            print(f"# Running Approach {app_id}: {ADVANCED_APPROACH_NAMES[app_id]} on {ds_key.upper()}")
            print(f"{'#'*70}")

            try:
                synthetic_df, results = runner(dataset_key=ds_key, do_eval=not args.no_eval)
                all_results[(app_id, ds_key)] = results
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                all_results[(app_id, ds_key)] = {"error": str(e)}

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All advanced experiments completed in {total_time:.0f}s")
    print(f"{'='*70}\n")

    if not args.no_eval and all_results:
        results_df = format_results_table(all_results)
        print(results_df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    main()
