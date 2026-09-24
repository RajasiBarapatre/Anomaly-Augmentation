"""
run_hybrid_experiments.py
==========================
Orchestrator for next-generation Hybrid and Ensemble approaches:
  - Approach 7: Multi-Generator Unified Mixture Ensemble (50% SMOTE + 35% TVAE + 15% CTGAN)
  - Approach 8: Cluster-Aware SMOTE + Adaptive Density Smoothing
  - Approach 9: Density-Filtered TVAE + SMOTE Hybrid

Usage:
  python3 run_hybrid_experiments.py             # Run 7, 8, 9 on both datasets
  python3 run_hybrid_experiments.py --approach 7 # Run only Approach 7
  python3 run_hybrid_experiments.py --dataset nslkdd # Run on NSL-KDD only
"""

import sys
import os
import time
import argparse
import traceback
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, OUTPUT_DIR

HYBRID_APPROACH_NAMES = {
    7: "Multi-Generator Ensemble (50% SMOTE + 35% TVAE + 15% CTGAN)",
    8: "Cluster-Aware SMOTE + Adaptive Density Smoothing",
    9: "Density-Filtered TVAE + SMOTE Hybrid",
}


def get_approach_runner(approach_id):
    if approach_id == 7:
        from approach_7_ensemble_mixture import run
    elif approach_id == 8:
        from approach_8_cluster_smote import run
    elif approach_id == 9:
        from approach_9_filtered_hybrid import run
    else:
        raise ValueError(f"Unknown hybrid approach ID: {approach_id}")
    return run


def format_results_table(all_results):
    rows = []
    for (approach_id, dataset_key), res in sorted(all_results.items()):
        if "score_S" in res:
            rows.append({
                "Approach": f"{approach_id}. {HYBRID_APPROACH_NAMES.get(approach_id, f'Approach {approach_id}')}",
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
    parser = argparse.ArgumentParser(description="Run hybrid and ensemble experiments")
    parser.add_argument("--approach", type=int, choices=[7, 8, 9], default=None,
                        help="Run only this approach ID")
    parser.add_argument("--dataset", type=str, choices=["nslkdd", "unsw"], default=None,
                        help="Run only on this dataset")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation, only generate synthetic data")
    args = parser.parse_args()

    approaches_to_run = [args.approach] if args.approach else [7, 8, 9]
    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())

    print("=" * 70)
    print("HYBRID & ENSEMBLE EXPERIMENTS RUNNER (Approaches 7, 8, 9)")
    print("=" * 70)
    print(f"Approaches: {[HYBRID_APPROACH_NAMES[a] for a in approaches_to_run]}")
    print(f"Datasets:   {datasets_to_run}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    total_start = time.time()

    for app_id in approaches_to_run:
        runner = get_approach_runner(app_id)
        for ds_key in datasets_to_run:
            print(f"\n{'#'*70}")
            print(f"# Running Approach {app_id}: {HYBRID_APPROACH_NAMES[app_id]} on {ds_key.upper()}")
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
    print(f"All hybrid experiments completed in {total_time:.0f}s")
    print(f"{'='*70}\n")

    if not args.no_eval and all_results:
        results_df = format_results_table(all_results)
        print(results_df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    main()
