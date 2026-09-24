"""
run_all_experiments.py
======================
Orchestrator: runs all 6 approaches on both datasets, collects results,
and writes a comprehensive comparison to findings.md.

Usage:
    python3 run_all_experiments.py              # Run all approaches, both datasets
    python3 run_all_experiments.py --quick       # Run only approaches 1, 2, 6 (fast ones)
    python3 run_all_experiments.py --approach 3  # Run only approach 3
    python3 run_all_experiments.py --dataset nslkdd  # Run only NSL-KDD
"""

import sys
import os
import time
import argparse
import traceback
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, APPROACH_NAMES, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Approach modules (lazy import to avoid loading all at startup)
# ---------------------------------------------------------------------------
def get_approach_runner(approach_id):
    """Dynamically import and return the run() function for an approach."""
    if approach_id == 1:
        from approach_1_baseline_jitter import run
    elif approach_id == 2:
        from approach_2_gaussian_copula import run
    elif approach_id == 3:
        from approach_3_ctgan import run
    elif approach_id == 4:
        from approach_4_tvae import run
    elif approach_id == 5:
        from approach_5_smote import run
    elif approach_id == 6:
        from approach_6_kde import run
    else:
        raise ValueError(f"Unknown approach ID: {approach_id}")
    return run


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------
def format_results_table(all_results):
    """Format results as a Markdown table."""
    rows = []
    for (approach_id, dataset_key), res in sorted(all_results.items()):
        if "score_S" in res:
            rows.append({
                "Approach": f"{approach_id}. {APPROACH_NAMES[approach_id]}",
                "Dataset": dataset_key.upper(),
                "AUPRC (ECOD)": f"{res.get('auprc_ecod', 0):.4f}",
                "AUPRC (IForest)": f"{res.get('auprc_iforest_mean', 0):.4f}",
                "Score S": f"{res.get('score_S', 0):.4f}",
                "Mean KS Dist": f"{res.get('mean_ks_distance', 0):.4f}",
                "Corr RMSE": f"{res.get('correlation_rmse', 0):.4f}",
                "Gen Time (s)": f"{res.get('generation_time', 0):.1f}",
            })
    return pd.DataFrame(rows)


def write_findings(all_results):
    """Write/update the findings.md file with results."""
    results_df = format_results_table(all_results)

    # Separate by dataset
    nslkdd_rows = results_df[results_df["Dataset"] == "NSLKDD"]
    unsw_rows = results_df[results_df["Dataset"] == "UNSW"]

    findings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "findings.md")

    md = """# Experiment Findings — Approach Comparison for Tabular Data Augmentation

## Objective
Compare 6 data augmentation approaches for generating synthetic **normal-only** network
traffic data that maximizes downstream anomaly detection performance (AUPRC) when used
to train fixed ECOD and Isolation Forest detectors.

## Evaluation Methodology
- **Local evaluation**: 80/20 stratified split of the labeled training set
- **Detectors**: ECOD (deterministic) + Isolation Forest (10 seeds × 500 trees, averaged)
- **Metric**: S = (AUPRC_ECOD + mean_AUPRC_IForest) / 2  (mirrors Kaggle scoring)
- **Statistical quality**: KS distance (marginal fidelity), Correlation RMSE (joint fidelity)
- Categoricals are **one-hot encoded** before detector training (mirrors Kaggle pipeline)

---

## Results — NSL-KDD (Target: 40,000 synthetic samples)

"""
    if len(nslkdd_rows) > 0:
        md += nslkdd_rows.to_markdown(index=False)
    else:
        md += "*No results yet — run `python3 run_all_experiments.py` to populate.*\n"

    md += "\n\n## Results — UNSW-NB15 (Target: 70,000 synthetic samples)\n\n"
    if len(unsw_rows) > 0:
        md += unsw_rows.to_markdown(index=False)
    else:
        md += "*No results yet — run `python3 run_all_experiments.py` to populate.*\n"

    md += """

---

## Approach Descriptions and Trade-offs

### 1. Baseline Jitter (Bootstrap + Gaussian Noise)
- **Method**: Resample with replacement + add Gaussian noise to numerics
- **Pros**: Dead simple, fast, no training needed
- **Cons**: No distribution learning; output is just noisy copies of input rows;
  cannot generate truly novel samples or fill distribution gaps
- **Use case**: Performance floor / sanity check

### 2. Gaussian Copula
- **Method**: Fit per-feature marginals + Gaussian copula for joint dependencies
- **Pros**: Naturally handles skew via marginal transforms; preserves pairwise correlations;
  fast training and sampling; handles mixed types
- **Cons**: Linear (Gaussian) dependence structure cannot capture non-linear or tail
  dependencies; single copula family
- **Use case**: Strong baseline when data is approximately Gaussian-copula-shaped

### 3. CTGAN (Conditional Tabular GAN)
- **Method**: Mode-specific normalization + conditional GAN training
- **Pros**: Handles multimodal/skewed numerics; conditional generation preserves
  categorical diversity; can capture complex non-linear relationships
- **Cons**: GAN training instability (mode collapse); slow; needs hyperparameter tuning
- **Use case**: When data has complex multimodal structure

### 4. TVAE (Tabular Variational Autoencoder)
- **Method**: Mode-specific normalization + VAE (ELBO training)
- **Pros**: More stable than GAN training; smoother latent space; diverse samples
- **Cons**: Can produce overly-smooth / blurry synthetic data; may underfit complex modes
- **Use case**: When training stability matters more than sharp mode capture

### 5. SMOTE (k-NN Interpolation)
- **Method**: Interpolate between k-nearest normal samples in feature space
- **Pros**: Generates within the convex hull of real data; simple; fast
- **Cons**: Cannot extrapolate or fill distribution gaps; approximate categorical handling;
  doesn't model the full joint distribution
- **Use case**: When data lies in a compact region and interpolation suffices

### 6. KDE (Kernel Density Estimation)
- **Method**: Non-parametric density estimation with log-transform preprocessing + PCA
- **Pros**: Adapts to any distribution shape; interpretable; handles skew via transforms
- **Cons**: Curse of dimensionality (mitigated by PCA); loses cat-numeric correlations;
  bandwidth sensitivity
- **Use case**: When distribution is smooth/unimodal and dimensionality is manageable

---

## Comparison Criteria

| Criterion | Weight | Rationale |
|---|---|---|
| Score S (AUPRC) | HIGH | Directly measures downstream utility — the Kaggle metric |
| Mean KS Distance | MEDIUM | Marginal distribution fidelity (lower = more faithful) |
| Correlation RMSE | MEDIUM | Joint dependency fidelity (lower = better) |
| Generation Time | LOW | Practical constraint, but not a primary concern |
| Transferability | HIGH | Must use one pipeline for both datasets |
| Explainability | MEDIUM | Required for project report and presentation |

---

## Final Intended Approach and Justification

> **To be populated after experiments complete.**
>
> Selection criteria:
> 1. Highest average Score S across both datasets
> 2. Good statistical fidelity (low KS distance and correlation RMSE)
> 3. Transferable (same pipeline for NSL-KDD and UNSW-NB15)
> 4. Explainable (can articulate why it works for the project report)
> 5. Reasonable training time

---

## Next Steps
1. Run `python3 run_all_experiments.py` to populate results tables above
2. Analyze results to select the best approach
3. Tune hyperparameters of the selected approach
4. Generate final Kaggle submissions

"""

    with open(findings_path, "w") as f:
        f.write(md)
    print(f"\nFindings written to: {findings_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run augmentation experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Run only fast approaches (1, 2, 6)")
    parser.add_argument("--approach", type=int, default=None,
                        help="Run only a specific approach (1-6)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only a specific dataset (nslkdd or unsw)")
    args = parser.parse_args()

    # Determine which approaches and datasets to run
    if args.approach:
        approach_ids = [args.approach]
    elif args.quick:
        approach_ids = [1, 2, 6]
    else:
        approach_ids = list(APPROACH_NAMES.keys())

    if args.dataset:
        dataset_keys = [args.dataset]
    else:
        dataset_keys = list(DATASETS.keys())

    print("=" * 70)
    print("TABULAR DATA AUGMENTATION — EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Approaches: {[APPROACH_NAMES[i] for i in approach_ids]}")
    print(f"Datasets:   {dataset_keys}")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    for approach_id in approach_ids:
        for dataset_key in dataset_keys:
            print(f"\n{'#'*70}")
            print(f"# Running Approach {approach_id}: {APPROACH_NAMES[approach_id]} "
                  f"on {dataset_key.upper()}")
            print(f"{'#'*70}")

            try:
                runner = get_approach_runner(approach_id)
                _, results = runner(dataset_key, do_eval=True)
                all_results[(approach_id, dataset_key)] = results
            except Exception as e:
                print(f"\n  ✗ FAILED: {e}")
                traceback.print_exc()
                all_results[(approach_id, dataset_key)] = {
                    "error": str(e),
                    "generation_time": 0,
                }

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All experiments completed in {total_elapsed:.0f}s")
    print(f"{'='*70}\n")

    # Display summary table
    results_df = format_results_table(all_results)
    if len(results_df) > 0:
        print(results_df.to_string(index=False))
    else:
        print("No successful experiments to display.")

    # Write findings
    write_findings(all_results)


if __name__ == "__main__":
    main()
