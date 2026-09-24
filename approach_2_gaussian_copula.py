"""
approach_2_gaussian_copula.py
=============================
Gaussian Copula approach using SDV's GaussianCopulaSynthesizer.

Method:
  1. Fit per-feature marginal distributions (auto-detected: beta, gamma, etc.)
  2. Transform each marginal to standard Normal via probability integral transform
  3. Fit a multivariate Gaussian to the transformed data (captures correlations)
  4. Sample from the joint Gaussian, then inverse-transform through marginals

Strengths:
  - Naturally handles skewed/zero-inflated features through marginal transforms
  - Preserves pairwise correlations
  - Fast training and sampling
  - Handles mixed (numeric + categorical) data

Weaknesses:
  - Linear (Gaussian) dependence model cannot capture non-linear relationships
  - Single copula family (Gaussian) may not suit heavy-tailed joint distributions
"""

import sys
import os
import time
import warnings
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset, local_train_eval_split, get_normal_data,
    validate_synthetic, add_id_column,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def generate_gaussian_copula(normal_df, col_types, target_n, seed=RANDOM_SEED):
    """
    Fit a Gaussian Copula model on normal-class data and generate synthetic samples.
    """
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import Metadata

    # Build SDV metadata with explicit column types to prevent misdetection
    metadata = Metadata.detect_from_dataframe(data=normal_df)

    # Explicitly set sdtype for all columns so SDV doesn't misclassify
    cat_set = set(col_types["categorical"])
    for col in normal_df.columns:
        if col in cat_set:
            metadata.update_column(column_name=col, sdtype="categorical")
        else:
            metadata.update_column(column_name=col, sdtype="numerical")

    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        numerical_distributions={},  # let SDV auto-detect
    )

    print("    Fitting Gaussian Copula model...")
    synthesizer.fit(normal_df)

    print(f"    Generating {target_n} synthetic samples...")
    synthetic = synthesizer.sample(num_rows=target_n)

    # Post-process: ensure integer columns are integers (only if actually numeric)
    for col in col_types["integer"] + col_types["binary"]:
        if col in synthetic.columns and pd.api.types.is_numeric_dtype(synthetic[col]):
            synthetic[col] = synthetic[col].round().astype(int)

    return add_id_column(synthetic)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 2: Gaussian Copula — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_gaussian_copula(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_2_{dataset_key}.csv")
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
