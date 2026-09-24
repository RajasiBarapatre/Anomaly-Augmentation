"""
approach_3_ctgan.py
===================
CTGAN (Conditional Tabular GAN) approach using SDV's CTGANSynthesizer.

Method:
  1. Mode-specific normalization: each numeric column is clustered into modes
     using a VGM (Variational Gaussian Mixture), then normalized per-mode
  2. A conditional GAN architecture generates rows conditioned on sampled
     categorical/mode values, ensuring balanced representation
  3. Discriminator sees both real and fake rows with their mode indicators

Strengths:
  - Handles multimodal/skewed numeric distributions naturally
  - Conditional training ensures categorical diversity
  - Can capture non-linear relationships

Weaknesses:
  - GAN training can be unstable (mode collapse, non-convergence)
  - Slower training than statistical methods
  - Requires hyperparameter tuning (epochs, batch size, etc.)
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
# Hyperparameters
# ---------------------------------------------------------------------------
CTGAN_EPOCHS = 300
CTGAN_BATCH_SIZE = 500
GENERATOR_DIM = (256, 256)
DISCRIMINATOR_DIM = (256, 256)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def generate_ctgan(normal_df, col_types, target_n, seed=RANDOM_SEED):
    """
    Fit CTGAN on normal-class data and generate synthetic samples.
    """
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import Metadata

    metadata = Metadata.detect_from_dataframe(data=normal_df)

    # Explicitly set sdtype for all columns so SDV doesn't misclassify
    cat_set = set(col_types["categorical"])
    for col in normal_df.columns:
        if col in cat_set:
            metadata.update_column(column_name=col, sdtype="categorical")
        else:
            metadata.update_column(column_name=col, sdtype="numerical")

    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        epochs=CTGAN_EPOCHS,
        batch_size=CTGAN_BATCH_SIZE,
        generator_dim=GENERATOR_DIM,
        discriminator_dim=DISCRIMINATOR_DIM,
        verbose=True,
    )

    print(f"    Fitting CTGAN ({CTGAN_EPOCHS} epochs)...")
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
    print(f"Approach 3: CTGAN — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_ctgan(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_3_{dataset_key}.csv")
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
