"""
approach_21_log_transformed_hybrid.py
=======================================
# REFINEMENT OF APPROACH 20, TARGETING UNSW'S HEAVY-TAILED FEATURES

Motivation:
  a18 -> a20 closed the gap between attack-repulsion and manifold-realism,
  but gains plateaued (a18: 0.93159/0.48398, a20: 0.92972/0.48588) --
  evidence that further tweaks to the *selection/filtering* stage of the
  pipeline are hitting diminishing returns. The generators feeding that
  pipeline (TVAE, Cluster-SMOTE) have not themselves been changed.

  UNSW-NB15 has several strongly heavy-tailed numeric columns (e.g. rate,
  sload, dload, sjit, djit, response_body_len). TVAE's mode-specific
  normalization and SMOTE's linear k-NN interpolation both implicitly
  assume roughly additive/linear structure; heavy-tailed, multiplicative-
  scale features are known to be poorly modeled this way -- interpolating
  two byte-count values linearly, for example, does not respect the fact
  that such quantities vary multiplicatively, not additively.

  Approach 21 keeps the exact selection pipeline validated in Approach 20
  (discriminative-weighted PCA + hard gate + composite realism/repulsion
  ranking), but changes WHAT feeds into it: both TVAE and Cluster-SMOTE now
  generate in a log1p-transformed feature space for automatically-detected
  skewed numeric columns, with values inverse-transformed back to the
  original scale before any repulsion/realism filtering or blending.

  *CRITICAL REQUIREMENT*: Universal pipeline for both datasets. Skewed
  columns and shift amounts are detected/learned per dataset automatically
  from the training data -- no dataset-specific hardcoding.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset, local_train_eval_split, get_normal_data,
    validate_synthetic, add_id_column,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics
from approach_4_tvae import generate_tvae
from approach_15_improved_cluster_smote import generate_improved_cluster_smote
from approach_20_direction_weighted_hybrid import (
    filter_by_direction_weighted_repulsion,
    final_sanity_clip,
    REALISM_WEIGHT,
    ENABLE_FINAL_CLIP,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OVER_SAMPLE_FACTOR = 1.5
BLEND_TVAE = 0.50
BLEND_SMOTE = 0.50

# A numeric column is treated as "skewed" (and log-transformed) if the
# absolute skewness of its real-normal-training values exceeds this.
SKEW_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# Log-transform helpers (fit on real normal data, applied/inverted per call)
# ---------------------------------------------------------------------------

def detect_skewed_columns(normal_df, numeric_cols, threshold=SKEW_THRESHOLD):
    """Return the subset of numeric_cols whose skewness (on real normal
    training data) exceeds `threshold` in absolute value."""
    skewed = []
    for col in numeric_cols:
        values = normal_df[col].to_numpy(dtype=float)
        if np.std(values) < 1e-12:
            continue
        s = skew(values)
        if np.isfinite(s) and abs(s) > threshold:
            skewed.append(col)
    return skewed


def fit_log_transform(normal_df, skewed_cols):
    """
    Learn a per-column shift so that log1p(x + shift) is always defined
    (i.e. x + shift >= 0 for all observed real-normal values, with a small
    margin). Returns {col: shift}.
    """
    shifts = {}
    for col in skewed_cols:
        min_val = normal_df[col].min()
        shifts[col] = max(0.0, -min_val) + 1e-6
    return shifts


def apply_log_transform(df, skewed_cols, shifts):
    df = df.copy()
    for col in skewed_cols:
        df[col] = np.log1p(df[col].to_numpy(dtype=float) + shifts[col])
    return df


def invert_log_transform(df, skewed_cols, shifts):
    df = df.copy()
    for col in skewed_cols:
        df[col] = np.expm1(df[col].to_numpy(dtype=float)) - shifts[col]
    return df


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_log_transformed_hybrid(
    train_df,
    label_col,
    col_types,
    target_n,
    over_factor=OVER_SAMPLE_FACTOR,
    blend_tvae=BLEND_TVAE,
    realism_weight=REALISM_WEIGHT,
    skew_threshold=SKEW_THRESHOLD,
    enable_final_clip=ENABLE_FINAL_CLIP,
    seed=RANDOM_SEED,
):
    feature_cols = col_types["all_feature"]
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]

    normal_df = train_df.loc[train_df[label_col] == 0, feature_cols].copy().reset_index(drop=True)
    anomaly_df = train_df.loc[train_df[label_col] == 1, feature_cols].copy().reset_index(drop=True)

    # --- Learn and apply the log transform on skewed numeric columns -----
    skewed_cols = detect_skewed_columns(normal_df, numeric_cols, threshold=skew_threshold)
    print(f"    [0/4] Skewed columns log-transformed ({len(skewed_cols)}/{len(numeric_cols)}): {skewed_cols}")
    shifts = fit_log_transform(normal_df, skewed_cols)

    normal_df_t = apply_log_transform(normal_df, skewed_cols, shifts)
    anomaly_df_t = apply_log_transform(anomaly_df, skewed_cols, shifts)

    n_tvae_target = int(round(target_n * blend_tvae))
    n_smote_target = target_n - n_tvae_target
    n_tvae_raw = int(np.ceil(n_tvae_target * over_factor))

    # --- Generate in transformed space, then invert back ------------------
    print(f"    [1/4] Sampling {n_tvae_raw} TVAE candidate points in log-space ({over_factor}x factor)...")
    synth_tvae_raw_t = generate_tvae(normal_df_t, col_types, n_tvae_raw, seed=seed).drop(columns=["id"], errors="ignore")
    synth_tvae_raw = invert_log_transform(synth_tvae_raw_t, skewed_cols, shifts)

    print(f"    [2/4] Applying Direction-Weighted Anomaly-Repulsion Gate (original scale)...")
    synth_tvae_filtered = filter_by_direction_weighted_repulsion(
        synth_tvae_raw,
        normal_df,
        anomaly_df,
        categorical_cols,
        keep_n=n_tvae_target,
        realism_weight=realism_weight,
    )

    print(f"    [3/4] Generating {n_smote_target} Cluster-Aware SMOTE anchor samples in log-space...")
    synth_smote_t = generate_improved_cluster_smote(normal_df_t, col_types, n_smote_target, seed=seed).drop(columns=["id"], errors="ignore")
    synth_smote = invert_log_transform(synth_smote_t, skewed_cols, shifts)

    print(f"    [4/4] Blending and finalizing...")
    combined = pd.concat(
        [synth_tvae_filtered[feature_cols], synth_smote[feature_cols]], ignore_index=True
    )
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if len(combined) != target_n:
        raise RuntimeError(f"Generated {len(combined)} rows; expected exactly {target_n}.")

    if enable_final_clip:
        print(f"    [+] Final sanity clip (stray-outlier repair)...")
        combined = final_sanity_clip(combined, normal_df, categorical_cols, seed=seed)

    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True, realism_weight=REALISM_WEIGHT, skew_threshold=SKEW_THRESHOLD):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 21: Log-Transformed Direction-Weighted Hybrid — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")
    print(f"  Realism weight: {realism_weight}, Skew threshold: {skew_threshold}")

    t0 = time.time()
    synthetic = generate_log_transformed_hybrid(
        train_df, cfg["label_col"], col_types, cfg["target_n"],
        realism_weight=realism_weight, skew_threshold=skew_threshold,
    )
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  \u2713 Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_21_{dataset_key}.csv")
    synthetic.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    results = {"generation_time": gen_time, "realism_weight": realism_weight, "skew_threshold": skew_threshold}
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
