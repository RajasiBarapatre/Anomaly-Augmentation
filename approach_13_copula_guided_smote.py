"""
approach_13_copula_guided_smote.py
==================================
Approach 13: Copula-Guided SMOTE

Hypothesis & Motivation:
  Gaussian Copula's greatest strength is its per-feature marginal transform (mapping skewed,
  heavy-tailed marginals to uniform/Gaussian space). Its main weakness is the linear Gaussian
  joint dependence assumption.
  
  SMOTE's greatest strength is non-parametric manifold interpolation, but its weakness is
  measuring Euclidean distance in heavily skewed original feature spaces.
  
  Approach 13 combines both:
    1. Apply empirical Probability Integral Transform (PIT) to map each numeric feature
       to uniform marginal space U(0, 1).
    2. Perform k-NN neighbor search and interpolation directly in the normalized copula space.
    3. Inverse-transform synthetic points back to native feature distributions via empirical
       quantile functions.
    4. Categoricals are preserved and mapped via nearest class boundaries.

  *CRITICAL REQUIREMENT*: Universal pipeline for both datasets.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset, local_train_eval_split, get_normal_data,
    validate_synthetic, add_id_column,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics

warnings.filterwarnings("ignore")

K_NEIGHBORS = 5


def generate_copula_guided_smote(normal_df, col_types, target_n, k_neighbors=K_NEIGHBORS, seed=RANDOM_SEED):
    """
    Generate synthetic data by interpolating in empirical copula (uniform marginal) space.
    """
    rng = np.random.default_rng(seed)
    feature_cols = col_types["all_feature"]
    cat_cols = col_types["categorical"]
    num_cols = col_types["numeric"]
    n_samples = len(normal_df)

    # 1. Transform numerics to uniform copula space [0, 1] using empirical CDF ranks
    print(f"    [1/3] Transforming {len(num_cols)} features into copula uniform space...")
    U_matrix = np.empty((n_samples, len(num_cols)))
    sorted_features = {}

    for j, col in enumerate(num_cols):
        vals = normal_df[col].to_numpy(dtype=float)
        # Fractional rank in (0, 1)
        ranks = (rankdata(vals, method="average") - 0.5) / n_samples
        U_matrix[:, j] = ranks
        sorted_features[col] = np.sort(vals)

    # 2. Encode categoricals to normalized [0, 1] interval
    cat_matrix = np.empty((n_samples, len(cat_cols)))
    label_encoders = {}
    for j, col in enumerate(cat_cols):
        le = LabelEncoder()
        encoded = le.fit_transform(normal_df[col].astype(str))
        n_classes = len(le.classes_)
        cat_matrix[:, j] = encoded / max(1, n_classes - 1)
        label_encoders[col] = le

    # Combined copula space representation
    X_copula = np.hstack([U_matrix, cat_matrix])

    # 3. Fit k-NN in copula space and interpolate
    print(f"    [2/3] Performing k-NN manifold interpolation in copula space (k={k_neighbors})...")
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto")
    nn.fit(X_copula)
    _, neighbors = nn.kneighbors(X_copula)

    synth_copula = np.empty((target_n, X_copula.shape[1]))
    for i in range(target_n):
        idx = rng.integers(0, n_samples)
        neighbor_idx = neighbors[idx, rng.integers(1, k_neighbors + 1)]
        lam = rng.uniform(0, 1)
        synth_copula[i] = X_copula[idx] + lam * (X_copula[neighbor_idx] - X_copula[idx])

    # 4. Inverse-transform copula points back to native distributions
    print(f"    [3/3] Inverse-transforming {target_n} samples to native distributions...")
    synth_dict = {}

    # Inverse CDF for numerics
    for j, col in enumerate(num_cols):
        u_vals = np.clip(synth_copula[:, j], 0.0, 1.0)
        sorted_real = sorted_features[col]
        # Map uniform u to empirical quantile
        indices = u_vals * (n_samples - 1)
        lower_idx = np.floor(indices).astype(int)
        upper_idx = np.ceil(indices).astype(int)
        weight = indices - lower_idx
        synth_vals = (1 - weight) * sorted_real[lower_idx] + weight * sorted_real[upper_idx]

        lo, hi = sorted_real[0], sorted_real[-1]
        synth_vals = np.clip(synth_vals, lo, hi)
        if col in col_types["integer"] or col in col_types["binary"]:
            synth_vals = np.rint(synth_vals).astype(int)
        synth_dict[col] = synth_vals

    # Inverse transform categoricals
    for j, col in enumerate(cat_cols):
        le = label_encoders[col]
        n_classes = len(le.classes_)
        norm_cat = np.clip(synth_copula[:, len(num_cols) + j], 0.0, 1.0)
        class_idx = np.clip(np.rint(norm_cat * (n_classes - 1)), 0, n_classes - 1).astype(int)
        synth_dict[col] = le.inverse_transform(class_idx)

    synthetic = pd.DataFrame(synth_dict)[feature_cols]
    return add_id_column(synthetic)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 13: Copula-Guided SMOTE — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_copula_guided_smote(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_13_{dataset_key}.csv")
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
