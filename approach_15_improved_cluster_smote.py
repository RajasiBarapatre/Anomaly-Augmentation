"""
approach_15_improved_cluster_smote.py
======================================
# IMPROVEMENT OF APPROACH 8

Hypothesis & Motivation:
  Approach 8 (Cluster-Aware SMOTE) achieved the highest score on NSL-KDD (0.93106) and 0.43917 on UNSW.
  Two key enhancements are made here:
    1. Higher Cluster Resolution (K=40 instead of K=20):
       Provides finer sub-manifold partitioning, especially critical for UNSW's 42 features
       and complex multimodal network states.
    2. Feature-Weighted Distance Metric for k-NN:
       Instead of uniform Euclidean distance, scale features by their inverse variance and
       univariate entropy weight, ensuring subtle low-variance protocol flags are not
       dominated by large byte-count dimensions.
    3. Adaptive Intra-Cluster Micro-Smoothing:
       Scaled to local cluster variance.

  *CRITICAL REQUIREMENT*: Universal pipeline for both datasets.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
N_CLUSTERS = 40
K_NEIGHBORS = 5
MICRO_SMOOTH_ALPHA = 0.04


def generate_improved_cluster_smote(
    normal_df,
    col_types,
    target_n,
    n_clusters=N_CLUSTERS,
    k_neighbors=K_NEIGHBORS,
    smooth_alpha=MICRO_SMOOTH_ALPHA,
    seed=RANDOM_SEED,
):
    """
    Generate synthetic data via fine-grained cluster-aware SMOTE with feature-weighted metric.
    """
    rng = np.random.default_rng(seed)
    feature_cols = col_types["all_feature"]
    cat_cols = col_types["categorical"]
    n_samples = len(normal_df)

    # 1. Label encode categoricals
    df_encoded = normal_df[feature_cols].copy()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    X = df_encoded.to_numpy(dtype=float)
    n_features = X.shape[1]

    # 2. Compute feature weights (inverse std with ceiling to prevent division by zero)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    feature_weights = 1.0 / np.sqrt(stds)
    feature_weights /= np.mean(feature_weights)

    # Weighted standardized matrix for distance and clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) * feature_weights

    # 3. Fine-grained clustering (K=40)
    actual_clusters = min(n_clusters, n_samples // (k_neighbors + 2))
    kmeans = MiniBatchKMeans(
        n_clusters=actual_clusters,
        random_state=seed,
        batch_size=min(1024, n_samples),
        n_init=3,
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Cluster target counts
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    cluster_props = cluster_counts / n_samples
    cluster_target_counts = np.round(cluster_props * target_n).astype(int)

    diff = target_n - cluster_target_counts.sum()
    if diff != 0:
        max_idx = np.argmax(cluster_counts)
        cluster_target_counts[max_idx] += diff

    print(f"    Partitioned data into {len(unique_clusters)} fine-grained weighted clusters (target_n={target_n})...")

    # 4. Intra-cluster feature-weighted SMOTE interpolation
    synthetic_rows = []
    for c_id, c_count, c_target in zip(unique_clusters, cluster_counts, cluster_target_counts):
        if c_target <= 0:
            continue

        c_indices = np.where(cluster_labels == c_id)[0]
        X_cluster = X[c_indices]
        X_cluster_scaled = X_scaled[c_indices]
        n_c = len(X_cluster)

        cluster_stds = np.std(X_cluster, axis=0)
        k_eff = min(k_neighbors, n_c - 1)

        if k_eff >= 1:
            nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto")
            nn.fit(X_cluster_scaled)
            _, neighbors = nn.kneighbors(X_cluster_scaled)
        else:
            neighbors = None

        for _ in range(c_target):
            idx_in_c = rng.integers(0, n_c)
            base_point = X_cluster[idx_in_c]

            if neighbors is not None:
                n_idx = neighbors[idx_in_c, rng.integers(1, k_eff + 1)]
                neighbor_point = X_cluster[n_idx]
                lam = rng.uniform(0, 1)
                synth_point = base_point + lam * (neighbor_point - base_point)
            else:
                synth_point = base_point.copy()

            # Local adaptive micro-noise
            noise = rng.normal(0, smooth_alpha * cluster_stds, size=n_features)
            synth_point = synth_point + noise
            synthetic_rows.append(synth_point)

    synthetic_matrix = np.array(synthetic_rows[:target_n])
    synthetic = pd.DataFrame(synthetic_matrix, columns=feature_cols)

    # 5. Post-process
    for col in col_types["binary"]:
        synthetic[col] = np.clip(np.rint(synthetic[col]), 0, 1).astype(int)

    for col in col_types["integer"]:
        lo, hi = normal_df[col].min(), normal_df[col].max()
        synthetic[col] = np.clip(np.rint(synthetic[col]), lo, hi).astype(int)

    for col in col_types["continuous"]:
        lo, hi = normal_df[col].min(), normal_df[col].max()
        synthetic[col] = synthetic[col].clip(lo, hi)

    for col in cat_cols:
        le = label_encoders[col]
        n_classes = len(le.classes_)
        cat_indices = np.clip(np.rint(synthetic[col]), 0, n_classes - 1).astype(int)
        synthetic[col] = le.inverse_transform(cat_indices)

    return add_id_column(synthetic)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 15: Improved Cluster-Aware SMOTE — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_improved_cluster_smote(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_15_{dataset_key}.csv")
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
