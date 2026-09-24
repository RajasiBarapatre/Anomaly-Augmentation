"""
approach_12_boundary_adversarial_smote.py
=========================================
Approach 12: Boundary-Adversarial SMOTE

Hypothesis & Motivation:
  In anomaly detection competitions, the most critical decision region for unsupervised
  detectors (ECOD and Isolation Forest) is the boundary separating normal traffic from
  anomalous attacks.
  
  Standard SMOTE samples uniformly across all normal points, spending most synthetic budget
  on dense interior regions where detectors are already certain.
  
  Approach 12 utilizes known training anomalies (A_seen) to:
    1. Measure the Euclidean proximity of every normal training sample to the nearest attack.
    2. Assign higher sampling probability to normal samples that lie along the attack boundary.
    3. Perform k-NN interpolation strictly between normal-normal pairs on the boundary.
    4. Densify normal coverage right where detectors are most prone to false alarms,
       without ever interpolating into the anomaly space itself.

  *CRITICAL REQUIREMENT*: The exact same pipeline is applied to both datasets.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
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
# Parameters
# ---------------------------------------------------------------------------
K_NEIGHBORS = 5
BOUNDARY_TEMPERATURE = 1.5  # Higher = stronger bias toward boundary normal points


def generate_boundary_adversarial_smote(
    train_df,
    label_col,
    col_types,
    target_n,
    k_neighbors=K_NEIGHBORS,
    temperature=BOUNDARY_TEMPERATURE,
    seed=RANDOM_SEED,
):
    """
    Generate synthetic normal data with boundary-adversarial sampling weights.
    """
    rng = np.random.default_rng(seed)
    feature_cols = col_types["all_feature"]
    cat_cols = col_types["categorical"]

    # Separate normal and anomaly sets from training data
    normal_df = train_df.loc[train_df[label_col] == 0, feature_cols].copy().reset_index(drop=True)
    anomaly_df = train_df.loc[train_df[label_col] == 1, feature_cols].copy().reset_index(drop=True)

    n_normal = len(normal_df)
    n_anomaly = len(anomaly_df)

    # Step 1: Label encode categoricals
    df_all = pd.concat([normal_df, anomaly_df], ignore_index=True)
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col].astype(str))
        label_encoders[col] = le

    X_norm = df_all.iloc[:n_normal].to_numpy(dtype=float)
    X_anom = df_all.iloc[n_normal:].to_numpy(dtype=float)

    # Step 2: Scale features
    scaler = StandardScaler()
    X_norm_scaled = scaler.fit_transform(X_norm)
    X_anom_scaled = scaler.transform(X_anom)

    # Step 3: Compute distance from each normal point to nearest anomaly point
    print(f"    [1/3] Computing boundary proximity to {n_anomaly} known attack samples...")
    nn_anom = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_anom.fit(X_anom_scaled)
    dist_to_anom, _ = nn_anom.kneighbors(X_norm_scaled)
    dist_to_anom = dist_to_anom.flatten()

    # Smaller distance to anomaly = closer to boundary = higher sampling weight
    median_dist = np.median(dist_to_anom) + 1e-10
    norm_dist = dist_to_anom / median_dist
    weights = np.exp(-temperature * norm_dist) + 0.2  # 0.2 baseline ensures interior is still covered
    weights /= weights.sum()

    # Step 4: Fit k-NN within NORMAL points
    print(f"    [2/3] Fitting normal manifold k-NN (k={k_neighbors})...")
    nn_norm = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto")
    nn_norm.fit(X_norm_scaled)
    _, norm_neighbors = nn_norm.kneighbors(X_norm_scaled)

    # Step 5: Sample and interpolate
    print(f"    [3/3] Generating {target_n} boundary-densified synthetic samples...")
    sampled_indices = rng.choice(n_normal, size=target_n, p=weights)
    synthetic_rows = np.empty((target_n, len(feature_cols)))

    for i, idx in enumerate(sampled_indices):
        base_point = X_norm[idx]
        neighbor_idx = norm_neighbors[idx, rng.integers(1, k_neighbors + 1)]
        neighbor_point = X_norm[neighbor_idx]
        lam = rng.uniform(0, 1)
        synthetic_rows[i] = base_point + lam * (neighbor_point - base_point)

    synthetic = pd.DataFrame(synthetic_rows, columns=feature_cols)

    # Post-process
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
    print(f"Approach 12: Boundary-Adversarial SMOTE — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_boundary_adversarial_smote(
        train_df, cfg["label_col"], col_types, cfg["target_n"]
    )
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_12_{dataset_key}.csv")
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
