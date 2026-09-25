"""
approach_22_bootstrap_robust_hybrid.py
========================================
# REFINEMENT OF APPROACH 18, TARGETING OVERFITTING TO A_seen

Motivation:
  Approaches 17/18/20 all score candidates using distance to the seen
  training attacks (A_seen). This is a real signal, but it is also the part
  of the pipeline most likely to overfit: A_seen is a specific, finite
  sample of attack instances, and the hidden validation/test sets contain
  anomaly types from A_val / A_test that were never observed. A repulsion
  score built on nearest-neighbor distance to a handful of specific attack
  points can end up encoding idiosyncrasies of THOSE particular points
  rather than a generalizable notion of "attack-like".

  Approach 22 makes the pipeline more conservative about attack-repulsion
  in two ways, while leaving the (attack-agnostic, inherently more
  generalizable) realism term as the dominant signal:

    1. BOOTSTRAP-ENSEMBLED ATTACK DISTANCE
       Instead of one nearest-neighbor lookup against the full A_seen set,
       we draw B bootstrap resamples of A_seen (each a random subset with
       replacement) and average the nearest-attack distance across all B
       resamples. This is a bagging-style variance reduction: a candidate
       that only looks "safe" because of quirks of a few specific attack
       rows will show much more volatile bootstrap distances than a
       candidate that is genuinely far from the attack distribution in
       general, and the averaging suppresses the former.

    2. REALISM-DOMINANT COMPOSITE SCORE
       The composite ranking score exposes attack-repulsion and realism as
       SEPARATELY weighted terms (previously realism was expressed as a
       fraction of a repulsion-dominant score). The default weighting now
       favors realism (attack-agnostic, generalizes to any anomaly type)
       over repulsion (attack-specific, risks overfitting to A_seen).

  Everything else -- TVAE oversampling, Cluster-Aware SMOTE blend, hard
  gate + composite ranking structure -- is carried over from Approach 18.
  No direction/discriminative weighting (Approach 20) is used here, since
  learning which directions separate normal from A_seen is itself an
  A_seen-specific fit that risks not transferring to A_val / A_test.

  *CRITICAL REQUIREMENT*: Universal pipeline for both datasets.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset, local_train_eval_split, get_normal_data,
    validate_synthetic, add_id_column, one_hot_encode,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics
from approach_4_tvae import generate_tvae
from approach_15_improved_cluster_smote import generate_improved_cluster_smote

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OVER_SAMPLE_FACTOR = 1.5
BLEND_TVAE = 0.50
BLEND_SMOTE = 0.50

# Bootstrap-ensembled attack distance
N_BOOTSTRAPS = 15
BOOTSTRAP_FRACTION = 0.70   # fraction of A_seen sampled (with replacement) per round
BOOTSTRAP_K = 3             # nearest-attack neighbors averaged within each bootstrap round

# Composite score = ATTACK_WEIGHT * z(attack_dist) - REALISM_WEIGHT * z(normal_dist)
# Realism dominant by default -- attack-repulsion contributes but does not
# drive the ranking, to reduce overfitting to the specific shape of A_seen.
ATTACK_WEIGHT = 0.5
REALISM_WEIGHT = 1.0
REALISM_K = 5

# Hard gate: still used as a conservative floor (never keep candidates that
# are unambiguously overlapping attacks), but bootstrap-averaged too.
ATTACK_OVERLAP_PERCENTILE = 2.5
MIN_TVAE_RETENTION = 0.85


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _fit_normal_scaled_space(normal_df, other_dfs, categorical_cols):
    normal_encoded, fit_categories = one_hot_encode(normal_df, categorical_cols)
    other_encoded_list = []
    for df in other_dfs:
        enc, _ = one_hot_encode(df, categorical_cols, fit_categories=fit_categories)
        other_encoded_list.append(enc)

    all_cols = set(normal_encoded.columns)
    for enc in other_encoded_list:
        all_cols |= set(enc.columns)
    all_cols = sorted(all_cols)

    def _align(enc):
        for c in all_cols:
            if c not in enc.columns:
                enc[c] = 0
        return enc[all_cols].to_numpy(dtype=float)

    X_normal = _align(normal_encoded)
    scaler = StandardScaler().fit(X_normal)
    return scaler, fit_categories, all_cols, scaler.transform(X_normal)


def _transform(df, categorical_cols, fit_categories, all_cols, scaler):
    encoded, _ = one_hot_encode(df, categorical_cols, fit_categories=fit_categories)
    for c in all_cols:
        if c not in encoded.columns:
            encoded[c] = 0
    return scaler.transform(encoded[all_cols].to_numpy(dtype=float))


def _nearest_distance(X_query, X_reference, k=1):
    if len(X_reference) == 0:
        return np.full(len(X_query), np.inf, dtype=float)
    k_eff = min(k, len(X_reference))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1)
    nn.fit(X_reference)
    distances, _ = nn.kneighbors(X_query)
    return distances.mean(axis=1)


def _zscore(x):
    mu, sigma = np.mean(x), np.std(x)
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma


def _bootstrap_attack_distance(
    X_query, X_anomaly, n_bootstraps=N_BOOTSTRAPS,
    fraction=BOOTSTRAP_FRACTION, k=BOOTSTRAP_K, seed=RANDOM_SEED,
):
    """
    Average nearest-attack distance across n_bootstraps random resamples of
    the attack set. Reduces sensitivity to any single attack instance's
    idiosyncrasies compared to a single nearest-neighbor lookup against the
    full A_seen set, at the cost of extra compute.
    """
    if len(X_anomaly) == 0:
        return np.full(len(X_query), np.inf, dtype=float)

    rng = np.random.RandomState(seed)
    n_sample = max(k, int(np.ceil(len(X_anomaly) * fraction)))
    all_dists = np.zeros((n_bootstraps, len(X_query)), dtype=float)

    for b in range(n_bootstraps):
        idx = rng.choice(len(X_anomaly), size=n_sample, replace=True)
        all_dists[b] = _nearest_distance(X_query, X_anomaly[idx], k=k)

    return all_dists.mean(axis=0)


# ---------------------------------------------------------------------------
# Bootstrap-robust, realism-dominant anomaly-repulsion gate
# ---------------------------------------------------------------------------

def filter_by_bootstrap_robust_repulsion(
    candidate_df,
    normal_df,
    anomaly_df,
    categorical_cols,
    keep_n,
    attack_weight=ATTACK_WEIGHT,
    realism_weight=REALISM_WEIGHT,
    realism_k=REALISM_K,
    n_bootstraps=N_BOOTSTRAPS,
    overlap_percentile=ATTACK_OVERLAP_PERCENTILE,
    min_retention=MIN_TVAE_RETENTION,
    seed=RANDOM_SEED,
):
    if len(candidate_df) <= keep_n:
        return candidate_df.copy().reset_index(drop=True)

    scaler, fit_categories, all_cols, X_normal = _fit_normal_scaled_space(
        normal_df, [anomaly_df, candidate_df], categorical_cols
    )
    X_anomaly = _transform(anomaly_df, categorical_cols, fit_categories, all_cols, scaler)
    X_candidate = _transform(candidate_df, categorical_cols, fit_categories, all_cols, scaler)

    # --- Stage 1: conservative hard gate (bootstrap-averaged too) ---------
    if len(X_anomaly) > 0:
        normal_attack_dist = _bootstrap_attack_distance(
            X_normal, X_anomaly, n_bootstraps=n_bootstraps, seed=seed
        )
        candidate_attack_dist = _bootstrap_attack_distance(
            X_candidate, X_anomaly, n_bootstraps=n_bootstraps, seed=seed
        )
        threshold = np.percentile(normal_attack_dist, overlap_percentile)
        suspicious_mask = candidate_attack_dist < threshold

        minimum_allowed = int(np.ceil(len(candidate_df) * min_retention))
        maximum_prunable = max(0, len(candidate_df) - minimum_allowed)
        suspicious_idx = np.where(suspicious_mask)[0]

        if len(suspicious_idx) > maximum_prunable:
            worst_first = suspicious_idx[np.argsort(candidate_attack_dist[suspicious_idx])]
            remove_idx = set(worst_first[:maximum_prunable].tolist())
        else:
            remove_idx = set(suspicious_idx.tolist())
    else:
        candidate_attack_dist = np.full(len(candidate_df), np.inf)
        remove_idx = set()

    survivors = np.array([i for i in range(len(candidate_df)) if i not in remove_idx])

    # --- Stage 2: realism-dominant composite ranking -----------------------
    realism_dist = _nearest_distance(X_candidate[survivors], X_normal, k=realism_k)
    attack_dist_survivors = candidate_attack_dist[survivors]

    composite = (
        attack_weight * _zscore(attack_dist_survivors)
        - realism_weight * _zscore(realism_dist)
    )
    order = np.argsort(-composite)
    selected = survivors[order][:keep_n]

    print(
        f"      Hard-gate removed: {len(remove_idx)}/{len(candidate_df)} "
        f"({100 * len(remove_idx) / len(candidate_df):.2f}%)"
    )
    print(f"      Composite-ranked pool: {len(survivors)}, kept: {len(selected)} "
          f"(attack_weight={attack_weight}, realism_weight={realism_weight})")

    return candidate_df.iloc[selected].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_bootstrap_robust_hybrid(
    train_df,
    label_col,
    col_types,
    target_n,
    over_factor=OVER_SAMPLE_FACTOR,
    blend_tvae=BLEND_TVAE,
    attack_weight=ATTACK_WEIGHT,
    realism_weight=REALISM_WEIGHT,
    seed=RANDOM_SEED,
):
    feature_cols = col_types["all_feature"]
    normal_df = train_df.loc[train_df[label_col] == 0, feature_cols].copy().reset_index(drop=True)
    anomaly_df = train_df.loc[train_df[label_col] == 1, feature_cols].copy().reset_index(drop=True)

    n_tvae_target = int(round(target_n * blend_tvae))
    n_smote_target = target_n - n_tvae_target
    n_tvae_raw = int(np.ceil(n_tvae_target * over_factor))

    print(f"    [1/3] Sampling {n_tvae_raw} TVAE candidate points ({over_factor}x factor)...")
    synth_tvae_raw = generate_tvae(normal_df, col_types, n_tvae_raw, seed=seed).drop(columns=["id"], errors="ignore")

    print(f"    [2/3] Applying Bootstrap-Robust, Realism-Dominant Repulsion Gate...")
    synth_tvae_filtered = filter_by_bootstrap_robust_repulsion(
        synth_tvae_raw,
        normal_df,
        anomaly_df,
        col_types["categorical"],
        keep_n=n_tvae_target,
        attack_weight=attack_weight,
        realism_weight=realism_weight,
        seed=seed,
    )

    print(f"    [3/3] Generating {n_smote_target} Cluster-Aware SMOTE anchor samples...")
    synth_smote = generate_improved_cluster_smote(normal_df, col_types, n_smote_target, seed=seed).drop(columns=["id"], errors="ignore")

    combined = pd.concat(
        [synth_tvae_filtered[feature_cols], synth_smote[feature_cols]], ignore_index=True
    )
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if len(combined) != target_n:
        raise RuntimeError(f"Generated {len(combined)} rows; expected exactly {target_n}.")

    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(
    dataset_key="nslkdd", do_eval=True,
    attack_weight=ATTACK_WEIGHT, realism_weight=REALISM_WEIGHT,
):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 22: Bootstrap-Robust Realism-Dominant Hybrid — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")
    print(f"  Attack weight: {attack_weight}, Realism weight: {realism_weight}")

    t0 = time.time()
    synthetic = generate_bootstrap_robust_hybrid(
        train_df, cfg["label_col"], col_types, cfg["target_n"],
        attack_weight=attack_weight, realism_weight=realism_weight,
    )
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  \u2713 Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_22_{dataset_key}.csv")
    synthetic.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    results = {"generation_time": gen_time, "attack_weight": attack_weight, "realism_weight": realism_weight}
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
