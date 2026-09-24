"""
approach_19_refined_a18.py
========================================
# IMPROVEMENT OF APPROACH 17 (v1 / v2)

Hypothesis & Motivation:
  Approach 17 (both v1 and v2) rank TVAE candidates purely by distance to the
  nearest seen attack(s) and keep the farthest ones. In practice this reduces
  to "keep the top-N farthest-from-attack candidates" in both versions once
  there is a comfortable oversample surplus -- the v2 percentile/threshold
  logic rarely changes the final selection because the final ranking step
  still sorts by descending attack-distance.

  The blind spot: maximizing distance-from-attack alone does not guarantee a
  candidate is a *realistic* normal point. TVAE can hallucinate candidates
  that are far from attacks AND far from the real normal manifold (empty /
  unrealistic feature space). Such points are easy for BOTH ECOD and
  IForest to isolate as anomalous during evaluation, since they are just as
  separable from real evaluation-set normals as they are from attacks.

  Approach 18 scores each candidate on two axes instead of one:
    - attack_score:  distance to nearest seen attack        (want LARGE)
    - realism_score: distance to nearest real normal points (want SMALL)

  Candidates are ranked by a composite z-scored objective that rewards being
  simultaneously safe (far from Aseen attacks) and realistic (close to the
  real normal manifold), rather than just "far from everything".

  A conservative hard gate (reused from v2's percentile idea) still removes
  candidates that are unambiguously overlapping the attack distribution
  before the composite ranking is applied.

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

OVER_SAMPLE_FACTOR = 1.5     # TVAE candidate pool size relative to TVAE quota
BLEND_TVAE = 0.50
BLEND_SMOTE = 0.50

# Hard gate: remove candidates whose distance to the nearest seen attack is
# below this percentile of the REAL-NORMAL -> nearest-attack distance
# distribution. This only removes unambiguous attack-overlap; it does not
# drive the main selection (that is the composite score below).
ATTACK_OVERLAP_PERCENTILE = 2.5
MIN_TVAE_RETENTION = 0.85    # never hard-prune more than 15% of the pool

# Weight on the realism term in the composite score.
# score = z(attack_dist) - REALISM_WEIGHT * z(normal_dist)
# Higher weight -> stronger preference for staying near the real normal
# manifold; lower weight -> behaves closer to plain attack-repulsion (v1).
REALISM_WEIGHT = 0.75

# k used for the "how close to the real normal manifold" realism score
REALISM_K = 5

# Final sanity-clip: after combining TVAE + SMOTE, any synthetic point whose
# distance to the real normal manifold exceeds this percentile (of the real
# normal internal nearest-neighbor spread) is considered a stray hallucination
# and is resampled from a real normal row + light jitter instead.
FINAL_CLIP_PERCENTILE = 99.9


# ---------------------------------------------------------------------------
# Encoding / distance-space helpers
# ---------------------------------------------------------------------------

def _fit_normal_scaled_space(normal_df, other_dfs, categorical_cols):
    """
    One-hot encode normal_df plus any number of other frames into a shared
    column space, then fit a StandardScaler on the NORMAL data only (so
    extreme attack/candidate values don't distort the scale).

    Returns the scaler, the fitted categories, the shared column list, and
    the scaled normal matrix.
    """
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
    """Mean distance from each query point to its k nearest reference points."""
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


# ---------------------------------------------------------------------------
# Manifold-balanced anomaly-repulsion gate
# ---------------------------------------------------------------------------

def filter_by_manifold_balanced_repulsion(
    candidate_df,
    normal_df,
    anomaly_df,
    categorical_cols,
    keep_n,
    overlap_percentile=ATTACK_OVERLAP_PERCENTILE,
    min_retention=MIN_TVAE_RETENTION,
    realism_weight=REALISM_WEIGHT,
    realism_k=REALISM_K,
):
    """
    Two-stage candidate selection:
      1. Hard gate: drop candidates that unambiguously overlap the seen
         attack distribution (closer to an attack than real normal points
         typically are, per a conservative percentile threshold).
      2. Composite ranking: among survivors, keep the candidates that best
         balance being far from attacks AND close to the real normal
         manifold (rather than just "farthest from attacks").
    """
    if len(candidate_df) <= keep_n:
        return candidate_df.copy().reset_index(drop=True)

    scaler, fit_categories, all_cols, X_normal = _fit_normal_scaled_space(
        normal_df, [anomaly_df, candidate_df], categorical_cols
    )
    X_anomaly = _transform(anomaly_df, categorical_cols, fit_categories, all_cols, scaler)
    X_candidate = _transform(candidate_df, categorical_cols, fit_categories, all_cols, scaler)

    # --- Stage 1: hard gate on unambiguous attack overlap -----------------
    if len(X_anomaly) > 0:
        normal_attack_dist = _nearest_distance(X_normal, X_anomaly, k=1)
        candidate_attack_dist = _nearest_distance(X_candidate, X_anomaly, k=1)
        threshold = np.percentile(normal_attack_dist, overlap_percentile)
        suspicious_mask = candidate_attack_dist < threshold

        minimum_allowed = int(np.ceil(len(candidate_df) * min_retention))
        maximum_prunable = max(0, len(candidate_df) - minimum_allowed)
        suspicious_idx = np.where(suspicious_mask)[0]

        if len(suspicious_idx) > maximum_prunable:
            # only remove the worst offenders, bounded by the safety floor
            worst_first = suspicious_idx[np.argsort(candidate_attack_dist[suspicious_idx])]
            remove_idx = set(worst_first[:maximum_prunable].tolist())
        else:
            remove_idx = set(suspicious_idx.tolist())
    else:
        candidate_attack_dist = np.full(len(candidate_df), np.inf)
        remove_idx = set()

    survivors = np.array([i for i in range(len(candidate_df)) if i not in remove_idx])

    # --- Stage 2: composite realism + repulsion ranking --------------------
    realism_dist = _nearest_distance(X_candidate[survivors], X_normal, k=realism_k)
    attack_dist_survivors = candidate_attack_dist[survivors]

    composite = _zscore(attack_dist_survivors) - realism_weight * _zscore(realism_dist)
    order = np.argsort(-composite)  # descending: best composite score first

    selected = survivors[order][:keep_n]

    print(
        f"      Hard-gate removed: {len(remove_idx)}/{len(candidate_df)} "
        f"({100 * len(remove_idx) / len(candidate_df):.2f}%)"
    )
    print(f"      Composite-ranked pool: {len(survivors)}, kept: {len(selected)}")

    return candidate_df.iloc[selected].copy().reset_index(drop=True)


def final_sanity_clip(
    combined_df,
    normal_df,
    categorical_cols,
    percentile=FINAL_CLIP_PERCENTILE,
    seed=RANDOM_SEED,
):
    """
    Catch stray hallucinated points that survived both the hard gate and
    the composite ranking (e.g. the isolated far-flung points visible in
    PCA plots of Approach 18 output).

    Threshold is learned from the REAL normal data's own internal nearest-
    neighbor spread (how far a typical real normal point is from its
    nearest real normal neighbor) -- so it flags synthetic points that are
    unusually far from the real manifold relative to the manifold's own
    natural spread, not an arbitrary fixed distance.

    Flagged rows are replaced by a real normal row plus small Gaussian
    jitter on numeric columns (cheap, safe fallback -- keeps exact row
    count and avoids re-running TVAE/SMOTE).
    """
    rng = np.random.RandomState(seed)

    scaler, fit_categories, all_cols, X_normal = _fit_normal_scaled_space(
        normal_df, [combined_df], categorical_cols
    )
    X_combined = _transform(combined_df, categorical_cols, fit_categories, all_cols, scaler)

    # Real normal points' distance to their own nearest neighbor (excluding self)
    nn_self = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=-1).fit(X_normal)
    self_dist, _ = nn_self.kneighbors(X_normal)
    normal_internal_spread = self_dist[:, 1]  # skip distance-to-self (0)

    threshold = np.percentile(normal_internal_spread, percentile)

    candidate_dist = _nearest_distance(X_combined, X_normal, k=1)
    stray_mask = candidate_dist > threshold
    n_stray = int(stray_mask.sum())

    if n_stray == 0:
        print("      No stray outliers detected.")
        return combined_df

    print(f"      Stray outliers detected and repaired: {n_stray}/{len(combined_df)}")

    numeric_cols = [c for c in combined_df.columns if c not in categorical_cols and c != "id"]
    result = combined_df.copy()
    stray_idx = np.where(stray_mask)[0]
    replacement_rows = normal_df.sample(n=n_stray, replace=True, random_state=seed).reset_index(drop=True)

    for pos, idx in enumerate(stray_idx):
        row = replacement_rows.iloc[pos].copy()
        for col in numeric_cols:
            col_std = normal_df[col].std()
            if col_std > 0:
                row[col] = row[col] + rng.normal(0, 0.01 * col_std)
        for col in combined_df.columns:
            if col in row.index:
                result.at[result.index[idx], col] = row[col]

    return result


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_manifold_balanced_hybrid(
    train_df,
    label_col,
    col_types,
    target_n,
    over_factor=OVER_SAMPLE_FACTOR,
    blend_tvae=BLEND_TVAE,
    realism_weight=REALISM_WEIGHT,
    seed=RANDOM_SEED,
):
    """
    Generate synthetic data via Manifold-Balanced Anomaly-Repulsion TVAE
    blended with Cluster-Aware SMOTE.
    """
    feature_cols = col_types["all_feature"]
    normal_df = train_df.loc[train_df[label_col] == 0, feature_cols].copy().reset_index(drop=True)
    anomaly_df = train_df.loc[train_df[label_col] == 1, feature_cols].copy().reset_index(drop=True)

    n_tvae_target = int(round(target_n * blend_tvae))
    n_smote_target = target_n - n_tvae_target
    n_tvae_raw = int(np.ceil(n_tvae_target * over_factor))

    print(f"    [1/3] Sampling {n_tvae_raw} TVAE candidate points ({over_factor}x factor)...")
    synth_tvae_raw = generate_tvae(normal_df, col_types, n_tvae_raw, seed=seed).drop(columns=["id"], errors="ignore")

    print(f"    [2/3] Applying Manifold-Balanced Anomaly-Repulsion Gate...")
    synth_tvae_filtered = filter_by_manifold_balanced_repulsion(
        synth_tvae_raw,
        normal_df,
        anomaly_df,
        col_types["categorical"],
        keep_n=n_tvae_target,
        realism_weight=realism_weight,
    )

    print(f"    [3/3] Generating {n_smote_target} Cluster-Aware SMOTE anchor samples...")
    synth_smote = generate_improved_cluster_smote(normal_df, col_types, n_smote_target, seed=seed).drop(columns=["id"], errors="ignore")

    combined = pd.concat(
        [synth_tvae_filtered[feature_cols], synth_smote[feature_cols]], ignore_index=True
    )
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if len(combined) != target_n:
        raise RuntimeError(f"Generated {len(combined)} rows; expected exactly {target_n}.")

    print(f"    [+] Final sanity clip (stray-outlier repair)...")
    combined = final_sanity_clip(combined, normal_df, col_types["categorical"], seed=seed)

    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True, realism_weight=REALISM_WEIGHT):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 19: Refined Manifold-Balanced Hybrid (with Stray-Outlier Repair) — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")
    print(f"  Realism weight: {realism_weight}")

    t0 = time.time()
    synthetic = generate_manifold_balanced_hybrid(
        train_df, cfg["label_col"], col_types, cfg["target_n"], realism_weight=realism_weight
    )
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  \u2713 Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_19_{dataset_key}.csv")
    synthetic.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    results = {"generation_time": gen_time, "realism_weight": realism_weight}
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