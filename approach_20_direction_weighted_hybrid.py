"""
approach_20_direction_weighted_hybrid.py
=========================================
# REFINEMENT OF APPROACH 18, MOTIVATED BY THE PCA VISUALIZATION

Motivation (from the before/after PCA plots):
  - On NSL-KDD, real attacks form several clusters that are mostly well
    separated from the real normal blob, with one thin attack streak
    overlapping the normal region. Plain Euclidean distance in the
    one-hot/standardized space is already a reasonable separability proxy
    here.
  - On UNSW-NB15, real normal and real attack points sit almost on top of
    each other along the SAME dominant diagonal trend (PC1 vs PC2). This
    means most of the variance in the data is along a direction that does
    NOT separate normal from attack -- so a plain Euclidean distance (as
    used in Approaches 17/18) mostly measures movement along that shared,
    non-discriminative direction, and under-weights the (smaller-variance)
    directions that actually distinguish normal from attack traffic.

  Approach 20 fixes this by learning a *discriminative direction weighting*
  before computing any attack-repulsion / realism distance:
    1. Fit PCA on the combined (real normal + real attack) scaled feature
       space to get a compact set of directions.
    2. For each PCA direction, compute a Fisher-style discriminative weight:
           w_i = |mean_attack_i - mean_normal_i| / (std_normal_i + std_attack_i + eps)
       Directions with large normal/attack mean separation relative to their
       spread get a high weight; shared "background" directions (like
       UNSW's dominant diagonal) get down-weighted automatically.
    3. All distance computations (the hard attack-overlap gate AND the
       composite realism/repulsion ranking from Approach 18) are performed
       in this weighted PCA space instead of raw standardized space.

  Everything else (TVAE oversampling, Cluster-Aware SMOTE blend, hard gate
  + composite ranking structure) is carried over unchanged from Approach 18.
  The Approach-18 "final sanity clip" is kept available but OFF by default,
  since it did not improve the Kaggle score when tested.

  *CRITICAL REQUIREMENT*: Universal pipeline for both datasets. Only the
  learned weights differ per dataset -- the algorithm itself does not.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Hard gate: remove candidates whose (weighted) distance to the nearest seen
# attack is below this percentile of the REAL-NORMAL -> nearest-attack
# weighted-distance distribution.
ATTACK_OVERLAP_PERCENTILE = 2.5
MIN_TVAE_RETENTION = 0.85    # never hard-prune more than 15% of the pool

# score = z(attack_dist) - REALISM_WEIGHT * z(normal_dist), both computed in
# the discriminative-weighted space.
REALISM_WEIGHT = 0.75
REALISM_K = 5

# PCA: keep enough components to explain this much variance of the combined
# normal+attack scaled data (bounded below by a hard cap for speed).
PCA_VARIANCE_TARGET = 0.95
PCA_MAX_COMPONENTS = 40
DISCRIMINATIVE_WEIGHT_EPS = 1e-6

# Optional final cleanup step from Approach 18. Left OFF by default: it did
# not move the Kaggle score when tested, and the flagged points appear to be
# harmless (a handful of rows out of tens of thousands).
ENABLE_FINAL_CLIP = False
FINAL_CLIP_PERCENTILE = 99.9


# ---------------------------------------------------------------------------
# Encoding / discriminative-weighted distance-space helpers
# ---------------------------------------------------------------------------

def _one_hot_align(df, categorical_cols, fit_categories, all_cols):
    encoded, _ = one_hot_encode(df, categorical_cols, fit_categories=fit_categories)
    for c in all_cols:
        if c not in encoded.columns:
            encoded[c] = 0
    return encoded[all_cols]


def fit_discriminative_space(normal_df, anomaly_df, categorical_cols):
    """
    Build the discriminative-weighted PCA space:
      1. One-hot encode normal + attack into a shared column space.
      2. Standard-scale using normal-data statistics only.
      3. Fit PCA on the combined (normal + attack) scaled data.
      4. Compute a Fisher-style discriminative weight per PCA component from
         how well that component separates normal vs. attack means relative
         to their spread.

    Returns a dict of fitted objects + the sqrt(weight)-scaled PCA
    projections of the normal and attack sets, ready for nearest-neighbor
    distance computations.
    """
    normal_encoded, fit_categories = one_hot_encode(normal_df, categorical_cols)
    all_cols = set(normal_encoded.columns)
    anom_encoded_raw, _ = one_hot_encode(anomaly_df, categorical_cols, fit_categories=fit_categories)
    all_cols |= set(anom_encoded_raw.columns)
    all_cols = sorted(all_cols)

    X_normal_raw = _one_hot_align(normal_df, categorical_cols, fit_categories, all_cols).to_numpy(dtype=float)
    X_anom_raw = _one_hot_align(anomaly_df, categorical_cols, fit_categories, all_cols).to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_normal_raw)
    X_normal_scaled = scaler.transform(X_normal_raw)
    X_anom_scaled = scaler.transform(X_anom_raw)

    n_components = min(PCA_MAX_COMPONENTS, X_normal_scaled.shape[1], X_normal_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(np.vstack([X_normal_scaled, X_anom_scaled]))

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cum_var, PCA_VARIANCE_TARGET) + 1)
    n_keep = max(2, min(n_keep, n_components))

    components = pca.components_[:n_keep]
    P_normal = X_normal_scaled @ components.T
    P_anom = X_anom_scaled @ components.T

    mean_normal, mean_anom = P_normal.mean(axis=0), P_anom.mean(axis=0)
    std_normal, std_anom = P_normal.std(axis=0), P_anom.std(axis=0)

    weights = np.abs(mean_anom - mean_normal) / (std_normal + std_anom + DISCRIMINATIVE_WEIGHT_EPS)
    weights = weights / (weights.max() + DISCRIMINATIVE_WEIGHT_EPS)  # normalize to [0, 1]
    sqrt_w = np.sqrt(weights)

    print(
        f"      Discriminative PCA space: kept {n_keep}/{n_components} components "
        f"({100 * cum_var[n_keep - 1]:.1f}% variance), "
        f"weight range [{weights.min():.3f}, {weights.max():.3f}]"
    )

    return {
        "fit_categories": fit_categories,
        "all_cols": all_cols,
        "scaler": scaler,
        "components": components,
        "sqrt_w": sqrt_w,
        "P_normal_weighted": P_normal * sqrt_w,
        "P_anom_weighted": P_anom * sqrt_w,
    }


def project_weighted(df, categorical_cols, space):
    """Project a new dataframe into the fitted discriminative-weighted PCA space."""
    X_raw = _one_hot_align(df, categorical_cols, space["fit_categories"], space["all_cols"]).to_numpy(dtype=float)
    X_scaled = space["scaler"].transform(X_raw)
    P = X_scaled @ space["components"].T
    return P * space["sqrt_w"]


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


# ---------------------------------------------------------------------------
# Direction-weighted anomaly-repulsion gate
# ---------------------------------------------------------------------------

def filter_by_direction_weighted_repulsion(
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
    Same two-stage selection as Approach 18 (hard attack-overlap gate, then
    composite realism + repulsion ranking), but all distances are computed
    in the discriminative-weighted PCA space instead of raw standardized
    one-hot space.
    """
    if len(candidate_df) <= keep_n:
        return candidate_df.copy().reset_index(drop=True)

    space = fit_discriminative_space(normal_df, anomaly_df, categorical_cols)
    P_normal = space["P_normal_weighted"]
    P_anom = space["P_anom_weighted"]
    P_candidate = project_weighted(candidate_df, categorical_cols, space)

    # --- Stage 1: hard gate on unambiguous attack overlap -----------------
    if len(P_anom) > 0:
        normal_attack_dist = _nearest_distance(P_normal, P_anom, k=1)
        candidate_attack_dist = _nearest_distance(P_candidate, P_anom, k=1)
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

    # --- Stage 2: composite realism + repulsion ranking --------------------
    realism_dist = _nearest_distance(P_candidate[survivors], P_normal, k=realism_k)
    attack_dist_survivors = candidate_attack_dist[survivors]

    composite = _zscore(attack_dist_survivors) - realism_weight * _zscore(realism_dist)
    order = np.argsort(-composite)
    selected = survivors[order][:keep_n]

    print(
        f"      Hard-gate removed: {len(remove_idx)}/{len(candidate_df)} "
        f"({100 * len(remove_idx) / len(candidate_df):.2f}%)"
    )
    print(f"      Composite-ranked pool: {len(survivors)}, kept: {len(selected)}")

    return candidate_df.iloc[selected].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Optional final sanity clip (carried over from Approach 18, OFF by default)
# ---------------------------------------------------------------------------

def final_sanity_clip(combined_df, normal_df, categorical_cols, percentile=FINAL_CLIP_PERCENTILE, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    normal_encoded, fit_categories = one_hot_encode(normal_df, categorical_cols)
    all_cols = sorted(normal_encoded.columns)
    scaler = StandardScaler().fit(normal_encoded[all_cols].to_numpy(dtype=float))
    X_normal = scaler.transform(normal_encoded[all_cols].to_numpy(dtype=float))
    X_combined = scaler.transform(
        _one_hot_align(combined_df, categorical_cols, fit_categories, all_cols).to_numpy(dtype=float)
    )

    nn_self = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=-1).fit(X_normal)
    self_dist, _ = nn_self.kneighbors(X_normal)
    threshold = np.percentile(self_dist[:, 1], percentile)

    candidate_dist = _nearest_distance(X_combined, X_normal, k=1)
    stray_idx = np.where(candidate_dist > threshold)[0]
    if len(stray_idx) == 0:
        print("      No stray outliers detected.")
        return combined_df

    print(f"      Stray outliers detected and repaired: {len(stray_idx)}/{len(combined_df)}")
    numeric_cols = [c for c in combined_df.columns if c not in categorical_cols and c != "id"]
    result = combined_df.copy()
    replacement_rows = normal_df.sample(n=len(stray_idx), replace=True, random_state=seed).reset_index(drop=True)
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

def generate_direction_weighted_hybrid(
    train_df,
    label_col,
    col_types,
    target_n,
    over_factor=OVER_SAMPLE_FACTOR,
    blend_tvae=BLEND_TVAE,
    realism_weight=REALISM_WEIGHT,
    enable_final_clip=ENABLE_FINAL_CLIP,
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

    print(f"    [2/3] Applying Direction-Weighted Anomaly-Repulsion Gate...")
    synth_tvae_filtered = filter_by_direction_weighted_repulsion(
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

    if enable_final_clip:
        print(f"    [+] Final sanity clip (stray-outlier repair)...")
        combined = final_sanity_clip(combined, normal_df, col_types["categorical"], seed=seed)

    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True, realism_weight=REALISM_WEIGHT):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 20: Direction-Weighted Anomaly-Repulsion Hybrid — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")
    print(f"  Realism weight: {realism_weight}")

    t0 = time.time()
    synthetic = generate_direction_weighted_hybrid(
        train_df, cfg["label_col"], col_types, cfg["target_n"], realism_weight=realism_weight
    )
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  \u2713 Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_20_{dataset_key}.csv")
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
