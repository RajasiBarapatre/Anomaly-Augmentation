"""
approach_17_anomaly_pruned_hybrid_v2.py
=======================================

Conservative Anomaly-Repulsion TVAE + Cluster-SMOTE Hybrid

Motivation
----------
Synthetic normal samples that substantially overlap with known attack
regions may weaken the downstream ECOD / Isolation Forest boundary and
increase false negatives.

However, hidden attacks may differ from the seen attack distribution.
Therefore, we DO NOT remove a fixed percentage of synthetic samples.

Instead:
    1. Train TVAE only on normal training data.
    2. Generate an oversampled candidate pool.
    3. Measure each candidate's distance to the nearest seen attack.
    4. Measure the same distance for real normal training samples.
    5. Define the attack-overlap region using a conservative percentile
       of the real-normal distance distribution.
    6. Remove only synthetic candidates that are unusually close to
       known attacks.
    7. Fill any missing TVAE quota using the safest remaining candidates.
    8. Combine with Cluster-Aware SMOTE.
    9. Return exactly target_n rows.

Important:
---------
The attack distribution is used only as a conservative boundary signal.
It is NOT used to model the synthetic normal distribution.
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
    load_dataset,
    local_train_eval_split,
    get_normal_data,
    validate_synthetic,
    add_id_column,
    one_hot_encode,
)

from local_evaluator import (
    evaluate_synthetic,
    compute_quality_metrics,
)

from approach_4_tvae import generate_tvae
from approach_15_improved_cluster_smote import (
    generate_improved_cluster_smote,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OVER_SAMPLE_FACTOR = 1.50

BLEND_TVAE = 0.50
BLEND_SMOTE = 0.50

# Conservative threshold.
#
# A synthetic point is considered suspicious only if its distance to the
# nearest seen attack is below this percentile of the REAL NORMAL distance
# distribution.
#
# Smaller percentile -> more conservative pruning.
#
# Examples:
#   1.0  -> remove only extreme overlap
#   2.5  -> conservative
#   5.0  -> moderate
#
ATTACK_OVERLAP_PERCENTILE = 2.5

# Minimum number of TVAE samples we require after pruning.
#
# This prevents the attack gate from completely dominating generation.
MIN_TVAE_RETENTION = 0.85


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_for_distance(
    candidate_df,
    reference_df,
    categorical_cols,
):
    """
    Encode candidate and reference data into a common numerical space.

    Categorical columns are one-hot encoded using categories learned from
    the candidate data, then missing columns are aligned.
    """

    candidate_encoded, fit_categories = one_hot_encode(
        candidate_df,
        categorical_cols,
    )

    reference_encoded, _ = one_hot_encode(
        reference_df,
        categorical_cols,
        fit_categories=fit_categories,
    )

    all_cols = sorted(
        set(candidate_encoded.columns)
        | set(reference_encoded.columns)
    )

    for col in all_cols:

        if col not in candidate_encoded.columns:
            candidate_encoded[col] = 0

        if col not in reference_encoded.columns:
            reference_encoded[col] = 0

    candidate_encoded = candidate_encoded[all_cols]
    reference_encoded = reference_encoded[all_cols]

    return (
        candidate_encoded.to_numpy(dtype=float),
        reference_encoded.to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def _fit_distance_space(
    normal_df,
    anomaly_df,
    categorical_cols,
):
    """
    Construct a standardized feature space for distance calculations.

    IMPORTANT:
    ----------
    Scaling parameters are fitted on NORMAL data rather than the pooled
    normal + anomaly data.

    This prevents extreme attack values from determining the scale of
    normal features.
    """

    normal_encoded, fit_categories = one_hot_encode(
        normal_df,
        categorical_cols,
    )

    anomaly_encoded, _ = one_hot_encode(
        anomaly_df,
        categorical_cols,
        fit_categories=fit_categories,
    )

    all_cols = sorted(
        set(normal_encoded.columns)
        | set(anomaly_encoded.columns)
    )

    for col in all_cols:

        if col not in normal_encoded.columns:
            normal_encoded[col] = 0

        if col not in anomaly_encoded.columns:
            anomaly_encoded[col] = 0

    normal_encoded = normal_encoded[all_cols]
    anomaly_encoded = anomaly_encoded[all_cols]

    X_normal = normal_encoded.to_numpy(dtype=float)
    X_anomaly = anomaly_encoded.to_numpy(dtype=float)

    # Fit scaler ONLY on normal observations.
    scaler = StandardScaler()
    scaler.fit(X_normal)

    X_normal_scaled = scaler.transform(X_normal)
    X_anomaly_scaled = scaler.transform(X_anomaly)

    return (
        scaler,
        fit_categories,
        all_cols,
        X_normal_scaled,
        X_anomaly_scaled,
    )


def _transform_candidates(
    candidate_df,
    categorical_cols,
    fit_categories,
    all_cols,
    scaler,
):
    """
    Transform candidate observations into the previously fitted
    normal-based distance space.
    """

    candidate_encoded, _ = one_hot_encode(
        candidate_df,
        categorical_cols,
        fit_categories=fit_categories,
    )

    for col in all_cols:

        if col not in candidate_encoded.columns:
            candidate_encoded[col] = 0

    candidate_encoded = candidate_encoded[all_cols]

    X_candidate = candidate_encoded.to_numpy(dtype=float)

    return scaler.transform(X_candidate)


def _nearest_attack_distance(
    X_query,
    X_attack,
):
    """
    Distance from every query point to its nearest seen attack.

    We deliberately use nearest-attack distance rather than averaging
    several neighbours.

    A point overlapping an attack cluster should be considered suspicious
    even if it happens to have other attacks relatively far away.
    """

    if len(X_attack) == 0:
        return np.full(
            len(X_query),
            np.inf,
            dtype=float,
        )

    n_neighbors = 1

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto",
        n_jobs=-1,
    )

    nn.fit(X_attack)

    distances, _ = nn.kneighbors(X_query)

    return distances[:, 0]


# ---------------------------------------------------------------------------
# Conservative anomaly-repulsion gate
# ---------------------------------------------------------------------------

def filter_by_conservative_anomaly_repulsion(
    candidate_df,
    normal_df,
    anomaly_df,
    categorical_cols,
    keep_n,
    overlap_percentile=ATTACK_OVERLAP_PERCENTILE,
    min_retention=MIN_TVAE_RETENTION,
):
    """
    Conservatively remove synthetic candidates that overlap unusually
    strongly with the seen attack distribution.

    The threshold is learned from the distribution of distances between
    REAL NORMAL samples and the nearest seen attack.

    This is fundamentally different from:
        "remove the closest 30% of synthetic points."

    Instead:
        "remove synthetic points whose attack proximity is unusually
         extreme compared with legitimate normal observations."
    """

    if len(candidate_df) <= keep_n:
        return candidate_df.copy().reset_index(drop=True)

    if len(anomaly_df) == 0:
        return candidate_df.iloc[:keep_n].copy().reset_index(drop=True)

    (
        scaler,
        fit_categories,
        all_cols,
        X_normal,
        X_anomaly,
    ) = _fit_distance_space(
        normal_df,
        anomaly_df,
        categorical_cols,
    )

    X_candidate = _transform_candidates(
        candidate_df,
        categorical_cols,
        fit_categories,
        all_cols,
        scaler,
    )

    # ---------------------------------------------------------------
    # Distance of REAL normal observations to seen attacks
    # ---------------------------------------------------------------

    normal_attack_dist = _nearest_attack_distance(
        X_normal,
        X_anomaly,
    )

    # ---------------------------------------------------------------
    # Distance of SYNTHETIC candidates to seen attacks
    # ---------------------------------------------------------------

    candidate_attack_dist = _nearest_attack_distance(
        X_candidate,
        X_anomaly,
    )

    # ---------------------------------------------------------------
    # Learn conservative threshold
    # ---------------------------------------------------------------

    threshold = np.percentile(
        normal_attack_dist,
        overlap_percentile,
    )

    # Only points below this threshold are considered suspicious.
    suspicious_mask = candidate_attack_dist < threshold

    suspicious_count = int(suspicious_mask.sum())

    # ---------------------------------------------------------------
    # Safety floor
    # ---------------------------------------------------------------

    minimum_allowed = int(
        np.ceil(
            len(candidate_df) * min_retention
        )
    )

    maximum_prunable = max(
        0,
        len(candidate_df) - minimum_allowed,
    )

    if suspicious_count > maximum_prunable:

        # If too many candidates fall into the suspicious region,
        # only remove the most attack-overlapping ones.
        suspicious_indices = np.where(
            suspicious_mask
        )[0]

        suspicious_sorted = suspicious_indices[
            np.argsort(
                candidate_attack_dist[suspicious_indices]
            )
        ]

        remove_indices = suspicious_sorted[
            :maximum_prunable
        ]

    else:

        remove_indices = np.where(
            suspicious_mask
        )[0]

    remove_set = set(remove_indices.tolist())

    keep_indices = np.array([
        i
        for i in range(len(candidate_df))
        if i not in remove_set
    ])

    # ---------------------------------------------------------------
    # If enough candidates survive, rank by attack distance only
    # as a tie-breaking / safety mechanism.
    # ---------------------------------------------------------------

    if len(keep_indices) >= keep_n:

        # Prefer points farther from attacks, but DO NOT remove
        # everything close to attacks indiscriminately.
        ordered = keep_indices[
            np.argsort(
                -candidate_attack_dist[keep_indices]
            )
        ]

        selected = ordered[:keep_n]

    else:

        # This should be rare because of the retention floor.
        #
        # If it occurs, fill the quota using the safest candidates
        # that were removed.
        remaining_indices = np.array([
            i
            for i in range(len(candidate_df))
            if i not in keep_indices
        ])

        selected = list(keep_indices)

        remaining_sorted = remaining_indices[
            np.argsort(
                -candidate_attack_dist[remaining_indices]
            )
        ]

        required = keep_n - len(selected)

        selected.extend(
            remaining_sorted[:required]
        )

        selected = np.asarray(selected)

    selected = selected[:keep_n]

    result = (
        candidate_df
        .iloc[selected]
        .copy()
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------

    actual_removed = len(candidate_df) - len(result)

    print(
        "      Attack-distance threshold: "
        f"{threshold:.4f}"
    )

    print(
        "      Suspicious candidates: "
        f"{suspicious_count}/{len(candidate_df)} "
        f"({100 * suspicious_count / len(candidate_df):.2f}%)"
    )

    print(
        "      Candidates removed: "
        f"{actual_removed}/{len(candidate_df)} "
        f"({100 * actual_removed / len(candidate_df):.2f}%)"
    )

    print(
        "      TVAE candidates retained: "
        f"{len(result)}/{len(candidate_df)}"
    )

    return result


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_anomaly_pruned_hybrid(
    train_df,
    label_col,
    col_types,
    target_n,
    over_factor=OVER_SAMPLE_FACTOR,
    seed=RANDOM_SEED,
    overlap_percentile=ATTACK_OVERLAP_PERCENTILE,
):
    """
    Generate:

        Conservative anomaly-repelled TVAE
        +
        Cluster-Aware SMOTE

    while maintaining exactly target_n rows.
    """

    feature_cols = col_types["all_feature"]

    # ---------------------------------------------------------------
    # Separate normal and seen attack observations
    # ---------------------------------------------------------------

    normal_df = (
        train_df
        .loc[
            train_df[label_col] == 0,
            feature_cols
        ]
        .copy()
        .reset_index(drop=True)
    )

    anomaly_df = (
        train_df
        .loc[
            train_df[label_col] == 1,
            feature_cols
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Exact mixture allocation
    # ---------------------------------------------------------------

    n_tvae_target = int(
        round(target_n * BLEND_TVAE)
    )

    n_smote_target = (
        target_n - n_tvae_target
    )

    n_tvae_raw = int(
        np.ceil(
            n_tvae_target * over_factor
        )
    )

    print(
        f"    Target: {target_n}"
    )

    print(
        f"    TVAE target: {n_tvae_target}"
    )

    print(
        f"    SMOTE target: {n_smote_target}"
    )

    print(
        f"    TVAE candidate pool: {n_tvae_raw}"
    )

    # ---------------------------------------------------------------
    # 1. Generate TVAE candidates
    # ---------------------------------------------------------------

    print(
        f"\n    [1/3] Generating "
        f"{n_tvae_raw} TVAE candidates..."
    )

    synth_tvae_raw = generate_tvae(
        normal_df,
        col_types,
        n_tvae_raw,
        seed=seed,
    )

    synth_tvae_raw = synth_tvae_raw.drop(
        columns=["id"],
        errors="ignore",
    )

    # ---------------------------------------------------------------
    # 2. Conservative anomaly gate
    # ---------------------------------------------------------------

    print(
        "\n    [2/3] Applying conservative "
        "anomaly-repulsion gate..."
    )

    synth_tvae_filtered = (
        filter_by_conservative_anomaly_repulsion(
            candidate_df=synth_tvae_raw,
            normal_df=normal_df,
            anomaly_df=anomaly_df,
            categorical_cols=col_types["categorical"],
            keep_n=n_tvae_target,
            overlap_percentile=overlap_percentile,
        )
    )

    # ---------------------------------------------------------------
    # 3. Cluster-Aware SMOTE
    # ---------------------------------------------------------------

    print(
        f"\n    [3/3] Generating "
        f"{n_smote_target} Cluster-Aware SMOTE samples..."
    )

    synth_smote = (
        generate_improved_cluster_smote(
            normal_df,
            col_types,
            n_smote_target,
            seed=seed,
        )
        .drop(
            columns=["id"],
            errors="ignore",
        )
    )

    # ---------------------------------------------------------------
    # Ensure exact column order
    # ---------------------------------------------------------------

    synth_tvae_filtered = (
        synth_tvae_filtered[feature_cols]
    )

    synth_smote = synth_smote[feature_cols]

    # ---------------------------------------------------------------
    # Combine
    # ---------------------------------------------------------------

    combined = pd.concat(
        [
            synth_tvae_filtered,
            synth_smote,
        ],
        ignore_index=True,
    )

    # Shuffle mixture components.
    combined = (
        combined
        .sample(
            frac=1.0,
            random_state=seed,
        )
        .reset_index(drop=True)
    )

    # Safety check.
    if len(combined) != target_n:
        raise RuntimeError(
            f"Generated {len(combined)} rows; "
            f"expected exactly {target_n}."
        )

    return add_id_column(combined)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    dataset_key="nslkdd",
    do_eval=True,
    overlap_percentile=ATTACK_OVERLAP_PERCENTILE,
):
    cfg = DATASETS[dataset_key]

    print(
        f"\n{'=' * 70}"
    )

    print(
        "Approach 17 v2: "
        "Conservative Anomaly-Repelled TVAE + "
        f"Cluster-SMOTE — {cfg['name']}"
    )

    print(
        f"{'=' * 70}"
    )

    df, col_types = load_dataset(cfg)

    train_df, eval_df = local_train_eval_split(
        df,
        cfg["label_col"],
    )

    normal_train = get_normal_data(
        train_df,
        cfg["label_col"],
    )

    print(
        f"  Normal training rows: "
        f"{len(normal_train)}"
    )

    print(
        f"  Target synthetic rows: "
        f"{cfg['target_n']}"
    )

    print(
        f"  Attack-overlap percentile: "
        f"{overlap_percentile}"
    )

    # ---------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------

    t0 = time.time()

    synthetic = generate_anomaly_pruned_hybrid(
        train_df=train_df,
        label_col=cfg["label_col"],
        col_types=col_types,
        target_n=cfg["target_n"],
        overlap_percentile=overlap_percentile,
    )

    gen_time = time.time() - t0

    print(
        f"\n  Generation time: "
        f"{gen_time:.1f}s"
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    validate_synthetic(
        synthetic,
        col_types,
        cfg["target_n"],
    )

    print(
        "  ✓ Validation passed"
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    out_path = os.path.join(
        OUTPUT_DIR,
        f"approach_17_v2_{dataset_key}.csv",
    )

    synthetic.to_csv(
        out_path,
        index=False,
    )

    print(
        f"  Saved to {out_path}"
    )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    results = {
        "generation_time": gen_time,
        "attack_overlap_percentile":
            overlap_percentile,
    }

    if do_eval:

        print(
            "\n  Evaluating..."
        )

        eval_results = evaluate_synthetic(
            synthetic,
            eval_df,
            cfg["categorical_cols"],
            cfg["label_col"],
        )

        quality = compute_quality_metrics(
            synthetic,
            normal_train,
            col_types,
        )

        results.update(
            eval_results
        )

        results["mean_ks_distance"] = (
            quality["mean_ks_distance"]
        )

        results["correlation_rmse"] = (
            quality["correlation_rmse"]
        )

    return synthetic, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    for key in DATASETS:

        run(
            dataset_key=key,
            overlap_percentile=
                ATTACK_OVERLAP_PERCENTILE,
        )