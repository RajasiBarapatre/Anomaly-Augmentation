"""
approach_23_conditional_statistical_manifold.py
===============================================

Approach 23 — Conditional Statistical Manifold Generator

Motivation
----------
Approaches 18–22 progressively improved the *selection* of TVAE/SMOTE
candidates, with Approach 20 giving the strongest UNSW result in the supplied
experiments. The next step is to change the generator rather than adding
another selection heuristic.

This pipeline is deliberately more statistics-heavy:

    50% density-aware Cluster-SMOTE
    30% cluster-conditioned rank / Gaussian dependency generator
    20% TVAE

The statistical generator is trained on REAL NORMAL DATA ONLY. Attack data is
used only by the final Approach-20-style candidate selector. This preserves
the normal-only generation requirement while allowing attack-aware selection.

Key idea
--------
A single global Gaussian copula performed poorly on UNSW. Approach 23 therefore
does NOT fit one global dependency model. Instead:

  1. Encode numeric features with StandardScaler and categoricals with one-hot.
  2. Partition real normals into local clusters with MiniBatchKMeans.
  3. Within each cluster, model numeric dependencies in rank/PIT space.
  4. Sample a correlated Gaussian latent vector.
  5. Map latent values through empirical numeric quantiles, preserving the
     observed marginal/tail shape.
  6. Sample categorical values conditional on the local cluster.
  7. Generate a large mixed candidate pool.
  8. Apply Approach-20 discriminative weighted PCA selection to the COMPLETE
     candidate pool, not just TVAE rows.

This is intended as a clean experiment: it tests whether better statistical
coverage plus local dependency modelling can beat the 50/50 TVAE+SMOTE
generator without adding another deep model.

Universal pipeline
------------------
No dataset-specific feature names, transforms, or constants are hard-coded.
The same algorithm is used for NSL-KDD and UNSW-NB15.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
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
from local_evaluator import evaluate_synthetic, compute_quality_metrics
from approach_4_tvae import generate_tvae
from approach_15_improved_cluster_smote import generate_improved_cluster_smote

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Final target mixture. These are generator proportions before final
# candidate selection.
BLEND_SMOTE = 0.50
BLEND_STATISTICAL = 0.30
BLEND_TVAE = 0.20

# Generate a modest surplus, then select the final target_n from the complete
# candidate pool. This lets the statistical candidates participate in the same
# attack-aware selection as TVAE candidates.
CANDIDATE_OVER_SAMPLE_FACTOR = 1.25

# Local statistical manifold.
MAX_CLUSTERS = 24
MIN_CLUSTER_SIZE = 80
KMEANS_BATCH_SIZE = 4096
KMEANS_N_INIT = 5

# Rank/correlation model.
CORRELATION_SHRINKAGE = 0.15
MIN_EIGENVALUE = 1e-5

# Final Approach-20-style selector.
ATTACK_OVERLAP_PERCENTILE = 2.5
MIN_CANDIDATE_RETENTION = 0.85

# Keep realism meaningful, but do not make it dominate attack geometry.
REALISM_WEIGHT = 0.75
REALISM_K = 5

PCA_VARIANCE_TARGET = 0.95
PCA_MAX_COMPONENTS = 40
DISCRIMINATIVE_WEIGHT_EPS = 1e-6

ENABLE_FINAL_CLIP = False
FINAL_CLIP_PERCENTILE = 99.9


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _nearest_distance(X_query, X_reference, k=1):
    if len(X_reference) == 0:
        return np.full(len(X_query), np.inf, dtype=float)

    k_eff = min(k, len(X_reference))
    nn = NearestNeighbors(
        n_neighbors=k_eff,
        algorithm="auto",
        n_jobs=-1,
    )
    nn.fit(X_reference)
    distances, _ = nn.kneighbors(X_query)
    return distances.mean(axis=1)


def _zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma


def _safe_covariance(X, shrinkage=CORRELATION_SHRINKAGE):
    """
    Return a numerically stable correlation-like covariance matrix.

    X is assumed to be approximately standard normal already.
    Shrinkage toward the identity prevents small local clusters from producing
    unstable covariance matrices.
    """
    d = X.shape[1]

    if d == 1:
        return np.ones((1, 1), dtype=float)

    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov).astype(float)

    if cov.shape != (d, d) or not np.all(np.isfinite(cov)):
        cov = np.eye(d)

    diag = np.sqrt(np.maximum(np.diag(cov), MIN_EIGENVALUE))
    corr = cov / np.outer(diag, diag)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(d)
    corr = 0.5 * (corr + corr.T)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, MIN_EIGENVALUE)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Re-normalize diagonal approximately to 1.
    diag = np.sqrt(np.maximum(np.diag(corr), MIN_EIGENVALUE))
    corr = corr / np.outer(diag, diag)
    corr = 0.5 * (corr + corr.T)

    return corr


def _nearest_psd_sqrt(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, MIN_EIGENVALUE)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _rank_to_normal_scores(values):
    """
    Empirical PIT -> approximately standard-normal scores.

    Ties are handled by average ranks. Values are clipped away from 0/1 so
    inverse-normal sampling remains finite.
    """
    from scipy.stats import rankdata, norm

    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=float)

    ranks = rankdata(values, method="average")
    u = (ranks - 0.5) / n
    u = np.clip(u, 1e-4, 1.0 - 1e-4)
    return norm.ppf(u)


def _normal_to_empirical_quantile(z, reference_values):
    """
    Map standard-normal latent samples back through the empirical quantile
    function of the reference values.

    np.quantile provides interpolation, so the generator is not restricted to
    exact row copies.
    """
    from scipy.stats import norm

    reference_values = np.asarray(reference_values, dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]

    if len(reference_values) == 0:
        return np.zeros(len(z), dtype=float)

    if len(reference_values) == 1 or np.std(reference_values) < 1e-12:
        return np.full(len(z), reference_values[0], dtype=float)

    sorted_values = np.sort(reference_values)
    u = norm.cdf(z)
    u = np.clip(u, 1e-5, 1.0 - 1e-5)

    grid = np.linspace(0.0, 1.0, len(sorted_values))
    return np.interp(u, grid, sorted_values)


# ---------------------------------------------------------------------------
# Cluster-conditioned statistical generator
# ---------------------------------------------------------------------------

class ConditionalStatisticalGenerator:
    """
    Local empirical-marginal + rank-dependency generator.

    The model is fit ONLY on normal_df.

    Cluster assignment is based on a compact mixed-type representation.
    Numeric dependencies are estimated within each cluster. Categorical
    variables are sampled conditionally on cluster membership.
    """

    def __init__(
        self,
        numeric_cols,
        categorical_cols,
        max_clusters=MAX_CLUSTERS,
        min_cluster_size=MIN_CLUSTER_SIZE,
        random_state=RANDOM_SEED,
    ):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state

        self.scaler = None
        self.fit_categories = None
        self.all_encoded_cols = None
        self.cluster_model = None
        self.cluster_ids = None
        self.cluster_sizes = None
        self.cluster_models = {}

    def _build_cluster_features(self, df, fit=False):
        """
        Numeric features are standardized. Categorical columns are one-hot
        encoded. This representation is only for local clustering; generation
        itself uses the original feature values.
        """
        pieces = []

        if self.numeric_cols:
            numeric = df[self.numeric_cols].to_numpy(dtype=float)

            if fit:
                self.scaler = StandardScaler().fit(numeric)

            numeric_scaled = self.scaler.transform(numeric)
            pieces.append(numeric_scaled)

        if self.categorical_cols:
            encoded, cats = one_hot_encode(
                df,
                self.categorical_cols,
                fit_categories=None if fit else self.fit_categories,
            )

            if fit:
                self.fit_categories = cats
                self.all_encoded_cols = list(encoded.columns)

            for col in self.all_encoded_cols:
                if col not in encoded.columns:
                    encoded[col] = 0

            pieces.append(
                encoded[self.all_encoded_cols].to_numpy(dtype=float)
            )

        if not pieces:
            return np.zeros((len(df), 1), dtype=float)

        X = np.hstack(pieces)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, normal_df):
        normal_df = normal_df.reset_index(drop=True)

        X = self._build_cluster_features(normal_df, fit=True)

        if len(normal_df) < 2:
            self.cluster_model = None
            self.cluster_ids = np.zeros(len(normal_df), dtype=int)
            self.cluster_sizes = {0: len(normal_df)}
            self._fit_cluster_model(0, normal_df)
            return self

        # Number of clusters is constrained by both sample count and the
        # minimum useful local sample size.
        n_by_size = max(1, len(normal_df) // self.min_cluster_size)
        n_clusters = min(self.max_clusters, n_by_size)

        # Very small datasets should remain a single statistical population.
        n_clusters = max(1, min(n_clusters, len(normal_df)))

        if n_clusters == 1:
            self.cluster_model = None
            self.cluster_ids = np.zeros(len(normal_df), dtype=int)
        else:
            self.cluster_model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                batch_size=min(KMEANS_BATCH_SIZE, len(normal_df)),
                n_init=KMEANS_N_INIT,
            )
            self.cluster_ids = self.cluster_model.fit_predict(X)

        self.cluster_sizes = {
            int(cid): int(np.sum(self.cluster_ids == cid))
            for cid in np.unique(self.cluster_ids)
        }

        for cid in sorted(self.cluster_sizes):
            cluster_df = normal_df.loc[self.cluster_ids == cid].reset_index(drop=True)

            # If MiniBatchKMeans creates a tiny cluster, merge its generation
            # model into the nearest sufficiently large global fallback by
            # fitting directly to the cluster but borrowing global marginals
            # later when necessary.
            self._fit_cluster_model(cid, cluster_df)

        print(
            f"      Conditional statistical model: "
            f"{len(self.cluster_sizes)} local clusters, "
            f"cluster size range "
            f"[{min(self.cluster_sizes.values())}, {max(self.cluster_sizes.values())}]"
        )

        return self

    def _fit_cluster_model(self, cid, cluster_df):
        model = {
            "n": len(cluster_df),
            "normal_df": cluster_df.copy(),
            "corr": None,
            "sqrt_corr": None,
            "latent_numeric": None,
            "categorical_probs": {},
        }

        if self.numeric_cols:
            X = cluster_df[self.numeric_cols].to_numpy(dtype=float)

            latent_columns = []
            for j in range(X.shape[1]):
                col = _rank_to_normal_scores(X[:, j])
                latent_columns.append(col)

            latent = np.column_stack(latent_columns)
            latent = np.nan_to_num(latent, nan=0.0, posinf=0.0, neginf=0.0)

            if len(cluster_df) >= max(10, len(self.numeric_cols) * 3):
                corr = _safe_covariance(latent)
            else:
                corr = np.eye(len(self.numeric_cols))

            model["corr"] = corr
            model["sqrt_corr"] = _nearest_psd_sqrt(corr)
            model["latent_numeric"] = latent

        for col in self.categorical_cols:
            values = cluster_df[col]

            counts = values.value_counts(dropna=False)

            if len(counts) == 0:
                model["categorical_probs"][col] = None
                continue

            probs = counts.to_numpy(dtype=float)
            probs = probs / probs.sum()

            model["categorical_probs"][col] = (
                counts.index.to_list(),
                probs,
            )

        self.cluster_models[int(cid)] = model

    def _sample_cluster_ids(self, n, rng):
        cluster_ids = np.array(sorted(self.cluster_sizes.keys()), dtype=int)
        sizes = np.array([self.cluster_sizes[c] for c in cluster_ids], dtype=float)
        probs = sizes / sizes.sum()
        return rng.choice(cluster_ids, size=n, replace=True, p=probs)

    def sample(self, n, seed=None):
        if n <= 0:
            return pd.DataFrame(columns=self.numeric_cols + self.categorical_cols)

        rng = np.random.RandomState(
            self.random_state if seed is None else seed
        )

        cluster_ids = self._sample_cluster_ids(n, rng)
        result = pd.DataFrame(index=np.arange(n))

        for cid in np.unique(cluster_ids):
            positions = np.where(cluster_ids == cid)[0]
            m = len(positions)

            model = self.cluster_models[int(cid)]

            if self.numeric_cols:
                d = len(self.numeric_cols)

                if model["n"] >= max(10, d * 3):
                    eps = rng.normal(size=(m, d))
                    latent = eps @ model["sqrt_corr"].T
                else:
                    latent = rng.normal(size=(m, d))

                ref = model["normal_df"]

                for j, col in enumerate(self.numeric_cols):
                    generated = _normal_to_empirical_quantile(
                        latent[:, j],
                        ref[col].to_numpy(dtype=float),
                    )
                    result.loc[positions, col] = generated

            for col in self.categorical_cols:
                values_probs = model["categorical_probs"].get(col)

                if values_probs is None:
                    continue

                values, probs = values_probs
                sampled = rng.choice(values, size=m, replace=True, p=probs)
                result.loc[positions, col] = sampled

        # Restore column ordering and original categorical dtypes as far as
        # possible. The repository's validation layer remains authoritative.
        return result[self.numeric_cols + self.categorical_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Approach-20 discriminative selection
# ---------------------------------------------------------------------------

def _one_hot_align(df, categorical_cols, fit_categories, all_cols):
    encoded, _ = one_hot_encode(
        df,
        categorical_cols,
        fit_categories=fit_categories,
    )

    for c in all_cols:
        if c not in encoded.columns:
            encoded[c] = 0

    return encoded[all_cols]


def fit_discriminative_space(normal_df, anomaly_df, categorical_cols):
    normal_encoded, fit_categories = one_hot_encode(
        normal_df,
        categorical_cols,
    )

    all_cols = set(normal_encoded.columns)

    anomaly_encoded, _ = one_hot_encode(
        anomaly_df,
        categorical_cols,
        fit_categories=fit_categories,
    )

    all_cols |= set(anomaly_encoded.columns)
    all_cols = sorted(all_cols)

    X_normal_raw = _one_hot_align(
        normal_df,
        categorical_cols,
        fit_categories,
        all_cols,
    ).to_numpy(dtype=float)

    X_anomaly_raw = _one_hot_align(
        anomaly_df,
        categorical_cols,
        fit_categories,
        all_cols,
    ).to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_normal_raw)

    X_normal = scaler.transform(X_normal_raw)
    X_anomaly = scaler.transform(X_anomaly_raw)

    n_components = min(
        PCA_MAX_COMPONENTS,
        X_normal.shape[1],
        max(1, X_normal.shape[0] - 1),
    )

    if n_components <= 1:
        components = np.eye(X_normal.shape[1])[:1]
        P_normal = X_normal[:, :1]
        P_anomaly = X_anomaly[:, :1]
        cum_var = np.array([1.0])
        n_keep = 1
    else:
        pca = PCA(
            n_components=n_components,
            random_state=RANDOM_SEED,
        )
        pca.fit(np.vstack([X_normal, X_anomaly]))

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(
            np.searchsorted(cum_var, PCA_VARIANCE_TARGET) + 1
        )
        n_keep = max(2, min(n_keep, n_components))

        components = pca.components_[:n_keep]
        P_normal = X_normal @ components.T
        P_anomaly = X_anomaly @ components.T

    mean_normal = P_normal.mean(axis=0)
    mean_anomaly = P_anomaly.mean(axis=0)

    std_normal = P_normal.std(axis=0)
    std_anomaly = P_anomaly.std(axis=0)

    weights = np.abs(mean_anomaly - mean_normal) / (
        std_normal + std_anomaly + DISCRIMINATIVE_WEIGHT_EPS
    )

    weights = weights / (
        np.max(weights) + DISCRIMINATIVE_WEIGHT_EPS
    )

    sqrt_w = np.sqrt(weights)

    explained = (
        float(cum_var[n_keep - 1])
        if len(cum_var)
        else 1.0
    )

    print(
        f"      Discriminative PCA: {n_keep} components, "
        f"{100 * explained:.1f}% variance, "
        f"weight range [{weights.min():.3f}, {weights.max():.3f}]"
    )

    return {
        "fit_categories": fit_categories,
        "all_cols": all_cols,
        "scaler": scaler,
        "components": components,
        "sqrt_w": sqrt_w,
        "P_normal_weighted": P_normal * sqrt_w,
        "P_anomaly_weighted": P_anomaly * sqrt_w,
    }


def project_weighted(df, categorical_cols, space):
    X_raw = _one_hot_align(
        df,
        categorical_cols,
        space["fit_categories"],
        space["all_cols"],
    ).to_numpy(dtype=float)

    X_scaled = space["scaler"].transform(X_raw)
    P = X_scaled @ space["components"].T

    return P * space["sqrt_w"]


def filter_candidates(
    candidate_df,
    normal_df,
    anomaly_df,
    categorical_cols,
    keep_n,
    overlap_percentile=ATTACK_OVERLAP_PERCENTILE,
    min_retention=MIN_CANDIDATE_RETENTION,
    realism_weight=REALISM_WEIGHT,
    realism_k=REALISM_K,
):
    """
    Apply the useful part of Approach 20 to ALL candidate generators.

    Stage 1:
      Conservative attack-overlap gate.

    Stage 2:
      z(attack distance) - realism_weight * z(normal-manifold distance).

    Unlike Approaches 18–22, candidates are not separated by generator type
    during selection. This lets the statistical generator compete directly
    with TVAE and SMOTE.
    """
    if len(candidate_df) <= keep_n:
        return candidate_df.copy().reset_index(drop=True)

    space = fit_discriminative_space(
        normal_df,
        anomaly_df,
        categorical_cols,
    )

    P_normal = space["P_normal_weighted"]
    P_anomaly = space["P_anomaly_weighted"]
    P_candidate = project_weighted(
        candidate_df,
        categorical_cols,
        space,
    )

    if len(P_anomaly):
        normal_attack_dist = _nearest_distance(
            P_normal,
            P_anomaly,
            k=1,
        )

        candidate_attack_dist = _nearest_distance(
            P_candidate,
            P_anomaly,
            k=1,
        )

        threshold = np.percentile(
            normal_attack_dist,
            overlap_percentile,
        )

        suspicious = candidate_attack_dist < threshold

        maximum_prunable = max(
            0,
            len(candidate_df)
            - int(np.ceil(len(candidate_df) * min_retention)),
        )

        suspicious_idx = np.where(suspicious)[0]

        if len(suspicious_idx) > maximum_prunable:
            worst = suspicious_idx[
                np.argsort(candidate_attack_dist[suspicious_idx])
            ]
            remove_idx = set(
                worst[:maximum_prunable].tolist()
            )
        else:
            remove_idx = set(suspicious_idx.tolist())
    else:
        candidate_attack_dist = np.full(
            len(candidate_df),
            np.inf,
        )
        remove_idx = set()

    survivors = np.array(
        [
            i
            for i in range(len(candidate_df))
            if i not in remove_idx
        ],
        dtype=int,
    )

    realism_dist = _nearest_distance(
        P_candidate[survivors],
        P_normal,
        k=realism_k,
    )

    attack_dist = candidate_attack_dist[survivors]

    composite = (
        _zscore(attack_dist)
        - realism_weight * _zscore(realism_dist)
    )

    order = np.argsort(-composite)
    selected = survivors[order][:keep_n]

    print(
        f"      Candidate gate removed "
        f"{len(remove_idx)}/{len(candidate_df)} "
        f"({100 * len(remove_idx) / len(candidate_df):.2f}%)"
    )
    print(
        f"      Selected {len(selected)} from "
        f"{len(survivors)} surviving candidates"
    )

    return candidate_df.iloc[selected].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Optional sanity clip
# ---------------------------------------------------------------------------

def final_sanity_clip(
    combined_df,
    normal_df,
    categorical_cols,
    percentile=FINAL_CLIP_PERCENTILE,
    seed=RANDOM_SEED,
):
    """
    Kept for controlled ablation only.

    It is OFF by default because the supplied experiments showed that the
    previous final clipping stage did not improve the leaderboard score.
    """
    rng = np.random.RandomState(seed)

    normal_encoded, fit_categories = one_hot_encode(
        normal_df,
        categorical_cols,
    )

    all_cols = sorted(normal_encoded.columns)

    scaler = StandardScaler().fit(
        normal_encoded[all_cols].to_numpy(dtype=float)
    )

    X_normal = scaler.transform(
        normal_encoded[all_cols].to_numpy(dtype=float)
    )

    X_combined = scaler.transform(
        _one_hot_align(
            combined_df,
            categorical_cols,
            fit_categories,
            all_cols,
        ).to_numpy(dtype=float)
    )

    nn_self = NearestNeighbors(
        n_neighbors=min(2, len(X_normal)),
        algorithm="auto",
        n_jobs=-1,
    ).fit(X_normal)

    self_dist, _ = nn_self.kneighbors(X_normal)

    if self_dist.shape[1] < 2:
        return combined_df

    threshold = np.percentile(
        self_dist[:, 1],
        percentile,
    )

    candidate_dist = _nearest_distance(
        X_combined,
        X_normal,
        k=1,
    )

    stray_idx = np.where(candidate_dist > threshold)[0]

    if len(stray_idx) == 0:
        print("      No stray outliers detected.")
        return combined_df

    print(
        f"      Stray outliers detected: "
        f"{len(stray_idx)}/{len(combined_df)}"
    )

    result = combined_df.copy()

    numeric_cols = [
        c
        for c in combined_df.columns
        if c not in categorical_cols and c != "id"
    ]

    replacements = normal_df.sample(
        n=len(stray_idx),
        replace=True,
        random_state=seed,
    ).reset_index(drop=True)

    for pos, idx in enumerate(stray_idx):
        row = replacements.iloc[pos].copy()

        for col in numeric_cols:
            if col in normal_df.columns:
                std = normal_df[col].std()
                if std > 0:
                    row[col] += rng.normal(
                        0,
                        0.01 * std,
                    )

        for col in combined_df.columns:
            if col in row.index:
                result.at[result.index[idx], col] = row[col]

    return result


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_conditional_statistical_hybrid(
    train_df,
    label_col,
    col_types,
    target_n,
    seed=RANDOM_SEED,
    realism_weight=REALISM_WEIGHT,
):
    feature_cols = col_types["all_feature"]
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]

    normal_df = (
        train_df.loc[
            train_df[label_col] == 0,
            feature_cols,
        ]
        .copy()
        .reset_index(drop=True)
    )

    anomaly_df = (
        train_df.loc[
            train_df[label_col] == 1,
            feature_cols,
        ]
        .copy()
        .reset_index(drop=True)
    )

    candidate_n = int(
        np.ceil(
            target_n * CANDIDATE_OVER_SAMPLE_FACTOR
        )
    )

    # Allocate the candidate pool using the intended generator mixture.
    n_smote = int(
        round(candidate_n * BLEND_SMOTE)
    )

    n_stat = int(
        round(candidate_n * BLEND_STATISTICAL)
    )

    n_tvae = candidate_n - n_smote - n_stat

    print(
        f"    Candidate pool: {candidate_n} "
        f"(SMOTE={n_smote}, statistical={n_stat}, TVAE={n_tvae})"
    )

    # ------------------------------------------------------------------
    # 1. Cluster-aware SMOTE
    # ------------------------------------------------------------------
    print(
        "    [1/4] Generating density-aware Cluster-SMOTE candidates..."
    )

    synth_smote = (
        generate_improved_cluster_smote(
            normal_df,
            col_types,
            n_smote,
            seed=seed,
        )
        .drop(columns=["id"], errors="ignore")
    )

    # ------------------------------------------------------------------
    # 2. Conditional statistical generator
    # ------------------------------------------------------------------
    print(
        "    [2/4] Fitting cluster-conditioned statistical generator..."
    )

    stat_generator = ConditionalStatisticalGenerator(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        max_clusters=MAX_CLUSTERS,
        min_cluster_size=MIN_CLUSTER_SIZE,
        random_state=seed,
    )

    stat_generator.fit(normal_df)

    print(
        "    [2/4] Sampling local rank/dependency candidates..."
    )

    synth_stat = stat_generator.sample(
        n_stat,
        seed=seed + 101,
    )

    # ------------------------------------------------------------------
    # 3. TVAE minority novelty component
    # ------------------------------------------------------------------
    print(
        "    [3/4] Generating TVAE novelty candidates..."
    )

    synth_tvae = (
        generate_tvae(
            normal_df,
            col_types,
            n_tvae,
            seed=seed + 202,
        )
        .drop(columns=["id"], errors="ignore")
    )

    # ------------------------------------------------------------------
    # 4. Combine + Approach-20 selection
    # ------------------------------------------------------------------
    print(
        "    [4/4] Combining generators and applying "
        "Approach-20 discriminative selection..."
    )

    candidates = pd.concat(
        [
            synth_smote[feature_cols],
            synth_stat[feature_cols],
            synth_tvae[feature_cols],
        ],
        ignore_index=True,
    )

    # Shuffle before selection so generator order cannot affect tie-breaking.
    candidates = candidates.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    selected = filter_candidates(
        candidates,
        normal_df,
        anomaly_df,
        categorical_cols,
        keep_n=target_n,
        realism_weight=realism_weight,
    )

    if len(selected) != target_n:
        raise RuntimeError(
            f"Generated {len(selected)} rows; "
            f"expected exactly {target_n}."
        )

    return add_id_column(selected)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(
    dataset_key="nslkdd",
    do_eval=True,
    realism_weight=REALISM_WEIGHT,
):
    cfg = DATASETS[dataset_key]

    print("\n" + "=" * 72)
    print(
        f"Approach 23: Conditional Statistical Manifold Hybrid — "
        f"{cfg['name']}"
    )
    print("=" * 72)

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
        f"  Normal training rows: {len(normal_train)}"
    )
    print(
        f"  Target synthetic rows: {cfg['target_n']}"
    )
    print(
        f"  Generator mix: "
        f"SMOTE={BLEND_SMOTE:.0%}, "
        f"statistical={BLEND_STATISTICAL:.0%}, "
        f"TVAE={BLEND_TVAE:.0%}"
    )
    print(
        f"  Candidate oversampling: "
        f"{CANDIDATE_OVER_SAMPLE_FACTOR:.2f}x"
    )

    t0 = time.time()

    synthetic = generate_conditional_statistical_hybrid(
        train_df,
        cfg["label_col"],
        col_types,
        cfg["target_n"],
        seed=RANDOM_SEED,
        realism_weight=realism_weight,
    )

    gen_time = time.time() - t0

    print(
        f"  Generation time: {gen_time:.1f}s"
    )

    validate_synthetic(
        synthetic,
        col_types,
        cfg["target_n"],
    )

    print("  ✓ Validation passed")

    out_path = os.path.join(
        OUTPUT_DIR,
        f"approach_23_{dataset_key}.csv",
    )

    synthetic.to_csv(
        out_path,
        index=False,
    )

    print(
        f"  Saved to {out_path}"
    )

    results = {
        "generation_time": gen_time,
        "realism_weight": realism_weight,
        "blend_smote": BLEND_SMOTE,
        "blend_statistical": BLEND_STATISTICAL,
        "blend_tvae": BLEND_TVAE,
    }

    if do_eval:
        print("\n  Evaluating...")

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

        results.update(eval_results)
        results["mean_ks_distance"] = quality[
            "mean_ks_distance"
        ]
        results["correlation_rmse"] = quality[
            "correlation_rmse"
        ]

        print(
            f"  Score S: {results.get('score', 'n/a')}"
        )
        print(
            f"  Mean KS: {results['mean_ks_distance']:.5f}"
        )
        print(
            f"  Corr RMSE: {results['correlation_rmse']:.5f}"
        )

    return synthetic, results


if __name__ == "__main__":
    for key in DATASETS:
        run(key)
