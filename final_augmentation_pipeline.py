'''
Command to run:
    python3 final_augmentation_pipeline.py --mode submission 
'''
from ast import For
import os
import sys
import time
import random
import json
from pathlib import Path
from scipy.stats import rankdata
from scipy.special import ndtr, ndtri

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from revised_experiment_config import (
    DATASETS,
    RANDOM_SEED,
    OUTPUT_DIR,
)

from revised_preprocessing import (
    load_dataset,
    local_train_eval_split,
    get_normal_data,
    validate_synthetic,
    add_id_column,
)

from revised_local_evaluator import (
    evaluate_synthetic,
    compute_quality_metrics,
)

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

#   Candidate pool is larger than the requested output
CANDIDATE_OVER_SAMPLE_FACTORS = {
    "nslkdd": 1.5,
    "unsw": 4.0,
}

GENERATION_SEEDS = {
    "nslkdd": 42,
    "unsw": 42,
}

REALISM_WEIGHTS = {
    "nslkdd": 0.75,
    "unsw": 1.0,     
}

#   Per-dataset base generator mixture (statistical copula, cluster-SMOTENC)
#   Mixture changes after adaptive strength applied
BASE_MIXTURES = {
    "nslkdd": (0.50, 0.50),
    "unsw": (0, 1.0),
}

#   Data-driven attack-coverage pre-filter on candidates
COVERAGE_FILTER = {
    "nslkdd": False,    
    "unsw": False,    
}
COVERAGE_ATTACK_MASS = 0.995
COVERAGE_MAX_NORMAL_SHARE = 0.60
COVERAGE_MAX_CARDINALITY = 16

#   TVAE is disabled. Kept for reference
USE_TVAE = {
    "nslkdd": False,
    "unsw": False,
}

#   Empirical is disabled. Kept for reference
EMPIRICAL_SHARES = {
    "nslkdd": 0.0,
    "unsw": 0.0,
}

MIXTURE_ADAPT_STRENGTHS = {
    "nslkdd": 0.25,
    "unsw": 0.45,
}

ATTACK_OVERLAP_PERCENTILES = {
    "nslkdd": 5.0,
    "unsw": 2.5,
}

#   Zone-based RFOD term in candidate selection
#   HYBRID_NEAR_FRACTION: share of gated candidates (closest to seen attacks) where RFOD applies
#   HYBRID_RFOD_WEIGHT: strength of the RFOD penalty inside that zone (0 disables the hybrid)
HYBRID_ENABLED = True
HYBRID_NEAR_FRACTION = 0.5
HYBRID_RFOD_WEIGHT = 1.0

HYBRID_RFOD_WEIGHTS = {
    "unsw": 1.0,
    "nslkdd": 0.0,             
}

RFOD_N_TREES = 10
RFOD_MIN_SAMPLES_LEAF = 10
RFOD_MAX_FEATURES = 0.5
RFOD_SEED = 0

#   Local clustering
MAX_CLUSTERS = 24
MIN_CLUSTER_SIZE = 80
KMEANS_BATCH_SIZE = 2048
KMEANS_N_INIT = 5
CLUSTER_STABILITY_RUNS = 4
CLUSTER_K_RADIUS = 2
STABILITY_TOLERANCE = 0.02
SILHOUETTE_SAMPLE_SIZE = 5000
TVAE_POOL_OVERSAMPLE = 1.25

#   Generator mixture is adapted around this baseline
#   A TVAE floor prevents adaptive routing from eliminating this generator in dense/stable clusters while retaining local adaptation
BASE_STATISTICAL = 0.50
BASE_SMOTE = 0.50
BASE_TVAE = 0.0


MIN_TVAE_WEIGHT = 0.0           

#   Statistical generator
CORR_SHRINKAGE = 0.15
MIN_STD = 1e-8


ATTACK_REWARD_CAP_PERCENTILE = 95.0
LOCAL_SCALE_FLOOR_PERCENTILE = 10.0

PCA_VARIANCE = 0.95
REALISM_K = 5
ATTACK_OVERLAP_PERCENTILE = 2.5

#   Final sanity clip
FINAL_CLIP = False 
FINAL_CLIP_PERCENTILE = 99.9


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def split_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def safe_numeric(df, cols):
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = out[c].median()
        out[c] = out[c].fillna(0.0 if pd.isna(med) else med)
    return out

# ============================================================
# PREPROCESSING
# ============================================================

def make_preprocessor(df, numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            ("num", StandardScaler(), numeric_cols)
        )

    if categorical_cols:
        try:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )
        transformers.append(("cat", ohe, categorical_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    ct.fit(df)
    return ct

def encode_for_geometry(df, preprocessor):
    x = preprocessor.transform(df)
    return np.asarray(x, dtype=float)

# ============================================================
# LOCAL CLUSTERING
# ============================================================

#   Measure variance captured by the first two PCs of mixed-type geometry
def measure_concentration(x_normal, sample_cap=5000, random_state=RANDOM_STATE):
    n = len(x_normal)
    if n < 10:
        return 1.0

    if n > sample_cap:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_cap, replace=False)
        sample = x_normal[idx]
    else:
        sample = x_normal

    n_comp = min(2, sample.shape[1])
    if n_comp < 1:
        return 1.0

    pca = PCA(n_components=n_comp, random_state=random_state)
    pca.fit(sample)
    return float(np.sum(pca.explained_variance_ratio_))

#   Concentration-aware heuristic used to define a search band
def choose_base_cluster_count(n, concentration=None):
    if n < MIN_CLUSTER_SIZE * 2:
        return 1

    k = int(np.sqrt(max(n, 1) / 250.0))

    if concentration is not None:
        scale = float(np.clip(1.3 - concentration, 0.5, 1.3))
        k = int(round(k * scale))

    k = max(2, k)
    k = min(MAX_CLUSTERS, k)
    k = min(k, max(2, n // MIN_CLUSTER_SIZE))
    return k

def _candidate_k_values(n, base_k):
    if base_k <= 1:
        return [1]
    lo = max(2, base_k - CLUSTER_K_RADIUS)
    hi = min(MAX_CLUSTERS, base_k + CLUSTER_K_RADIUS, n // MIN_CLUSTER_SIZE)
    return list(range(lo, max(lo, hi) + 1))

#   Estimate cluster stability over several independent K-means starts
#   Stability is the mean pairwise Adjusted Rand Index (ARI)
#   Silhouette is used as a secondary criterion among k values whose stability is close to the best observed value
def _cluster_stability_and_silhouette(x, k, random_state):
    if k <= 1:
        return 1.0, 0.0, None, None

    labels_runs = []
    models = []
    for r in range(CLUSTER_STABILITY_RUNS):
        seed = int(random_state + 100 * r + 17 * k)
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=min(KMEANS_BATCH_SIZE, max(256, len(x))),
            n_init=1,
        )
        labels = km.fit_predict(x)
        labels_runs.append(labels)
        models.append(km)

    if len(labels_runs) >= 2:
        scores = []
        for i in range(len(labels_runs)):
            for j in range(i + 1, len(labels_runs)):
                scores.append(
                    adjusted_rand_score(labels_runs[i], labels_runs[j])
                )
        stability = float(np.mean(scores))
    else:
        stability = 1.0

    rng = np.random.default_rng(random_state + 999 + k)
    if len(x) > SILHOUETTE_SAMPLE_SIZE:
        idx = rng.choice(len(x), size=SILHOUETTE_SAMPLE_SIZE, replace=False)
        xs = x[idx]
        ls = labels_runs[0][idx]
    else:
        xs = x
        ls = labels_runs[0]

    try:
        sil = float(silhouette_score(xs, ls, metric="euclidean"))
    except Exception:
        sil = -1.0

    final_km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=min(KMEANS_BATCH_SIZE, max(256, len(x))),
        n_init=KMEANS_N_INIT,
    )
    final_labels = final_km.fit_predict(x)
    return stability, sil, final_km, final_labels

def fit_local_clusters(x_normal, random_state=RANDOM_STATE):
    n = len(x_normal)
    concentration = measure_concentration(
        x_normal, random_state=random_state
    )
    base_k = choose_base_cluster_count(n, concentration=concentration)

    if base_k <= 1:
        labels = np.zeros(n, dtype=int)
        centers = np.mean(x_normal, axis=0, keepdims=True)
        print(
            f"      Mixed-type PC1+PC2 concentration: {concentration:.3f} "
            f"-> cluster count: 1"
        )
        return None, labels, centers

    candidates = _candidate_k_values(n, base_k)
    results = []
    for k in candidates:
        stability, sil, km, labels = _cluster_stability_and_silhouette(
            x_normal, k, random_state
        )
        results.append({
            "k": k,
            "stability": stability,
            "silhouette": sil,
            "km": km,
            "labels": labels,
        })

    max_stability = max(r["stability"] for r in results)
    stable = [
        r for r in results
        if r["stability"] >= max_stability - STABILITY_TOLERANCE
    ]
    chosen = max(
        stable,
        key=lambda r: (
            r["silhouette"],
            -abs(r["k"] - base_k),
            -r["k"],
        )
    )

    print(
        f"      Mixed-type PC1+PC2 concentration: {concentration:.3f} "
        f"-> base k: {base_k}, stable k: {chosen['k']} "
        f"(stability={chosen['stability']:.3f}, "
        f"silhouette={chosen['silhouette']:.3f})"
    )

    return chosen["km"], chosen["labels"], chosen["km"].cluster_centers_

# ============================================================
# STATISTICAL LOCAL GENERATOR
# ============================================================

#   Average ranks preserve ties
#   Constant marginals have no latent variance
def rank_normal_scores(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1 or np.ptp(values) == 0:
        return np.zeros(len(values))
    return ndtri((rankdata(values, method="average") - 0.5) / len(values))

#   Generalized inverse empirical CDF preserves discrete support
def inverse_rank_normal(scores, reference):
    ref = np.sort(np.asarray(reference, dtype=float))
    if not len(ref):
        return np.zeros_like(scores, dtype=float)
    probabilities = np.clip(ndtr(np.asarray(scores, dtype=float)), 0.0, 1.0)
    if np.all(np.abs(ref - np.rint(ref)) < 1e-9):
        
        indices = np.minimum((probabilities * len(ref)).astype(int), len(ref)-1)
        return ref[indices]
    return np.quantile(ref, probabilities)

def nearest_psd(cov, eps=1e-6):
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0

    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T

def gaussian_local_numeric_sample(cluster_df, numeric_cols, n_samples,
                                  random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)

    if n_samples <= 0:
        return pd.DataFrame(columns=numeric_cols)

    x = safe_numeric(cluster_df, numeric_cols).to_numpy(float)

    if len(x) == 0:
        raise ValueError("Cannot generate numeric candidates from an empty cluster.")

    if len(numeric_cols) == 0:
        return pd.DataFrame(index=np.arange(n_samples), columns=numeric_cols)

    if len(x) < 3:
        if len(x) < 2:
            raise ValueError(
                "A singleton cluster has no interpolation partner; refusing to copy its only row."
            )
        anchors = rng.integers(0, len(x), size=n_samples)
        partners = 1 - anchors if len(x) == 2 else rng.integers(0, len(x), size=n_samples)
        alpha = rng.uniform(0.05, 0.95, size=n_samples)
        values = x[anchors] + alpha[:, None] * (x[partners] - x[anchors])
        for j, col in enumerate(numeric_cols):
            ref = x[:, j]
            if np.allclose(ref, np.rint(ref)):
                values[:, j] = np.rint(values[:, j])
        return pd.DataFrame(values, columns=numeric_cols)

    #   Rank-normalize each marginal
    z = np.zeros_like(x)
    for j in range(x.shape[1]):
        z[:, j] = rank_normal_scores(x[:, j])

    #   Shrunk local correlation/covariance
    if z.shape[1] == 1:
        cov = np.array([[1.0]])
    else:
        cov = np.cov(z, rowvar=False)
        cov = np.atleast_2d(cov)
        diag = np.diag(np.diag(cov))
        cov = (1.0 - CORR_SHRINKAGE) * cov + CORR_SHRINKAGE * diag

    active = np.ptp(x, axis=0) > 0
    sampled_z = np.zeros((n_samples, x.shape[1]))
    if active.any():
        active_cov = nearest_psd(cov[np.ix_(active, active)])
        sampled_z[:, active] = rng.multivariate_normal(
            np.mean(z[:, active], axis=0), active_cov, size=n_samples
        )

    out = {}
    for j, c in enumerate(numeric_cols):
        out[c] = inverse_rank_normal(sampled_z[:, j], x[:, j])

    return pd.DataFrame(out)

def categorical_local_sample(cluster_df, categorical_cols, n_samples,
                             random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    out = pd.DataFrame(index=np.arange(n_samples))

    for c in categorical_cols:
        s = cluster_df[c].astype("object")
        counts = s.value_counts(dropna=False)

        if len(counts) == 0:
            out[c] = np.nan
            continue

        values = counts.index.to_list()
        probs = counts.to_numpy(float)
        probs = probs / probs.sum()

        out[c] = rng.choice(values, size=n_samples, p=probs)

    return out

def statistical_generate_cluster(cluster_df, numeric_cols, categorical_cols,
                                  n_samples, random_state=RANDOM_STATE):
    if n_samples <= 0:
        return pd.DataFrame(columns=cluster_df.columns)
    if len(cluster_df) < 3:
        return _cluster_local_knn_fallback(
            cluster_df, numeric_cols, categorical_cols, n_samples,
            random_state=random_state,
        )
    num = gaussian_local_numeric_sample(
        cluster_df, numeric_cols, n_samples, random_state
    )
    if categorical_cols and n_samples and len(cluster_df):
        rng = np.random.default_rng(random_state + 1)
        if numeric_cols:
            scaler = StandardScaler().fit(safe_numeric(cluster_df, numeric_cols))
            reference = scaler.transform(safe_numeric(cluster_df, numeric_cols))
            query = scaler.transform(num[numeric_cols])
            neighbors = NearestNeighbors(n_neighbors=min(5, len(cluster_df))).fit(reference)
            donor_options = neighbors.kneighbors(query, return_distance=False)
            donor = donor_options[np.arange(n_samples), rng.integers(0, donor_options.shape[1], n_samples)]
        else:
            donor = rng.integers(0, len(cluster_df), n_samples)
        cat = cluster_df.iloc[donor][categorical_cols].reset_index(drop=True)
    else:
        cat = pd.DataFrame(index=np.arange(n_samples), columns=categorical_cols)

    pieces = []
    if numeric_cols:
        pieces.append(num[numeric_cols])
    if categorical_cols:
        pieces.append(cat[categorical_cols])

    if not pieces:
        return pd.DataFrame(index=np.arange(n_samples))

    return pd.concat(pieces, axis=1)

# ============================================================
# CLUSTER-CONDITIONED SMOTENC
# ============================================================

#   Integer-encode categorical columns for SMOTENC and return reverse maps
def _encode_categorical_for_smote(df, categorical_cols):
    work = df.copy().reset_index(drop=True)
    reverse_maps = {}

    for c in categorical_cols:
        values = pd.Series(work[c].astype("object").fillna("__MISSING__").unique())
        mapping = {value: i for i, value in enumerate(values.tolist())}
        reverse_maps[c] = {i: value for value, i in mapping.items()}
        work[c] = (
            work[c]
            .astype("object")
            .fillna("__MISSING__")
            .map(mapping)
            .astype(int)
        )

    return work, reverse_maps

#   Restore original categorical values after SMOTENC generation
def _decode_categorical_from_smote(df, categorical_cols, reverse_maps):
    out = df.copy()
    for c in categorical_cols:
        out[c] = out[c].round().astype(int).map(reverse_maps[c])

        if out[c].isna().any():
            fallback = next(iter(reverse_maps[c].values()))
            out[c] = out[c].fillna(fallback)
    return out

#   Generate new local rows by interpolating within identical categorical strata.
#   Used only when SMOTENC cannot fill a cluster quota or a cluster is too small for the statistical generator
#   Exact source-row resampling is forbidden. 
#   A stratum with fewer than two records cannot support interpolation and raises.
def _cluster_local_knn_fallback(cluster_df, numeric_cols, categorical_cols,
                                n_samples, random_state=RANDOM_STATE,
                                geometry=None):
    if n_samples <= 0:
        return pd.DataFrame(columns=cluster_df.columns)

    work = cluster_df.copy().reset_index(drop=True)
    if len(work) < 2:
        raise ValueError(
            "Cannot make a copy-free fallback from a cluster with fewer than two rows."
        )
    if not numeric_cols:
        raise ValueError(
            "Copy-free interpolation requires at least one numeric feature."
        )

    rng = np.random.default_rng(random_state)
    num_work = safe_numeric(work, numeric_cols)
    scaler = StandardScaler().fit(num_work)
    numeric_geometry = scaler.transform(num_work)

    if categorical_cols:
        group_key = work[categorical_cols].astype("string").fillna("<NA>").agg("\x1f".join, axis=1)
    else:
        group_key = pd.Series("__ALL__", index=work.index)

    groups = [np.asarray(idx, dtype=int) for _, idx in group_key.groupby(group_key).groups.items()]
    groups = [idx for idx in groups if len(idx) >= 2]
    if not groups:
        raise ValueError(
            "No categorical stratum has two normal rows; refusing to copy or mix incompatible categories."
        )

    group_sizes = np.asarray([len(idx) for idx in groups], dtype=float)
    group_probs = group_sizes / group_sizes.sum()
    anchors = np.empty(n_samples, dtype=int)
    partners = np.empty(n_samples, dtype=int)

    # Pre-fit local neighbours independently within each categorical stratum
    nn_by_group = []
    for idx in groups:
        k = min(5, len(idx) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(numeric_geometry[idx])
        nn_by_group.append((idx, nn, k))

    chosen_groups = rng.choice(len(groups), size=n_samples, p=group_probs)
    for g_idx, (idx, nn, k) in enumerate(nn_by_group):
        rows = np.flatnonzero(chosen_groups == g_idx)
        if not len(rows):
            continue
        local_anchor_pos = rng.integers(0, len(idx), size=len(rows))
        local_neighbors = nn.kneighbors(
            numeric_geometry[idx[local_anchor_pos]], return_distance=False
        )
        choice = rng.integers(1, k + 1, size=len(rows))
        local_partner_pos = local_neighbors[np.arange(len(rows)), choice]
        anchors[rows] = idx[local_anchor_pos]
        partners[rows] = idx[local_partner_pos]

    alpha = rng.uniform(0.05, 0.95, size=n_samples)
    out = pd.DataFrame(index=np.arange(n_samples), columns=work.columns)
    int_cols = [c for c in numeric_cols if np.allclose(
        pd.to_numeric(work[c], errors="coerce"),
        np.rint(pd.to_numeric(work[c], errors="coerce"))
    )]
    binary_cols = [c for c in numeric_cols if set(
        pd.to_numeric(work[c], errors="coerce").dropna().unique().tolist()
    ).issubset({0, 1})]

    for c in numeric_cols:
        a = pd.to_numeric(work.iloc[anchors][c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(work.iloc[partners][c], errors="coerce").to_numpy(float)
        if c in binary_cols:
            vals = np.where(rng.random(n_samples) < alpha, b, a)
        else:
            vals = a + alpha * (b - a)
            if c in int_cols:
                vals = np.rint(vals)
        out[c] = vals

    for c in categorical_cols:
        out[c] = work.iloc[anchors][c].to_numpy()

    return out[work.columns].reset_index(drop=True)

#   Generate requested additional samples per normal cluster using cluster-conditioned SMOTENC
#   SMOTENC only interpolates between observations belonging to the same learned normal data cluster
def cluster_conditioned_smotenc_generate(
    normal_df,
    labels,
    quota_by_cluster,
    numeric_cols,
    categorical_cols,
    random_state=RANDOM_STATE,
):
    from imblearn.over_sampling import SMOTENC, SMOTE
    normal_df = normal_df.reset_index(drop=True)
    labels = np.asarray(labels, dtype=int)
    if len(normal_df) != len(labels):
        raise ValueError("normal_df and labels must have the same length")

    quotas = {
        int(k): max(0, int(v))
        for k, v in quota_by_cluster.items()
    }
    generated = {
        k: pd.DataFrame(columns=normal_df.columns)
        for k in quotas
    }

    counts = pd.Series(labels).value_counts().to_dict()
    strategy = {}
    for k, quota in quotas.items():
        if quota <= 0:
            continue
        size = int(counts.get(k, 0))
        #   SMOTENC cannot generate from a singleton cluster
        if size >= 2:
            strategy[k] = size + quota

    smotenc_generated = 0
    smotenc_clusters = 0
    smotenc_by_cluster = {int(k): 0 for k in quotas}
    fallback_by_cluster = {int(k): 0 for k in quotas}
    sampler_failed = False

    if strategy and len(np.unique(labels)) >= 2:
        work = normal_df[numeric_cols + categorical_cols].copy()

        for c in numeric_cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")
            med = work[c].median()
            work[c] = work[c].fillna(0.0 if pd.isna(med) else med)

        numeric_scaler = StandardScaler().fit(work[numeric_cols]) if numeric_cols else None
        if numeric_scaler is not None:
            work[numeric_cols] = numeric_scaler.transform(work[numeric_cols])
        encoded, reverse_maps = _encode_categorical_for_smote(
            work, categorical_cols
        )

        cat_idx = [encoded.columns.get_loc(c) for c in categorical_cols]
        min_class_size = min(int(counts[k]) for k in strategy)
        k_neighbors = max(1, min(5, min_class_size - 1))

        try:
            if categorical_cols:
                sampler = SMOTENC(
                    categorical_features=cat_idx,
                    sampling_strategy=strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                )
            else:
                sampler = SMOTE(
                    sampling_strategy=strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                )

            x_res, y_res = sampler.fit_resample(encoded, labels)
            x_res = pd.DataFrame(x_res, columns=encoded.columns)
            y_res = np.asarray(y_res, dtype=int)

            #   imbalanced-learn appends generated samples after the original observations for this over-sampling operation
            x_syn = x_res.iloc[len(normal_df):].reset_index(drop=True)
            y_syn = y_res[len(normal_df):]

            if categorical_cols:
                x_syn = _decode_categorical_from_smote(
                    x_syn, categorical_cols, reverse_maps
                )

            if numeric_scaler is not None:
                x_syn[numeric_cols] = numeric_scaler.inverse_transform(x_syn[numeric_cols])
            x_syn = x_syn[normal_df.columns].copy()

            for k, quota in quotas.items():
                if quota <= 0 or k not in strategy:
                    continue
                idx = np.where(y_syn == k)[0]
                if len(idx) == 0:
                    continue
                take = idx[:quota]
                generated[k] = x_syn.iloc[take].copy().reset_index(drop=True)
                smotenc_generated += len(generated[k])
                smotenc_by_cluster[int(k)] += len(generated[k])
                if len(generated[k]) > 0:
                    smotenc_clusters += 1

        except Exception as exc:
            print(f"      Cluster-SMOTENC warning: {exc}")
            sampler_failed = True

    #   Any cluster that could not be handled by SMOTENC is filled with a local same category KNN interpolation fallback
    #   Singleton/unsupported strata fail closed rather than copying an observed row
    fallback_generated = 0
    for k, quota in quotas.items():
        if quota <= 0:
            continue
        deficit = max(0, quota - len(generated[k]))
        if deficit == 0:
            continue

        idx = np.where(labels == k)[0]
        cluster_df = normal_df.iloc[idx].reset_index(drop=True)
        fallback = _cluster_local_knn_fallback(
            cluster_df,
            numeric_cols,
            categorical_cols,
            deficit,
            random_state + 5000 + int(k),
            geometry=None,
        )

        if len(fallback):
            if len(generated[k]):
                generated[k] = pd.concat(
                    [generated[k], fallback], ignore_index=True
                )
            else:
                generated[k] = fallback.reset_index(drop=True)
            fallback_generated += len(fallback)
            fallback_by_cluster[int(k)] += len(fallback)

    return generated, {
        "generated_by_smotenc": int(smotenc_generated),
        "fallback_generated": int(fallback_generated),
        "smotenc_by_cluster": smotenc_by_cluster,
        "fallback_by_cluster": fallback_by_cluster,
        "smotenc_clusters": int(smotenc_clusters),
        "sampler_failed": bool(sampler_failed),
    }

# ============================================================
# TVAE -> REMOVED FROM FINAL PIPELINE (SETTINGS DISABLED); CODE KEPT FOR REFERENCE
# ============================================================

#   Generate synthetic rows with explicit mixed-type metadata
def tvae_generate(normal_df, n_samples, random_state=RANDOM_STATE):
    if n_samples <= 0:
        return pd.DataFrame(columns=normal_df.columns)

    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    set_seed(random_state)
    normal_df = normal_df.copy()

    numeric_cols = normal_df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = [
        c for c in normal_df.columns if c not in numeric_cols
    ]

    for col in numeric_cols:
        converted = pd.to_numeric(normal_df[col], errors="coerce")

        if converted.isna().any() and normal_df[col].notna().any():
            raise ValueError(
                f"Training column {col!r} contains invalid numeric values."
            )

        normal_df[col] = converted

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(normal_df)

    for col in numeric_cols:
        metadata.update_column(
            column_name=col,
            sdtype="numerical",
        )

    for col in categorical_cols:
        metadata.update_column(
            column_name=col,
            sdtype="categorical",
        )

    print("TVAE numeric columns:", numeric_cols)
    print("TVAE categorical columns:", categorical_cols)

    if "ct_state_ttl" in numeric_cols:
        print(
            "ct_state_ttl metadata:",
            metadata.to_dict()["columns"]["ct_state_ttl"],
        )

    model = TVAESynthesizer(
        metadata,
        epochs=300,
        enforce_min_max_values=True,
        enforce_rounding=True,
    )

    model.fit(normal_df)
    synth = model.sample(num_rows=n_samples)

    synth = synth[normal_df.columns].copy()

    for col in numeric_cols:
        converted = pd.to_numeric(synth[col], errors="coerce")
        invalid = converted.isna() & synth[col].notna()

        if invalid.any():
            examples = synth.loc[invalid, col].astype(str).unique()[:10]
            raise ValueError(
                f"TVAE generated non-numeric values in {col!r}: "
                f"{examples.tolist()}"
            )

        synth[col] = converted

    return synth

# ============================================================
# ADAPTIVE MIXTURE
# ============================================================

#   Estimate local structure from the normal data only
#   The returned features are dataset-agnostic:
#     - density: relative local cluster size
#     - stability: inverse robust dispersion
#     - skewness: average absolute robust skew
#     - separation: how isolated this cluster is from the global normal center
#   These are used only to continuously adapt generator weights
def cluster_structure_features(cluster_df, all_normal_df, numeric_cols, categorical_cols):
    
    if len(cluster_df) == 0:
        return {
            "density": 1.0,
            "stability": 0.5,
            "skewness": 0.0,
            "separation": 0.5,
            "categorical_entropy": 0.0
        }

    density = len(cluster_df) / max(len(all_normal_df), 1)

    if not numeric_cols or len(cluster_df) < 5:
        return {
            "density": float(density),
            "stability": 0.5,
            "skewness": 0.0,
            "separation": 0.5,
            "categorical_entropy": 0.0
        }

    x = safe_numeric(cluster_df, numeric_cols).to_numpy(float)
    global_x = safe_numeric(all_normal_df, numeric_cols).to_numpy(float)

    #   Dispersion relative to the global marginal scale
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) + MIN_STD

    global_med = np.median(global_x, axis=0)
    global_mad = np.median(np.abs(global_x - global_med), axis=0)
    global_iqr = np.percentile(global_x, 75, axis=0) - np.percentile(global_x, 25, axis=0)
    global_std = np.std(global_x, axis=0)
    global_mad = np.where(global_mad > MIN_STD, global_mad,
                          np.where(global_iqr > MIN_STD, global_iqr, global_std))
    global_mad = np.where(global_mad > MIN_STD, global_mad, 1.0)

    relative_dispersion = np.median(
        mad / global_mad
    )

    stability = float(
        1.0 / (1.0 + max(relative_dispersion, 0.0))
    )
    stability = float(np.clip(stability, 0.0, 1.0))

    #   Skew proxy: mean/median asymmetry normalized by MAD
    q25 = np.percentile(x, 25, axis=0)
    q50 = np.percentile(x, 50, axis=0)
    q75 = np.percentile(x, 75, axis=0)

    robust_skew = np.abs(
        (q75 + q25 - 2.0 * q50) /
        (q75 - q25 + MIN_STD)
    )
    skewness = float(np.mean(np.clip(robust_skew, 0.0, 5.0)) / 5.0)

    #   Separation from the global normal center
    global_scale = global_mad
    center_offset = np.median(
        np.abs(med - global_med) / global_scale
    )
    separation = float(
        np.clip(center_offset / 5.0, 0.0, 1.0)
    )

    #   Average normalized categorical entropy
    cat_entropy = 0.0
    cat_count = 0
    for c in categorical_cols:
        probs = cluster_df[c].astype("object").value_counts(normalize=True, dropna=False)
        if len(probs) > 1:
            h = -float(np.sum(probs.to_numpy() * np.log(probs.to_numpy() + 1e-12)))
            hmax = np.log(len(probs))
            cat_entropy += h / max(hmax, 1e-12)
            cat_count += 1
    categorical_entropy = cat_entropy / cat_count if cat_count else 0.0

    return {
        "density": float(density),
        "stability": stability,
        "skewness": skewness,
        "separation": separation,
        "categorical_entropy": float(np.clip(categorical_entropy, 0.0, 1.0))
    }

#   Compute adaptive mixture weights for generators based on local cluster features
def adaptive_weights(features, median_cluster_size, mixture_adapt_strength, use_tvae=True,
                     base_mixture=None):
    density = features["density"]
    stability = features["stability"]
    skewness = features["skewness"]
    separation = features["separation"]
    categorical_entropy = features.get("categorical_entropy", 0.0)

    #   Convert density to a bounded relative-density signal
    rel_density = density / max(median_cluster_size, 1e-12)
    density_signal = np.tanh(np.log1p(max(rel_density, 1e-8)))

    #   Heterogeneity signal: skew + instability + separation
    heterogeneity = np.clip(
        0.35 * skewness +
        0.25 * (1.0 - stability) +
        0.20 * separation +
        0.20 * categorical_entropy,
        0.0,
        1.0
    )

    #   Continuous local adjustments
    stat, smote = base_mixture if base_mixture is not None else (BASE_STATISTICAL, BASE_SMOTE)
    tvae = BASE_TVAE

    #   Dense/stable regions -> statistical generator
    stat += mixture_adapt_strength * (
        0.10 * density_signal +
        0.08 * (stability - 0.5)
    )

    #   Sparse/separated regions -> SMOTE
    smote += mixture_adapt_strength * (
        -0.08 * density_signal +
        0.10 * (separation - 0.5)
    )

    #   Heterogeneous/skewed regions -> slightly more TVAE when TVAE is enabled
    tvae += mixture_adapt_strength * (
        0.08 * (heterogeneity - 0.5)
    )

    if use_tvae:
        weights = np.clip(np.array([stat, smote, tvae], dtype=float), 0.05, None)
        weights /= weights.sum()
        if weights[2] < MIN_TVAE_WEIGHT:
            deficit = MIN_TVAE_WEIGHT - weights[2]
            remaining = weights[0] + weights[1]
            if remaining > 1e-8:
                weights[0] -= deficit * (weights[0] / remaining)
                weights[1] -= deficit * (weights[1] / remaining)
            weights[2] = MIN_TVAE_WEIGHT
            weights = np.clip(weights, 0.0, None)
            weights /= weights.sum()
    else:
        weights = np.clip(np.array([stat, smote], dtype=float), 0.05, None)
        weights /= weights.sum()
        weights = np.r_[weights, 0.0]

    return {
        "statistical": float(weights[0]),
        "smote": float(weights[1]),
        "tvae": float(weights[2])
    }

# ============================================================
# DISCRIMINATIVE PCA GEOMETRY 
# ============================================================

def fit_discriminative_pca(normal_df, attack_df, numeric_cols,
                           categorical_cols):
    combined = pd.concat(
        [normal_df.copy(), attack_df.copy()],
        ignore_index=True
    )

    pre = make_preprocessor(combined, numeric_cols, categorical_cols)
    x_all = encode_for_geometry(combined, pre)

    n_normal = len(normal_df)
    x_n = x_all[:n_normal]
    x_a = x_all[n_normal:]

    if x_all.shape[0] < 2 or x_all.shape[1] < 1:
        raise ValueError("Discriminative PCA requires at least two rows and one feature.")

    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=RANDOM_STATE)
    z_all = pca.fit_transform(x_all)

    z_n = z_all[:n_normal]
    z_a = z_all[n_normal:]

    mean_n = np.mean(z_n, axis=0)
    mean_a = np.mean(z_a, axis=0)

    var_n = np.var(z_n, axis=0) + 1e-8
    var_a = np.var(z_a, axis=0) + 1e-8

    fisher = np.abs(mean_a - mean_n) / np.sqrt(
        0.5 * (var_n + var_a)
    )

    #   Normalize weights around mean 1 to preserve comparable distance scales
    fisher = fisher / (np.mean(fisher) + 1e-8)

    return pre, pca, fisher, z_n, z_a

def transform_weighted(df, pre, pca, fisher):
    x = encode_for_geometry(df, pre)
    z = pca.transform(x)
    return z * np.sqrt(fisher)

def knn_distance(points, reference, k=5):
    if len(points) == 0 or len(reference) == 0:
        return np.zeros(len(points))

    kk = max(1, min(k, len(reference)))
    nn = NearestNeighbors(n_neighbors=kk)
    nn.fit(reference)
    d, _ = nn.kneighbors(points)
    return d.mean(axis=1)

#   Distance relative to spacing between distinct real normal locations
def density_normalized_realism(points, reference, k=5):
    reference = np.unique(np.asarray(reference, dtype=float), axis=0)
    points = np.asarray(points, dtype=float)
    if len(reference) < 2:
        distance = knn_distance(points, reference, k=1)
        return distance, distance, {"distinct_normal_locations": len(reference), "scale_floor": None}
    kk = min(k, len(reference)-1)
    nn = NearestNeighbors(n_neighbors=kk+1).fit(reference)
    distances, _ = nn.kneighbors(reference)
    spacing = distances[:, 1:].mean(axis=1)
    positive = spacing[spacing > 0]
    floor = max(float(np.percentile(positive, LOCAL_SCALE_FLOOR_PERCENTILE)), 1e-12)
    spacing = np.maximum(spacing, floor)
    distances, indices = nn.kneighbors(points, n_neighbors=kk)
    raw = distances.mean(axis=1)
    local_scale = spacing[indices].mean(axis=1)
    ratio = raw / local_scale
    return ratio, raw, {"distinct_normal_locations": len(reference), "scale_floor": floor}

# ============================================================
# RFOD CONDITIONAL-NORMALITY SCORER 
# ============================================================

#   Feature-wise conditional reconstruction (RFOD-style), fitted on training normals only
#   For every feature j a small random forest predicts x_j from all other features (regressor for numeric j, classifier for categorical j; categoricals enter as ordinal codes)
#   Cell discrepancy: |x_j - x_hat_j| / range_j for numeric j, 1 - P(x_j | x_-j) for categorical j
#   Row score: mean cell discrepancy. High = the row breaks feature-to-feature relationships seen in normal traffic, even if every individual value is in range (rare combination)
class RFODScorer:
    def __init__(self, normal_df, numeric_cols, categorical_cols, seed=RFOD_SEED):
        self.nums, self.cats = list(numeric_cols), list(categorical_cols)
        self.cols = self.nums + self.cats
        self.levels = {c: sorted(normal_df[c].astype(str).unique()) for c in self.cats}
        Xo = self._ordinal(normal_df[self.cols])
        rng = np.random.RandomState(seed)
        self.models, self.range = {}, {}
        for j in self.cols:
            others = [c for c in self.cols if c != j]
            X = Xo[others].to_numpy(float)
            if j in self.cats:
                m = RandomForestClassifier(
                    RFOD_N_TREES, min_samples_leaf=RFOD_MIN_SAMPLES_LEAF,
                    max_features=RFOD_MAX_FEATURES, n_jobs=-1,
                    random_state=rng.randint(1e6),
                ).fit(X, Xo[j].to_numpy())
            else:
                y = pd.to_numeric(normal_df[j], errors="coerce").to_numpy(float)
                m = RandomForestRegressor(
                    RFOD_N_TREES, min_samples_leaf=RFOD_MIN_SAMPLES_LEAF,
                    max_features=RFOD_MAX_FEATURES, n_jobs=-1,
                    random_state=rng.randint(1e6),
                ).fit(X, y)
                self.range[j] = max(float(np.ptp(y)), 1e-9)
            self.models[j] = (m, others)

    def _ordinal(self, df):
        X = df.copy()
        for c in self.cats:
            if c in X:
                X[c] = pd.Categorical(X[c].astype(str), categories=self.levels[c]).codes
        for c in self.nums:
            if c in X:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def row_scores(self, df):
        df = df.reset_index(drop=True)
        Xo = self._ordinal(df[self.cols])
        cells = np.zeros((len(df), len(self.cols)))
        for k, j in enumerate(self.cols):
            m, others = self.models[j]
            X = Xo[others].to_numpy(float)
            if j in self.cats:
                proba = m.predict_proba(X)
                pos = {c: i for i, c in enumerate(m.classes_)}
                col = np.array([pos.get(c, -1) for c in Xo[j].to_numpy()])
                p_true = np.where(col >= 0, proba[np.arange(len(df)), np.maximum(col, 0)], 0.0)
                cells[:, k] = 1.0 - p_true
            else:
                cells[:, k] = np.abs(Xo[j].to_numpy(float) - m.predict(X)) / self.range[j]
        return cells.mean(axis=1)

#   Refined distance terms
#   Inside the near-attack zone, an RFOD conditional-normality penalty is added
#   Far-zone candidates keep the original score
def select_candidates(normal_real, attack_real, candidates,
                      numeric_cols, categorical_cols, realism_weight,
                      attack_overlap_percentile):
    pre, pca, fisher, z_n, z_a = fit_discriminative_pca(
        normal_real, attack_real, numeric_cols, categorical_cols
    )
    z_n_w = z_n * np.sqrt(fisher)
    z_a_w = z_a * np.sqrt(fisher)
    z_c_w = transform_weighted(candidates, pre, pca, fisher)
    attack_dist = knn_distance(z_c_w, z_a_w, k=3)
    relative_realism, realism_dist, density_diag = density_normalized_realism(
        z_c_w, z_n_w, k=REALISM_K
    )
    normal_to_attack = knn_distance(z_n_w, z_a_w, k=3)
    if not 0.0 <= attack_overlap_percentile <= 100.0:
        raise ValueError("attack_overlap_percentile must be between 0 and 100.")
    threshold = np.percentile(normal_to_attack, attack_overlap_percentile)
    keep = attack_dist >= threshold
    initial_keep = keep.copy()
    gate_relaxed = False
    if keep.sum() < max(50, int(0.20 * len(candidates))):
        threshold = np.percentile(normal_to_attack, 1.0)
        keep = attack_dist >= threshold
        gate_relaxed = True
    idx = np.flatnonzero(keep)
    gate_bypassed = len(idx) == 0
    if gate_bypassed:
        keep = np.ones(len(candidates), dtype=bool)
        idx = np.arange(len(candidates))

    cap = max(float(np.percentile(normal_to_attack, ATTACK_REWARD_CAP_PERCENTILE)), 1e-12)
    attack_reward = np.log1p(np.minimum(attack_dist, cap) / cap)
    realism_penalty = np.log1p(relative_realism)
    ad, rd = attack_reward[idx], realism_penalty[idx]
    ad_z = (ad - ad.mean()) / max(float(ad.std()), 1e-8)
    rd_z = (rd - rd.mean()) / max(float(rd.std()), 1e-8)
    score = ad_z - realism_weight * rd_z

    hybrid_diag = {"hybrid_enabled": bool(HYBRID_ENABLED and HYBRID_RFOD_WEIGHT > 0)}
    if hybrid_diag["hybrid_enabled"] and len(idx) > 1:
        t_rf = time.time()
        rfod = RFODScorer(normal_real, numeric_cols, categorical_cols)
        rf = np.log1p(rfod.row_scores(candidates.iloc[idx]))
        rf_z = (rf - rf.mean()) / max(float(rf.std()), 1e-8)
        # rank by distance to seen attacks: 0 = closest
        attack_rank = (np.argsort(np.argsort(attack_dist[idx], kind="stable"), kind="stable") + 0.5) / len(idx)
        near = attack_rank < HYBRID_NEAR_FRACTION
        score = score - HYBRID_RFOD_WEIGHT * near.astype(float) * rf_z
        hybrid_diag.update({
            "near_fraction": float(HYBRID_NEAR_FRACTION),
            "rfod_weight": float(HYBRID_RFOD_WEIGHT),
            "near_candidates": int(near.sum()),
            "rfod_mean_near": float(rf[near].mean()) if near.any() else None,
            "rfod_mean_far": float(rf[~near].mean()) if (~near).any() else None,
        })
        print(
            f"      Hybrid RFOD selection: {int(near.sum()):,} near-attack candidates penalised "
            f"(RFOD near {hybrid_diag['rfod_mean_near']:.4f} vs far {hybrid_diag['rfod_mean_far']:.4f}; "
            f"{time.time() - t_rf:.0f}s)"
        )

    order = idx[np.argsort(-score, kind="stable")]
    return order, {
        "hybrid": hybrid_diag,
        "attack_distance": attack_dist,
        "realism_distance": realism_dist,
        "density_normalized_realism": relative_realism,
        "initial_gate_mask": initial_keep,
        "gate_mask": keep,
        "gate_threshold": float(threshold),
        "gate_relaxed": gate_relaxed,
        "gate_bypassed": gate_bypassed,
        "kept_after_gate": int(keep.sum()),
        "attack_overlap_percentile": float(attack_overlap_percentile),
        "attack_reward_cap": cap,
        "density_diagnostics": density_diag,
    }

# ============================================================
# CLIPPING AND DOMAIN ENFORCEMENT
# ============================================================

def enforce_numeric_domains(frame, reference, numeric_cols):
    result = frame.copy()
    for c in numeric_cols:
        ref = pd.to_numeric(reference[c], errors="raise").to_numpy(float)
        values = pd.to_numeric(result[c], errors="raise").to_numpy(float)
        values = np.clip(values, ref.min(), ref.max())
        if np.all(np.abs(ref - np.rint(ref)) < 1e-9):
            values = np.rint(values)
        result[c] = values
    return result

def restore_numerical_zeros(frame, reference, numeric_cols):
    out = frame.copy()
    for c in numeric_cols:
        ref = pd.to_numeric(reference[c], errors="coerce").to_numpy(float)
        ref = ref[np.isfinite(ref)]
        if not len(ref) or np.mean(ref == 0) < 0.05:
            continue
        nonzero = np.abs(ref[ref != 0])
        if not len(nonzero):
            tolerance = 1e-10
        else:
            tolerance = min(
                1e-10 * max(1.0, float(np.median(nonzero))),
                0.01 * float(np.quantile(nonzero, 0.01)),
            )
        values = pd.to_numeric(out[c], errors="coerce").to_numpy(float).copy()
        values[np.isfinite(values) & (np.abs(values) <= tolerance)] = 0.0
        out[c] = values
    return out

#   Preserve an exact training-normal feature identity, when it exists
#   Apply the repair before candidate selection so ranking sees the same relationship as the output
def repair_tcp_timing_identity(frame, reference):
    columns = ("tcprtt", "synack", "ackdat")
    if not all(c in frame.columns and c in reference.columns for c in columns):
        return frame
    ref = reference.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if ref.isna().any().any() or not np.isfinite(ref.to_numpy(float)).all():
        return frame
    discrepancy = np.abs(ref["tcprtt"] - ref["synack"] - ref["ackdat"])
    if not (discrepancy <= 1e-6).all():
        return frame
    out = frame.copy()
    a = pd.to_numeric(out["synack"], errors="raise").to_numpy(float).copy()
    b = pd.to_numeric(out["ackdat"], errors="raise").to_numpy(float).copy()
    total = a + b
    hi = float(ref["tcprtt"].max())
    if hi <= 0:
        a[:] = 0.0
        b[:] = 0.0
    else:
        excess = total > hi
        factor = np.ones(len(total))
        factor[excess] = hi / total[excess]
        a *= factor
        b *= factor
    out["synack"] = a
    out["ackdat"] = b
    out["tcprtt"] = a + b
    return out

def final_sanity_clip(synth, normal_df, numeric_cols):
    if not FINAL_CLIP or not numeric_cols:
        return synth

    out = synth.copy()

    for c in numeric_cols:
        ref = pd.to_numeric(normal_df[c], errors="coerce").dropna()
        if len(ref) < 10:
            continue

        lo = np.percentile(ref, 100 - FINAL_CLIP_PERCENTILE)
        hi = np.percentile(ref, FINAL_CLIP_PERCENTILE)

        out[c] = pd.to_numeric(out[c], errors="coerce")
        if np.all(np.abs(ref.to_numpy() - np.rint(ref.to_numpy())) < 1e-9):
            lo, hi = np.floor(lo), np.ceil(hi)
        out[c] = out[c].clip(lo, hi)

    return out

# ============================================================
# ATTACK-COVERAGE PRE-FILTER 
# ============================================================

#   Low-cardinality features whose smallest attack-covering value set is narrow
#   For feature f, values are added in decreasing attack/normal lift until they hold COVERAGE_ATTACK_MASS of all training attacks
#   If that set V_f holds at most COVERAGE_MAX_NORMAL_SHARE of normals, "x_f in V_f" is a property of (almost) every observed attack and V_f is treated as attack territory
def coverage_rules(normal_df, attack_df, columns):
    rules = {}
    for c in columns:
        values = pd.concat([normal_df[c], attack_df[c]], ignore_index=True)
        if values.nunique() > COVERAGE_MAX_CARDINALITY:
            continue
        pa = attack_df[c].value_counts(normalize=True)
        pn = normal_df[c].value_counts(normalize=True).reindex(pa.index).fillna(0.0)
        chosen, attack_mass, normal_mass = [], 0.0, 0.0
        for v in (pa / (pn + 1e-6)).sort_values(ascending=False).index:
            chosen.append(v)
            attack_mass += pa[v]
            normal_mass += pn[v]
            if attack_mass >= COVERAGE_ATTACK_MASS:
                break
        if attack_mass >= COVERAGE_ATTACK_MASS and normal_mass <= COVERAGE_MAX_NORMAL_SHARE:
            rules[c] = chosen
    return rules

#   Drop candidates lying inside every attack-covering value set
def apply_coverage_filter(candidates, sources, clusters, normal_df, attack_df):
    rules = coverage_rules(normal_df, attack_df, list(normal_df.columns))
    if not rules:
        print("      Coverage filter: no qualifying feature (no-op)")
        return candidates, sources, clusters, {}
    inside = np.all([candidates[c].isin(v).to_numpy() for c, v in rules.items()], axis=0)
    keep = ~inside
    desc = ", ".join(f"{c} in {v}" for c, v in rules.items())
    print(f"      Coverage filter: {desc} -> removed {int(inside.sum()):,} of {len(candidates):,} candidates")
    return (candidates[keep].reset_index(drop=True), sources[keep],
            np.asarray(clusters)[keep], {c: [str(x) for x in v] for c, v in rules.items()})


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_synthetic_normals(
    normal_df,
    attack_df,
    candidate_over_sample_factor,
    target_n=None,
    random_state=RANDOM_STATE,
    realism_weight=None,
    mixture_adapt_strength=None,
    attack_overlap_percentile=None,
    use_tvae=False,
    coverage_filter=False,
    empirical_share=0.0,
    base_mixture=None,
):
    realism_weight = 0.75 if realism_weight is None else realism_weight
    mixture_adapt_strength = 0.45 if mixture_adapt_strength is None else mixture_adapt_strength
    attack_overlap_percentile = ATTACK_OVERLAP_PERCENTILE if attack_overlap_percentile is None else attack_overlap_percentile
    if candidate_over_sample_factor < 1:
        raise ValueError("Candidate oversampling factor must be at least 1.")
    if len(normal_df) < 2 or len(attack_df) < 1:
        raise ValueError("Training requires at least two normals and one seen attack.")
    set_seed(random_state)

    normal_df = normal_df.copy().reset_index(drop=True)
    attack_df = attack_df.copy().reset_index(drop=True)

    if target_n is None:
        target_n = len(normal_df)

    numeric_cols, categorical_cols = split_columns(normal_df)

    attack_df = attack_df[normal_df.columns].copy()

    #   Local clustering of real normals
    pre_local = make_preprocessor(
        normal_df,
        numeric_cols,
        categorical_cols
    )
    x_normal = encode_for_geometry(normal_df, pre_local)

    km, labels, centers = fit_local_clusters(
        x_normal,
        random_state
    )

    unique_labels = np.unique(labels)
    cluster_sizes = {
        int(k): int(np.sum(labels == k))
        for k in unique_labels
    }
    median_cluster_size = float(np.median(list(cluster_sizes.values())))

    #   Decide how many candidates each local cluster gets
    candidate_n = int(np.ceil(
        target_n * candidate_over_sample_factor
    ))

    raw_alloc = {}
    for k in unique_labels:
        size = cluster_sizes[int(k)]
        raw_alloc[int(k)] = (
            size * (0.70 + 0.30 * np.sqrt(
                median_cluster_size / max(size, 1)
            ))
        )

    total_raw = sum(raw_alloc.values())

    allocations = {}
    for k in unique_labels:
        allocations[int(k)] = int(round(
            candidate_n * raw_alloc[int(k)] / total_raw
        ))

    diff = candidate_n - sum(allocations.values())
    if diff != 0:
        largest = max(allocations, key=allocations.get)
        allocations[largest] += diff

    #   Generate each cluster with an adaptive mixture
    all_candidates = []
    all_candidate_sources = []
    all_candidate_clusters = []
    diagnostics = []
    requested_tvae = {}
    tvae_failed = False

    for k in unique_labels:
        idx_k = np.where(labels == k)[0]
        cluster_df = normal_df.iloc[idx_k].reset_index(drop=True)
        structure = cluster_structure_features(
            cluster_df, normal_df, numeric_cols, categorical_cols
        )
        weights = adaptive_weights(structure, median_cluster_size / len(normal_df),
                                   mixture_adapt_strength=mixture_adapt_strength,
                                   use_tvae=use_tvae,
                                   base_mixture=base_mixture)
        n_k = allocations[int(k)]
        # Enforce the TVAE floor only when the learned TVAE generator is enabled
        if not use_tvae and empirical_share > 0:
            raise ValueError(
                "Empirical bootstrap is disabled. Set empirical_share=0 or explicitly enable TVAE."
            )
        n_tvae = min(
            n_k,
            max(1 if n_k > 0 else 0, int(np.ceil(n_k * max(MIN_TVAE_WEIGHT, weights["tvae"]))))
        ) if use_tvae else 0
        remaining_quota = n_k - n_tvae
        stat_share = weights["statistical"] / max(
            weights["statistical"] + weights["smote"], 1e-12
        )
        n_stat = int(round(remaining_quota * stat_share))
        n_smote = remaining_quota - n_stat
        requested_tvae[int(k)] = n_tvae
        diagnostics.append({
            "cluster": int(k),
            "size": int(len(cluster_df)),
            "density": structure["density"],
            "stability": structure["stability"],
            "skewness": structure["skewness"],
            "separation": structure["separation"],
            "categorical_entropy": structure.get("categorical_entropy", 0.0),
            "weight_statistical": weights["statistical"],
            "weight_smote": weights["smote"],
            "weight_tvae": weights["tvae"],
            "n_statistical": int(n_stat),
            "n_smote": int(n_smote),
            "n_tvae": int(n_tvae),
        })

    smote_quotas = {
        int(item["cluster"]): int(item["n_smote"])
        for item in diagnostics
    }
    cluster_smote, smote_diag = cluster_conditioned_smotenc_generate(
        normal_df=normal_df,
        labels=labels,
        quota_by_cluster=smote_quotas,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        random_state=random_state + 3000,
    )

    print(
        f"      Cluster-SMOTENC generated={smote_diag['generated_by_smotenc']} "
        f"fallback={smote_diag['fallback_generated']} "
        f"clusters={smote_diag['smotenc_clusters']}"
    )

    tvae_needed = int(sum(requested_tvae.values()))
    global_tvae = None
    tvae_assignments = {}
    if tvae_needed > 0 and not use_tvae:
        raise RuntimeError("Internal quota error: TVAE/empirical quota must be zero when TVAE is disabled.")
    elif tvae_needed > 0:
        try:
            pool_n = max(tvae_needed, int(np.ceil(tvae_needed * TVAE_POOL_OVERSAMPLE)))
            global_tvae = tvae_generate(normal_df, pool_n, random_state + 1000)
            z_tvae = encode_for_geometry(global_tvae, pre_local)

            remaining = set(range(len(global_tvae)))
            cluster_order = sorted(
                unique_labels,
                key=lambda k: requested_tvae[int(k)],
                reverse=True
            )
            for k in cluster_order:
                need = requested_tvae[int(k)]
                if need <= 0:
                    tvae_assignments[int(k)] = pd.DataFrame(columns=normal_df.columns)
                    continue
                center_idx = int(k)
                distances = np.sum((z_tvae - centers[center_idx]) ** 2, axis=1)
                available = np.array(sorted(remaining), dtype=int)
                if len(available):
                    take = available[np.argsort(distances[available])[:need]]
                    for j in take.tolist():
                        remaining.discard(int(j))
                    tvae_assignments[int(k)] = global_tvae.iloc[take].copy().reset_index(drop=True)
                else:
                    tvae_assignments[int(k)] = pd.DataFrame(columns=normal_df.columns)
        except Exception as exc:
            import traceback
            print("TVAE failed. Full traceback:")
            traceback.print_exc()
            raise

    for k in unique_labels:
        idx_k = np.where(labels == k)[0]
        cluster_df = normal_df.iloc[idx_k].reset_index(drop=True)
        n_k = allocations[int(k)]
        d = next(item for item in diagnostics if item["cluster"] == int(k))
        n_stat = d["n_statistical"]
        n_smote = d["n_smote"]
        n_tvae = d["n_tvae"]

        stat_df = statistical_generate_cluster(
            cluster_df, numeric_cols, categorical_cols, n_stat,
            random_state + 2000 + int(k)
        )

        smote_df = cluster_smote.get(
            int(k),
            pd.DataFrame(columns=normal_df.columns)
        ).iloc[:n_smote].copy()

        pieces = []
        source_labels = []
        if len(stat_df):
            pieces.append(stat_df)
            source_labels.extend(["statistical"] * len(stat_df))
        if len(smote_df):
            pieces.append(smote_df)
            true_smote_n = min(
                len(smote_df),
                int(smote_diag.get("smotenc_by_cluster", {}).get(int(k), 0)),
            )
            smote_fallback_n = len(smote_df) - true_smote_n
            source_labels.extend(["smotenc"] * true_smote_n)
            source_labels.extend(["smote_local_fallback"] * smote_fallback_n)

        tpart = tvae_assignments.get(int(k), pd.DataFrame(columns=normal_df.columns))
        tpart = tpart.iloc[:n_tvae].copy()
        if len(tpart):
            pieces.append(tpart)
            source_labels.extend(["tvae"] * len(tpart))

        if pieces:
            cand_k = pd.concat(pieces, ignore_index=True)
        else:
            cand_k = pd.DataFrame(columns=normal_df.columns)

        if len(cand_k) < n_k:
            n_fallback = n_k - len(cand_k)
            fallback = _cluster_local_knn_fallback(
                cluster_df, numeric_cols, categorical_cols, n_fallback,
                random_state=random_state + 4000 + int(k),
            )
            cand_k = pd.concat([cand_k, fallback], ignore_index=True)
            source_labels.extend(["smote_local_fallback"] * n_fallback)

        cand_k = cand_k.iloc[:n_k].copy()
        source_labels = source_labels[:n_k]
        all_candidates.append(cand_k)
        all_candidate_sources.extend(source_labels)
        all_candidate_clusters.extend([int(k)] * len(cand_k))

    candidates = pd.concat(all_candidates, ignore_index=True)
    candidate_sources = np.asarray(all_candidate_sources, dtype=object)

    #   Candidate sanitation
    for c in numeric_cols:
        candidates[c] = pd.to_numeric(
            candidates[c],
            errors="coerce"
        )

        med = normal_df[c].median()
        candidates[c] = candidates[c].fillna(
            0.0 if pd.isna(med) else med
        )

    for c in categorical_cols:
        observed = set(normal_df[c].astype("object").dropna().unique())
        if observed:
            fallback = normal_df[c].mode(dropna=True)
            fallback = fallback.iloc[0] if len(fallback) else next(iter(observed))
            candidates[c] = candidates[c].where(
                candidates[c].isin(observed),
                fallback
            )

    candidates = enforce_numeric_domains(candidates, normal_df, numeric_cols)
    candidates = restore_numerical_zeros(candidates, normal_df, numeric_cols)
    candidates = candidates[normal_df.columns].reset_index(drop=True)

    candidates, candidate_sources, all_candidate_clusters, consistency_diag = (
        filter_consistent_candidates(
            candidates=candidates,
            sources=candidate_sources,
            cluster_ids=all_candidate_clusters,
            normal_df=normal_df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_n=target_n,
        )
    )

    coverage_diag = {}
    if coverage_filter:
        candidates, candidate_sources, all_candidate_clusters, coverage_diag = apply_coverage_filter(
            candidates, candidate_sources, all_candidate_clusters, normal_df, attack_df
        )

    #   Discriminative manifold selection
    order, selection_diag = select_candidates(
        normal_df,
        attack_df,
        candidates,
        numeric_cols,
        categorical_cols,
        realism_weight=realism_weight,
        attack_overlap_percentile=attack_overlap_percentile,
    )

    if len(order) < target_n:
        raise ValueError(
            f"Only {len(order)} candidates survived for {target_n} rows. "
            "Increase candidate oversampling; rejected rows will not be reused."
        )
    selected_n = target_n
    selected_indices = np.asarray(order[:selected_n], dtype=int)
    chosen = candidates.iloc[selected_indices].copy()

    selected_sources = candidate_sources[selected_indices]
    candidate_clusters = np.asarray(all_candidate_clusters)
    for item in diagnostics:
        k = item["cluster"]
        item["candidate_count"] = int(np.sum(candidate_clusters == k))
        item["selected_count"] = int(np.sum(candidate_clusters[selected_indices] == k))
        item["acceptance_rate"] = item["selected_count"] / max(item["candidate_count"], 1)
        origin = candidate_clusters == k
        item["initial_gate_survivors"] = int(np.sum(origin & selection_diag["initial_gate_mask"]))
        item["effective_gate_survivors"] = int(np.sum(origin & selection_diag["gate_mask"]))
        item["gate_rejected"] = item["candidate_count"] - item["effective_gate_survivors"]
        item["ranking_rejected"] = item["effective_gate_survivors"] - item["selected_count"]
    source_order = ["statistical", "smotenc", "smote_local_fallback", "tvae"]
    source_counts = {name: int(np.sum(selected_sources == name)) for name in source_order}
    denom = max(len(selected_sources), 1)
    print("      Selected generator contribution:")
    for name in source_order:
        print(f"        {name}: {source_counts[name]:,} rows ({100.0 * source_counts[name] / denom:.1f}%)")

    chosen = chosen[normal_df.columns].reset_index(drop=True)
    chosen_geometry = encode_for_geometry(chosen, pre_local)
    actual_labels = (km.predict(chosen_geometry) if km is not None
                     else np.zeros(len(chosen), dtype=int))
    for item in diagnostics:
        item["selected_actual_cluster_count"] = int(np.sum(actual_labels == item["cluster"]))
        item["selected_actual_cluster_fraction"] = item["selected_actual_cluster_count"] / max(len(chosen), 1)

    return chosen, {
        "n_normal": len(normal_df),
        "n_attack": len(attack_df),
        "n_candidates": len(candidates),
        "n_selected": len(chosen),
        "n_clusters": len(unique_labels),
        "cluster_diagnostics": pd.DataFrame(diagnostics),
        "smotenc_diagnostics": smote_diag,
        "selection_diagnostics": selection_diag,
        "coverage_rules": coverage_diag,
        "consistency": consistency_diag,
        "tvae_failed": tvae_failed,
        "requested_tvae_rows": int(tvae_needed),
        "actual_tvae_rows": int(sum(len(v) for v in tvae_assignments.values())),
        "selected_generator_counts": source_counts,
        "selected_generator_percentages": {
            name: (100.0 * count / denom) for name, count in source_counts.items()
        },
        "candidate_generator_counts": {
            name: int(np.sum(candidate_sources == name)) for name in source_order
        }
    }

# ============================================================
# DOMAIN CONSISTENCY + NOVELTY FILTERS
# ============================================================

#   Learn constraints from normal trainign rows
#   Candidate identities are reconstructed before selection; candidates that remain invalid are rejected, never replaced by sampled observed rows
#   Empirical inequalities are diagnostic only
RULE_MIN_GROUP = 50
ENFORCE_MINED_INEQUALITIES = False
IDENTITY_DISCOVERY_TOL = 0.02
IDENTITY_VALIDATION_TOL = 1e-6
RULE_NUMERIC_TOL = 1e-9

def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        b_arr = np.asarray(b)
        return np.where(b_arr != 0, np.asarray(a) / np.where(b_arr == 0, 1, b_arr), np.nan)

#   Some common deterministic identities in network traffic features
IDENTITY_CANDIDATES = [
    ("smean", ["sbytes", "spkts"], lambda t: np.round(_safe_div(t.sbytes, t.spkts)), lambda t: t.spkts > 0),
    ("dmean", ["dbytes", "dpkts"], lambda t: np.round(_safe_div(t.dbytes, t.dpkts)), lambda t: t.dpkts > 0),
    ("rate", ["spkts", "dpkts", "dur"], lambda t: _safe_div(t.spkts + t.dpkts - 1, t.dur), lambda t: t.dur > 0),
    ("tcprtt", ["synack", "ackdat"], lambda t: t.synack + t.ackdat, lambda t: np.ones(len(t), dtype=bool)),
]

#   Keep candidate identities only when at least 99.9% of valid normals support them
def mine_identities(normal_df):
    found = []
    for target, deps, fn, guard in IDENTITY_CANDIDATES:
        if not all(c in normal_df.columns for c in [target] + deps):
            continue
        try:
            mask = np.asarray(guard(normal_df), dtype=bool)
            if not mask.any():
                continue
            expected = np.asarray(fn(normal_df), dtype=float)[mask]
            observed = pd.to_numeric(normal_df[target], errors="coerce").to_numpy(float)[mask]
            finite = np.isfinite(expected) & np.isfinite(observed)
            if not finite.any():
                continue
            close = np.abs(observed[finite] - expected[finite]) <= (
                IDENTITY_DISCOVERY_TOL * np.maximum(np.abs(expected[finite]), 1.0)
            )
            if float(close.mean()) >= 0.999:
                found.append((target, fn, guard))
        except (AttributeError, TypeError, ValueError):
            continue
    return found

#   Mine only deterministic categorical-to-numeric dependencies by default
def mine_hard_rules(normal_df, numeric_cols, categorical_cols):
    ineq = []
    if ENFORCE_MINED_INEQUALITIES:
        X = normal_df[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        for i, a_name in enumerate(numeric_cols):
            for j, b_name in enumerate(numeric_cols):
                if i >= j:
                    continue
                a, b = X[:, i], X[:, j]
                finite = np.isfinite(a) & np.isfinite(b)
                if finite.sum() < len(normal_df) or np.ptp(a[finite]) == 0 or np.ptp(b[finite]) == 0:
                    continue
                if (a[finite] <= b[finite] + RULE_NUMERIC_TOL).all():
                    if (a[finite] < b[finite] - RULE_NUMERIC_TOL).mean() > 0.05:
                        ineq.append((a_name, b_name))

    deps = []
    for c in categorical_cols:
        for t in numeric_cols:
            grouped = normal_df.groupby(c, dropna=False)[t].agg(["nunique", "size"])
            deterministic = grouped[(grouped["nunique"] == 1) & (grouped["size"] >= RULE_MIN_GROUP)]
            if not len(deterministic):
                continue
            if deterministic["size"].sum() / max(len(normal_df), 1) <= 0.05:
                continue
            for level in deterministic.index:
                if pd.isna(level):
                    rows = normal_df.loc[normal_df[c].isna(), t]
                else:
                    rows = normal_df.loc[normal_df[c] == level, t]
                if len(rows):
                    deps.append((c, level, t, rows.iloc[0]))
    return ineq, deps

def _reconstruct_identities(frame, identities, reference):
    out = frame.copy()
    out_of_domain = np.zeros(len(out), dtype=bool)
    for target, fn, guard in identities:
        mask = np.asarray(guard(out), dtype=bool)
        expected = np.asarray(fn(out), dtype=float)
        ref = pd.to_numeric(reference[target], errors="coerce").to_numpy(float)
        finite_ref = ref[np.isfinite(ref)]
        if not len(finite_ref):
            continue
        valid_expected = mask & np.isfinite(expected)
        out_of_domain |= mask & ~np.isfinite(expected)
        out_of_domain |= valid_expected & (
            (expected < finite_ref.min() - RULE_NUMERIC_TOL) |
            (expected > finite_ref.max() + RULE_NUMERIC_TOL)
        )
        values = pd.to_numeric(out[target], errors="coerce").to_numpy(float).copy()
        values[valid_expected] = expected[valid_expected]
        if np.all(np.abs(finite_ref - np.rint(finite_ref)) <= 1e-9):
            values[valid_expected] = np.rint(values[valid_expected])
            out[target] = values.astype(reference[target].dtype)
        else:
            out[target] = values
    return out, out_of_domain

def _consistency_mask(frame, identities, inequalities, dependencies, numeric_cols):
    bad = np.zeros(len(frame), dtype=bool)
    for c in numeric_cols:
        values = pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
        bad |= ~np.isfinite(values)
    for target, fn, guard in identities:
        mask = np.asarray(guard(frame), dtype=bool)
        expected = np.asarray(fn(frame), dtype=float)
        observed = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
        bad |= mask & (~np.isfinite(expected) | ~np.isfinite(observed))
        valid = mask & np.isfinite(expected) & np.isfinite(observed)
        bad |= valid & (np.abs(observed - expected) > (
            IDENTITY_VALIDATION_TOL * np.maximum(np.abs(expected), 1.0)
        ))
    for a, b in inequalities:
        av = pd.to_numeric(frame[a], errors="coerce").to_numpy(float)
        bv = pd.to_numeric(frame[b], errors="coerce").to_numpy(float)
        bad |= av > bv + RULE_NUMERIC_TOL
    for c, level, target, value in dependencies:
        if pd.isna(level):
            level_mask = frame[c].isna().to_numpy()
        else:
            level_mask = frame[c].eq(level).fillna(False).to_numpy()
        target_values = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
        bad |= level_mask & (np.abs(target_values - float(value)) > RULE_NUMERIC_TOL)
    return ~bad

#   Hash exact feature rows after normalizing
def _canonical_hashes(frame, feature_cols, numeric_cols):
    canonical = pd.DataFrame(index=np.arange(len(frame)))
    numeric_set = set(numeric_cols)
    for c in feature_cols:
        if c in numeric_set:
            canonical[c] = pd.to_numeric(frame[c], errors="coerce").astype(float).to_numpy()
        else:
            canonical[c] = frame[c].astype("string").fillna("<NA>").astype(str).to_numpy()
    return pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype=np.uint64)

#   Reconstruct supported identities, then reject invalid/duplicate/copied candidates
def filter_consistent_candidates(candidates, sources, cluster_ids, normal_df,
                                 numeric_cols, categorical_cols, target_n):
    candidates = candidates.copy().reset_index(drop=True)
    sources = np.asarray(sources, dtype=object)
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    if not (len(candidates) == len(sources) == len(cluster_ids)):
        raise ValueError("Candidate rows, source labels and cluster IDs are not aligned.")

    feature_cols = list(normal_df.columns)
    identities = mine_identities(normal_df)
    inequalities, dependencies = mine_hard_rules(normal_df, numeric_cols, categorical_cols)
    candidates, out_of_domain = _reconstruct_identities(candidates, identities, normal_df)
    valid = _consistency_mask(candidates, identities, inequalities, dependencies, numeric_cols)
    valid &= ~out_of_domain

    valid &= ~candidates.duplicated(subset=feature_cols, keep="first").to_numpy()
    candidate_hash = _canonical_hashes(candidates, feature_cols, numeric_cols)
    normal_hash = _canonical_hashes(normal_df, feature_cols, numeric_cols)
    valid &= ~np.isin(candidate_hash, normal_hash)

    rejected = int((~valid).sum())
    filtered = candidates.loc[valid].reset_index(drop=True)
    filtered_sources = sources[valid]
    filtered_clusters = cluster_ids[valid]
    diagnostics = {
        "identities": [item[0] for item in identities],
        "inequalities_enforced": bool(ENFORCE_MINED_INEQUALITIES),
        "inequalities": int(len(inequalities)),
        "categorical_dependencies": int(len(dependencies)),
        "candidate_rows_before": int(len(candidates)),
        "candidate_rows_rejected": rejected,
        "candidate_rows_after": int(len(filtered)),
        "candidate_rejection_fraction": float(rejected / max(len(candidates), 1)),
        "exact_training_matches_removed": int(np.sum(np.isin(candidate_hash, normal_hash))),
    }
    print(
        "      Candidate audit: "
        f"checked={len(candidates):,}, rejected={rejected:,} "
        f"({diagnostics['candidate_rejection_fraction']:.2%}), "
        f"kept={len(filtered):,}, identities={diagnostics['identities']}"
    )
    if len(filtered) < target_n:
        raise ValueError(
            f"Only {len(filtered):,} novel, rule-compliant candidates remain for "
            f"{target_n:,} requested rows. Increase candidate oversampling or revise "
            "the generator; will not silently insert copied training rows."
        )
    return filtered, filtered_sources, filtered_clusters, diagnostics

def audit_domain_consistency(frame, normal_df, numeric_cols, categorical_cols):
    identities = mine_identities(normal_df)
    inequalities, dependencies = mine_hard_rules(normal_df, numeric_cols, categorical_cols)
    valid = _consistency_mask(frame, identities, inequalities, dependencies, numeric_cols)
    if not valid.all():
        raise AssertionError(f"Final output contains {(~valid).sum()} domain-inconsistent rows.")
    for c in numeric_cols:
        ref = pd.to_numeric(normal_df[c], errors="coerce").to_numpy(float)
        values = pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
        if len(ref) and ((values < np.nanmin(ref) - RULE_NUMERIC_TOL) |
                         (values > np.nanmax(ref) + RULE_NUMERIC_TOL)).any():
            raise AssertionError(f"Final output column {c!r} exceeds observed normal support.")
    return {
        "identities": [item[0] for item in identities],
        "inequalities": int(len(inequalities)),
        "categorical_dependencies": int(len(dependencies)),
        "rows_checked": int(len(frame)),
        "rows_invalid": 0,
    }

def audit_no_copies_or_duplicates(frame, normal_df, feature_cols, numeric_cols):
    features = frame.drop(columns=["id"], errors="ignore")[feature_cols].reset_index(drop=True)
    output_hash = _canonical_hashes(features, feature_cols, numeric_cols)
    normal_hash = _canonical_hashes(normal_df[feature_cols], feature_cols, numeric_cols)
    exact_copies = int(np.isin(output_hash, normal_hash).sum())
    duplicate_rows = int(features.duplicated(subset=feature_cols).sum())
    if exact_copies:
        raise AssertionError(f"Final output contains {exact_copies} exact training-normal copies.")
    if duplicate_rows:
        raise AssertionError(f"Final output contains {duplicate_rows} duplicate rows.")
    return {
        "rows_checked": int(len(features)),
        "exact_training_row_copies": exact_copies,
        "duplicate_rows": duplicate_rows,
        "unique_fraction": 1.0 if len(features) else 0.0,
    }

# ============================================================
# NORMAL-COVERAGE RE-WEIGHTING
# ============================================================

#   Final set is resampled from the domain-consistent synthetic set with per-row weights
#   Corrects how often are-but-real normal behaviour appears, which drives false alarms
#   Tilts are chosen on internal splits of the training data; bagged over REWEIGHT_BAGS splits (majority vote per column) and gated by REWEIGHT_MIN_GAIN
REWEIGHT_STRENGTHS = {
    "unsw": 1.5,
    "nslkdd": 0.0,
}
REWEIGHT_STRENGTH = 1.5       
REWEIGHT_ESS_MIN = 0.3
REWEIGHT_BAGS = 3
REWEIGHT_QUORUM = 2
REWEIGHT_MIN_GAIN = 0.01
REWEIGHT_VAL_FRAC = 0.25
REWEIGHT_PASSES = 2

#   Odd values (1-3 numeric features pushed into the outer 1% tail of normal)
#   Rare combinations (half the numeric features copied from a normal row with different categorical values)
def _pseudo_anomalies(Hn, ref, feats, cats, frac=0.4, seed=0):
    rng = np.random.default_rng(seed)
    nums = [c for c in feats if c not in cats and ref[c].nunique() > 1]
    m = int(frac * len(Hn)); half = m // 2
    odd = Hn.sample(half, random_state=seed).reset_index(drop=True)
    odd[nums] = odd[nums].astype(float)
    for i in range(half):
        for c in rng.choice(nums, rng.integers(1, 4), replace=False):
            v = ref[c].to_numpy(float)
            hi = np.quantile(v, 0.995); lo = np.quantile(v, 0.005)
            tail = v[v >= hi] if (hi > np.median(v) or lo == np.median(v)) else v[v <= lo]
            val = rng.choice(tail) if len(tail) else v.max()
            odd.at[i, c] = val * (1 + rng.uniform(0, .5)) if val > 0 else val
    comb = Hn.sample(m - half, random_state=seed + 1).reset_index(drop=True)
    comb[nums] = comb[nums].astype(float)
    donor = ref.sample(4 * (m - half), random_state=seed + 2).reset_index(drop=True)
    for i in range(m - half):
        cand = donor.iloc[4 * i: 4 * i + 4]
        diff = cand[(cand[cats].to_numpy() != comb.loc[i, cats].to_numpy()).any(1)]
        d = (diff if len(diff) else cand).iloc[0]
        cols = rng.choice(nums, len(nums) // 2, replace=False)
        comb.loc[i, cols] = d[cols].to_numpy()
    return pd.concat([odd, comb], ignore_index=True)

def _weighted_ecod_term(syn_col, w, val_col, sk):
    o = np.argsort(syn_col, kind="stable"); xs = syn_col[o]; cw = np.cumsum(w[o]); W = cw[-1]
    nv = len(val_col); vs = np.sort(val_col); ns = len(syn_col)
    a = ns / (ns + nv); b = 1 - a
    le = np.searchsorted(xs, val_col, "right"); lt = np.searchsorted(xs, val_col, "left")
    Fs = np.where(le > 0, cw[np.maximum(le - 1, 0)], 0) / W
    Ss = 1 - np.where(lt > 0, cw[np.maximum(lt - 1, 0)], 0) / W
    Fv = np.searchsorted(vs, val_col, "right") / nv
    Sv = (nv - np.searchsorted(vs, val_col, "left")) / nv
    F = a * Fs + b * Fv; S = a * Ss + b * Sv
    ul, ur = -np.log(np.maximum(F, 1e-12)), -np.log(np.maximum(S, 1e-12))
    us = ul * -1 * np.sign(sk - 1) + ur * np.sign(sk + 1)
    return np.nan_to_num(np.maximum(np.maximum(ul, ur), us))

def _tailness(syn, cols):
    return {c: np.abs(2 * (rankdata(syn[c]) - 0.5) / len(syn) - 1) for c in cols}

def reweight_search(syn, V, yv, P, feats, cats):
    from scipy.stats import skew
    from sklearn.metrics import average_precision_score as ap
    nums = [c for c in feats if c not in cats and syn[c].nunique() > 1]
    E = pd.concat([V, P], ignore_index=True); n_v = len(V)
    y_p = np.r_[np.zeros((yv == 0).sum()), np.ones(len(P))]
    nmask = np.r_[yv == 0, np.zeros(len(P), bool)]
    G = _tailness(syn, nums)
    cat_cols = []
    for c in cats:
        for lv in sorted(syn[c].astype(str).unique()):
            a = (syn[c].astype(str).to_numpy(dtype=object) == lv).astype(float)
            b = (E[c].astype(str).to_numpy(dtype=object) == lv).astype(float)
            cat_cols.append((a, b, np.sign(skew(np.r_[a, b]))))
    SK = {c: np.sign(skew(np.r_[syn[c].to_numpy(float), E[c].to_numpy(float)])) for c in nums}

    def total(t):
        lw = sum((t[c] * G[c] for c in nums if t[c]), np.zeros(len(syn)))
        w = np.exp(lw - lw.max()); ess = w.sum() ** 2 / (w ** 2).sum()
        tt = sum(_weighted_ecod_term(syn[c].to_numpy(float), w, E[c].to_numpy(float), SK[c]) for c in nums)
        tt = tt + sum(_weighted_ecod_term(a, w, b, k) for a, b, k in cat_cols)
        return tt, ess

    def scores(tt):
        return ap(yv, tt[:n_v]), ap(y_p, np.r_[tt[nmask], tt[n_v:]])

    t = {c: 0.0 for c in nums}
    s_seen, s_ps = scores(total(t)[0]); start = (s_seen, s_ps)
    for _ in range(REWEIGHT_PASSES):
        for c in nums:
            best = None
            for v in sorted({REWEIGHT_STRENGTH, -REWEIGHT_STRENGTH, 0.0}):
                if v == t[c]:
                    continue
                t2 = dict(t); t2[c] = v
                tt, ess = total(t2)
                if ess < REWEIGHT_ESS_MIN * len(syn):
                    continue
                a, b = scores(tt)
                if a > s_seen + 1e-4 and b >= s_ps and (best is None or a > best[1][0]):
                    best = (v, (a, b))
            if best:
                t[c] = best[0]; s_seen, s_ps = best[1]
    return t, start, (s_seen, s_ps)

REWEIGHT_ANCHOR = {
    "unsw": "uniform",     #    tilt applied once
    "nslkdd": "weighted",  #    tilt also guides where new rows are generated
}
REWEIGHT_POOL_MULT = 1.5   #    new rows generated = 1.5 x N, before rule checks
REWEIGHT_K = 5             #    same-category neighbours used for interpolation

def _rank_space(df, feats, cats, ref):
    parts = []
    for c in feats:
        if c in cats:
            lv = np.asarray(ref[c].astype(str).unique(), dtype=object)
            parts.append((df[c].astype(str).to_numpy(dtype=object)[:, None] == lv[None]).astype(float))
        else:
            r = np.sort(ref[c].to_numpy(float)); x = df[c].to_numpy(float)
            parts.append(((np.searchsorted(r, x, "left") + np.searchsorted(r, x, "right")) / (2 * len(r)))[:, None])
    return np.hstack(parts)

def _tailness_vs(ref, df, cols):
    out = {}
    for c in cols:
        r = np.sort(ref[c].to_numpy(float)); x = df[c].to_numpy(float)
        F = (np.searchsorted(r, x, "left") + np.searchsorted(r, x, "right")) / (2 * len(r))
        out[c] = np.abs(2 * F - 1)
    return out

#   Realise tilted target distribution with new, distinct, rule-checked rows (no copies):
#   1. ADASYN-style pool: anchors from base set, each new row a convex interpolation with one of REWEIGHT_K nearest base rows sharing all categorical values
#   2. Every new row is passed through mined identities and hard rules; violators are discarded
#   3. Pool = base + new, de-duplicated; N rows drawn without replacement with probability proportional to tilt weight (Efraimidis-Spirakis weighted sampling)
def utility_adasyn(base, normal_df, feats, cats, tilts, N, anchor_mode, seed=RANDOM_STATE):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    num = [c for c in feats if c not in cats]
    base = base.copy(); base.attrs = {}
    S = base[feats].drop_duplicates().reset_index(drop=True)
    G = _tailness_vs(S, S, list(tilts))
    lw = sum(tilts[c] * G[c] for c in tilts)
    wS = np.exp(lw - lw.max())
    Z = _rank_space(S, feats, cats, S)
    key = S[cats].astype(str).agg("|".join, axis=1).to_numpy()
    nn = NearestNeighbors(n_neighbors=REWEIGHT_K + 1).fit(Z)
    intcols = [c for c in num if np.allclose(normal_df[c], np.round(normal_df[c]))]
    bincols = [c for c in num if set(np.unique(normal_df[c])) <= {0, 1}]
    identities = mine_identities(normal_df)
    ineq, deps = mine_hard_rules(normal_df, num, cats)
    target = int(REWEIGHT_POOL_MULT * N); new = []
    for _ in range(8):
        m = target - sum(len(x) for x in new)
        if m <= 0:
            break
        pa = wS / wS.sum() if anchor_mode == "weighted" else None
        a = rng.choice(len(S), int(m * 1.6), p=pa)
        _, nb = nn.kneighbors(Z[a])
        b = nb[np.arange(len(a)), rng.integers(1, REWEIGHT_K + 1, len(a))]
        ok = key[a] == key[b]; a, b = a[ok], b[ok]
        A, B = S.iloc[a].reset_index(drop=True), S.iloc[b].reset_index(drop=True)
        t = rng.uniform(0, 1, len(A)); out = A.copy()
        for c in num:
            if c in bincols:
                out[c] = np.where(rng.random(len(A)) < t, B[c], A[c])
            else:
                v = A[c].to_numpy(float) + t * (B[c].to_numpy(float) - A[c].to_numpy(float))
                out[c] = np.rint(v) if c in intcols else v
        for target_col, fn, guard in identities:
            g = np.asarray(guard(out)); v = np.asarray(fn(out), dtype=float)
            x = out[target_col].to_numpy(float).copy(); x[g] = v[g]
            ref = normal_df[target_col].to_numpy(float)
            x = np.clip(np.nan_to_num(x, nan=0.0), ref.min(), ref.max())
            out[target_col] = np.rint(x) if np.allclose(ref, np.round(ref)) else x
        bad = np.zeros(len(out), bool)
        for target_col, fn, guard in identities:
            g = np.asarray(guard(out)); v = np.asarray(fn(out), dtype=float); x = out[target_col].to_numpy(float)
            bad |= g & ~(np.abs(x - v) <= IDENTITY_VALIDATION_TOL * np.maximum(np.abs(v), 1))
        for c1, c2 in ineq:
            bad |= (out[c1] > out[c2] + 1e-9).to_numpy()
        for c, lvl, tc, val in deps:
            bad |= ((out[c].astype(str) == lvl) & (out[tc] != val)).to_numpy()
        new.append(out[~bad])
    P = pd.concat([S] + new, ignore_index=True)
    for c in num:
        P[c] = P[c].astype(normal_df[c].dtype)
    P = P.drop_duplicates().reset_index(drop=True)

    P_hash = _canonical_hashes(P, feats, num)
    normal_hash = _canonical_hashes(normal_df[feats], feats, num)
    P = P.loc[~np.isin(P_hash, normal_hash)].reset_index(drop=True)
    if len(P) < N:
        raise ValueError(
            f"Utility reweighting left only {len(P)} novel rows for {N} outputs; "
            "refusing to resample or duplicate rows."
        )
    G2 = _tailness_vs(S, P, list(tilts))
    lw2 = sum(tilts[c] * G2[c] for c in tilts)
    w = np.exp(lw2 - lw2.max())
    keys = np.log(rng.random(len(P))) / w
    idx = np.argsort(-keys)[:N]
    return P.iloc[idx].reset_index(drop=True), len(P)

def reweight_stage(train_df, label_col, feature_cols, categorical_cols, synthetic, generate_fn,
                   anchor_mode="uniform", seed=RANDOM_STATE):
    nrm = train_df[train_df[label_col] == 0]; att = train_df[train_df[label_col] == 1]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    votes, gains = [], []
    for b in range(REWEIGHT_BAGS):
        vn = nrm.sample(frac=REWEIGHT_VAL_FRAC, random_state=b)
        va = att.sample(frac=REWEIGHT_VAL_FRAC, random_state=b)
        tr_gen = train_df.drop(vn.index).drop(va.index)
        gn = tr_gen[tr_gen[label_col] == 0][feature_cols]
        print(f"      Re-weighting bag {b + 1}/{REWEIGHT_BAGS}: re-fitting generator on internal split ...")
        syn_in = generate_fn(tr_gen)
        syn_in = syn_in.drop(columns=["id"], errors="ignore")[feature_cols]
        V = pd.concat([vn, va])[feature_cols]; yv = np.r_[np.zeros(len(vn)), np.ones(len(va))]
        P = _pseudo_anomalies(vn[feature_cols].reset_index(drop=True), gn, feature_cols, categorical_cols, seed=b)
        t, s0, s1 = reweight_search(syn_in, V, yv, P, feature_cols, categorical_cols)
        print(f"        bag {b + 1}: seen AUPRC {s0[0]:.4f} -> {s1[0]:.4f}, pseudo {s0[1]:.4f} -> {s1[1]:.4f}; "
              f"tilts {[(c, v) for c, v in t.items() if v]}")
        votes.append(t); gains.append(s1[0] - s0[0])
    mean_gain = float(np.mean(gains))
    tilts = {}
    #   Union of columns over all bags, in first-seen order
    for c in dict.fromkeys(col for vote in votes for col in vote):
        cnt = pd.Series([v.get(c, 0.0) for v in votes]).value_counts().drop(0.0, errors="ignore")
        tilts[c] = float(cnt.index[0]) if len(cnt) and cnt.iloc[0] >= REWEIGHT_QUORUM else 0.0
    active = {c: v for c, v in tilts.items() if v}
    if mean_gain < REWEIGHT_MIN_GAIN or not active:
        print(f"      Re-weighting gate: mean gain {mean_gain:+.4f}, tilts {active} -> not applied")
        synthetic.attrs["reweight"] = {"mean_gain": mean_gain, "applied": False}
        return synthetic
    base = synthetic.drop(columns=["id"], errors="ignore")[feature_cols]
    out, pool_size = utility_adasyn(base, nrm[feature_cols], feature_cols, categorical_cols, active,
                                    len(base), anchor_mode, seed=seed)
    print(f"      Re-weighting gate: mean gain {mean_gain:+.4f} -> applied; tilts {active}; anchor mode '{anchor_mode}'; "
          f"pool {pool_size} distinct rule-checked rows -> {len(out)} drawn without replacement "
          f"({len(out.drop_duplicates())} unique)")
    out.attrs = dict(synthetic.attrs)
    out.attrs["reweight"] = {"mean_gain": mean_gain, "applied": True, "tilts": active,
                             "anchor_mode": anchor_mode, "pool_size": int(pool_size)}
    return add_id_column(out)

# ============================================================
# PIPELINE RUNNERS
# ============================================================

def generate_dataset_synthetic_normals(
    train_df,
    label_col,
    col_types,
    target_n,
    candidate_over_sample_factor,
    seed=RANDOM_SEED,
    realism_weight=None,
    mixture_adapt_strength=None,
    attack_overlap_percentile=None,
    use_tvae=False,
    coverage_filter=False,
    empirical_share=0.0,
    base_mixture=None,
):
    feature_cols = col_types["all_feature"]
    normal_df = train_df.loc[train_df[label_col] == 0, feature_cols].copy().reset_index(drop=True)
    attack_df = train_df.loc[train_df[label_col] == 1, feature_cols].copy().reset_index(drop=True)

    synth, diag = generate_synthetic_normals(
        normal_df=normal_df,
        attack_df=attack_df,
        target_n=target_n,
        candidate_over_sample_factor=candidate_over_sample_factor,
        random_state=seed,
        realism_weight=realism_weight,
        mixture_adapt_strength=mixture_adapt_strength,
        attack_overlap_percentile=attack_overlap_percentile,
        use_tvae=use_tvae,
        coverage_filter=coverage_filter,
        empirical_share=empirical_share,
        base_mixture=base_mixture,
    )
    synth.attrs["generation_diagnostics"] = diag
    return add_id_column(synth)

def run(
    dataset_key=None,
    do_eval=True,
    seed=None,
    realism_weight=None,
    mixture_adapt_strength=None,
    attack_overlap_percentile=None,
):
    global HYBRID_RFOD_WEIGHT
    HYBRID_RFOD_WEIGHT = HYBRID_RFOD_WEIGHTS[dataset_key]
    import imblearn
    if USE_TVAE[dataset_key]:
        import sdv
        import torch
    if do_eval:
        from pyod.models.ecod import ECOD
    cfg = DATASETS[dataset_key]
    candidate_factor = CANDIDATE_OVER_SAMPLE_FACTORS[dataset_key]
    if realism_weight is None:
        realism_weight = REALISM_WEIGHTS[dataset_key]

    if mixture_adapt_strength is None:
        mixture_adapt_strength = MIXTURE_ADAPT_STRENGTHS[dataset_key]
    
    if seed is None:
        seed = GENERATION_SEEDS[dataset_key]

    if attack_overlap_percentile is None:
        attack_overlap_percentile = ATTACK_OVERLAP_PERCENTILES[dataset_key]

    if not 0.0 <= attack_overlap_percentile <= 100.0:
        raise ValueError(
            "attack_overlap_percentile must be between 0 and 100."
        )

    print("\n" + "=" * 72)
    print(f"Dataset:                  {dataset_key}")
    print(f"Seed:                     {seed}")
    print(f"Realism weight:           {realism_weight}")
    print(f"Mixture adapt strength:   {mixture_adapt_strength}")
    print(f"Candidate oversample:      {candidate_factor}")
    print(f"Attack overlap percentile: {attack_overlap_percentile}")
    print(f"Hybrid RFOD selection:     enabled={HYBRID_ENABLED}, near_fraction={HYBRID_NEAR_FRACTION}, "
          f"weight={HYBRID_RFOD_WEIGHT}")
    print(f"Re-weighting strength:     {REWEIGHT_STRENGTHS[dataset_key]}")
    print(
        f"Stable Mixed-Type Clustered Synthetic-Normal Generation — "
        f"{cfg['name']} (seed={seed})"
    )
    print("=" * 72)

    df, col_types = load_dataset(cfg)

    if do_eval:
        train_df, eval_df = local_train_eval_split(df, cfg["label_col"], random_state=seed)
    else:
        train_df, eval_df = df, None

    normal_train = get_normal_data(
        train_df,
        cfg["label_col"],
    )

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")
    print(
        f"  Base generator mix (pre-normalization): "
        f"statistical={BASE_STATISTICAL:.0%}, "
        f"Cluster-SMOTENC={BASE_SMOTE:.0%}, "
        f"TVAE={BASE_TVAE:.0%} (floor={MIN_TVAE_WEIGHT:.0%})"
    )
    print(
        f"  Candidate oversampling: "
        f"{candidate_factor:.2f}x, "
        f"adapt_strength={mixture_adapt_strength}"
    )

    t0 = time.time()

    def generate_fn(frame):
        return generate_dataset_synthetic_normals(
            frame,
            cfg["label_col"],
            col_types,
            cfg["target_n"],
            seed=seed,
            candidate_over_sample_factor=candidate_factor,
            realism_weight=realism_weight,
            mixture_adapt_strength=mixture_adapt_strength,
            attack_overlap_percentile=attack_overlap_percentile,
            use_tvae=USE_TVAE[dataset_key],
            coverage_filter=COVERAGE_FILTER[dataset_key],
            empirical_share=EMPIRICAL_SHARES[dataset_key],
            base_mixture=BASE_MIXTURES[dataset_key],
        )

    synthetic = generate_fn(train_df)

    normal_train_df = train_df[train_df[cfg["label_col"]] == 0][col_types["all_feature"]].copy()
    numeric_feature_cols = [c for c in col_types["all_feature"] if c not in cfg["categorical_cols"]]

    global REWEIGHT_STRENGTH
    REWEIGHT_STRENGTH = REWEIGHT_STRENGTHS[dataset_key]
    if True: 
        synthetic = reweight_stage(
            train_df, cfg["label_col"], col_types["all_feature"], cfg["categorical_cols"],
            synthetic, generate_fn, anchor_mode=REWEIGHT_ANCHOR[dataset_key],
        )

    final_domain_audit = audit_domain_consistency(
        synthetic.drop(columns=["id"], errors="ignore")[col_types["all_feature"]],
        normal_train_df,
        numeric_feature_cols,
        cfg["categorical_cols"],
    )
    final_novelty_audit = audit_no_copies_or_duplicates(
        synthetic,
        normal_train_df,
        col_types["all_feature"],
        numeric_feature_cols,
    )
    synthetic.attrs["final_domain_audit"] = final_domain_audit
    synthetic.attrs["final_novelty_audit"] = final_novelty_audit

    gen_time = time.time() - t0

    print(
        f"  Generation time: {gen_time:.1f}s"
    )

    if len(synthetic) != int(cfg["target_n"]):
        raise AssertionError(
            f"Final output row count {len(synthetic)} != required {cfg['target_n']}."
        )

    validate_synthetic(
        synthetic,
        col_types,
        cfg["target_n"],
    )

    print("  ✓ Validation passed")

    out_path = os.path.join(
        OUTPUT_DIR,
        f"synthetic_normals_{dataset_key}_{'validation' if do_eval else 'submission'}.csv",
    )

    synthetic.to_csv(
        out_path,
        index=False,
    )

    print(f"  Saved to {out_path}")
    diagnostics = synthetic.attrs.get("generation_diagnostics", {})
    cluster_report = diagnostics.get("cluster_diagnostics")
    if cluster_report is not None:
        cluster_report.to_csv(out_path.replace(".csv", "_clusters.csv"), index=False)
    import importlib.metadata
    versions = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "sdv", "torch", "pyod"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    Path(out_path.replace(".csv", "_run.json")).write_text(json.dumps({
        "mode": "validation" if do_eval else "submission", "seed": seed,
        "training_rows": len(train_df), "versions": versions,
        "selection_version": "density_normalized_bounded_reward_v2",
        "pipeline_version": "no_exact_row_bootstrap_copy_free_candidates", "consistency": diagnostics.get("consistency", {}), "final_domain_audit": synthetic.attrs.get("final_domain_audit", {}), "final_novelty_audit": synthetic.attrs.get("final_novelty_audit", {}), "reweight": synthetic.attrs.get("reweight", {}), "base_mixture": BASE_MIXTURES[dataset_key], "use_tvae": USE_TVAE[dataset_key],
        "coverage_filter": COVERAGE_FILTER[dataset_key],
        "empirical_share": EMPIRICAL_SHARES[dataset_key],
        "copy_free_generation": True,
        "coverage_rules": diagnostics.get("coverage_rules", {}),
        "tcp_timing_identity_repair_applied": False,
        "final_clip_applied": False,
        "hybrid_rfod": {"enabled": HYBRID_ENABLED, "near_fraction": HYBRID_NEAR_FRACTION,
                        "weight": HYBRID_RFOD_WEIGHT, "n_trees": RFOD_N_TREES,
                        "min_samples_leaf": RFOD_MIN_SAMPLES_LEAF, "max_features": RFOD_MAX_FEATURES,
                        "seed": RFOD_SEED,
                        "selection": diagnostics.get("selection_diagnostics", {}).get("hybrid", {})},
        "reweight_strength": REWEIGHT_STRENGTHS[dataset_key],
        "attack_reward_cap_percentile": ATTACK_REWARD_CAP_PERCENTILE,
        "local_scale_floor_percentile": LOCAL_SCALE_FLOOR_PERCENTILE,
        "selection_summary": {k: v for k, v in diagnostics.get("selection_diagnostics", {}).items()
                              if not isinstance(v, np.ndarray)},
        "candidate_factor": candidate_factor, "realism_weight": realism_weight,
        "mixture_adapt_strength": mixture_adapt_strength,
        "attack_overlap_percentile": attack_overlap_percentile,
        "selected_generator_counts": diagnostics.get("selected_generator_counts", {}),
    }, indent=2))

    results = {
        "generation_time": gen_time,
        "base_statistical": BASE_STATISTICAL,
        "base_smote": BASE_SMOTE,
        "base_tvae": BASE_TVAE,
        "min_tvae_weight": MIN_TVAE_WEIGHT,
        "seed": seed,
        "realism_weight": realism_weight,
        "mixture_adapt_strength": mixture_adapt_strength,
        "attack_overlap_percentile": float(attack_overlap_percentile),
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
            f"  Score S: {results.get('score_S', 'n/a')}"
        )
        print(
            f"  Mean KS: {results['mean_ks_distance']:.5f}"
        )
        print(
            f"  Corr RMSE: {results['correlation_rmse']:.5f}"
        )

    Path(out_path.replace(".csv", "_metrics.json")).write_text(json.dumps(results, indent=2))
    return synthetic, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["unsw", "nslkdd", "both"], default="both")
    parser.add_argument("--mode", choices=["validation", "submission"], default="submission")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.data_dir is not None:
        DATASETS["unsw"]["path"] = str(args.data_dir / "unsw_nb15_train.csv")
        DATASETS["nslkdd"]["path"] = str(args.data_dir / "nslkdd_train.csv")
    if args.output_dir is not None:
        OUTPUT_DIR = str(args.output_dir.resolve())
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    keys = ["unsw", "nslkdd"] if args.dataset == "both" else [args.dataset]
    for key in keys:
        run(dataset_key=key, seed=args.seed, do_eval=args.mode == "validation")