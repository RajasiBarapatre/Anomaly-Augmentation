"""
Approach 26 — TVAE-Floor + Concentration-Adaptive Cluster Mixture
(built directly on Approach 25's Adaptive Local Generator Mixture)

Goal
----
Approach 25 raised combined-mean score substantially (UNSW-NB15 jumped from
0.484 to 0.532) but NSL-KDD regressed slightly (0.93159 -> 0.92637). PCA
visualization of Approach 25's synthetic output showed NSL-KDD's synthetic
points concentrated tightly on the dense core of the normal blob, with
thinner coverage of the sparser, more scattered normal points -- consistent
with the code-level finding that dense/stable clusters (NSL-KDD's profile)
get pushed to very low TVAE shares by Approach 25's heterogeneity-driven
adaptive weighting, on top of an already-thin 15% global TVAE baseline.

Approach 26 makes three targeted, dataset-agnostic changes on top of
Approach 25's pipeline (clustering -> adaptive mixture -> Approach-20-style
selection -> final clip), all driven by measured properties of the data
rather than any dataset-specific branching:

  1. TVAE BASELINE + FLOOR
     BASE_TVAE raised from 0.15 to 0.30 (BASE_STATISTICAL trimmed to
     compensate), and a hard per-cluster floor (MIN_TVAE_WEIGHT) is enforced
     after adaptive weighting, so no cluster -- however dense/stable -- ever
     drops below a meaningful TVAE share feeding the selection stage.

  2. REDUCED ADAPTATION STRENGTH
     MIXTURE_ADAPT_STRENGTH lowered from 0.75 to 0.45, so dense/stable
     clusters are not pushed as aggressively away from TVAE as before,
     while UNSW-NB15's heterogeneous clusters still get adapted mixtures.

  3. CONCENTRATION-ADAPTIVE CLUSTER COUNT
     choose_cluster_count() now scales with a measured statistic -- the
     fraction of normal-data variance captured by the first two principal
     components. High concentration (data lies mostly along a few dominant
     axes, e.g. NSL-KDD's well-separated blobs) yields fewer, larger
     clusters instead of over-fragmenting already-coherent structure; low
     concentration (UNSW-NB15's entangled continuum) keeps the finer split
     that helped it in Approach 25.

The attack set is still used for candidate selection only, not for fitting
any generator. All three changes are continuous functions of measured data
properties -- no dataset name or dataset-specific rule appears anywhere.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTENC
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset,
    local_train_eval_split,
    get_normal_data,
    validate_synthetic,
    add_id_column,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

# Candidate pool is deliberately larger than the requested output.
CANDIDATE_OVER_SAMPLE_FACTOR = 1.50

# Local clustering.
MAX_CLUSTERS = 24
MIN_CLUSTER_SIZE = 80
KMEANS_BATCH_SIZE = 2048
KMEANS_N_INIT = 5

# Generator mixture is adapted around this baseline.
# NOTE (Approach 26): Approach 25's BASE_TVAE=0.15 starved dense/stable
# clusters (e.g. NSL-KDD's well-separated blobs) of TVAE material, since the
# heterogeneity-driven adaptation pushes TVAE weight even lower than the
# baseline for such clusters -- exactly where a18 showed TVAE-based
# selection worked best. Raising the baseline and adding a floor (below)
# restores a meaningful TVAE presence everywhere without removing the
# adaptive routing that helped UNSW-NB15.
BASE_STATISTICAL = 0.35
BASE_SMOTE = 0.35
BASE_TVAE = 0.30

# Local adaptation strength. Reduced from 0.75 so dense/stable clusters are
# not pushed as aggressively away from TVAE.
MIXTURE_ADAPT_STRENGTH = 0.45

# Hard floor: no cluster's final TVAE weight is allowed to drop below this,
# regardless of how the adaptive heuristic scores it. Guarantees the
# Approach-20-style selection stage always has a real TVAE pool to draw
# strong candidates from.
MIN_TVAE_WEIGHT = 0.20             

# Statistical generator.
CORR_SHRINKAGE = 0.15
MIN_STD = 1e-8

# Approach-20-style selection.
PCA_VARIANCE = 0.95
PCA_MAX_COMPONENTS = 40
REALISM_WEIGHT = 0.75
REALISM_K = 5
ATTACK_OVERLAP_PERCENTILE = 2.5
MIN_TVAE_RETENTION = 0.85  # retained for compatibility / reporting

# Final sanity clip.
FINAL_CLIP = True
FINAL_CLIP_PERCENTILE = 99.9


# ============================================================
# BASIC HELPERS
# ============================================================

def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)


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
        # sparse_output is not available in some older sklearn versions.
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

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    ct.fit(df)
    return ct


def encode_for_geometry(df, preprocessor):
    x = preprocessor.transform(df)
    return np.asarray(x, dtype=float)


# ============================================================
# LOCAL CLUSTERING
# ============================================================

def measure_concentration(x_normal, sample_cap=5000, random_state=RANDOM_STATE):
    """
    Approach 26: measure how much of the normal data's variance is captured
    by its first two principal components. High concentration (data lies
    mostly along one or two dominant axes, e.g. NSL-KDD's few well-separated
    blobs) means the natural structure is already low-dimensional/coarse,
    so it should not be fragmented into many small local clusters. Low
    concentration (variance spread across many directions, e.g. UNSW's
    entangled continuum) benefits from the current finer-grained split.

    This is measured directly from the data, with no dataset-specific
    branching -- it is used purely to scale the existing cluster-count
    formula.
    """
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


def choose_cluster_count(n, concentration=None):
    if n < MIN_CLUSTER_SIZE * 2:
        return 1

    k = int(np.sqrt(max(n, 1) / 250.0))

    if concentration is not None:
        # concentration in [0, 1]; scale factor in roughly [0.5, 1.3].
        # High concentration -> fewer clusters; low concentration -> keep
        # (or slightly increase) the base cluster count.
        scale = float(np.clip(1.3 - concentration, 0.5, 1.3))
        k = int(round(k * scale))

    k = max(2, k)
    k = min(MAX_CLUSTERS, k)
    k = min(k, max(2, n // MIN_CLUSTER_SIZE))
    return k


def fit_local_clusters(x_normal, random_state=RANDOM_STATE):
    n = len(x_normal)
    concentration = measure_concentration(x_normal, random_state=random_state)
    k = choose_cluster_count(n, concentration=concentration)
    print(f"      Measured PC1+PC2 concentration: {concentration:.3f} -> cluster count: {k}")

    if k <= 1:
        labels = np.zeros(n, dtype=int)
        centers = np.mean(x_normal, axis=0, keepdims=True)
        return None, labels, centers

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=min(KMEANS_BATCH_SIZE, max(256, n)),
        n_init=KMEANS_N_INIT
    )
    labels = km.fit_predict(x_normal)
    return km, labels, km.cluster_centers_


# ============================================================
# STATISTICAL LOCAL GENERATOR
# ============================================================

def rank_normal_scores(values):
    """
    Convert a numeric vector to approximately N(0,1) scores through ranks.
    This preserves empirical ordering while making local covariance easier
    to estimate.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n <= 1:
        return np.zeros(n)

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)

    p = (ranks + 0.5) / n
    # scipy is intentionally avoided here; use an accurate approximation
    # through sklearn-compatible normal quantiles from numpy where possible.
    from statistics import NormalDist
    nd = NormalDist()
    return np.array([nd.inv_cdf(float(q)) for q in p], dtype=float)


def inverse_rank_normal(scores, reference):
    scores = np.asarray(scores, dtype=float)
    ref = np.asarray(reference, dtype=float)

    if len(ref) == 0:
        return np.zeros_like(scores)

    ref_sorted = np.sort(ref)
    from statistics import NormalDist
    nd = NormalDist()

    out = np.empty(len(scores), dtype=float)
    for i, s in enumerate(scores):
        p = min(max(nd.cdf(float(s)), 1e-6), 1 - 1e-6)
        pos = p * (len(ref_sorted) - 1)
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        if lo == hi:
            out[i] = ref_sorted[lo]
        else:
            w = pos - lo
            out[i] = (1 - w) * ref_sorted[lo] + w * ref_sorted[hi]
    return out


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
        return pd.DataFrame(
            np.zeros((n_samples, len(numeric_cols))),
            columns=numeric_cols
        )

    if len(x) < 3 or len(numeric_cols) == 0:
        idx = rng.integers(0, len(x), size=n_samples)
        return pd.DataFrame(x[idx], columns=numeric_cols)

    # Rank-normalize each marginal.
    z = np.zeros_like(x)
    for j in range(x.shape[1]):
        z[:, j] = rank_normal_scores(x[:, j])

    # Shrunk local correlation/covariance.
    if z.shape[1] == 1:
        cov = np.array([[1.0]])
    else:
        cov = np.cov(z, rowvar=False)
        cov = np.atleast_2d(cov)
        diag = np.diag(np.diag(cov))
        cov = (1.0 - CORR_SHRINKAGE) * cov + CORR_SHRINKAGE * diag

    cov = nearest_psd(cov)

    mean = np.mean(z, axis=0)
    sampled_z = rng.multivariate_normal(mean, cov, size=n_samples)

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
    num = gaussian_local_numeric_sample(
        cluster_df, numeric_cols, n_samples, random_state
    )
    cat = categorical_local_sample(
        cluster_df, categorical_cols, n_samples, random_state + 1
    )

    pieces = []
    if numeric_cols:
        pieces.append(num[numeric_cols])
    if categorical_cols:
        pieces.append(cat[categorical_cols])

    if not pieces:
        return pd.DataFrame(index=np.arange(n_samples))

    return pd.concat(pieces, axis=1)


# ============================================================
# CLUSTER-SMOTE GENERATION
# ============================================================

def smote_generate_cluster(cluster_df, numeric_cols, categorical_cols,
                            n_samples, random_state=RANDOM_STATE):
    """
    Cluster-local SMOTENC. If a cluster is too small or lacks categorical
    variation, fall back to local interpolation / bootstrap.
    """
    if n_samples <= 0:
        return pd.DataFrame(columns=cluster_df.columns)

    work = cluster_df.copy().reset_index(drop=True)

    if len(work) < 3 or len(numeric_cols) == 0:
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, len(work), size=n_samples)
        return work.iloc[idx].reset_index(drop=True)

    # For SMOTENC, categorical columns must be integer encoded.
    enc = work.copy()

    category_maps = {}
    for c in categorical_cols:
        vals = pd.Series(enc[c].astype("object").unique())
        mapping = {v: i for i, v in enumerate(vals)}
        category_maps[c] = {i: v for v, i in mapping.items()}
        enc[c] = enc[c].astype("object").map(mapping).fillna(0).astype(int)

    # Build a synthetic minority target only for the local normal cluster.
    # SMOTENC requires class labels; use a 1/0 split created from the local
    # data itself, then discard the labels. To avoid inventing semantics,
    # the split is based on a random half and only provides interpolation.
    rng = np.random.default_rng(random_state)
    y = rng.integers(0, 2, size=len(enc))

    if len(np.unique(y)) < 2:
        idx = rng.integers(0, len(work), size=n_samples)
        return work.iloc[idx].reset_index(drop=True)

    k_neighbors = max(1, min(5, len(enc) // 4))

    cat_idx = [enc.columns.get_loc(c) for c in categorical_cols]

    try:
        sm = SMOTENC(
            categorical_features=cat_idx if cat_idx else "auto",
            k_neighbors=k_neighbors,
            random_state=random_state
        )

        x_res, y_res = sm.fit_resample(enc, y)

        extra = len(x_res) - len(enc)
        if extra <= 0:
            idx = rng.integers(0, len(work), size=n_samples)
            return work.iloc[idx].reset_index(drop=True)

        synth = pd.DataFrame(x_res.iloc[len(enc):] if hasattr(x_res, "iloc")
                             else x_res[len(enc):],
                             columns=enc.columns)

        # If SMOTENC generated fewer points than requested, bootstrap.
        if len(synth) < n_samples:
            idx = rng.integers(0, len(synth), size=n_samples - len(synth))
            synth = pd.concat(
                [synth, synth.iloc[idx].copy()],
                ignore_index=True
            )

        synth = synth.iloc[:n_samples].copy()

        for c in categorical_cols:
            synth[c] = synth[c].round().astype(int).map(category_maps[c])

        # Numeric columns may have tiny floating artifacts.
        for c in numeric_cols:
            synth[c] = pd.to_numeric(synth[c], errors="coerce")

        for c in work.columns:
            if c not in synth:
                synth[c] = work[c].iloc[0]

        return synth[work.columns].reset_index(drop=True)

    except Exception:
        # Robust fallback: random local interpolation between two rows.
        out = pd.DataFrame(index=np.arange(n_samples), columns=work.columns)
        a = rng.integers(0, len(work), size=n_samples)
        b = rng.integers(0, len(work), size=n_samples)
        alpha = rng.random(n_samples)

        for c in work.columns:
            if c in numeric_cols:
                va = pd.to_numeric(work.iloc[a][c], errors="coerce").to_numpy(float)
                vb = pd.to_numeric(work.iloc[b][c], errors="coerce").to_numpy(float)
                out[c] = (1 - alpha) * va + alpha * vb
            else:
                choose_a = rng.random(n_samples) < 0.5
                va = work.iloc[a][c].to_numpy()
                vb = work.iloc[b][c].to_numpy()
                out[c] = np.where(choose_a, va, vb)

        return out.reset_index(drop=True)


# ============================================================
# TVAE
# ============================================================

def tvae_generate(normal_df, n_samples, random_state=RANDOM_STATE):
    if n_samples <= 0:
        return pd.DataFrame(columns=normal_df.columns)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(normal_df)

    model = TVAESynthesizer(
        metadata,
        epochs=300,
        enforce_min_max_values=True,
        enforce_rounding=True
    )

    model.fit(normal_df)
    synth = model.sample(num_rows=n_samples)

    return synth[normal_df.columns].copy()


# ============================================================
# ADAPTIVE MIXTURE
# ============================================================

def cluster_structure_features(cluster_df, all_normal_df, numeric_cols):
    """
    Estimate local structure from the normal data only.

    The returned features are dataset-agnostic:
      - density: relative local cluster size
      - stability: inverse robust dispersion
      - skewness: average absolute robust skew
      - separation: how isolated this cluster is from the global normal center

    These are used only to continuously adapt generator weights. No dataset
    name or dataset-specific rule is used.
    """
    if len(cluster_df) == 0:
        return {
            "density": 1.0,
            "stability": 0.5,
            "skewness": 0.0,
            "separation": 0.5
        }

    density = len(cluster_df) / max(len(all_normal_df), 1)

    if not numeric_cols or len(cluster_df) < 5:
        return {
            "density": float(density),
            "stability": 0.5,
            "skewness": 0.0,
            "separation": 0.5
        }

    x = safe_numeric(cluster_df, numeric_cols).to_numpy(float)
    global_x = safe_numeric(all_normal_df, numeric_cols).to_numpy(float)

    # Robust dispersion relative to the global marginal scale.
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) + MIN_STD

    global_med = np.median(global_x, axis=0)
    global_mad = np.median(
        np.abs(global_x - global_med),
        axis=0
    ) + MIN_STD

    relative_dispersion = np.median(
        mad / global_mad
    )

    stability = float(
        1.0 / (1.0 + max(relative_dispersion, 0.0))
    )
    stability = float(np.clip(stability, 0.0, 1.0))

    # Robust skew proxy: mean/median asymmetry normalized by MAD.
    q25 = np.percentile(x, 25, axis=0)
    q50 = np.percentile(x, 50, axis=0)
    q75 = np.percentile(x, 75, axis=0)

    robust_skew = np.abs(
        (q75 + q25 - 2.0 * q50) /
        (q75 - q25 + MIN_STD)
    )
    skewness = float(np.mean(np.clip(robust_skew, 0.0, 5.0)) / 5.0)

    # Separation from the global normal center.
    global_scale = np.median(
        global_mad + MIN_STD
    )
    center_offset = np.median(
        np.abs(med - global_med) / global_scale
    )
    separation = float(
        np.clip(center_offset / 5.0, 0.0, 1.0)
    )

    return {
        "density": float(density),
        "stability": stability,
        "skewness": skewness,
        "separation": separation
    }


def adaptive_weights(features, median_cluster_size):
    """
    Universal continuous generator routing.

    No dataset-specific branch exists.

    Statistical generation is favored when the local empirical distribution
    is dense/stable. Cluster-SMOTE is favored for sparse or well-separated
    regions. TVAE receives a small increase where local structure is
    heterogeneous/skewed, but never dominates.

    The transformation is deliberately bounded so that Approach 24's strong
    UNSW behavior is not destroyed by an overly aggressive routing change.
    """
    density = features["density"]
    stability = features["stability"]
    skewness = features["skewness"]
    separation = features["separation"]

    # Convert density to a bounded relative-density signal.
    rel_density = density / max(median_cluster_size, 1.0)
    density_signal = np.tanh(np.log1p(max(rel_density, 1e-8)))

    # Heterogeneity signal: skew + instability + separation.
    heterogeneity = np.clip(
        0.45 * skewness +
        0.30 * (1.0 - stability) +
        0.25 * separation,
        0.0,
        1.0
    )

    # Start from Approach 24's useful mixture, then make only modest,
    # continuous local adjustments.
    stat = BASE_STATISTICAL
    smote = BASE_SMOTE
    tvae = BASE_TVAE

    # Dense/stable regions -> empirical/statistical generator.
    stat += MIXTURE_ADAPT_STRENGTH * (
        0.10 * density_signal +
        0.08 * (stability - 0.5)
    )

    # Sparse/separated regions -> SMOTE to preserve local support.
    smote += MIXTURE_ADAPT_STRENGTH * (
        -0.08 * density_signal +
        0.10 * (separation - 0.5)
    )

    # Heterogeneous/skewed regions -> slightly more TVAE.
    tvae += MIXTURE_ADAPT_STRENGTH * (
        0.08 * (heterogeneity - 0.5)
    )

    weights = np.array([stat, smote, tvae], dtype=float)
    weights = np.clip(weights, 0.05, None)
    weights /= weights.sum()

    # Enforce the TVAE floor (Approach 26): if adaptation pushed TVAE below
    # MIN_TVAE_WEIGHT, top it back up and rescale statistical/SMOTE
    # proportionally so the three weights still sum to 1.
    if weights[2] < MIN_TVAE_WEIGHT:
        deficit = MIN_TVAE_WEIGHT - weights[2]
        remaining = weights[0] + weights[1]
        if remaining > 1e-8:
            weights[0] -= deficit * (weights[0] / remaining)
            weights[1] -= deficit * (weights[1] / remaining)
        weights[2] = MIN_TVAE_WEIGHT
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()

    return {
        "statistical": float(weights[0]),
        "smote": float(weights[1]),
        "tvae": float(weights[2])
    }


# ============================================================
# DISCRIMINATIVE PCA GEOMETRY — APPROACH 20 STYLE
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

    pca_full = PCA(
        n_components=min(PCA_MAX_COMPONENTS, x_all.shape[1]),
        random_state=RANDOM_STATE
    )
    z_all = pca_full.fit_transform(x_all)

    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = np.searchsorted(cumulative, PCA_VARIANCE) + 1
    n_comp = max(1, min(n_comp, z_all.shape[1]))

    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
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

    # Normalize weights around mean 1 to preserve comparable distance scales.
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


def select_candidates(normal_real, attack_real, candidates,
                      numeric_cols, categorical_cols):
    pre, pca, fisher, z_n, z_a = fit_discriminative_pca(
        normal_real,
        attack_real,
        numeric_cols,
        categorical_cols
    )

    z_n_w = z_n * np.sqrt(fisher)
    z_a_w = z_a * np.sqrt(fisher)
    z_c_w = transform_weighted(candidates, pre, pca, fisher)

    attack_dist = knn_distance(z_c_w, z_a_w, k=3)
    realism_dist = knn_distance(z_c_w, z_n_w, k=REALISM_K)

    # Hard overlap gate:
    # reject points closer to attacks than almost all genuine normals are.
    normal_to_attack = knn_distance(z_n_w, z_a_w, k=3)
    threshold = np.percentile(
        normal_to_attack,
        ATTACK_OVERLAP_PERCENTILE
    )

    keep = attack_dist >= threshold

    if keep.sum() < max(50, int(0.20 * len(candidates))):
        # Never allow the gate to collapse the candidate pool completely.
        # Fall back to a very conservative percentile gate.
        threshold = np.percentile(normal_to_attack, 1.0)
        keep = attack_dist >= threshold

    idx = np.where(keep)[0]

    if len(idx) == 0:
        idx = np.arange(len(candidates))

    # Standardized two-objective score.
    ad = attack_dist[idx]
    rd = realism_dist[idx]

    ad_z = (ad - np.mean(ad)) / (np.std(ad) + 1e-8)
    rd_z = (rd - np.mean(rd)) / (np.std(rd) + 1e-8)

    score = ad_z - REALISM_WEIGHT * rd_z

    order = idx[np.argsort(-score)]

    return order, {
        "attack_distance": attack_dist,
        "realism_distance": realism_dist,
        "gate_threshold": threshold,
        "kept_after_gate": int(keep.sum())
    }


# ============================================================
# FINAL CLIP
# ============================================================

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
        out[c] = out[c].clip(lo, hi)

    return out


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_approach24(
    normal_df,
    attack_df,
    target_n=None,
    random_state=RANDOM_STATE
):
    """
    Parameters
    ----------
    normal_df : pandas.DataFrame
        Real normal-only training data.
    attack_df : pandas.DataFrame
        Seen attacks, used only for candidate selection.
    target_n : int or None
        Number of synthetic normal rows. Defaults to len(normal_df).

    Returns
    -------
    pandas.DataFrame
        Synthetic normal traffic.
    dict
        Diagnostics.
    """
    set_seed(random_state)

    normal_df = normal_df.copy().reset_index(drop=True)
    attack_df = attack_df.copy().reset_index(drop=True)

    if target_n is None:
        target_n = len(normal_df)

    numeric_cols, categorical_cols = split_columns(normal_df)

    # Ensure attack data has matching columns.
    attack_df = attack_df[normal_df.columns].copy()

    # --------------------------------------------------------
    # 1. Local clustering of REAL NORMALS
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 2. Decide how many candidates each local cluster gets
    # --------------------------------------------------------
    candidate_n = int(np.ceil(
        target_n * CANDIDATE_OVER_SAMPLE_FACTOR
    ))

    # Allocation proportional to cluster size, with a mild
    # square-root boost for smaller clusters.
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

    # Correct rounding.
    diff = candidate_n - sum(allocations.values())
    if diff != 0:
        largest = max(allocations, key=allocations.get)
        allocations[largest] += diff

    # --------------------------------------------------------
    # 3. Generate each cluster with an adaptive mixture
    # --------------------------------------------------------
    all_candidates = []
    diagnostics = []

    # TVAE is trained once globally rather than once per cluster.
    # This keeps the approach computationally practical.
    tvae_total = max(0, int(round(candidate_n * BASE_TVAE)))

    global_tvae = None
    if tvae_total > 0:
        try:
            global_tvae = tvae_generate(
                normal_df,
                tvae_total,
                random_state + 1000
            )
        except Exception:
            global_tvae = None

    tvae_cursor = 0

    for k in unique_labels:
        cluster_df = normal_df.iloc[
            np.where(labels == k)[0]
        ].reset_index(drop=True)

        n_k = allocations[int(k)]

        structure = cluster_structure_features(
            cluster_df,
            normal_df,
            numeric_cols
        )

        weights = adaptive_weights(
            structure,
            median_cluster_size
        )

        n_stat = int(round(n_k * weights["statistical"]))
        n_smote = int(round(n_k * weights["smote"]))
        n_tvae = n_k - n_stat - n_smote

        # Statistical generator.
        stat_df = statistical_generate_cluster(
            cluster_df,
            numeric_cols,
            categorical_cols,
            n_stat,
            random_state + 2000 + int(k)
        )

        # Cluster-SMOTE.
        smote_df = smote_generate_cluster(
            cluster_df,
            numeric_cols,
            categorical_cols,
            n_smote,
            random_state + 3000 + int(k)
        )

        pieces = [p for p in [stat_df, smote_df] if len(p)]

        # Pull the local share of global TVAE samples.
        if n_tvae > 0 and global_tvae is not None:
            take_end = min(tvae_cursor + n_tvae, len(global_tvae))
            if take_end > tvae_cursor:
                tpart = global_tvae.iloc[
                    tvae_cursor:take_end
                ].copy()
                pieces.append(tpart)
            tvae_cursor = take_end

        if pieces:
            cand_k = pd.concat(
                pieces,
                ignore_index=True
            )

            # If TVAE failed / ran short, fill from local bootstrap.
            if len(cand_k) < n_k:
                rng = np.random.default_rng(
                    random_state + 4000 + int(k)
                )
                idx = rng.integers(
                    0,
                    len(cluster_df),
                    size=n_k - len(cand_k)
                )
                cand_k = pd.concat(
                    [cand_k, cluster_df.iloc[idx]],
                    ignore_index=True
                )

            cand_k = cand_k.iloc[:n_k].copy()
            all_candidates.append(cand_k)

        diagnostics.append({
            "cluster": int(k),
            "size": int(len(cluster_df)),
            "density": structure["density"],
            "stability": structure["stability"],
            "skewness": structure["skewness"],
            "separation": structure["separation"],
            "weight_statistical": weights["statistical"],
            "weight_smote": weights["smote"],
            "weight_tvae": weights["tvae"],
            "n_statistical": int(n_stat),
            "n_smote": int(n_smote),
            "n_tvae": int(n_tvae)
        })

    candidates = pd.concat(
        all_candidates,
        ignore_index=True
    )

    # --------------------------------------------------------
    # 4. Candidate sanitation
    # --------------------------------------------------------
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
        # Match observed categories whenever possible.
        observed = set(normal_df[c].astype("object").dropna().unique())
        if observed:
            fallback = normal_df[c].mode(dropna=True)
            fallback = fallback.iloc[0] if len(fallback) else next(iter(observed))
            candidates[c] = candidates[c].where(
                candidates[c].isin(observed),
                fallback
            )

    candidates = candidates[normal_df.columns].reset_index(drop=True)

    # --------------------------------------------------------
    # 5. Approach-20 discriminative manifold selection
    # --------------------------------------------------------
    order, selection_diag = select_candidates(
        normal_df,
        attack_df,
        candidates,
        numeric_cols,
        categorical_cols
    )

    selected_n = min(target_n, len(order))
    chosen = candidates.iloc[order[:selected_n]].copy()

    # --------------------------------------------------------
    # 6. Conservative final clipping
    # --------------------------------------------------------
    chosen = final_sanity_clip(
        chosen,
        normal_df,
        numeric_cols
    )

    chosen = chosen[normal_df.columns].reset_index(drop=True)

    return chosen, {
        "n_normal": len(normal_df),
        "n_attack": len(attack_df),
        "n_candidates": len(candidates),
        "n_selected": len(chosen),
        "n_clusters": len(unique_labels),
        "cluster_diagnostics": pd.DataFrame(diagnostics),
        "selection_diagnostics": selection_diag
    }


# ============================================================
# PROJECT PIPELINE RUNNERS
# ============================================================

def generate_tvae_floor_adaptive_cluster_hybrid(
    train_df,
    label_col,
    col_types,
    target_n,
    seed=RANDOM_SEED,
):
    feature_cols = col_types["all_feature"]
    normal_df = train_df.loc[train_df[label_col] == 0, feature_cols].copy().reset_index(drop=True)
    attack_df = train_df.loc[train_df[label_col] == 1, feature_cols].copy().reset_index(drop=True)

    synth, diag = generate_approach24(
        normal_df=normal_df,
        attack_df=attack_df,
        target_n=target_n,
        random_state=seed,
    )
    return add_id_column(synth)


# Aliases (kept for compatibility with the Approach 24/25 lineage)
generate_approach25 = generate_approach24
generate_approach26 = generate_approach24
generate_improved_adaptive_local_mixture_hybrid = generate_tvae_floor_adaptive_cluster_hybrid


def run(
    dataset_key="nslkdd",
    do_eval=True,
):
    cfg = DATASETS[dataset_key]

    print("\n" + "=" * 72)
    print(
        f"Approach 26: TVAE-Floor + Concentration-Adaptive Cluster Mixture — "
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
        f"  Base generator mix: "
        f"statistical={BASE_STATISTICAL:.0%}, "
        f"SMOTE={BASE_SMOTE:.0%}, "
        f"TVAE={BASE_TVAE:.0%} (floor={MIN_TVAE_WEIGHT:.0%})"
    )
    print(
        f"  Candidate oversampling: "
        f"{CANDIDATE_OVER_SAMPLE_FACTOR:.2f}x, "
        f"adapt_strength={MIXTURE_ADAPT_STRENGTH}"
    )

    t0 = time.time()

    synthetic = generate_tvae_floor_adaptive_cluster_hybrid(
        train_df,
        cfg["label_col"],
        col_types,
        cfg["target_n"],
        seed=RANDOM_SEED,
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
        f"approach_26_{dataset_key}.csv",
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
        "base_statistical": BASE_STATISTICAL,
        "base_smote": BASE_SMOTE,
        "base_tvae": BASE_TVAE,
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

