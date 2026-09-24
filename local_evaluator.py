"""
local_evaluator.py
==================
Local evaluation harness that mirrors the Kaggle scoring pipeline:
  1. One-hot encode categorical features in both synthetic (train) and eval sets
  2. Train ECOD on synthetic data → score eval set → compute AUPRC
  3. Train IForest (10 fixed seeds, 500 trees each) on synthetic data
     → score eval set → compute AUPRC per seed → average
  4. Final score S = (AUPRC_ECOD + mean_AUPRC_IForest) / 2

Also computes statistical quality metrics for analysis.
"""

import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from experiment_config import IFOREST_SEEDS, IFOREST_N_ESTIMATORS
from preprocessing import one_hot_encode

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ECOD detector (deterministic, unsupervised)
# ---------------------------------------------------------------------------
def _train_ecod(X_train):
    """
    Train a simple ECOD-style detector.
    ECOD uses empirical cumulative distribution functions per feature to score
    anomalies. We implement a lightweight version: for each feature, compute
    the empirical CDF on the training data, then score test points by summing
    negative log-likelihoods of their tail probabilities.
    """
    try:
        from pyod.models.ecod import ECOD
        model = ECOD(contamination=0.1, n_jobs=-1)
        model.fit(X_train)
        return model
    except ImportError:
        # Fallback: simple tail-probability scoring
        return _ECODFallback(X_train)


class _ECODFallback:
    """Lightweight ECOD implementation when pyod is not installed."""

    def __init__(self, X_train):
        self.X_train = X_train.copy()
        self.n_train = len(X_train)

    def decision_function(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        X_train = np.asarray(self.X_train, dtype=float)
        scores = np.zeros(len(X_test))

        for j in range(X_train.shape[1]):
            col_train = X_train[:, j]
            col_test = X_test[:, j]
            # Left-tail and right-tail ECDF probabilities
            left = np.array([np.mean(col_train <= v) for v in col_test])
            right = np.array([np.mean(col_train >= v) for v in col_test])
            # Clip to avoid log(0)
            left = np.clip(left, 1e-10, 1.0)
            right = np.clip(right, 1e-10, 1.0)
            # Sum of negative log tail probabilities (both tails)
            scores += -np.log(np.minimum(left, right))

        return scores


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def evaluate_synthetic(
    synthetic_df,
    eval_df,
    categorical_cols,
    label_col,
    verbose=True,
):
    """
    Evaluate synthetic normal data against a labeled eval set.

    Parameters
    ----------
    synthetic_df : pd.DataFrame
        Generated normal-only data (feature columns only, no label).
    eval_df : pd.DataFrame
        Held-out labeled data with both normal and anomaly rows.
    categorical_cols : list[str]
        Categorical column names to one-hot encode.
    label_col : str
        Name of the label column in eval_df.
    verbose : bool
        Print progress messages.

    Returns
    -------
    results : dict
        Keys: auprc_ecod, auprc_iforest_mean, auprc_iforest_seeds,
              score_S, eval_time_seconds
    """
    t0 = time.time()

    # --- One-hot encode (drop id and label columns) ---
    synth_features = synthetic_df.drop(columns=["id"], errors="ignore")
    synth_encoded, categories = one_hot_encode(synth_features, categorical_cols)
    eval_features = eval_df.drop(columns=[label_col, "id"], errors="ignore")
    eval_encoded, _ = one_hot_encode(eval_features, categorical_cols, fit_categories=categories)

    # Align columns (eval may have categories not seen in synthetic and vice versa)
    all_cols = sorted(set(synth_encoded.columns) | set(eval_encoded.columns))
    for col in all_cols:
        if col not in synth_encoded.columns:
            synth_encoded[col] = 0
        if col not in eval_encoded.columns:
            eval_encoded[col] = 0
    synth_encoded = synth_encoded[all_cols].astype(float)
    eval_encoded = eval_encoded[all_cols].astype(float)

    y_true = eval_df[label_col].values

    # --- ECOD ---
    if verbose:
        print("    Training ECOD...")
    ecod_model = _train_ecod(synth_encoded.values)
    ecod_scores = ecod_model.decision_function(eval_encoded.values)
    auprc_ecod = average_precision_score(y_true, ecod_scores)
    if verbose:
        print(f"    ECOD AUPRC: {auprc_ecod:.4f}")

    # --- Isolation Forest (10 seeds) ---
    iforest_auprcs = []
    for seed in IFOREST_SEEDS:
        if verbose:
            print(f"    Training IForest seed={seed}...", end="")
        iforest = IsolationForest(
            n_estimators=IFOREST_N_ESTIMATORS,
            random_state=seed,
            contamination="auto",
            n_jobs=-1,
        )
        iforest.fit(synth_encoded.values)
        # IForest decision_function returns negative scores (more negative = more anomalous)
        # We negate so that higher = more anomalous for AUPRC computation
        if_scores = -iforest.decision_function(eval_encoded.values)
        auprc_if = average_precision_score(y_true, if_scores)
        iforest_auprcs.append(auprc_if)
        if verbose:
            print(f" AUPRC={auprc_if:.4f}")

    auprc_iforest_mean = np.mean(iforest_auprcs)
    score_S = (auprc_ecod + auprc_iforest_mean) / 2.0

    elapsed = time.time() - t0
    if verbose:
        print(f"    IForest mean AUPRC: {auprc_iforest_mean:.4f}")
        print(f"    >>> Combined Score S: {score_S:.4f} <<<")
        print(f"    Evaluation took {elapsed:.1f}s")

    return {
        "auprc_ecod": auprc_ecod,
        "auprc_iforest_mean": auprc_iforest_mean,
        "auprc_iforest_seeds": iforest_auprcs,
        "score_S": score_S,
        "eval_time_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Statistical quality metrics
# ---------------------------------------------------------------------------
def compute_quality_metrics(synthetic_df, real_normal_df, col_types):
    """
    Compute statistical similarity between synthetic and real normal data.

    Returns a dict with:
      - mean_ks_distance: average KS statistic across numeric features
      - per_feature_ks: dict of feature -> KS statistic
      - correlation_rmse: RMSE between correlation matrices
      - marginal_stats: DataFrame comparing mean/std/skew between real and synthetic
    """
    numeric_cols = col_types["numeric"]

    # --- Per-feature KS distance ---
    ks_distances = {}
    for col in numeric_cols:
        if col in synthetic_df.columns and col in real_normal_df.columns:
            stat, _ = ks_2samp(real_normal_df[col].values, synthetic_df[col].values)
            ks_distances[col] = stat

    mean_ks = np.mean(list(ks_distances.values())) if ks_distances else np.nan

    # --- Correlation matrix RMSE ---
    synth_num = synthetic_df[numeric_cols].astype(float)
    real_num = real_normal_df[numeric_cols].astype(float)
    corr_synth = synth_num.corr().values
    corr_real = real_num.corr().values
    # Handle NaN correlations (constant columns)
    mask = np.isfinite(corr_synth) & np.isfinite(corr_real)
    if mask.any():
        corr_rmse = np.sqrt(np.mean((corr_synth[mask] - corr_real[mask]) ** 2))
    else:
        corr_rmse = np.nan

    # --- Marginal statistics comparison ---
    rows = []
    for col in numeric_cols:
        if col in synthetic_df.columns and col in real_normal_df.columns:
            rows.append({
                "feature": col,
                "real_mean": real_normal_df[col].mean(),
                "synth_mean": synthetic_df[col].mean(),
                "real_std": real_normal_df[col].std(),
                "synth_std": synthetic_df[col].std(),
                "real_skew": real_normal_df[col].skew(),
                "synth_skew": synthetic_df[col].skew(),
            })
    marginal_stats = pd.DataFrame(rows)

    return {
        "mean_ks_distance": mean_ks,
        "per_feature_ks": ks_distances,
        "correlation_rmse": corr_rmse,
        "marginal_stats": marginal_stats,
    }


if __name__ == "__main__":
    # Quick test: evaluate the starter-kit baseline
    from experiment_config import DATASETS
    from preprocessing import load_dataset, local_train_eval_split, get_normal_data

    cfg = DATASETS["nslkdd"]
    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    # Simple baseline: just sample with replacement
    synthetic = normal_train.sample(n=cfg["target_n"], replace=True,
                                     random_state=42).reset_index(drop=True)
    print("=== Quick baseline test (NSL-KDD) ===")
    results = evaluate_synthetic(synthetic, eval_df, cfg["categorical_cols"], cfg["label_col"])
    print(f"\nFinal Score S = {results['score_S']:.4f}")
