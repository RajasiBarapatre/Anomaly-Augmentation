"""
Local evaluation harness that implements the documented detector scheme:
  1. One-hot encode categorical features in both synthetic (train) and eval sets
  2. Train ECOD on synthetic data -> score eval set -> compute AUPRC
  3. Train IForest (10 fixed seeds, 500 trees each) on synthetic data -> score eval set -> compute AUPRC per seed -> average
  4. Final score S = (AUPRC_ECOD + mean_AUPRC_IForest) / 2
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from revised_experiment_config import (
    IFOREST_SEEDS,
    IFOREST_N_ESTIMATORS,
)

from revised_preprocessing import one_hot_encode


#   ECOD detector (deterministic, unsupervised)
#   For each feature, compute the empirical CDF on the training data, then score test points by summing negative log-likelihoods of their tail probabilities
def _train_ecod(X_train):
    try:
        from pyod.models.ecod import ECOD
        model = ECOD(contamination=0.1, n_jobs=-1)
        model.fit(X_train)
        return model
    except ImportError as exc:
        raise ImportError("PyOD is required for scored evaluation. Install the organizer's version; no approximate fallback is used.") from exc


#   Core evaluation
#   Evaluate synthetic data against a labeled eval set using ECOD and Isolation Forest
#   synthetic_df : Generated normal-only data (feature columns only, no label)
#   eval_df : Held-out labeled data with both normal and anomaly rows
def evaluate_synthetic(
    synthetic_df,
    eval_df,
    categorical_cols,
    label_col,
    verbose=True,
):
    t0 = time.time()

    #   One-hot encode (drop id and label columns)
    synth_features = synthetic_df.drop(columns=["id"], errors="ignore")
    synth_encoded, categories = one_hot_encode(synth_features, categorical_cols)
    eval_features = eval_df.drop(columns=[label_col, "id"], errors="ignore")
    eval_encoded, _ = one_hot_encode(eval_features, categorical_cols, fit_categories=categories)

    #   Align columns
    all_cols = sorted(set(synth_encoded.columns) | set(eval_encoded.columns))
    for col in all_cols:
        if col not in synth_encoded.columns:
            synth_encoded[col] = 0
        if col not in eval_encoded.columns:
            eval_encoded[col] = 0
    synth_encoded = synth_encoded[all_cols].astype(float)
    eval_encoded = eval_encoded[all_cols].astype(float)

    y_true = eval_df[label_col].values

    #   ECOD
    if verbose:
        print("    Training ECOD...")
    ecod_model = _train_ecod(synth_encoded.values)
    ecod_scores = ecod_model.decision_function(eval_encoded.values)
    auprc_ecod = average_precision_score(y_true, ecod_scores)
    if verbose:
        print(f"    ECOD AUPRC: {auprc_ecod:.4f}")

    #   Isolation Forest (10 seeds)
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
        #   IForest decision_function returns negative scores (more negative = more anomalous)
        #   Negate so that higher = more anomalous for AUPRC computation
        if_scores = -iforest.decision_function(eval_encoded.values)
        auprc_if = average_precision_score(y_true, if_scores)
        iforest_auprcs.append(auprc_if)
        if verbose:
            print(f" AUPRC={auprc_if:.4f}")

    auprc_iforest_mean = np.mean(iforest_auprcs)
    score_S = (auprc_ecod + auprc_iforest_mean) / 2.0

    elapsed = time.time() - t0
    if verbose:
        print(f"    AUPRC ECOD:    {auprc_ecod:.5f}")
        print(f"    AUPRC IFOREST: {auprc_iforest_mean:.5f}")
        print(f"    >>> Final Combined Score = (ECOD + IForest) / 2: {score_S:.5f} <<<")
        print(f"    Evaluation took {elapsed:.1f}s")

    return {
        "auprc_ecod": auprc_ecod,
        "auprc_iforest_mean": auprc_iforest_mean,
        "auprc_iforest_seeds": iforest_auprcs,
        "score_S": score_S,
        "eval_time_seconds": elapsed,
    }


#   Statistical quality metrics for analysis
#   Compute statistical similarity between synthetic and real normal data.
def compute_quality_metrics(synthetic_df, real_normal_df, col_types):
    
    numeric_cols = col_types["numeric"]

    #   Per-feature KS distance
    ks_distances = {}
    for col in numeric_cols:
        if col in synthetic_df.columns and col in real_normal_df.columns:
            stat, _ = ks_2samp(real_normal_df[col].values, synthetic_df[col].values)
            ks_distances[col] = stat

    mean_ks = np.mean(list(ks_distances.values())) if ks_distances else np.nan

    #   Correlation matrix RMSE
    synth_num = synthetic_df[numeric_cols].astype(float)
    real_num = real_normal_df[numeric_cols].astype(float)
    corr_synth = synth_num.corr().values
    corr_real = real_num.corr().values
    mask = np.isfinite(corr_synth) & np.isfinite(corr_real)
    if mask.any():
        corr_rmse = np.sqrt(np.mean((corr_synth[mask] - corr_real[mask]) ** 2))
    else:
        corr_rmse = np.nan

    #   Marginal statistics comparison
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
    from revised_experiment_config import DATASETS
    from revised_preprocessing import (
        load_dataset,
        local_train_eval_split,
        get_normal_data,
    )
    cfg = DATASETS["nslkdd"]
    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    synthetic = normal_train.sample(n=cfg["target_n"], replace=True,
                                     random_state=42).reset_index(drop=True)
    print("=== Quick baseline test (NSL-KDD) ===")
    results = evaluate_synthetic(synthetic, eval_df, cfg["categorical_cols"], cfg["label_col"])
    print(f"\nFinal Score S = {results['score_S']:.4f}")
