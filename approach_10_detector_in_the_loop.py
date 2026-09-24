"""
approach_10_detector_in_the_loop.py
====================================
Approach 10: Detector-in-the-Loop Candidate Selection

Hypothesis & Motivation:
  Rather than unguided generation or simple geometric filtering, this approach explicitly
  optimizes the synthetic dataset against a local proxy of the downstream detectors:
    1. Over-generate a large candidate pool (2.5x target N) using a diverse mixture
       (SMOTE + TVAE + Local Jitter).
    2. Fit local proxy ECOD and Isolation Forest detectors on the real normal training data.
    3. Score every synthetic candidate point with both detectors:
         - ECOD score: measures empirical tail rarity (lower score = higher normal typicality).
         - IForest score: measures average isolation path depth (higher score = deeper normal point).
    4. Compute a combined normalized detector-confidence score and select the top N candidates
       that exhibit high normal typicality across BOTH detectors simultaneously.
    5. Prepend the mandatory 'id' column as the first column.

  *CRITICAL REQUIREMENT*: The exact same pipeline and selection criteria are applied
  to both NSL-KDD and UNSW-NB15 without dataset-specific branching.
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, RANDOM_SEED, OUTPUT_DIR
from preprocessing import (
    load_dataset, local_train_eval_split, get_normal_data,
    validate_synthetic, add_id_column, one_hot_encode,
)
from local_evaluator import evaluate_synthetic, compute_quality_metrics, _train_ecod
from approach_4_tvae import generate_tvae
from approach_5_smote import generate_smote
from approach_1_baseline_jitter import generate_baseline_jitter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Parameters (Universal for all datasets)
# ---------------------------------------------------------------------------
OVER_GENERATION_FACTOR = 2.5
CANDIDATE_RATIOS = {"smote": 0.45, "tvae": 0.40, "jitter": 0.15}


def generate_detector_in_the_loop(
    normal_df,
    col_types,
    target_n,
    over_factor=OVER_GENERATION_FACTOR,
    seed=RANDOM_SEED,
):
    """
    Generate synthetic data via detector-in-the-loop scoring and selection.
    """
    total_candidates = int(round(target_n * over_factor))
    n_smote = int(round(total_candidates * CANDIDATE_RATIOS["smote"]))
    n_tvae = int(round(total_candidates * CANDIDATE_RATIOS["tvae"]))
    n_jitter = total_candidates - n_smote - n_tvae

    feature_cols = col_types["all_feature"]
    cat_cols = col_types["categorical"]

    print(f"    [1/4] Generating {total_candidates} candidate pool ({n_smote} SMOTE, {n_tvae} TVAE, {n_jitter} Jitter)...")
    
    # 1. Generate diverse candidates
    synth_smote = generate_smote(normal_df, col_types, n_smote, seed=seed).drop(columns=["id"], errors="ignore")
    synth_tvae = generate_tvae(normal_df, col_types, n_tvae, seed=seed).drop(columns=["id"], errors="ignore")
    synth_jitter = generate_baseline_jitter(normal_df, col_types, n_jitter, seed=seed).drop(columns=["id"], errors="ignore")

    candidates_df = pd.concat([synth_smote[feature_cols], synth_tvae[feature_cols], synth_jitter[feature_cols]], ignore_index=True)

    # 2. One-hot encode real normal data and candidate pool
    print("    [2/4] Fitting local proxy ECOD and Isolation Forest detectors on real normal data...")
    real_encoded, fit_cats = one_hot_encode(normal_df[feature_cols], cat_cols)
    cand_encoded, _ = one_hot_encode(candidates_df, cat_cols, fit_categories=fit_cats)

    all_cols = sorted(set(real_encoded.columns) | set(cand_encoded.columns))
    for c in all_cols:
        if c not in real_encoded.columns:
            real_encoded[c] = 0
        if c not in cand_encoded.columns:
            cand_encoded[c] = 0

    X_real = real_encoded[all_cols].to_numpy(dtype=float)
    X_cand = cand_encoded[all_cols].to_numpy(dtype=float)

    # 3. Train proxy detectors on real normal data
    ecod = _train_ecod(X_real)
    iforest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    iforest.fit(X_real)

    # 4. Score candidates
    print("    [3/4] Scoring candidates against proxy detector typicality...")
    # ECOD returns anomaly scores (lower = more typical normal)
    ecod_scores = ecod.decision_function(X_cand)
    # IForest decision_function returns positive for normal, negative for anomaly
    if_scores = iforest.decision_function(X_cand)

    # Normalize scores to [0, 1] range (higher = more typical normal)
    ecod_rank = (ecod_scores.max() - ecod_scores) / (ecod_scores.max() - ecod_scores.min() + 1e-10)
    if_rank = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)

    # Combined detector typicality score
    combined_score = 0.5 * ecod_rank + 0.5 * if_rank

    # 5. Select top target_n candidates
    print(f"    [4/4] Selecting top {target_n} detector-aligned synthetic samples...")
    selected_indices = np.argsort(-combined_score)[:target_n]
    selected_synthetic = candidates_df.iloc[selected_indices].copy().reset_index(drop=True)

    return add_id_column(selected_synthetic)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run(dataset_key="nslkdd", do_eval=True):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*60}")
    print(f"Approach 10: Detector-in-the-Loop Selection — {cfg['name']}")
    print(f"{'='*60}")

    df, col_types = load_dataset(cfg)
    train_df, eval_df = local_train_eval_split(df, cfg["label_col"])
    normal_train = get_normal_data(train_df, cfg["label_col"])

    print(f"  Normal training rows: {len(normal_train)}")
    print(f"  Target synthetic rows: {cfg['target_n']}")

    t0 = time.time()
    synthetic = generate_detector_in_the_loop(normal_train, col_types, cfg["target_n"])
    gen_time = time.time() - t0
    print(f"  Generation time: {gen_time:.1f}s")

    validate_synthetic(synthetic, col_types, cfg["target_n"])
    print("  ✓ Validation passed")

    out_path = os.path.join(OUTPUT_DIR, f"approach_10_{dataset_key}.csv")
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
