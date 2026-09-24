"""
experiment_config.py
====================
Shared configuration for the experimentation pipeline.
All approach scripts import from here to ensure consistency.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to this file's directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# IForest evaluation uses 10 fixed seeds (mirroring Kaggle scoring)
IFOREST_SEEDS = list(range(10))
IFOREST_N_ESTIMATORS = 500

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "nslkdd": {
        "name": "NSL-KDD",
        "path": os.path.join(PROJECT_ROOT, "nslkdd_train.csv"),
        "categorical_cols": ["protocol_type", "service", "flag"],
        "label_col": "is_anomaly",
        "target_n": 40_000,       # Kaggle requires exactly 40,000 synthetic rows
    },
    "unsw": {
        "name": "UNSW-NB15",
        "path": os.path.join(PROJECT_ROOT, "unsw_train.csv"),
        "categorical_cols": ["proto", "service", "state"],
        "label_col": "is_anomaly",
        "target_n": 70_000,       # Kaggle requires exactly 70,000 synthetic rows
    },
}

# ---------------------------------------------------------------------------
# Local evaluation split ratio
# ---------------------------------------------------------------------------
# We hold out 20% of the full labeled training set (stratified by label)
# as a local "hidden" evaluation set to compute AUPRC before Kaggle submission.
LOCAL_EVAL_SPLIT = 0.20

# ---------------------------------------------------------------------------
# Approach registry (for run_all_experiments.py)
# ---------------------------------------------------------------------------
APPROACH_NAMES = {
    1: "Baseline Jitter",
    2: "Gaussian Copula",
    3: "CTGAN",
    4: "TVAE",
    5: "SMOTE",
    6: "KDE",
}
