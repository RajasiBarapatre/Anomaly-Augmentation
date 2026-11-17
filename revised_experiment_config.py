"""
Shared configuration for experiments
"""

import os

# PATHS
PROJECT_DIR = os.environ.get("CS5344_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
NSLKDD_PATH = os.path.join(PROJECT_DIR, "data","nslkdd_train.csv")
UNSW_PATH = os.path.join(PROJECT_DIR, "data", "unsw_nb15_train.csv")
OUTPUT_DIR = os.environ.get("CS5344_OUTPUT_DIR", os.path.join(PROJECT_DIR, "outputs"))

# EXPERIMENT SETTINGS
RANDOM_SEED = 42
IFOREST_SEEDS = list(range(10))
IFOREST_N_ESTIMATORS = 500

# DATASET CONFIGURATION
DATASETS = {
    "nslkdd": {
        "name": "NSL-KDD",
        "path": NSLKDD_PATH,
        "categorical_cols": [
            "protocol_type",
            "service",
            "flag",
        ],
        "label_col": "is_anomaly",
        "target_n": 40_000,
    },

    "unsw": {
        "name": "UNSW-NB15",
        "path": UNSW_PATH,
        "categorical_cols": [
            "proto",
            "service",
            "state",
        ],
        "label_col": "is_anomaly",
        "target_n": 70_000,
    },
}

LOCAL_EVAL_SPLIT = 0.20

