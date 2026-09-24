## Overview
Experimentation pipeline for comparing tabular data augmentation approaches for the
CS5344 anomaly detection project. The goal is to generate synthetic **normal-only** data
that maximizes downstream AUPRC when used to train ECOD and Isolation Forest detectors.

## Datasets
| Dataset | Train Rows | Features | Categorical | Target Synthetic N |
|---|---|---|---|---|
| NSL-KDD | 30,000 | 41 (38 num + 3 cat) | protocol_type, service, flag | **40,000** |
| UNSW-NB15 | 50,000 | 42 (39 num + 3 cat) | proto, service, state | **70,000** |

## Approaches (17 total)

### Single-Generator Baselines (1–6)
| # | Approach | Type | Speed | Key Idea |
|---|---|---|---|---|
| 1 | Baseline Jitter | Statistical | ⚡ Fast | Bootstrap + Gaussian noise (floor) |
| 2 | Gaussian Copula | Statistical | ⚡ Fast | Marginal transforms + Gaussian joint |
| 3 | CTGAN | Deep Learning | 🐢 Slow | Mode-specific norm + conditional GAN |
| 4 | TVAE | Deep Learning | 🐢 Slow | Mode-specific norm + VAE |
| 5 | SMOTE | Instance-based | ⚡ Fast | k-NN interpolation |
| 6 | KDE | Non-parametric | ⚡ Fast | Kernel density + PCA + transforms |

### Hybrid & Ensemble Generators (7–9)
| # | Approach | Type | Speed | Key Idea |
|---|---|---|---|---|
| 7 | **Unified Mixture Ensemble** | Ensemble | 🐢 Moderate | 50% SMOTE + 35% TVAE + 15% CTGAN blend |
| 8 | **Cluster-Aware SMOTE** | Instance-based | ⚡ Fast | Intra-cluster SMOTE + local density smoothing |
| 9 | **Density-Filtered Hybrid** | Hybrid Filter | 🐢 Moderate | TVAE (1.5x) + k-NN manifold density gate + SMOTE |

### Advanced & Improved Approaches (10–19)
| # | Approach | Type | Key Idea |
|---|---|---|---|
| 10 | **Detector-in-the-Loop** | Active Selection | 2.5x mixture pool + proxy ECOD/IForest ranking |
| 11 | **Per-Detector Dual Blend** | Tail + Boundary | 50% ECOD quantile match + 50% IForest boundary SMOTE |
| 12 | **Boundary-Adversarial SMOTE** | Boundary Bias | Sampling weighted near $A_{seen}$ attack borders |
| 13 | **Copula-Guided SMOTE** | Statistical PIT | PIT marginal CDF transforms + k-NN joint sampling |
| 14 | **Density-Adaptive Jitter** | Statistical | Jitter scaled inversely by local k-NN density |
| 15 | **Improved Cluster-SMOTE** | Geometric | $K=40$ fine clusters + feature-weighted metric [IMP OF 8] |
| 16 | **Upgraded 4-Way Mixture** | 4-Way Ensemble | 45% Cluster-SMOTE + 30% TVAE + 15% CTGAN + 10% Jitter [IMP OF 7] |
| 17 | **Anomaly-Pruned Hybrid (v1)** | Advanced Hybrid | TVAE ($1.3\times$) + Fixed Anomaly Gate + Cluster-SMOTE [IMP OF 9] |
| 17v2 | **Conservative Anomaly-Repelled Hybrid (v2)** | Advanced Hybrid | TVAE ($1.5\times$) + 2.5% Normal-Calibrated Gate + Cluster-SMOTE |
| 18 | **Manifold-Balanced Anomaly-Repulsion Hybrid** | Advanced Hybrid | Dual-Objective (Attack Safety + Normal Realism) + Cluster-SMOTE [🏆 CHAMPION] |
| 19 | **Refined Manifold-Balanced Hybrid** | Advanced Hybrid | Approach 18 + 99.9% Internal Spread Stray-Outlier Repair [🥈 RUNNER-UP] |

## File Structure
```
experiment_config.py           # Shared configuration (paths, seeds, targets)
preprocessing.py               # Data loading, column typing, train/eval split, id helper
local_evaluator.py             # ECOD + IForest(10 seeds) → AUPRC evaluation
approach_1_baseline_jitter.py ... approach_19_refined_a18.py
generate_final_submission.py   # Final submission generator on 100% training data
run_all_experiments.py         # Orchestrator for approaches 1–6
run_hybrid_experiments.py      # Orchestrator for approaches 7–9
run_advanced_experiments.py    # Orchestrator for approaches 10–19 (and 17v2)
findings.md                    # Complete results comparison and final recommendation
outputs/                       # Generated synthetic CSVs (with 'id' column as col 0)
```

## How to Run

### Run Advanced Approaches (10–17)
```bash
cd AExperimentsForApproaches
python3 run_advanced_experiments.py
```

### Run Fast Approaches (11, 12, 13, 14, 15)
```bash
python3 run_advanced_experiments.py --fast
```

### Run Specific Approach
```bash
python3 run_advanced_experiments.py --approach 17
```

## Evaluation Methodology
The local evaluator mirrors the Kaggle scoring pipeline:
1. **One-hot encode** categorical features in both synthetic and eval data
2. Train **ECOD** on synthetic data → score eval set → AUPRC
3. Train **Isolation Forest** (10 seeds × 500 trees) on synthetic data → AUPRC per seed → average
4. **Score S** = (AUPRC_ECOD + mean_AUPRC_IForest) / 2

## Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn, scipy
- sdv (for approaches 2, 3, 4)
- matplotlib, seaborn (optional, for visualization)
- pyod (optional, for official ECOD; falls back to custom implementation)
