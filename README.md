# Synthetic Normal Data Generation for Unsupervised Anomaly Detection

This codebase generates **synthetic normal-only sets** for two network-intrusion datasets
(**NSL-KDD** and **UNSW-NB15**). Detectors trained only on the synthetic data are scored on how well
they separate normal traffic from attacks, including attack types the generator never saw.


**Scoring scheme**

1. One-hot encode the categorical features.
2. Train **ECOD** on the synthetic data, score the evaluation set, and compute AUPRC.
3. Train **Isolation Forest** (10 fixed seeds, 500 trees each), compute AUPRC per seed, and average.
4. Final score: `S = (AUPRC_ECOD + mean AUPRC_IForest) / 2`.

| Dataset   | Synthetic rows required (`target_n`) |
|-----------|--------------------------------------|
| NSL-KDD   | 40,000                               |
| UNSW-NB15 | 70,000                               | 

**Final synthetic data CSV paths**

NSL-KDD: ./outputs/synthetic_normals_nslkdd_submission.csv
UNSW-NB15: ./outputs/synthetic_normals_unsw_submission.csv

---

## Repository layout

| File | Purpose |
|------|---------|
| `final_augmentation_pipeline.py` | The generator. Produces the synthetic normals and is the entry point for validation and submission runs |
| `revised_experiment_config.py` | Dataset paths, categorical columns, `target_n`, IForest settings, output directory |
| `revised_preprocessing.py` | Dataset loading, column-type detection, stratified local split, one-hot encoding, output validation |
| `revised_local_evaluator.py` | Local implementation of the ECOD + IForest scoring, plus KS and correlation quality metrics |
| `outputs/` | Generated CSVs, JSON reports and evaluation folds |
| `outputs/synthetic_normals_nslkdd_submission.csv` | Synthetic NSL-KDD (40,000 rows) |
| `outputs/synthetic_normals_unsw_submission.csv` | Synthetic UNSW-NB15 (70,000 rows) | 
| `requirements.txt` | Dependencies |
| `evaluations/cluster_holdout_evaluator.py` | For reference ONLY. Main unseen-attack evaluator: 8 pseudo attack types, 3 held out per fold, with the full generator run on each fold |
| `evaluations/calibrated_proxy.py` | For reference ONLY. Lighter, faster variant of the holdout evaluator (4 folds, 2 hidden clusters, no re-weighting stage) |

---
## Data

Place the training CSVs in the `data` folder in the project directory:

```
nslkdd_train.csv
unsw_nb15_train.csv
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

### Generate synthetic normals

```bash
# MAIN COMMAND TO RUN; Submission: trains on the full dataset
python final_augmentation_pipeline.py --mode submission --dataset both

# Validation: holds out a 20% stratified slice, generates, then scores locally
python final_augmentation_pipeline.py --mode validation --dataset both
```

| Flag | Values | Default |
|------|--------|---------|
| `--dataset` | `unsw`, `nslkdd`, `both` | `both` |
| `--mode` | `validation`, `submission` | `submission` |
| `--seed` | integer | `42` |
| `--data-dir` | directory containing the CSVs | config default |
| `--output-dir` | output directory | config default |

## Outputs

All outputs are written under `outputs/` (or `CS5344_OUTPUT_DIR`).

**From `final_augmentation_pipeline.py`**

| File | Contents |
|------|----------|
| `synthetic_normals_<dataset>_<mode>.csv` | The synthetic normal data (`id` first, then feature columns). |
| `synthetic_normals_<dataset>_<mode>_clusters.csv` | Per-cluster diagnostics. |
| `synthetic_normals_<dataset>_<mode>_run.json` | Run provenance: seed, package versions, configuration, audits, selection and re-weighting summaries, generator counts. |
| `synthetic_normals_<dataset>_<mode>_metrics.json` | Generation time and settings; in validation mode also AUPRC (ECOD, IForest), `score_S`, mean KS distance and correlation RMSE. |

`<dataset>` is `unsw` or `nslkdd`; `<mode>` is `validation` or `submission`.

---

## How the generator works

All stages are in `final_augmentation_pipeline.py`.

1. **Local clustering.** Real normals are encoded (scaled numerics plus one-hot categoricals) and split into local clusters using MiniBatchKMeans, with `k` chosen by a stability and silhouette check.
2. **Candidate generation.** Each cluster generates candidates with a mix of two generators:
   a Gaussian-copula-style *statistical* sampler and *cluster-conditioned SMOTENC*.
   Per-cluster mixture weights adapt to cluster structure (`MIXTURE_ADAPT_STRENGTHS`).
3. **Candidate selection.** The pool is oversampled (`CANDIDATE_OVER_SAMPLE_FACTORS`: 1.5x NSL-KDD, 4.0x UNSW) and then filtered. Scoring combines a density-normalised realism term with a bounded reward for staying away from known attacks (`ATTACK_OVERLAP_PERCENTILES`). A zone-based **RFOD** penalty (random-forest outlier detection) is applied to the half of the candidates closest to seen attacks.
4. **Rule mining and consistency.** Identities and dependencies mined from real normals
   (for example ratio and timing relationships) are used to drop inconsistent candidates.
5. **Normal-coverage re-weighting.** The final set is resampled with per-row weights, with tilts chosen on internal splits, bagged and gated by a minimum gain. This targets how often rare-but-real normal behaviour appears, which drives false alarms.
6. **Audits and validation.** The output is checked for row count, columns, NaN/inf, binary and integer columns, domain consistency, and no exact copies or duplicates of real normals.

---

## Evaluation design

### Stratified 80/20 is the base initial evaluator.

`revised_local_evaluator.py` called through `final_augmentation_pipeline.py`.

### Quick Calibrated proxy (FOR REFERENCE ONLY)

`evaluations/calibrated_proxy.py` - 4 folds, 2 hidden clusters per fold, and a 0.3 attack prevalence for both the `seen` and `unseen` subset scores.

```bash
python calibrated_proxy.py --dataset unsw           # IForest: 3 seeds x 200 trees
python calibrated_proxy.py --dataset unsw --full    # IForest: 10 seeds x 500 trees
```


### Cluster-holdout evaluation (unseen attack types) (FOR REFERENCE ONLY)

`evaluations/cluster_holdout_evaluator.py`

1. Training attacks are clustered into 8 pseudo attack types (k-means on log-scaled numerics and one-hot categoricals, standardised).
2. Each fold hides 3 of the 8 types entirely (different set per fold).
3. The generator sees 70% of normals and 70% of the seen-type attacks.
4. The fold test set holds the other 30% of normals, the other 30% of seen-type attacks and all unseen-type attacks.
5. It reports `S` (all test rows) and `S_unseen` (normals plus unseen-type attacks only), which is the guardrail metric.


```bash
python cluster_holdout_evaluator.py                      # both datasets, folds 0,1,2
python cluster_holdout_evaluator.py --dataset unsw       # one dataset
python cluster_holdout_evaluator.py --folds 0            # one fold
python cluster_holdout_evaluator.py --step score         # re-score existing folds only
python cluster_holdout_evaluator.py --scoring official   # 10-seed IForest, AUPRC per seed
```

`--step` can be `split`, `generate`, `score` or `all` (default), so can generate once and re-score cheaply.
`--scoring local` (default) averages IForest scores over 3 seeds (500 trees) before computing AUPRC.
`--scoring official` mirrors the real scheme: AUPRC per seed over 10 seeds, then averaged.

#### Outputs (`outputs/cluster_holdout/`)

| File | Contents |
|------|----------|
| `<dataset>/fold<k>/train.csv` | Fold training part given to the generator |
| `<dataset>/fold<k>/test.csv` | Fold test set with `pseudo_type` and `unseen` flags |
| `<dataset>/fold<k>/synthetic.csv` | Synthetic normals generated for that fold |
| `<dataset>/fold<k>/split.json` | Held-out pseudo-types and row counts |
| `results.csv` | One row per dataset and fold: `S`, `AUPRC_ECOD`, `AUPRC_IForest`, `S_unseen`, `test_attack_share`, `synthetic_rows`. Re-runs overwrite matching rows and keep the others. |

---

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CS5344_DATA_DIR` | directory of the config file | Project root; input CSVs are read from `<root>/data/` |
| `CS5344_OUTPUT_DIR` | `<root>/outputs` | Where results are written |

---

## Reproducibility

- Generation seeds are fixed (`GENERATION_SEEDS`, 42); the holdout k-means seed is 0 and the unseen-type draw is seeded by fold.
- Dependency versions are pinned in `requirements.txt` and recorded in each `synthetic_normals_<dataset>_<mode>_run.json`.
- IForest uses seeds 0-9 and 500 trees in the official scheme.