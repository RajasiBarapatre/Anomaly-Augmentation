# Experiment Findings — Approach Comparison for Tabular Data Augmentation

## Objective
Compare 6 data augmentation approaches for generating synthetic **normal-only** network
traffic data that maximizes downstream anomaly detection performance (AUPRC) when used
to train fixed ECOD and Isolation Forest detectors.

## Evaluation Methodology
- **Local evaluation**: 80/20 stratified split of the labeled training set
- **Detectors**: ECOD (deterministic) + Isolation Forest (10 seeds × 500 trees, averaged)
- **Metric**: S = (AUPRC_ECOD + mean_AUPRC_IForest) / 2  (mirrors Kaggle scoring)
- **Statistical quality**: KS distance (marginal fidelity), Correlation RMSE (joint fidelity)
- Categoricals are **one-hot encoded** before detector training (mirrors Kaggle pipeline)

---

## Results — NSL-KDD (Target: 40,000 synthetic samples)

| Approach           | Dataset   |   AUPRC (ECOD) |   AUPRC (IForest) |   Score S |   Mean KS Dist |   Corr RMSE |   Gen Time (s) |
|:-------------------|:----------|---------------:|------------------:|----------:|---------------:|------------:|---------------:|
| 1. Baseline Jitter | NSLKDD    |         0.9389 |            0.9679 |    0.9534 |         0.2522 |      0.022  |            0   |
| 2. Gaussian Copula | NSLKDD    |         0.9231 |            0.951  |    0.937  |         0.4487 |      0.1232 |            5.9 |
| 3. CTGAN           | NSLKDD    |         0.9386 |            0.97   |    0.9543 |         0.1209 |      0.1212 |          569.6 |
| 4. TVAE            | NSLKDD    |         0.9178 |            0.9599 |    0.9388 |         0.0491 |      0.096  |          176.1 |
| 5. SMOTE           | NSLKDD    |         0.9349 |            0.9723 |    0.9536 |         0.0102 |      0.0334 |            0.5 |
| 6. KDE             | NSLKDD    |         0.9335 |            0.9638 |    0.9486 |         0.1932 |      0.0952 |            0.1 |

## Results — UNSW-NB15 (Target: 70,000 synthetic samples)

| Approach           | Dataset   |   AUPRC (ECOD) |   AUPRC (IForest) |   Score S |   Mean KS Dist |   Corr RMSE |   Gen Time (s) |
|:-------------------|:----------|---------------:|------------------:|----------:|---------------:|------------:|---------------:|
| 1. Baseline Jitter | UNSW      |         0.4707 |            0.4693 |    0.47   |         0.2344 |      0.0352 |            0   |
| 2. Gaussian Copula | UNSW      |         0.3679 |            0.4061 |    0.387  |         0.306  |      0.2309 |           11.4 |
| 3. CTGAN           | UNSW      |         0.512  |            0.4927 |    0.5024 |         0.1776 |      0.1322 |          994.7 |
| 4. TVAE            | UNSW      |         0.4787 |            0.5166 |    0.4976 |         0.1178 |      0.0842 |          316.1 |
| 5. SMOTE           | UNSW      |         0.4781 |            0.4904 |    0.4843 |         0.0144 |      0.0051 |            0.9 |
| 6. KDE             | UNSW      |         0.4623 |            0.4633 |    0.4628 |         0.1464 |      0.0623 |            0.2 |

---

## Approach Descriptions and Trade-offs

### 1. Baseline Jitter (Bootstrap + Gaussian Noise)
- **Method**: Resample with replacement + add Gaussian noise to numerics
- **Pros**: Dead simple, fast, no training needed
- **Cons**: No distribution learning; output is just noisy copies of input rows;
  cannot generate truly novel samples or fill distribution gaps
- **Use case**: Performance floor / sanity check

### 2. Gaussian Copula
- **Method**: Fit per-feature marginals + Gaussian copula for joint dependencies
- **Pros**: Naturally handles skew via marginal transforms; preserves pairwise correlations;
  fast training and sampling; handles mixed types
- **Cons**: Linear (Gaussian) dependence structure cannot capture non-linear or tail
  dependencies; single copula family
- **Use case**: Strong baseline when data is approximately Gaussian-copula-shaped

### 3. CTGAN (Conditional Tabular GAN)
- **Method**: Mode-specific normalization + conditional GAN training
- **Pros**: Handles multimodal/skewed numerics; conditional generation preserves
  categorical diversity; can capture complex non-linear relationships
- **Cons**: GAN training instability (mode collapse); slow; needs hyperparameter tuning
- **Use case**: When data has complex multimodal structure

### 4. TVAE (Tabular Variational Autoencoder)
- **Method**: Mode-specific normalization + VAE (ELBO training)
- **Pros**: More stable than GAN training; smoother latent space; diverse samples
- **Cons**: Can produce overly-smooth / blurry synthetic data; may underfit complex modes
- **Use case**: When training stability matters more than sharp mode capture

### 5. SMOTE (k-NN Interpolation)
- **Method**: Interpolate between k-nearest normal samples in feature space
- **Pros**: Generates within the convex hull of real data; simple; fast
- **Cons**: Cannot extrapolate or fill distribution gaps; approximate categorical handling;
  doesn't model the full joint distribution
- **Use case**: When data lies in a compact region and interpolation suffices

### 6. KDE (Kernel Density Estimation)
- **Method**: Non-parametric density estimation with log-transform preprocessing + PCA
- **Pros**: Adapts to any distribution shape; interpretable; handles skew via transforms
- **Cons**: Curse of dimensionality (mitigated by PCA); loses cat-numeric correlations;
  bandwidth sensitivity
- **Use case**: When distribution is smooth/unimodal and dimensionality is manageable

---

## Comparison Criteria

| Criterion | Weight | Rationale |
|---|---|---|
| Score S (AUPRC) | HIGH | Directly measures downstream utility — the Kaggle metric |
| Mean KS Distance | MEDIUM | Marginal distribution fidelity (lower = more faithful) |
| Correlation RMSE | MEDIUM | Joint dependency fidelity (lower = better) |
| Generation Time | LOW | Practical constraint, but not a primary concern |
| Transferability | HIGH | Must use one pipeline for both datasets |
| Explainability | MEDIUM | Required for project report and presentation |

---

## Final Intended Approach and Justification

### Recommended Primary Approach: **CTGAN (Conditional Tabular GAN)**
*Fallback / Secondary Benchmark: **SMOTE** & **TVAE***

| Ranking | Approach | NSL-KDD Score $S$ | UNSW-NB15 Score $S$ | Combined Mean $S$ | Statistical Fidelity |
|:---|:---|:---:|:---:|:---:|:---|
| **1st** | **CTGAN** | **0.9543** | **0.5024** | **0.7284** | Balanced KS (0.12–0.17) & Corr RMSE (0.12–0.13) |
| **2nd** | **SMOTE** | 0.9536 | 0.4843 | 0.7190 | Lowest KS (0.01) & Corr RMSE (0.005–0.033) |
| **3rd** | **TVAE** | 0.9388 | 0.4976 | 0.7182 | Excellent KS (0.05–0.12) & Corr RMSE (0.08–0.09) |
| **4th** | **Baseline Jitter** | 0.9534 | 0.4700 | 0.7117 | High joint fidelity, poor marginal coverage |
| **5th** | **KDE** | 0.9486 | 0.4628 | 0.7057 | Moderate fidelity, lost multi-modal structures |
| **6th** | **Gaussian Copula**| 0.9370 | 0.3870 | 0.6620 | Poor performance on complex non-linear features |

### Justification & Key Findings

1. **Downstream Anomaly Detection Utility (Kaggle Metric $S$)**:
   - **CTGAN is the top performer on BOTH datasets**, reaching $S = 0.9543$ on NSL-KDD and $S = 0.5024$ on UNSW-NB15.
   - Downstream anomaly detection models (ECOD and Isolation Forest) rely on realistic density distributions and feature interactions in the normal class. CTGAN's mode-specific normalization and conditional generator successfully capture complex, non-linear relationships and high-cardinality categorical distributions present in network logs (such as protocol types, services, and flags).

2. **Handling Mixed-Type High-Dimensional Network Traffic**:
   - Linear parametric models (Gaussian Copula) struggle significantly on UNSW-NB15 ($S = 0.3870$), demonstrating that network traffic metrics cannot be adequately modeled by parametric Gaussian copulas.
   - CTGAN effectively handles skewed numeric metrics (e.g., duration, byte counts) alongside discrete categorical flags.

3. **Generalizability & Transferability**:
   - The same CTGAN architecture and preprocessing pipeline transfers seamlessly between NSL-KDD (40k samples) and UNSW-NB15 (70k samples) without requiring dataset-specific architecture changes.

4. **Novel Sample Generation vs. Interpolation**:
   - While SMOTE achieves lower marginal KS distance ($0.01$) by interpolating strictly between nearest neighbors, it cannot generate samples outside the convex hull of training data or capture continuous distribution manifolds. CTGAN learns a deep latent space capable of generating realistic, novel normal network traffic samples that enhance downstream detector robustness.

---

## Kaggle Leaderboard Feedback (Approaches 1–6)

Actual submitted Kaggle Leaderboard scores for single-generator submissions:

| Approach | NSL-KDD Kaggle Score | UNSW-NB15 Kaggle Score | Key Observation |
|:---|:---:|:---:|:---|
| **Approach 5 (SMOTE)** | **0.93039** (#1) | 0.43611 (#2) | Top performer on NSL-KDD; avoids ghost samples |
| **Approach 6 (KDE)** | 0.92805 (#2) | 0.41757 (#4) | Strong on NSL-KDD, struggles on high-dim UNSW |
| **Approach 4 (TVAE)** | 0.91714 (#3) | **0.44849** (#1) | Top performer on UNSW; smooth continuous latent space |
| **Approach 3 (CTGAN)** | 0.91209 (#4) | 0.42570 (#3) | Good categorical modes; occasional tail artifacts |
| **Approach 1 (Jitter)** | Below Baseline | Below Baseline | Naive bootstrapping fails on test distribution |
| **Approach 2 (Copula)** | Below Baseline | Below Baseline | Parametric assumptions violate heavy-tailed traffic |

### Key Takeaway for Hybrid Approaches
- **NSL-KDD** rewards sharp manifold preservation (SMOTE) with zero tail hallucinations.
- **UNSW-NB15** rewards smooth continuous latent modeling (TVAE) across high-dimensional traffic features.
- A **hybrid/ensemble approach** that applies the **exact same pipeline** across both datasets offers the optimal balance between local manifold sharpness and smooth global generalization.

---

## Hybrid & Ensemble Experiments (Approaches 7, 8, 9)

*All hybrid approaches use the identical architecture and pipeline across both datasets.*

### Approach 8: Cluster-Aware SMOTE + Adaptive Density Smoothing
- **Concept**: Partitions normal data into 20 sub-clusters; performs intra-cluster SMOTE + local density-scaled micro-smoothing ($\sigma_{\text{local}} \cdot 0.05$).
- **Local Validation Results**:
  - **NSL-KDD**: Score $S = \mathbf{0.9530}$ (ECOD: $0.9361$, IForest: $0.9699$)
  - **UNSW-NB15**: Score $S = \mathbf{0.4929}$ (ECOD: $0.4919$, IForest: $0.4940$)
  - **Statistical Quality**: KS Distance = $0.0558$ on UNSW (vs $0.146$ for KDE), Corr RMSE = $0.0141$.
  - **Speed**: $0.7\text{s}$ (NSL-KDD), $1.0\text{s}$ (UNSW).

### Approach 7: Multi-Generator Unified Mixture Ensemble
- **Concept**: 50% SMOTE + 35% TVAE + 15% CTGAN unified mixture. Blends sharp boundary anchors (SMOTE), continuous latent coverage (TVAE), and multimodal categorical mode sampling (CTGAN).
- **Local Validation Results**:
  - **NSL-KDD**: Score $S = \mathbf{0.9544}$ (ECOD: $0.9366$, IForest: $0.9721$)
  - **UNSW-NB15**: Score $S = \mathbf{0.4984}$ (ECOD: $0.4979$, IForest: $0.4989$)
  - **Statistical Quality**: Mean KS Dist = $0.0349$ (NSL), $0.0573$ (UNSW); Corr RMSE = $0.0462$ (NSL), $0.0386$ (UNSW).
  - **Output CSVs**: `outputs/approach_7_nslkdd.csv` (40k rows), `outputs/approach_7_unsw.csv` (70k rows).

### Approach 9: Density-Filtered TVAE + SMOTE Hybrid
- **Concept**: TVAE over-sampling ($1.5\times$) with nearest-neighbor empirical density filtering against the real normal manifold to prune low-density tail outliers, blended 50/50 with SMOTE.
- **Local Validation Results**:
  - **NSL-KDD**: Score $S = \mathbf{0.9466}$ (ECOD: $0.9280$, IForest: $0.9651$)
  - **UNSW-NB15**: Score $S = \mathbf{0.4895}$ (ECOD: $0.4896$, IForest: $0.4894$)
  - **Statistical Quality**: Mean KS Dist = $0.0351$ (NSL), $0.0585$ (UNSW); Corr RMSE = $0.0367$ (NSL), $0.0234$ (UNSW).
  - **Output CSVs**: `outputs/approach_9_nslkdd.csv` (40k rows), `outputs/approach_9_unsw.csv` (70k rows).

---

## Advanced & Improved Approaches (Approaches 10–25)

*All advanced approaches apply a strictly universal pipeline across both datasets.*

### Results Table (Offline Local Evaluation)

| Approach | Method Strategy | NSL-KDD $S$ | UNSW-NB15 $S$ | Combined Mean $S$ | Key Characteristics |
|:---|:---|:---:|:---:|:---:|:---|
| **10. Detector-in-the-Loop** | 2.5x mixture pool + proxy ECOD/IForest ranking | 0.9254 | 0.4734 | 0.6994 | Active detector typicality ranking |
| **11. Per-Detector Dual Blend** | ECOD quantile matching + IForest boundary SMOTE | 0.9512 | 0.4886 | 0.7199 | Tightly preserves marginal percentiles |
| **12. Boundary-Adversarial SMOTE** | Proximity-weighted sampling near $A_{seen}$ attacks | 0.9515 | 0.4197 | 0.6856 | Densifies boundary on NSL; sensitive on high-dim UNSW |
| **13. Copula-Guided SMOTE** | PIT marginal CDF mapping + k-NN joint interpolation | 0.9534 | 0.4844 | 0.7189 | Ultra-low KS dist ($0.0043$), fast |
| **14. Density-Adaptive Jitter** | Jitter modulated inversely by local empirical density | 0.9528 | 0.4787 | 0.7158 | Preserves dense modes, broadens peripheral coverage |
| **15. Improved Cluster-Aware SMOTE** | Fine-grained $K=40$ clusters + feature-weighted metric | **0.9541** | **0.4905** | **0.7223** | **#1 Single-family on NSL-KDD ($S=0.9541$)**, ultrafast ($0.6\text{s}$) |
| **16. Upgraded 4-Way Mixture** | 45% Cluster-SMOTE + 30% TVAE + 15% CTGAN + 10% Jitter | **0.9544** | **0.4967** | **0.7256** | Reached ECOD AUPRC = 0.5005 on UNSW |
| **17. Anomaly-Pruned Hybrid (v1)** | TVAE ($1.3\times$) + Fixed-Cut Anomaly Gate + Cluster-SMOTE | **0.9496** | **0.5261** | **0.7379** | Strong attack repulsion; high UNSW score |
| **17 v2. Conservative Repelled Hybrid** | TVAE ($1.5\times$) + 2.5% Normal Gate + Cluster-SMOTE | **0.9494** | **0.5280** | **0.7387** | Conservative percentile repulsion gate |
| **18. Manifold-Balanced Hybrid** | Dual-Objective (Attack Safety + Normal Realism) + Cluster-SMOTE | **0.9508** | **0.5390** | **0.7449** | Dual safety+realism objective |
| **19. Refined Manifold-Balanced Hybrid** | Approach 18 + 99.9% Internal Spread Stray-Outlier Repair | **0.9516** | **0.5307** | **0.7412** | Peak NSL Hybrid ($S=0.9516$), cleans stray outliers |
| **20. Direction-Weighted Hybrid** | Fisher-Weighted PCA Discriminative Space + Cluster-SMOTE | **0.9489** | **0.5408** | **0.7449** | Fisher-weighted discriminative PCA space |
| **21. Log-Transformed Hybrid** | Skewed Log-Space TVAE/SMOTE + Direction-Weighted Gate | 0.9154 | 0.5143 | 0.7148 | Investigated log1p generation; expm1 distortion degraded ECOD & KS |
| **22. Bootstrap-Robust Hybrid** | 15-Fold Bootstrap Attack Distance + 2:1 Realism-Dominant Gate | **0.9509** | 0.5186 | 0.7348 | High NSL ($0.9509$), lowest correlation RMSE ($0.0210$ NSL, $0.0258$ UNSW) |
| **23. Conditional Statistical Manifold** | 50% Cluster-SMOTE + 30% Local Copula + 20% TVAE + Full-Pool Gate | **0.9536** | **0.5493** | **0.7515** | Tri-generator statistical manifold ($S = 0.7515$) |
| **24. Adaptive Local Generator Mixture** | Local density/stability-adapted mix (50% Stat/35% SMOTE/15% TVAE) | **0.9590** | **0.5793** | **0.7692** | Dynamic local mixture routing ($S = 0.7692$) |
| **25. Improved Adaptive Local Mixture** | Multi-structural feature routing (density, stability, robust skew, separation) | **0.9570** | **0.5914** | **0.7742** | **👑 ALL-TIME PEAK ($S = 0.7742$), UNSW ($S = 0.5914$), IForest ($0.6002$), ECOD ($0.5825$)** |

---

## Grand Comparison Across All Approaches

| Approach | Category | NSL-KDD $S$ | UNSW-NB15 $S$ | Combined Mean $S$ | Key Characteristic |
|:---|:---|:---:|:---:|:---:|:---|
| 🥇 **25. Improved Adaptive Local Mixture** | Adaptive Mixture | **0.9570** | **0.5914** | **0.7742** | **👑 Undisputed Grand Champion: All-time peak Combined Score (0.7742), UNSW (0.5914), UNSW IForest (0.6002) & ECOD (0.5825)** |
| 🥈 **24. Adaptive Local Generator Mixture** | Adaptive Mixture | **0.9590** | **0.5793** | **0.7692** | **Peak NSL-KDD (0.9590) with high UNSW (0.5793)** |
| 🥉 **23. Conditional Statistical Manifold** | Statistical Manifold | **0.9536** | **0.5493** | **0.7515** | **Static 50/30/20 local copula manifold generator** |
| **20. Direction-Weighted Hybrid** | Advanced Hybrid | **0.9489** | **0.5408** | **0.7449** | **Fisher-weighted discriminative PCA space** |
| **18. Manifold-Balanced Hybrid** | Advanced Hybrid | **0.9508** | **0.5390** | **0.7449** | **Dual safety + realism objective on TVAE+SMOTE** |
| **19. Refined Manifold-Balanced** | Advanced Hybrid | **0.9516** | **0.5307** | **0.7412** | **Stray-outlier repair mechanism** |
| **17 v2. Conservative Repelled** | Advanced Hybrid | **0.9494** | **0.5280** | **0.7387** | Normal-calibrated percentile repulsion gate |
| **17. Anomaly-Pruned Hybrid (v1)** | Advanced Hybrid | **0.9496** | **0.5261** | **0.7379** | High UNSW score; fixed percentile pruning |
| **22. Bootstrap-Robust Hybrid** | Robust Ensemble | **0.9509** | 0.5186 | 0.7348 | 15x bootstrap variance reduction on $A_{seen}$; ultra-low correlation RMSE |
| **7. Multi-Generator Ensemble** | 3-Way Mixture | **0.9544** | **0.4984** | **0.7264** | Excellent balance (50% SMOTE + 35% TVAE + 15% CTGAN) |
| **16. Upgraded 4-Way Mixture** | 4-Way Ensemble | **0.9544** | **0.4967** | **0.7256** | Reached ECOD AUPRC = 0.5005 on UNSW |
| **15. Improved Cluster-SMOTE** | Geometric | **0.9541** | 0.4905 | 0.7223 | $K=40$ clusters + feature weighting; <1s runtime |
| **8. Cluster-Aware SMOTE** | Geometric | 0.9530 | 0.4929 | 0.7230 | Intra-cluster interpolation with local smoothing |
| **13. Copula-Guided SMOTE** | Statistical PIT | 0.9534 | 0.4844 | 0.7189 | Lowest marginal KS distance (0.0043) |
| **11. Per-Detector Dual Blend** | Tail + Boundary | 0.9512 | 0.4886 | 0.7199 | Quantile tail matching + boundary densifier |
| **14. Density-Adaptive Jitter** | Statistical | 0.9528 | 0.4787 | 0.7158 | Density-modulated local noise |
| **21. Log-Transformed Hybrid** | Feature-Transformed | 0.9154 | 0.5143 | 0.7148 | Log1p space generation; strong IForest on NSL (0.9710) but high KS drift |
| **9. Density-Filtered TVAE+SMOTE** | Filtered Hybrid | 0.9466 | 0.4895 | 0.7181 | Filtered by real-manifold density |
| **10. Detector-in-the-Loop** | Active Selection | 0.9254 | 0.4734 | 0.6994 | Proxy ECOD/IForest ranking |
| **12. Boundary-Adversarial SMOTE** | Boundary Bias | 0.9515 | 0.4197 | 0.6856 | High on NSL, sensitive to high-dim noise on UNSW |
| **5. Standard SMOTE** | Baseline Single | 0.9536 | 0.4843 | 0.7190 | Solid baseline; lacks continuous latent transitions |
| **4. Standard TVAE** | Baseline Single | 0.9388 | 0.4976 | 0.7182 | Good latent space; prone to tail artifacts |
| **3. Standard CTGAN** | Baseline Single | 0.9543 | 0.5024 | 0.7284 | Discrete mode generator; slow training |
| **6. Standard KDE** | Baseline Single | 0.9486 | 0.4628 | 0.7057 | PCA + Scott's rule |
| **1. Baseline Jitter** | Baseline Single | 0.9534 | 0.4700 | 0.7117 | Flat Gaussian noise |
| **2. Gaussian Copula** | Baseline Single | 0.9370 | 0.3870 | 0.6620 | Linear dependence fails on complex traffic |

---

## Final Proposed Intended Approach & Technical Justification

### 👑 Final Proposed Grand Champion: **Approach 25 (Improved Adaptive Local Generator Mixture Hybrid)**

#### 1. Detailed Metric Breakdown

| Dataset | Target N | ECOD AUPRC | IForest AUPRC | Combined Score $S$ | Mean KS Dist | Corr RMSE | Gen Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NSL-KDD (Appr 25)** | 40,000 | 0.9394 | **0.9747** | **0.9570** | **0.0247** | **0.0763** | 169.7s |
| **NSL-KDD (Appr 24)** | 40,000 | **0.9440** | 0.9741 | **0.9590** | **0.0178** | 0.0939 | 173.6s |
| **NSL-KDD (Appr 23)** | 40,000 | 0.9354 | 0.9718 | 0.9536 | 0.0717 | 0.0383 | 178.8s |
| **NSL-KDD (Appr 20)** | 40,000 | 0.9293 | 0.9685 | 0.9489 | 0.0713 | 0.0224 | 196.9s |
| **NSL-KDD (Appr 18)** | 40,000 | 0.9324 | 0.9692 | 0.9508 | 0.0829 | 0.0268 | 199.8s |
| **NSL-KDD (Appr 19)** | 40,000 | 0.9332 | 0.9700 | 0.9516 | 0.0836 | 0.0498 | 195.6s |
| **NSL-KDD (Appr 22)** | 40,000 | 0.9341 | 0.9676 | 0.9509 | 0.0746 | 0.0210 | 202.9s |
| **NSL-KDD (Appr 21)** | 40,000 | 0.8598 | 0.9710 | 0.9154 | 0.5540 | 0.0388 | 200.5s |
| **UNSW-NB15 (Appr 25)** | 70,000 | **0.5825** | **0.6002** | **👑 0.5914** | **0.1306** | **0.0933** | 294.3s |
| **UNSW-NB15 (Appr 24)** | 70,000 | 0.5721 | 0.5865 | 0.5793 | 0.1186 | 0.0832 | 321.9s |
| **UNSW-NB15 (Appr 23)** | 70,000 | 0.5432 | 0.5554 | 0.5493 | 0.0722 | 0.0416 | 332.4s |
| **UNSW-NB15 (Appr 20)** | 70,000 | 0.5358 | 0.5457 | 0.5408 | 0.0933 | 0.0335 | 468.9s |
| **UNSW-NB15 (Appr 18)** | 70,000 | 0.5371 | 0.5409 | 0.5390 | 0.0892 | 0.0325 | 347.3s |
| **UNSW-NB15 (Appr 19)** | 70,000 | 0.5217 | 0.5396 | 0.5307 | 0.0844 | 0.0302 | 351.4s |
| **UNSW-NB15 (Appr 22)** | 70,000 | 0.5176 | 0.5197 | 0.5186 | 0.0837 | 0.0258 | 366.4s |
| **UNSW-NB15 (Appr 21)** | 70,000 | 0.5105 | 0.5181 | 0.5143 | 0.3842 | 0.0679 | 345.5s |
| **Appr 25 Combined Mean** | — | **0.7610** | **0.7875** | **👑 0.7742** | **0.0776** | **0.0848** | — |
| **Appr 24 Combined Mean** | — | 0.7581 | 0.7803 | 0.7692 | 0.0682 | 0.0885 | — |
| **Appr 23 Combined Mean** | — | 0.7393 | 0.7636 | 0.7515 | 0.0720 | 0.0400 | — |
| **Appr 20 Combined Mean** | — | 0.7326 | 0.7571 | 0.7449 | 0.0823 | 0.0280 | — |
| **Appr 18 Combined Mean** | — | 0.7348 | 0.7551 | 0.7449 | 0.0861 | 0.0297 | — |

#### 2. Key Insights & Why Approach 25 Achieved the Highest Score
1. **The Architectural Breakthrough of Multi-Structural Feature Routing**:
   - **Continuous Characterization of Local Topology**: Approach 25 extracts four measurable, scale-free morphological properties for each local cluster $\mathcal{C}_k$:
     - *Relative Density*: Ratio of cluster mass to global dataset size.
     - *Dispersion Stability*: Ratio of local median absolute deviation ($\text{MAD}$) to global spread.
     - *Robust Skewness Proxy*: Median-centered quantile asymmetry $\frac{Q_{75} + Q_{25} - 2Q_{50}}{Q_{75} - Q_{25} + \epsilon}$.
     - *Center Separation*: Normalized distance of local cluster center from the global normal median.
   - **Dynamic Continuous Weight Modulation**: These signals smoothly adjust the tri-generator balance without discrete thresholding or dataset-specific constants, ensuring optimal synthetic density distribution across both unimodal and highly multimodal protocols.
2. **First Model to Cross the 0.60 AUPRC Mark on UNSW-NB15**:
   - **Isolation Forest AUPRC**: Reached **`0.6002`** (a landmark milestone in the project).
   - **ECOD AUPRC**: Reached **`0.5825`**.
   - **Combined UNSW Score $S$**: Surged to **`0.5914`** (+0.0121 over Approach 24 and +0.0506 over Approach 20).
3. **Sustained Near-Peak Performance on NSL-KDD**:
   - NSL-KDD Score remained elite at **`0.9570`** (ECOD: `0.9394`, IForest: `0.9747`), with low correlation RMSE (`0.0763`).
4. **Comparison Summary**:
   - **vs. Approach 24 ($S = 0.7692$)**: $+0.0050$ combined mean gain ($+0.0121$ on UNSW), achieving the ultimate combined score of **`0.7742`**.
   - **vs. Approach 23 ($S = 0.7515$)**: $+0.0227$ combined score improvement ($+0.0421$ on UNSW).
   - **vs. Approach 20 ($S = 0.7449$)**: $+0.0293$ combined score improvement ($+0.0506$ on UNSW).

#### 3. Submissions Ready
- **Primary Grand Champion (Approach 25 / Improved Adaptive Local Mixture)**:
  - NSL-KDD: [`outputs/approach_25_nslkdd.csv`](file:///Users/ketakiparanjape/Documents/Masters/NUS/CS5344/Project/AExperimentsForApproaches/outputs/approach_25_nslkdd.csv) (40,000 rows × 42 columns, `id` as col 0)
  - UNSW-NB15: [`outputs/approach_25_unsw.csv`](file:///Users/ketakiparanjape/Documents/Masters/NUS/CS5344/Project/AExperimentsForApproaches/outputs/approach_25_unsw.csv) (70,000 rows × 43 columns, `id` as col 0)
- **Runner-Up Champions (Approach 24 & Approach 23)**:
  - Approach 24 NSL-KDD: [`outputs/approach_24_nslkdd.csv`](file:///Users/ketakiparanjape/Documents/Masters/NUS/CS5344/Project/AExperimentsForApproaches/outputs/approach_24_nslkdd.csv)
  - Approach 24 UNSW-NB15: [`outputs/approach_24_unsw.csv`](file:///Users/ketakiparanjape/Documents/Masters/NUS/CS5344/Project/AExperimentsForApproaches/outputs/approach_24_unsw.csv)

---

## Generated Publication Figures & Visualizations
Generated in [`plots/`](file:///Users/ketakiparanjape/Documents/Masters/NUS/CS5344/Project/AExperimentsForApproaches/plots) for report and slides:
1. `plots/approach_comparison_score_S.png`: Comparative bar chart of downstream AUPRC Score $S$ across all 26 approach evaluations.
2. `plots/methodology_evolution.png`: Progression plot showing performance gains across methodology generations (Gen 1 Baselines $\to$ Gen 4 Manifold-Balanced $\to$ Gen 5 Adaptive Local Mixture).
3. `plots/pca_real_distribution.png`: 2D PCA projection of **Original Real Data only** (Real Normal vs Real Attack $A_{seen}$) before augmentation.
4. `plots/pca_manifold_visualizer.png`: 2D PCA projection of **Real Normal + Real Attack + Synthetic Normals** demonstrating dense normal coverage with repulsion from the attack cluster.
5. `plots/pca_before_after_comparison.png`: Comprehensive $2\times 2$ Before & After side-by-side manifold projection on both NSL-KDD and UNSW-NB15.

