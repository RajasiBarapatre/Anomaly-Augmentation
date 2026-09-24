"""
generate_summary_plots.py
=========================
Generates publication-quality figures and plots for the CS5344 project report,
slides, and final submission analysis.

Plots Generated:
  1. plots/approach_comparison_score_S.png
     - Comparative bar chart of downstream AUPRC Score S across all 17 approaches.
  2. plots/methodology_evolution.png
     - Grouped progression chart showing performance gains from baselines -> hybrids -> anomaly-pruned models.
  3. plots/pca_manifold_visualizer.png
     - 2D PCA projection comparing Real Normals, Real Attacks (A_seen), and Approach 17 Synthetic Normals.

Usage:
  python3 generate_summary_plots.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import DATASETS, OUTPUT_DIR
from preprocessing import load_dataset, one_hot_encode

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set global aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11

# Empirical Score S dictionary across all 17 approaches
RESULTS_DATA = [
    # Baseline single models
    {"id": 1, "name": "1. Jitter", "category": "Single Baseline", "nsl": 0.9534, "unsw": 0.4700},
    {"id": 2, "name": "2. Copula", "category": "Single Baseline", "nsl": 0.9370, "unsw": 0.3870},
    {"id": 3, "name": "3. CTGAN", "category": "Single Baseline", "nsl": 0.9543, "unsw": 0.5024},
    {"id": 4, "name": "4. TVAE", "category": "Single Baseline", "nsl": 0.9388, "unsw": 0.4976},
    {"id": 5, "name": "5. SMOTE", "category": "Single Baseline", "nsl": 0.9536, "unsw": 0.4843},
    {"id": 6, "name": "6. KDE", "category": "Single Baseline", "nsl": 0.9486, "unsw": 0.4628},
    # Initial hybrids
    {"id": 7, "name": "7. Mixture Ensemble", "category": "Initial Hybrid", "nsl": 0.9544, "unsw": 0.4984},
    {"id": 8, "name": "8. Cluster-SMOTE", "category": "Initial Hybrid", "nsl": 0.9530, "unsw": 0.4929},
    {"id": 9, "name": "9. Density-Filtered", "category": "Initial Hybrid", "nsl": 0.9466, "unsw": 0.4895},
    # Advanced approaches
    {"id": 10, "name": "10. Detector-in-Loop", "category": "Advanced", "nsl": 0.9254, "unsw": 0.4734},
    {"id": 11, "name": "11. Dual-Blend", "category": "Advanced", "nsl": 0.9512, "unsw": 0.4886},
    {"id": 12, "name": "12. Boundary-SMOTE", "category": "Advanced", "nsl": 0.9515, "unsw": 0.4197},
    {"id": 13, "name": "13. Copula-SMOTE", "category": "Advanced", "nsl": 0.9534, "unsw": 0.4844},
    {"id": 14, "name": "14. Adaptive-Jitter", "category": "Advanced", "nsl": 0.9528, "unsw": 0.4787},
    # Targeted improvements
    {"id": 15, "name": "15. Improved Cluster-SMOTE", "category": "Optimized Hybrid", "nsl": 0.9541, "unsw": 0.4905},
    {"id": 16, "name": "16. 4-Way Mixture", "category": "Optimized Hybrid", "nsl": 0.9544, "unsw": 0.4967},
    {"id": 17, "name": "17. Anomaly-Pruned (v1)", "category": "Optimized Hybrid", "nsl": 0.9496, "unsw": 0.5261},
    {"id": 18, "name": "17v2. Conservative Repelled", "category": "Optimized Hybrid", "nsl": 0.9494, "unsw": 0.5280},
    {"id": 19, "name": "18. Manifold-Balanced", "category": "Optimized Hybrid", "nsl": 0.9508, "unsw": 0.5390},
    {"id": 20, "name": "19. Refined Manifold-Balanced", "category": "Optimized Hybrid", "nsl": 0.9516, "unsw": 0.5307},
]


def plot_approach_comparison():
    """Bar chart comparing Score S across all 17 approaches."""
    df = pd.DataFrame(RESULTS_DATA)
    df["mean_score"] = (df["nsl"] + df["unsw"]) / 2

    # Sort by mean score
    df_sorted = df.sort_values("mean_score", ascending=True).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    # Palette
    cat_colors = {
        "Single Baseline": "#6c757d",
        "Initial Hybrid": "#0d6efd",
        "Advanced": "#6f42c1",
        "Optimized Hybrid": "#198754",
    }
    colors = [cat_colors[c] for c in df_sorted["category"]]

    # NSL-KDD
    bars1 = ax1.barh(df_sorted["name"], df_sorted["nsl"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax1.set_xlim(0.91, 0.96)
    ax1.set_xlabel("Score S (AUPRC)")
    ax1.set_title("NSL-KDD Performance (Target: 40k)")
    ax1.axvline(0.950, color="red", linestyle="--", alpha=0.7, label="Strong Benchmark (0.950)")
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.0005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=8)

    # UNSW-NB15
    bars2 = ax2.barh(df_sorted["name"], df_sorted["unsw"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax2.set_xlim(0.35, 0.55)
    ax2.set_xlabel("Score S (AUPRC)")
    ax2.set_title("UNSW-NB15 Performance (Target: 70k)")
    ax2.axvline(0.500, color="red", linestyle="--", alpha=0.7, label="0.50 Threshold")
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=8, fontweight="bold" if w > 0.52 else "normal")

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=cat)
        for cat, color in cat_colors.items()
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(PLOTS_DIR, "approach_comparison_score_S.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_methodology_evolution():
    """Progression plot grouped by methodology generations."""
    df = pd.DataFrame(RESULTS_DATA)
    grouped = df.groupby("category", sort=False).agg({"nsl": "mean", "unsw": "mean"}).reset_index()
    grouped["combined"] = (grouped["nsl"] + grouped["unsw"]) / 2

    # Add Approach 18 explicit callout
    app18 = df[df["name"].str.contains("18")].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(grouped))
    width = 0.35

    ax.bar(x - width/2, grouped["nsl"], width, label="NSL-KDD (Mean S)", color="#2b5c8f", alpha=0.9, edgecolor="black")
    ax.bar(x + width/2, grouped["unsw"], width, label="UNSW-NB15 (Mean S)", color="#d9534f", alpha=0.9, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["category"], fontweight="bold")
    ax.set_ylabel("Score S (AUPRC)")
    ax.set_title("Progression by Generative Methodology Category")
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc="upper left")

    # Annotate Approach 18 winner
    ax.annotate(
        f"Approach 18 Peak:\nUNSW S = {app18['unsw']:.4f}\nNSL S = {app18['nsl']:.4f}",
        xy=(3 + width/2, app18["unsw"]),
        xytext=(2.3, 0.65),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=8),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffc107", lw=1.5),
        fontweight="bold",
    )

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "methodology_evolution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_real_manifold_only():
    """2D PCA projection of ONLY Real Normal and Real Attack data (Before Augmentation)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, (ds_key, ax) in enumerate(zip(["nslkdd", "unsw"], axes)):
        cfg = DATASETS[ds_key]
        df, col_types = load_dataset(cfg)
        feature_cols = col_types["all_feature"]
        cat_cols = col_types["categorical"]

        n_vis = 2500
        normal_real = df[df[cfg["label_col"]] == 0][feature_cols].sample(min(n_vis, (df[cfg["label_col"]] == 0).sum()), random_state=42)
        anomaly_real = df[df[cfg["label_col"]] == 1][feature_cols].sample(min(n_vis, (df[cfg["label_col"]] == 1).sum()), random_state=42)

        # One-hot encode on real normal fit
        real_enc, cats = one_hot_encode(normal_real, cat_cols)
        anom_enc, _ = one_hot_encode(anomaly_real, cat_cols, fit_categories=cats)

        all_cols = sorted(set(real_enc.columns) | set(anom_enc.columns))
        for c in all_cols:
            if c not in real_enc: real_enc[c] = 0
            if c not in anom_enc: anom_enc[c] = 0

        X_real_norm = real_enc[all_cols].to_numpy(dtype=float)
        X_real_anom = anom_enc[all_cols].to_numpy(dtype=float)

        # PCA fit on combined real data
        scaler = StandardScaler()
        X_combined = np.vstack([X_real_norm, X_real_anom])
        scaler.fit(X_combined)

        pca = PCA(n_components=2, random_state=42)
        pca.fit(scaler.transform(X_combined))

        pca_norm = pca.transform(scaler.transform(X_real_norm))
        pca_anom = pca.transform(scaler.transform(X_real_anom))

        ax.scatter(pca_anom[:, 0], pca_anom[:, 1], c="#dc3545", alpha=0.45, s=16, label="Real Attack (A_seen)", marker="x")
        ax.scatter(pca_norm[:, 0], pca_norm[:, 1], c="#0d6efd", alpha=0.45, s=16, label="Real Normal (Train)", marker="o")

        ax.set_title(f"{cfg['name']}: Original Real Distribution (Before Augmentation)")
        ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.legend(loc="upper right", markerscale=2)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "pca_real_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_pca_manifold():
    """2D PCA projection comparing Real Normal, Real Attacks, and Approach 18 Synthetic Normals."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, (ds_key, ax) in enumerate(zip(["nslkdd", "unsw"], axes)):
        cfg = DATASETS[ds_key]
        df, col_types = load_dataset(cfg)
        feature_cols = col_types["all_feature"]
        cat_cols = col_types["categorical"]

        # Synthetic Approach 18 (fallback to 17v2 or 17)
        synth_path = os.path.join(OUTPUT_DIR, f"approach_18_{ds_key}.csv")
        if not os.path.exists(synth_path):
            synth_path = os.path.join(OUTPUT_DIR, f"approach_17_v2_{ds_key}.csv")
        if not os.path.exists(synth_path):
            synth_path = os.path.join(OUTPUT_DIR, f"approach_17_{ds_key}.csv")
        if not os.path.exists(synth_path):
            continue
        synth_df = pd.read_csv(synth_path).drop(columns=["id"], errors="ignore")

        # Sample subsets for fast visualization
        n_vis = 2500
        normal_real = df[df[cfg["label_col"]] == 0][feature_cols].sample(min(n_vis, (df[cfg["label_col"]] == 0).sum()), random_state=42)
        anomaly_real = df[df[cfg["label_col"]] == 1][feature_cols].sample(min(n_vis, (df[cfg["label_col"]] == 1).sum()), random_state=42)
        synth_vis = synth_df[feature_cols].sample(n_vis, random_state=42)

        # One-hot encode on real normal fit
        real_enc, cats = one_hot_encode(normal_real, cat_cols)
        anom_enc, _ = one_hot_encode(anomaly_real, cat_cols, fit_categories=cats)
        synth_enc, _ = one_hot_encode(synth_vis, cat_cols, fit_categories=cats)

        all_cols = sorted(set(real_enc.columns) | set(anom_enc.columns) | set(synth_enc.columns))
        for c in all_cols:
            if c not in real_enc: real_enc[c] = 0
            if c not in anom_enc: anom_enc[c] = 0
            if c not in synth_enc: synth_enc[c] = 0

        X_real_norm = real_enc[all_cols].to_numpy(dtype=float)
        X_real_anom = anom_enc[all_cols].to_numpy(dtype=float)
        X_synth = synth_enc[all_cols].to_numpy(dtype=float)

        # PCA fit on combined real data
        scaler = StandardScaler()
        X_combined = np.vstack([X_real_norm, X_real_anom])
        scaler.fit(X_combined)

        pca = PCA(n_components=2, random_state=42)
        pca.fit(scaler.transform(X_combined))

        pca_norm = pca.transform(scaler.transform(X_real_norm))
        pca_anom = pca.transform(scaler.transform(X_real_anom))
        pca_synth = pca.transform(scaler.transform(X_synth))

        ax.scatter(pca_anom[:, 0], pca_anom[:, 1], c="#dc3545", alpha=0.35, s=12, label="Real Attack (A_seen)", marker="x")
        ax.scatter(pca_norm[:, 0], pca_norm[:, 1], c="#0d6efd", alpha=0.35, s=12, label="Real Normal (Train)", marker="o")
        ax.scatter(pca_synth[:, 0], pca_synth[:, 1], c="#198754", alpha=0.45, s=15, label="Synthetic Normal (Appr 18)", marker="^")

        ax.set_title(f"{cfg['name']}: Manifold Alignment (After Augmentation)")
        ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.legend(loc="upper right", markerscale=2)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "pca_manifold_visualizer.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_pca_before_after():
    """2x2 Before & After Manifold Comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col_idx, ds_key in enumerate(["nslkdd", "unsw"]):
        cfg = DATASETS[ds_key]
        df, col_types = load_dataset(cfg)
        feature_cols = col_types["all_feature"]
        cat_cols = col_types["categorical"]

        synth_path = os.path.join(OUTPUT_DIR, f"approach_18_{ds_key}.csv")
        if not os.path.exists(synth_path):
            synth_path = os.path.join(OUTPUT_DIR, f"approach_17_v2_{ds_key}.csv")
        if not os.path.exists(synth_path):
            synth_path = os.path.join(OUTPUT_DIR, f"approach_17_{ds_key}.csv")
        if not os.path.exists(synth_path):
            continue
        synth_df = pd.read_csv(synth_path).drop(columns=["id"], errors="ignore")

        n_vis = 2500
        normal_real = df[df[cfg["label_col"]] == 0][feature_cols].sample(min(n_vis, (df[cfg["label_col"]] == 0).sum()), random_state=42)
        anomaly_real = df[df[cfg["label_col"]] == 1][feature_cols].sample(min(n_vis, (df[cfg["label_col"]] == 1).sum()), random_state=42)
        synth_vis = synth_df[feature_cols].sample(n_vis, random_state=42)

        real_enc, cats = one_hot_encode(normal_real, cat_cols)
        anom_enc, _ = one_hot_encode(anomaly_real, cat_cols, fit_categories=cats)
        synth_enc, _ = one_hot_encode(synth_vis, cat_cols, fit_categories=cats)

        all_cols = sorted(set(real_enc.columns) | set(anom_enc.columns) | set(synth_enc.columns))
        for c in all_cols:
            if c not in real_enc: real_enc[c] = 0
            if c not in anom_enc: anom_enc[c] = 0
            if c not in synth_enc: synth_enc[c] = 0

        X_real_norm = real_enc[all_cols].to_numpy(dtype=float)
        X_real_anom = anom_enc[all_cols].to_numpy(dtype=float)
        X_synth = synth_enc[all_cols].to_numpy(dtype=float)

        scaler = StandardScaler()
        X_combined = np.vstack([X_real_norm, X_real_anom])
        scaler.fit(X_combined)

        pca = PCA(n_components=2, random_state=42)
        pca.fit(scaler.transform(X_combined))

        pca_norm = pca.transform(scaler.transform(X_real_norm))
        pca_anom = pca.transform(scaler.transform(X_real_anom))
        pca_synth = pca.transform(scaler.transform(X_synth))

        # Top row: BEFORE (Real Only)
        ax_before = axes[0, col_idx]
        ax_before.scatter(pca_anom[:, 0], pca_anom[:, 1], c="#dc3545", alpha=0.45, s=14, label="Real Attack (A_seen)", marker="x")
        ax_before.scatter(pca_norm[:, 0], pca_norm[:, 1], c="#0d6efd", alpha=0.45, s=14, label="Real Normal (Train)", marker="o")
        ax_before.set_title(f"{cfg['name']} — BEFORE (Original Real Data)", fontweight="bold")
        ax_before.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax_before.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax_before.legend(loc="upper right", markerscale=2)

        # Bottom row: AFTER (Real + Approach 18 Synthetic)
        ax_after = axes[1, col_idx]
        ax_after.scatter(pca_anom[:, 0], pca_anom[:, 1], c="#dc3545", alpha=0.35, s=12, label="Real Attack (A_seen)", marker="x")
        ax_after.scatter(pca_norm[:, 0], pca_norm[:, 1], c="#0d6efd", alpha=0.35, s=12, label="Real Normal (Train)", marker="o")
        ax_after.scatter(pca_synth[:, 0], pca_synth[:, 1], c="#198754", alpha=0.45, s=15, label="Synthetic Normal (Appr 18)", marker="^")
        ax_after.set_title(f"{cfg['name']} — AFTER (Augmented with Approach 18)", fontweight="bold")
        ax_after.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax_after.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax_after.legend(loc="upper right", markerscale=2)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "pca_before_after_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def main():
    print("=" * 70)
    print("GENERATING SUMMARY PLOTS FOR REPORT & PRESENTATION")
    print("=" * 70)
    plot_approach_comparison()
    plot_methodology_evolution()
    plot_real_manifold_only()
    plot_pca_manifold()
    plot_pca_before_after()
    print("=" * 70)
    print("ALL PLOTS SUCCESSFULLY GENERATED IN: plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
