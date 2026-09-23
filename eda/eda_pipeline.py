"""
eda_pipeline.py
================
One EDA pipeline, run identically on both NSL-KDD and UNSW-NB15, producing a
single combined Markdown report (report.md) for the CS5344 tabular data
augmentation project.

Run:
    python3 eda_pipeline.py

Inputs expected (edit DATASETS below if your paths differ):
    ./nslkdd_train.csv
    ./unsw_train.csv

Output:
    eda_output/
        report.md                          <- the single combined report
        nslkdd/*.png, *.csv                <- supporting figures/tables
        unsw/*.png, *.csv                  <- supporting figures/tables
    (report.md references the images via relative paths, so keep the folder
    structure intact when viewing it)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, skew, kurtosis
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130

# --------------------------------------------------------------------------
# Config — the only thing that changes between datasets
# --------------------------------------------------------------------------
OUT_ROOT = "eda_output"

DATASETS = {
    "nslkdd": {
        "name": "NSL-KDD",
        "path": "./nslkdd_train.csv",
        "categorical_cols": ["protocol_type", "service", "flag"],
        "label_col": "is_anomaly",
    },
    "unsw": {
        "name": "UNSW-NB15",
        "path": "./unsw_train.csv",
        "categorical_cols": ["proto", "service", "state"],
        "label_col": "is_anomaly",
    },
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan


def iqr_outlier_fraction(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((series < lo) | (series > hi)).mean())


# --------------------------------------------------------------------------
# Pipeline steps (all take a `df`/config and write into out_dir)
# --------------------------------------------------------------------------
def load_data(cfg):
    df = pd.read_csv(cfg["path"])
    label_col = cfg["label_col"]
    cat_cols = cfg["categorical_cols"]
    num_cols = [c for c in df.columns if c not in cat_cols + [label_col]]
    return df, num_cols, cat_cols, label_col


def basic_info(df, out_dir):
    info = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isnull().sum().sum()),
    }
    missing = df.isnull().sum().to_frame("n_missing")
    missing["pct_missing"] = (missing["n_missing"] / len(df) * 100).round(3)
    missing.to_csv(os.path.join(out_dir, "02_missing_values.csv"))
    return info, missing


def class_balance(df, label_col, name, out_dir):
    counts = df[label_col].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Normal (0)", "Anomaly (1)"], counts.values,
                   color=["#4C72B0", "#C44E52"])
    for b, p in zip(bars, pct.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{int(b.get_height())}\n({p}%)", ha="center", va="bottom")
    ax.set_title(f"{name}: Class Balance (Training Set)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_class_balance.png"))
    plt.close(fig)
    return counts, pct


def numeric_summary(df, num_cols, out_dir):
    summary = df[num_cols].describe().T
    summary["skew"] = df[num_cols].skew()
    summary["n_unique"] = df[num_cols].nunique()
    summary["pct_zero"] = (df[num_cols] == 0).mean() * 100
    summary.to_csv(os.path.join(out_dir, "03_numeric_summary.csv"))
    return summary


def categorical_summary(df, cat_cols, out_dir):
    rows = []
    for c in cat_cols:
        vc = df[c].value_counts()
        rows.append({
            "feature": c,
            "n_unique": df[c].nunique(),
            "top_value": vc.index[0],
            "top_value_pct": round(vc.iloc[0] / len(df) * 100, 2),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "04_categorical_summary.csv"), index=False)
    return summary


def correlation_heatmap(df, num_cols, name, out_dir, max_features=40):
    cols = num_cols[:max_features]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(min(0.35 * len(cols) + 3, 20),
                                     min(0.35 * len(cols) + 3, 20)))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.6})
    ax.set_title(f"{name}: Numeric Feature Correlation (All Classes)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_correlation_heatmap.png"))
    plt.close(fig)
    return corr


def feature_discriminative_power(df, num_cols, cat_cols, label_col, name, out_dir, top_n=15):
    y = df[label_col].values
    scores = []
    for c in num_cols:
        x = df[c].values
        try:
            auc = roc_auc_score(y, x)
            auc = max(auc, 1 - auc)
        except Exception:
            auc = np.nan
        scores.append({"feature": c, "type": "numeric", "score": auc, "metric": "AUC"})
    for c in cat_cols:
        try:
            v = cramers_v(df[c], df[label_col])
        except Exception:
            v = np.nan
        scores.append({"feature": c, "type": "categorical", "score": v, "metric": "Cramer's V"})

    ranked = pd.DataFrame(scores).sort_values("score", ascending=False)
    ranked.to_csv(os.path.join(out_dir, "07_feature_discriminative_power.csv"), index=False)

    top = ranked.head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(top))))
    colors = ["#4C72B0" if t == "numeric" else "#DD8452" for t in top["type"]]
    ax.barh(top["feature"][::-1], top["score"][::-1], color=colors[::-1])
    ax.set_xlabel("Univariate discriminative score (AUC for numeric, Cramer's V for categorical)")
    ax.set_title(f"{name}: Top {top_n} Most Discriminative Features")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_top_discriminative.png"))
    plt.close(fig)
    return ranked


def distributions_grid(df, label_col, out_dir, feats, filename, title, per_fig=12):
    feats = feats[:per_fig]
    n = len(feats)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, feat in enumerate(feats):
        ax = axes[i]
        data = df[[feat, label_col]].copy()
        vals = data[feat]
        if (vals >= 0).all() and vals.skew() > 3:
            data[feat] = np.log1p(vals)
            xlabel = f"log1p({feat})"
        else:
            xlabel = feat
        sns.kdeplot(data=data, x=feat, hue=label_col, ax=ax, common_norm=False,
                    fill=True, alpha=0.35, legend=(i == 0))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("")
        ax.set_title(feat, fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)


def categorical_vs_label(df, cat_cols, label_col, name, out_dir, top_k=8):
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, cat_cols):
        top_cats = df[c].value_counts().head(top_k).index
        sub = df[df[c].isin(top_cats)]
        ct = pd.crosstab(sub[c], sub[label_col], normalize="index")
        ct = ct.loc[top_cats]
        ct.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#C44E52"], legend=False)
        ax.set_title(f"{c} (top {top_k} categories)")
        ax.set_ylabel("Proportion")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in ["#4C72B0", "#C44E52"]]
    fig.legend(handles, ["Normal", "Anomaly"], loc="upper right")
    fig.suptitle(f"{name}: Anomaly Rate by Categorical Feature")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_categorical_vs_label.png"))
    plt.close(fig)


def outlier_analysis(df, num_cols, name, out_dir, top_n=20):
    fracs = {c: iqr_outlier_fraction(df[c]) for c in num_cols}
    out = pd.Series(fracs, name="outlier_fraction_iqr").sort_values(ascending=False)
    out.to_csv(os.path.join(out_dir, "10_outlier_iqr_summary.csv"))

    top = out.head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(top))))
    ax.barh(top.index[::-1], top.values[::-1] * 100, color="#55A868")
    ax.set_xlabel("% of rows flagged as IQR outliers")
    ax.set_title(f"{name}: Features with Most IQR Outliers (top {top_n})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "11_outlier_iqr_summary.png"))
    plt.close(fig)
    return out


def normal_only_shape_stats(df, num_cols, label_col, ranked, out_dir, top_n=15):
    """Skew/kurtosis/zero-inflation of the NORMAL class only, for the features a
    generator will most need to reproduce faithfully. Directly informs the
    augmentation method (e.g. heavy skew/zero-inflation -> avoid plain Gaussian
    generators; multi-modal features -> avoid single-component models)."""
    normal = df[df[label_col] == 0]
    top_feats = ranked[ranked["type"] == "numeric"]["feature"].tolist()[:top_n]
    rows = []
    for c in top_feats:
        v = normal[c]
        rows.append({
            "feature": c,
            "mean": v.mean(),
            "std": v.std(),
            "skew": skew(v),
            "excess_kurtosis": kurtosis(v),
            "pct_zero": (v == 0).mean() * 100,
            "n_unique": v.nunique(),
        })
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(out_dir, "12_normal_only_shape_stats.csv"), index=False)
    return stats, top_feats


def run_pipeline(key, cfg):
    out_dir = os.path.join(OUT_ROOT, key)
    os.makedirs(out_dir, exist_ok=True)

    df, num_cols, cat_cols, label_col = load_data(cfg)
    name = cfg["name"]
    print(f"[{name}] loaded shape={df.shape}")

    info, missing = basic_info(df, out_dir)
    counts, pct = class_balance(df, label_col, name, out_dir)
    numeric_summary(df, num_cols, out_dir)
    cat_summary = categorical_summary(df, cat_cols, out_dir)
    correlation_heatmap(df, num_cols, name, out_dir)
    ranked = feature_discriminative_power(df, num_cols, cat_cols, label_col, name, out_dir)

    top_numeric = ranked[ranked["type"] == "numeric"]["feature"].tolist()
    distributions_grid(
        df, label_col, out_dir, top_numeric, "08_distributions_normal_vs_anomaly.png",
        f"{name}: Normal vs Anomaly — Top Discriminative Numeric Features",
    )
    categorical_vs_label(df, cat_cols, label_col, name, out_dir)
    outliers = outlier_analysis(df, num_cols, name, out_dir)

    normal_stats, normal_top_feats = normal_only_shape_stats(df, num_cols, label_col, ranked, out_dir)
    distributions_grid(
        df[df[label_col] == 0], label_col, out_dir, normal_top_feats,
        "13_normal_only_distributions.png",
        f"{name}: NORMAL-CLASS-ONLY Distributions — Features to Preserve for Synthetic Data",
    )

    print(f"[{name}] done -> {out_dir}")
    return {
        "key": key, "name": name, "info": info, "counts": counts, "pct": pct,
        "num_cols": num_cols, "cat_cols": cat_cols, "cat_summary": cat_summary,
        "ranked": ranked, "outliers": outliers, "normal_stats": normal_stats,
    }


# --------------------------------------------------------------------------
# Single combined report
# --------------------------------------------------------------------------
def dataset_section(r):
    p = r["key"]  # relative image path prefix, e.g. "nslkdd/xxx.png"
    top_feats = r["ranked"].head(8)
    top_lines = "\n".join(
        f"- **{row.feature}** ({row.type}): {row.metric} = {row.score:.3f}"
        for row in top_feats.itertuples()
    )
    worst_outliers = r["outliers"].head(5)
    outlier_lines = "\n".join(
        f"- {feat}: {frac*100:.1f}% of rows flagged" for feat, frac in worst_outliers.items()
    )
    skewed = r["normal_stats"].reindex(
        r["normal_stats"]["skew"].abs().sort_values(ascending=False).index
    ).head(5)
    skew_lines = "\n".join(
        f"- {row.feature}: skew={row.skew:.2f}, excess kurtosis={row.excess_kurtosis:.2f}, "
        f"% zero={row.pct_zero:.1f}%"
        for row in skewed.itertuples()
    )

    return f"""
## {r['name']}

### Overview
- Rows: {r['info']['n_rows']:,} | Columns: {r['info']['n_cols']}
- Numeric features: {len(r['num_cols'])} | Categorical features: {len(r['cat_cols'])}
- Duplicate rows: {r['info']['n_duplicate_rows']:,} | Missing cells: {r['info']['total_missing_cells']}
- Class balance: Normal {r['counts'].get(0,0):,} ({r['pct'].get(0,0)}%) / Anomaly {r['counts'].get(1,0):,} ({r['pct'].get(1,0)}%)
  — reflects `A_seen` attack types only, per the project's train/valid/test split design.

![Class balance]({p}/01_class_balance.png)

### Categorical Features
{r['cat_summary'].to_markdown(index=False)}

![Anomaly rate by categorical feature]({p}/09_categorical_vs_label.png)

### Most Discriminative Features (Normal vs Anomaly, univariate)
{top_lines}

![Top discriminative features]({p}/06_top_discriminative.png)

![Distributions by class]({p}/08_distributions_normal_vs_anomaly.png)

### Correlation Structure
Pairwise numeric correlations: `{p}/05_correlation_heatmap.png` — useful for spotting
redundant/derived features (e.g. rate features computed from raw counts) that a
generator must reproduce jointly, not independently.

### Outliers (IQR rule, full training set)
{outlier_lines}

Note: in intrusion data, IQR "outliers" often coincide with the anomaly class rather
than being data-quality noise — not something to clean away by default.

### Normal-Class-Only Shape (what the generator must reproduce)
Since the submission must be **normal-only**, the shape of each feature's *normal*
distribution — not the normal-vs-anomaly contrast above — is what a generator needs
to match. Most skewed/zero-inflated features among the top discriminative ones:
{skew_lines}

![Normal-only distributions]({p}/13_normal_only_distributions.png)
"""


def write_combined_report(results):
    r1, r2 = list(results.values())
    comp_rows = [
        ("Rows", lambda r: f"{r['info']['n_rows']:,}"),
        ("Numeric features", lambda r: len(r["num_cols"])),
        ("Categorical features", lambda r: len(r["cat_cols"])),
        ("Anomaly %", lambda r: f"{r['pct'].get(1,0)}%"),
        ("Duplicate rows", lambda r: f"{r['info']['n_duplicate_rows']:,}"),
        ("Top discriminative feature", lambda r: r["ranked"].iloc[0]["feature"]),
        ("Most skewed normal-class feature (top-8 discriminative)",
         lambda r: r["normal_stats"].reindex(r["normal_stats"]["skew"].abs().sort_values(ascending=False).index).iloc[0]["feature"]),
    ]
    comp_table = "| Metric | " + f"{r1['name']} | {r2['name']} |\n"
    comp_table += "|---|---|---|\n"
    for label, fn in comp_rows:
        comp_table += f"| {label} | {fn(r1)} | {fn(r2)} |\n"

    md = f"""# EDA Report — NSL-KDD & UNSW-NB15 (Tabular Data Augmentation for Anomaly Detection)

Both datasets are analyzed with the **same pipeline** (`eda_pipeline.py`), so results
are directly comparable. Per the problem formulation, only the labeled training set
is available; it contains normal samples plus anomalies from the training-visible
attack types (`A_seen`) only — validation/test add unseen attack types, so training-set
statistics (e.g. class balance, per-feature normal-vs-anomaly separation) may not fully
transfer to validation/test.

## Cross-Dataset Comparison
{comp_table}
Both datasets are network-traffic intrusion data with the same general schema —
numeric connection/flow statistics, 3 categorical protocol-related fields, and a binary
`is_anomaly` label — so the same generation/augmentation approach can reasonably be
applied to both, with dataset-specific feature scaling/encoding.

{dataset_section(r1)}
{dataset_section(r2)}

## Implications for the Augmentation Task
1. **Target the normal-class distribution specifically.** The submission must contain
   only synthetic normal samples; the "Normal-Class-Only Shape" sections above (not the
   normal-vs-anomaly contrast plots) are what a generator needs to match.
2. **Several top-discriminative numeric features are heavily skewed / zero-inflated**
   even within the normal class (see skew/kurtosis tables) — a plain multivariate
   Gaussian generator will misrepresent these; log-transforms, copula-based methods, or
   models like CTGAN/TVAE that handle mixed/skewed tabular data are better starting
   points than naive Gaussian sampling.
3. **Categorical features have a few dominant categories** (see categorical summaries)
   — the generator should preserve normal-class category frequencies rather than
   sampling categories uniformly or from the full (normal+anomaly) distribution.
4. **Numeric features are correlated** (correlation heatmaps) — several are arithmetic
   derivatives of others (e.g. `*_rate` features derived from counts in both datasets).
   A generator that samples features independently will break these dependencies and
   produce internally-inconsistent synthetic rows; joint/multivariate generation is
   preferred over per-column generation.
5. **Downstream detectors will likely lean on the top-ranked discriminative features**
   identified here — synthetic normal data that fails to reproduce their normal-class
   marginal and joint behavior is likely to hurt downstream AUPRC most.

## Files
- `eda_pipeline.py` — the pipeline script (run with `python3 eda_pipeline.py`)
- `eda_output/report.md` — this report
- `eda_output/<nslkdd|unsw>/*.png`, `*.csv` — supporting figures and tables per dataset
"""
    with open(os.path.join(OUT_ROOT, "report.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = {}
    for key, cfg in DATASETS.items():
        results[key] = run_pipeline(key, cfg)
    write_combined_report(results)
    print("\nCombined report written to:", os.path.join(OUT_ROOT, "report.md"))
