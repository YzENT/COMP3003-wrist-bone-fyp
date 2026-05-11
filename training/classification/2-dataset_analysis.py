"""
Analyses feature discriminability between healthy and fractured classes
Identify which features are worth keeping b4 training

Outputs:
  - per_class_stats.csv
  - feature_discriminability.png
  - distribution_plots.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys
import os

def load_data(csv_path):
    drop_cols = ["dataset_id"]
    label_col = "pathology_label"

    full_df = pd.read_csv(csv_path)
    feature_cols = [c for c in full_df.columns if c != label_col and c not in drop_cols]
    feature_df = full_df[feature_cols]
    mapping = {0: 0, 1: 1, 2: 2}
    labels = df[label_col].map(mapping)
    
    return full_df, feature_df, labels, feature_cols

def per_class_stats(feature_df, labels, feature_cols):
    healthy_df = feature_df[labels == 0]
    fractured_df = feature_df[labels == 1]
 
    stat_rows = []
    for col in feature_cols:
        healthy_vals = healthy_df[col].dropna()
        fractured_vals = fractured_df[col].dropna()
 
        mean_diff  = abs(healthy_vals.mean() - fractured_vals.mean())

        # mannwhitney (doesnt assumr normality)
        _, p_value = stats.mannwhitneyu(healthy_vals, fractured_vals, alternative='two-sided')

        # Cohen's d
        pooled_std = np.sqrt((healthy_vals.std()**2 + fractured_vals.std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
 
        if p_value < 0.05:
            significance = "YES"
        else:
            significance = "NO"
 
        stat_rows.append({
            "feature":           col,
            "mean_healthy":      round(healthy_vals.mean(), 4),
            "std_healthy":       round(healthy_vals.std(), 4),
            "mean_pathological": round(fractured_vals.mean(), 4),
            "std_pathological":  round(fractured_vals.std(), 4),
            "mean_difference":   round(mean_diff, 4),
            "mannwhitney_p":     round(p_value, 4),
            "cohens_d":          round(cohens_d, 4),
            "is_significant":    significance,
        })
 
    return pd.DataFrame(stat_rows).sort_values("cohens_d", ascending=False)


def plot_discriminability(stats_df, out_dir):
    num_features = len(stats_df)
    feature_indices = np.arange(num_features)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(feature_indices - width/2,
           stats_df["mean_healthy"],
           width,
           yerr=stats_df["std_healthy"],
           label="Healthy",
           color="steelblue", alpha=0.8, capsize=4)

    ax.bar(feature_indices + width/2,
           stats_df["mean_pathological"],
           width,
           yerr=stats_df["std_pathological"],
           label="Pathological",
           color="coral", alpha=0.8, capsize=4)

    ax.set_xticks(feature_indices)
    ax.set_xticklabels(stats_df["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Per-Class Mean +/- Std (sorted by Cohen's d)")
    ax.set_ylabel("Feature Value")
    ax.legend()
    plt.tight_layout()

    path = out_dir / "feature_discriminability.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_distributions(X, y, feature_cols, out_dir):
    n_cols = 4
    n_rows = int(np.ceil(len(feature_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    healthy = X[y == 0]
    fractured = X[y == 1]

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        sns.kdeplot(healthy[col], ax=ax, label="Healthy", color="steelblue", fill=True, alpha=0.4)
        sns.kdeplot(fractured[col], ax=ax, label="Fractured", color="coral", fill=True, alpha=0.4)
        ax.set_title(col.replace("original_", ""), fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=13, y=1.01)
    plt.tight_layout()

    path = out_dir / "distribution_plots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 1:
        print("Usage: python 2-dataset_analysis.py <input_csv_cleaned_v2>")
        sys.exit(1)

    input_csv = args[0]
    output_path = Path(input_csv).parent / "dataset analysis script output"
    os.makedirs(output_path, exist_ok=True)

    df, X, y, feature_cols = load_data(input_csv)

    print(f"Healthy: {(y==0).sum()}, Fractured: {(y==1).sum()}")

    stats_df = per_class_stats(X, y, feature_cols)
    stats_path = output_path / "per_class_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved: {stats_path}")

    plot_discriminability(stats_df, output_path)
    plot_distributions(X, y, feature_cols, output_path)