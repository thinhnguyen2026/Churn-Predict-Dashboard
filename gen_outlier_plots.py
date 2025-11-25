#!/usr/bin/env python3
"""
Generate simple BEFORE/AFTER scatter plots to show outliers
by capping extremes at the 1st/99th percentiles (winsorizing).

Outputs:
  reports/figures/scatter_balance_vs_age_before_after.png
  reports/figures/scatter_creditscore_vs_age_before_after.png
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cap_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    """Cap a numeric series at given quantiles (winsorize)."""
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower=lower, upper=upper)

def make_before_after_scatter(df, x, y, x_label, y_label, title_left, title_right, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Raw data (before)
    x_raw = df[x].astype(float)
    y_raw = df[y].astype(float)

    # Cleaned (after): cap extremes at 1st/99th percentiles
    x_cap = cap_series(x_raw, 0.01, 0.99)
    y_cap = cap_series(y_raw, 0.01, 0.99)

    # Use same axis limits on both subplots for a fair comparison
    x_min, x_max = float(x_raw.min()), float(x_raw.max())
    y_min, y_max = float(y_raw.min()), float(y_raw.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    # BEFORE
    axes[0].scatter(x_raw, y_raw, s=10, alpha=0.4)
    axes[0].set_title(title_left)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    # start axes at 0 if nonnegative data
    if x_min >= 0: axes[0].set_xlim(left=0)
    if y_min >= 0: axes[0].set_ylim(bottom=0)
    # keep full raw range for visibility
    axes[0].set_xlim(left=(0 if x_min >= 0 else x_min), right=x_max)
    axes[0].set_ylim(bottom=(0 if y_min >= 0 else y_min), top=y_max)
    axes[0].grid(True, linewidth=0.3, alpha=0.5)

    # AFTER
    axes[1].scatter(x_cap, y_cap, s=10, alpha=0.4, color="#2c7fb8")
    axes[1].set_title(title_right)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    # keep the SAME limits as the raw plot to visually show the effect
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].grid(True, linewidth=0.3, alpha=0.5)

    fig.suptitle(f"{y_label} vs {x_label} — Before vs After Capping (1st/99th pct)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")

def main(raw_csv, figdir):
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_csv(raw_csv)
    # Keep just the columns we need
    needed = ["Age", "Balance", "CreditScore"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {raw_csv}: {missing}")

    # Plot 1: Balance vs Age
    make_before_after_scatter(
        df=df,
        x="Age", y="Balance",
        x_label="Age", y_label="Balance",
        title_left="Before Cleaning (Raw)",
        title_right="After Cleaning (Capped 1%/99%)",
        outfile=os.path.join(figdir, "scatter_balance_vs_age_before_after.png"),
    )

    # Plot 2: CreditScore vs Age
    make_before_after_scatter(
        df=df,
        x="Age", y="CreditScore",
        x_label="Age", y_label="Credit Score",
        title_left="Before Cleaning (Raw)",
        title_right="After Cleaning (Capped 1%/99%)",
        outfile=os.path.join(figdir, "scatter_creditscore_vs_age_before_after.png"),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple before/after outlier scatter plots with winsorizing.")
    parser.add_argument("--raw_csv", default="churn.csv", help="Path to churn CSV")
    parser.add_argument("--figdir", default="reports/figures", help="Directory to save figures")
    args = parser.parse_args()
    main(args.raw_csv, args.figdir)
