#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

def plot_missing_counts(df: pd.DataFrame, title: str, outfile: str, order: List[str] | None = None):
    counts = df.isna().sum()
    if order is None:
        counts = counts.sort_values(ascending=False)
        order = list(counts.index)
    counts = counts.reindex(order)

    fig, ax = plt.subplots(figsize=(10, 4))

    total_missing = int(counts.sum())
    all_zero = (total_missing == 0)

    # Use a tiny epsilon to draw visible bars for the all-zero case
    eps = 1e-6 if all_zero else 0.0
    heights = [eps if v == 0 else int(v) for v in counts.values]

    bars = ax.bar(range(len(order)), heights, color=("#d0d0d0" if all_zero else "#d9534f"), alpha=0.85)

    # Add value labels (show "0" cleanly when all_zero or value==0)
    for idx, rect in enumerate(bars):
        val = counts.values[idx]
        label_val = "0" if val == 0 else str(int(val))
        # place slightly above 0 for visibility in all-zero case
        y_pos = rect.get_height() + (0.02 if all_zero else 0.0)
        ax.text(rect.get_x() + rect.get_width()/2, y_pos, label_val,
                ha="center", va="bottom", fontsize=9, color="#333333")

    # Title & axes
    ax.set_title(title)
    ax.set_ylabel("Missing count")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")

    # *** Force y-axis to start at 0 ***
    if all_zero:
        # make a clear visible axis when everything is zero
        ax.set_ylim(0, 1)   # 0..1 clean range so labels/bars show; no negatives
    else:
        ymax = max(heights) if len(heights) else 1
        ax.set_ylim(0, ymax * 1.10)  # small headroom above tallest bar

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved: {outfile}")

def main(raw_csv, figdir):
    os.makedirs(figdir, exist_ok=True)

    df_raw = pd.read_csv(raw_csv)

    # Define the "after cleaning" modeling columns (your pipeline keep list)
    keep_cols = [
        "CreditScore","Geography","Gender","Age","Tenure","Balance",
        "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"
    ]
    df_after = df_raw[keep_cols].copy()

    # Use a consistent column order between before/after for easier visual compare
    before_order = list(df_raw.columns)
    # If some columns from keep_cols don't exist in df_raw, ignore quietly
    after_order = [c for c in keep_cols if c in df_after.columns]

    plot_missing_counts(df_raw,
                        title="Missing Values per Column (Before Cleaning)",
                        outfile=os.path.join(figdir, "missing_before.png"),
                        order=before_order)

    plot_missing_counts(df_after,
                        title="Missing Values per Column (After Cleaning)",
                        outfile=os.path.join(figdir, "missing_after.png"),
                        order=after_order)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate missing-values before/after figures (y-axis starts at 0).")
    p.add_argument("--raw_csv", default="churn.csv", help="Path to raw churn CSV")
    p.add_argument("--figdir", default="reports/figures", help="Output figures directory")
    args = p.parse_args()
    main(args.raw_csv, args.figdir)
