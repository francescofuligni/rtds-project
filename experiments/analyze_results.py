"""Aggregate results/final_experiment.csv across seeds and plot
reconstruction error vs. number of released trees (m), one line per
background-knowledge condition, with baseline reference lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "final_experiment.csv"
FIGURES_DIR = ROOT / "report" / "figures"

CONDITION_STYLE = {
    "none": dict(color="#555555", marker="o", label="None known"),
    "A": dict(color="#1b9e77", marker="s", label="A known (marital-status)"),
    "B": dict(color="#d95f02", marker="^", label="B known (education)"),
    "A+B": dict(color="#7570b3", marker="D", label="A+B known"),
}


def main():
    df = pd.read_csv(RESULTS_CSV)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    agg = df.groupby(["m", "condition"])["reconstruction_error"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / agg["count"] ** 0.5

    uniform_mean = df["uniform_baseline_error"].mean()
    marginal_mean = df["marginal_baseline_error"].mean()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for condition, style in CONDITION_STYLE.items():
        sub = agg[agg["condition"] == condition].sort_values("m")
        ax.errorbar(sub["m"], sub["mean"], yerr=sub["sem"], capsize=3, linewidth=1.5,
                    **style)

    ax.axhline(uniform_mean, color="#999999", linestyle=":", linewidth=1.2,
               label=f"Uniform random baseline ({uniform_mean:.2f})")
    ax.axhline(marginal_mean, color="#999999", linestyle="--", linewidth=1.2,
               label=f"Holdout-marginal baseline ({marginal_mean:.2f})")

    ax.set_xscale("log")
    ax.set_xticks(sorted(df["m"].unique()))
    ax.set_xticklabels([str(m) for m in sorted(df["m"].unique())])
    ax.set_xlabel("Number of released trees (m)")
    ax.set_ylabel("Reconstruction error on T = sex\n(mean $\\pm$ SEM over 10 seeds)")
    ax.set_title("Reconstruction error vs. released trees, by background knowledge", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / "reconstruction_error_vs_m.pdf"
    fig.savefig(out_path)
    print(f"Wrote {out_path}")

    print("\nAggregate table (mean error by m, condition):")
    pivot = agg.pivot(index="m", columns="condition", values="mean").round(3)
    print(pivot)

    print(f"\nUniform baseline mean: {uniform_mean:.3f}")
    print(f"Marginal baseline mean: {marginal_mean:.3f}")

    paired = df.pivot_table(index=["seed", "m"], columns="condition", values="reconstruction_error")
    print("\nFraction of seeds where condition beats 'none' (paired, per m):")
    for condition in ["A", "B", "A+B"]:
        frac = (paired[condition] < paired["none"]).groupby(level="m").mean()
        print(f"  {condition}: {dict(frac.round(2))}")


if __name__ == "__main__":
    main()
