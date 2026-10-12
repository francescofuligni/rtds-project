"""Aggregate results/final_experiment.csv across seeds and plot
reconstruction error vs. number of released trees (m), one line per
background-knowledge condition, with baseline reference lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "final_experiment.csv"
FIGURES_DIR = ROOT / "report" / "figures"

def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm's step-down procedure. Returns, in the same order as
    `pvalues`, whether each test is significant after correcting for
    `len(pvalues)` simultaneous comparisons."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    k = len(pvalues)
    significant = [False] * k
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (k - rank):
            significant[i] = True
        else:
            break  # Holm stops at the first failure to reject
    return significant


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
    n_seeds = df["seed"].nunique()
    ax.set_ylabel(f"Reconstruction error on T = sex\n(mean $\\pm$ SEM over {n_seeds} seeds)")
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

    print("\nWilcoxon signed-rank test (condition vs. None), pooled across all m:")
    for condition in ["A", "B", "A+B"]:
        diffs = (paired[condition] - paired["none"]).dropna()
        stat, p = wilcoxon(diffs)
        print(f"  {condition}: n={len(diffs)}, mean_diff={diffs.mean():+.4f}, p={p:.4g}")

    print("\nWilcoxon signed-rank test (condition vs. None), per m, with Holm-Bonferroni correction:")
    per_m_tests = []
    for m in sorted(df["m"].unique()):
        sub = paired.xs(m, level="m").dropna()
        for condition in ["A", "B", "A+B"]:
            diffs = sub[condition] - sub["none"]
            if len(diffs) >= 5 and diffs.abs().sum() > 0:
                stat, p = wilcoxon(diffs)
                per_m_tests.append((m, condition, len(diffs), diffs.mean(), p))

    pvalues = [t[4] for t in per_m_tests]
    sig = holm_bonferroni(pvalues)
    for (m, condition, n, mean_diff, p), is_sig in zip(per_m_tests, sig):
        flag = "*" if is_sig else " "
        print(f"  m={m:<4} {condition:<4} n={n:<3} mean_diff={mean_diff:+.4f} p={p:.4g} {flag}")
    print(f"  ({sum(sig)}/{len(sig)} tests survive Holm-Bonferroni correction at alpha=0.05)")


if __name__ == "__main__":
    main()
