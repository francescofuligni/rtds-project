"""Summarise a saved main experiment and optionally plot error by release size.

Paired comparisons use available seeds separately at each release size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from analysis_output import condition_labels, run_analysis
from output_paths import figure_output_dir, latest_result_dir
from provenance import describe, load_results, print_outcome_coverage
from scipy.stats import wilcoxon

EXPERIMENT = "final_experiment"


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
    "none": {"color": "#555555", "marker": "o"},
    "A": {"color": "#1b9e77", "marker": "s"},
    "B": {"color": "#d95f02", "marker": "^"},
    "A+B": {"color": "#7570b3", "marker": "D"},
}


def main(result_dir: Path, figure_dir: Path, *, plots: bool = True):
    try:
        df, manifest = load_results(result_dir, EXPERIMENT)
    except FileNotFoundError as exc:
        print_outcome_coverage(result_dir)
        print(f"No valid data available: {exc}")
        return
    print(describe(manifest) + "\n")
    print_outcome_coverage(result_dir)
    if df.reconstruction_error.notna().sum() == 0:
        print("No valid reconstruction outcomes; statistics and plots unavailable.")
        return
    labels = condition_labels(manifest["protocol"])

    agg = (
        df.groupby(["m", "condition"])["reconstruction_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / agg["count"] ** 0.5

    uniform_mean = df["uniform_baseline_error"].mean()
    marginal_mean = df["marginal_baseline_error"].mean()

    if plots:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for condition, style in CONDITION_STYLE.items():
            sub = agg[agg["condition"] == condition].sort_values("m")
            ax.errorbar(
                sub["m"],
                sub["mean"],
                yerr=sub["sem"],
                capsize=3,
                linewidth=1.5,
                label=labels[condition],
                **style,
            )

        ax.axhline(
            uniform_mean,
            color="#999999",
            linestyle=":",
            linewidth=1.2,
            label="Uniform baseline",
        )
        ax.axhline(
            marginal_mean,
            color="#999999",
            linestyle="--",
            linewidth=1.2,
            label="Marginal baseline",
        )

        ax.set_xscale("log")
        ax.set_xticks(sorted(df["m"].unique()))
        ax.set_xticklabels([str(m) for m in sorted(df["m"].unique())])
        ax.set_xlabel("Number of released trees (m)")
        n_seeds = df["seed"].nunique()
        ax.set_ylabel(
            f"Reconstruction error on T = sex\n(mean $\\pm$ SEM over {n_seeds} seeds)"
        )
        ax.set_title(
            "Reconstruction error vs. released trees, by background knowledge",
            fontsize=11,
        )
        ax.grid(alpha=0.3)
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            fontsize=8,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.17, 1, 1))

        figure_dir.mkdir(parents=True, exist_ok=True)
        out_path = figure_dir / "reconstruction_error_vs_m.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Wrote {out_path}")

    print("\nAggregate table (mean error by m, condition):")
    pivot = (
        agg.pivot(index="m", columns="condition", values="mean")
        .rename(columns=labels)
        .round(3)
    )
    print(pivot)

    print("\nAggregate table (mean +/- SEM, for Table 1 in the report):")
    for condition in ["none", "A", "B", "A+B"]:
        sub = agg[agg["condition"] == condition].sort_values("m")
        cells = [f"{r['mean']:.3f}+/-{r['sem']:.3f}" for _, r in sub.iterrows()]
        print(f"  {labels[condition]:<40} {cells}")

    print(f"\nUniform baseline mean: {uniform_mean:.3f}")
    print(f"Marginal baseline mean: {marginal_mean:.3f}")

    paired_statistics(df, labels)


def paired_statistics(df, labels):
    """Each seed contributes once to each per-m comparison; never pool release sizes."""
    paired = df.pivot(
        index=["seed", "m"], columns="condition", values="reconstruction_error"
    ).reindex(columns=["none", "A", "B", "A+B"])
    tests = []
    print("\nAvailable paired comparisons vs. None (per m):")
    for m in sorted(df["m"].unique()):
        sub = paired.xs(m, level="m")
        for condition in ["A", "B", "A+B"]:
            pairs = sub[["none", condition]].dropna()
            diffs = pairs[condition] - pairs["none"]
            wins = int((diffs < 0).sum())
            fraction = f"{wins / len(pairs):.3f}" if len(pairs) else "unavailable"
            print(
                f"  m={m} {labels[condition]}: wins={wins}, paired={len(pairs)}, "
                f"missing={len(sub) - len(pairs)}, fraction={fraction}"
            )
            if len(diffs) == 0:
                print("    Wilcoxon unavailable: no valid pairs")
            elif len(diffs) < 5:
                print("    Wilcoxon unavailable: fewer than 5 valid pairs")
            elif diffs.abs().sum() == 0:
                print("    Wilcoxon: all differences zero; p=1")
                tests.append((m, condition, len(diffs), 0.0, 1.0))
            else:
                _, p = wilcoxon(diffs)
                tests.append((m, condition, len(diffs), diffs.mean(), p))
    sig = holm_bonferroni([t[4] for t in tests])
    print("\nPer-m Wilcoxon tests with Holm-Bonferroni correction:")
    for (m, condition, n, mean_diff, p), significant in zip(tests, sig):
        print(
            f"  m={m} {labels[condition]}: n={n}, mean_diff={mean_diff:+.4f}, "
            f"p={p:.4g}, significant={significant}"
        )
    print(f"  ({sum(sig)}/{len(sig)} tests survive at alpha=0.05)")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="a stamped result set (directory with manifest.json)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="write numerical summaries only"
    )
    args = parser.parse_args()
    result_dir = args.results_dir or latest_result_dir(EXPERIMENT)
    run_analysis(
        lambda: main(result_dir, figure_output_dir(), plots=not args.no_plots),
        result_dir / "summary_analyze_results.txt",
    )


if __name__ == "__main__":
    cli()
