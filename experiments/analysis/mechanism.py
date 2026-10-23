"""Test the mechanism hypothesized in the report: background knowledge
of A/B should only reduce error on T where a tree's root-to-leaf path
jointly splits on T and A/B. Correlates each (seed, m) cell's
joint-split fraction (logged by run_final_experiment.py) against the
observed error reduction (None's error minus the condition's error).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from analysis_output import condition_labels, run_analysis
from output_paths import figure_output_dir, latest_result_dir
from provenance import describe, load_results, print_outcome_coverage
from scipy.stats import pearsonr, spearmanr

EXPERIMENT = "final_experiment"


def main(result_dir: Path, figure_dir: Path, *, plots: bool = True):
    try:
        df, manifest = load_results(result_dir, EXPERIMENT)
    except FileNotFoundError as exc:
        print_outcome_coverage(result_dir)
        print(f"No valid data available: {exc}")
        return
    print(describe(manifest) + "\n")
    print_outcome_coverage(result_dir)
    labels = condition_labels(manifest["protocol"])

    error = df.pivot_table(
        index=["seed", "m"], columns="condition", values="reconstruction_error"
    )
    error = error.reindex(columns=["none", "A", "B", "A+B"])
    joint = df.groupby(["seed", "m"])[
        ["joint_split_frac_A", "joint_split_frac_B"]
    ].first()
    merged = error.join(joint).dropna(subset=["none"])

    merged["reduction_A"] = merged["none"] - merged["A"]
    merged["reduction_B"] = merged["none"] - merged["B"]

    if merged.empty:
        print("No valid reconstruction outcomes; mechanism analysis unavailable.")
        return

    if plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for index, (attr, color) in enumerate([("A", "#1b9e77"), ("B", "#d95f02")]):
        feature = labels[attr].split(": ", maxsplit=1)[1]
        sub = merged.dropna(subset=[f"reduction_{attr}", f"joint_split_frac_{attr}"])
        annotation = (
            f"n={len(sub)}; correlation unavailable (insufficient or constant data)"
        )
        if (
            len(sub) >= 3
            and sub[f"joint_split_frac_{attr}"].std() > 0
            and sub[f"reduction_{attr}"].std() > 0
        ):
            r_p, p_p = pearsonr(
                sub[f"joint_split_frac_{attr}"], sub[f"reduction_{attr}"]
            )
            r_s, p_s = spearmanr(
                sub[f"joint_split_frac_{attr}"], sub[f"reduction_{attr}"]
            )
            annotation = (
                f"Pearson r={r_p:.3f} (p={p_p:.3g}), "
                f"Spearman r={r_s:.3f} (p={p_s:.3g}), n={len(sub)}"
            )
        print(f"{labels[attr]}: {annotation}")
        if plots:
            ax = axes[index]
            ax.scatter(
                sub[f"joint_split_frac_{attr}"],
                sub[f"reduction_{attr}"],
                alpha=0.35,
                s=18,
                color=color,
            )
            ax.axhline(0, color="#999999", linewidth=1, linestyle=":")
            ax.set_xlabel(f"Trees splitting on T and {feature}")
            ax.set_title(labels[attr])
            ax.text(
                0.05,
                0.95,
                annotation,
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                wrap=True,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
    if plots:
        axes[0].set_ylabel("Error reduction vs. None\n(None error - condition error)")
        fig.suptitle(
            "Does joint T+attribute splitting predict the background-knowledge benefit?",
            fontsize=11,
        )
        fig.tight_layout()
        figure_dir.mkdir(parents=True, exist_ok=True)
        out_path = figure_dir / "mechanism_joint_split_correlation.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"\nWrote {out_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="a stamped result set (directory with manifest.json)",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result_dir = args.results_dir or latest_result_dir(EXPERIMENT)
    run_analysis(
        lambda: main(result_dir, figure_output_dir(), plots=not args.no_plots),
        result_dir / "summary_analyze_mechanism.txt",
    )


if __name__ == "__main__":
    cli()
