"""Reliability diagram for the solver-agreement confidence proxy:
bins rows by their agreement rate (fraction of repeated solves, with
different solver seeds, that agree on the row's reconstructed T after
row-matching) and checks whether higher agreement actually predicts
higher empirical accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from analysis_output import run_analysis
from output_paths import latest_result_dir
from provenance import describe, load_results
from scipy.stats import spearmanr

EXPERIMENT = "calibration_experiment"


def main(result_dir: Path):
    df, manifest = load_results(result_dir, EXPERIMENT)
    print(describe(manifest) + "\n")

    m_values = sorted(df["m"].unique())
    fig, axes = plt.subplots(
        1, len(m_values), figsize=(5.5 * len(m_values), 4.2), sharey=True
    )
    if len(m_values) == 1:
        axes = [axes]

    print("Agreement-rate reliability table (pooled across conditions/seeds):")
    for ax, m in zip(axes, m_values):
        sub = df[df["m"] == m]
        agg = sub.groupby("agreement")["correct"].agg(["mean", "count"]).reset_index()
        agg = agg.sort_values("agreement")
        print(f"\nm={m}:")
        print(agg.rename(columns={"mean": "empirical_accuracy"}).to_string(index=False))

        ax.scatter(
            agg["agreement"],
            agg["mean"],
            s=(agg["count"] / agg["count"].max() * 300 + 20),
            alpha=0.7,
            color="#1b9e77",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#999999",
            linewidth=1,
            label="perfect calibration",
        )
        ax.set_xlim(0.4, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Agreement rate across repeated solves\n(confidence proxy)")
        ax.set_title(f"m = {m}")
        ax.grid(alpha=0.3)

        rho, p = spearmanr(sub["agreement"], sub["correct"])
        ax.text(
            0.42,
            0.95,
            f"Spearman $\\rho$={rho:.2f}\n(p={p:.2g}, n={len(sub)})",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        print(
            f"  row-level Spearman rho(agreement, correct) = {rho:.3f} (p={p:.3g}, n={len(sub)})"
        )

    axes[0].set_ylabel("Empirical accuracy\n(fraction of rows correctly reconstructed)")
    fig.suptitle(
        "Does solver agreement predict reconstruction correctness?", fontsize=11
    )
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    out_path = result_dir / "calibration_reliability.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="a stamped result set (directory with manifest.json)",
    )
    args = parser.parse_args()
    result_dir = args.results_dir or latest_result_dir(EXPERIMENT)
    run_analysis(
        lambda: main(result_dir),
        result_dir / "summary_analyze_calibration.txt",
    )
