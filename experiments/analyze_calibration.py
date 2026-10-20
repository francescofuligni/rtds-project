"""Plot diagnostic agreement against reconstruction accuracy.

The score uses evaluator-assisted alignment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from analysis_output import run_analysis
from experiment_io import (
    analysis_arguments,
    diagnostic_scores,
    load_results,
    print_availability,
)
from scipy.stats import spearmanr


def main(input_paths: list[Path], output_dir: Path):
    attempts = load_results(input_paths, diagnostic=True)
    print_availability(attempts)
    df = diagnostic_scores(attempts)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        rho, p = float("nan"), float("nan")
        if sub["agreement"].nunique() > 1 and sub["correct"].nunique() > 1:
            rho, p = spearmanr(sub["agreement"], sub["correct"])
        else:
            print("  Spearman correlation is undefined for constant input.")
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
    fig.suptitle("Diagnostic only: truth-aligned solver agreement", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    out_path = output_dir / "calibration_reliability.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    args = analysis_arguments(diagnostic=True)
    run_analysis(
        lambda: main(args.input, args.output_dir),
        args.output_dir / "summaries" / "analyze_calibration_summary.txt",
    )
