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
from analysis_output import run_analysis
from output_paths import latest_result_dir
from provenance import describe, load_results
from scipy.stats import pearsonr, spearmanr

EXPERIMENT = "final_experiment"


def main(result_dir: Path):
    df, manifest = load_results(result_dir, EXPERIMENT)
    print(describe(manifest) + "\n")

    error = df.pivot_table(
        index=["seed", "m"], columns="condition", values="reconstruction_error"
    )
    joint = df.groupby(["seed", "m"])[
        ["joint_split_frac_A", "joint_split_frac_B"]
    ].first()
    merged = error.join(joint).dropna(subset=["none"])

    merged["reduction_A"] = merged["none"] - merged["A"]
    merged["reduction_B"] = merged["none"] - merged["B"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (attr, color) in zip(axes, [("A", "#1b9e77"), ("B", "#d95f02")]):
        sub = merged.dropna(subset=[f"reduction_{attr}", f"joint_split_frac_{attr}"])
        ax.scatter(
            sub[f"joint_split_frac_{attr}"],
            sub[f"reduction_{attr}"],
            alpha=0.35,
            s=18,
            color=color,
        )
        ax.axhline(0, color="#999999", linewidth=1, linestyle=":")
        ax.set_xlabel(f"Fraction of trees with a joint T+{attr} split leaf")
        ax.set_title(f"Attribute {attr}")
        if len(sub) >= 3 and sub[f"joint_split_frac_{attr}"].std() > 0:
            r_p, p_p = pearsonr(
                sub[f"joint_split_frac_{attr}"], sub[f"reduction_{attr}"]
            )
            r_s, p_s = spearmanr(
                sub[f"joint_split_frac_{attr}"], sub[f"reduction_{attr}"]
            )
            ax.text(
                0.05,
                0.95,
                f"Pearson r={r_p:.2f} (p={p_p:.2g})\nSpearman r={r_s:.2f} (p={p_s:.2g})\nn={len(sub)}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            print(
                f"{attr}: Pearson r={r_p:.3f} (p={p_p:.3g}), Spearman r={r_s:.3f} (p={p_s:.3g}), n={len(sub)}"
            )

    axes[0].set_ylabel("Error reduction vs. None\n(None error - condition error)")
    fig.suptitle(
        "Does joint T+attribute splitting predict the background-knowledge benefit?",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = result_dir / "mechanism_joint_split_correlation.pdf"
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
        result_dir / "summary_analyze_mechanism.txt",
    )
