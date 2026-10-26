"""Overconfidence of the record confidence with 200 vs 2,000 public records.

Usage: .venv/bin/python experiments/plot_public_data_calibration.py
       (writes figures/public_data_calibration.png)

Bars are per-seed means of stated confidence minus accuracy (prior-guided
attacker, points) with 95% t-intervals over seeds, for a plain public prior and
the Dirichlet record prior, at m = 10, 25, 50.

The 2,000-record plain bars come from the official random record-confidence
run, which commit 9d56b3a moved out of main. Restore it with
    git restore --source=9d56b3a~1 --worktree -- results/record_confidence/random
or pass --official-dir. Every other bar reads a timestamped set under results/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MS = (10, 25, 50)

INK_2, MUTED, GRID = "#52514e", "#a3a29b", "#e6e5e0"
# (public records, prior) -> (bar colour, legend label), in left-to-right order
SERIES = {
    (200, "plain"): ("#6f6e68", "200 records, plain"),
    (200, "dirichlet"): ("#2a78d6", "200 records, Dirichlet"),
    (2000, "plain"): ("#b9b8b1", "2,000 records, plain"),
    (2000, "dirichlet"): ("#8db9f0", "2,000 records, Dirichlet"),
}
# Result sets per (public records, prior) and m, relative to results/; None
# marks the official run passed as --official-dir.
SETS = {
    (200, "plain"): {
        10: "20261020_210941/record_confidence/m010",
        25: "20261022_080000/record_confidence/m025",
        50: "20261022_080000/record_confidence/m050",
    },
    (200, "dirichlet"): {
        10: "20261020_224806/record_confidence/m010",
        25: "20261022_080000_01/record_confidence/m025",
        50: "20261022_080000_01/record_confidence/m050",
    },
    (2000, "plain"): {m: None for m in MS},
    (2000, "dirichlet"): {
        10: "20261020_234026/record_confidence/m010",
        25: "20261023_130000/record_confidence/m025",
        50: "20261023_130000/record_confidence/m050",
    },
}

plt.rcParams.update(
    {
        "font.family": "Inter",
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 1,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def overconfidence(directory: Path) -> tuple[float, float]:
    """Per-seed mean of confidence minus correctness, with its 95% half-width."""
    rows = pd.concat(pd.read_csv(f) for f in sorted(directory.glob("seed*.csv")))
    by_seed = rows[rows.method == "prior_guided"].groupby("seed")
    v = (by_seed.confidence.mean() - by_seed.correct.mean()).to_numpy() * 100
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * stats.sem(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-dir", type=Path, default=RESULTS / "record_confidence" / "random"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "figures" / "public_data_calibration.png"
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    width = 0.17
    # two pairs per m, public data growing left to right: 200 records, gap, 2,000
    for (key, (color, label)), offset in zip(
        SERIES.items(), (-0.29, -0.12, 0.12, 0.29)
    ):
        bars = [
            overconfidence(
                args.official_dir / f"m{m:03d}" if d is None else RESULTS / d
            )
            for m, d in SETS[key].items()
        ]
        xs = np.arange(len(MS)) + offset
        means, halves = zip(*bars)
        ax.bar(
            xs,
            means,
            width * 0.92,
            color=color,
            yerr=halves,
            ecolor=INK_2,
            capsize=3,
            error_kw={"lw": 1.2},
            label=label,
        )
        for x, mean, half in zip(xs, means, halves):
            ax.annotate(
                f"{mean:.1f}",
                (x, mean + half),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#0b0b0b",
            )
    ax.set_axisbelow(True)
    # under each pair its public-data size, under each group its m
    pair_x = [g + d for g in range(len(MS)) for d in (-0.205, 0.205)]
    ax.set_xticks(pair_x, ["200", "2,000"] * len(MS), fontsize=11)
    ax.tick_params(axis="x", length=0, pad=4)
    for g, m in enumerate(MS):
        ax.text(
            g,
            -0.17,
            f"m = {m}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=13,
            color=INK_2,
        )
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlim(-0.5, len(MS) - 0.5)
    ax.set_ylim(0, 11)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("overconfidence (points)")
    ax.legend(
        frameon=False,
        fontsize=11,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        borderaxespad=0,
        columnspacing=1.6,
        handlelength=1.2,
    )
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
