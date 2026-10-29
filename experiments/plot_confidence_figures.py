"""Record-confidence figures for the report.

Usage: .venv/bin/python experiments/plot_confidence_figures.py

Writes figures/confidence_reliability_m50.png (stated confidence against
observed accuracy at m = 50, prior-guided sampling vs public statistics only)
and figures/confidence_public_data.png (overconfidence with 2,000 vs 200
public records, plain vs Dirichlet prior, at m = 10, 25, 50). Overconfidence
is the per-seed mean confidence minus accuracy; error bars are 95%
t-intervals over seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_paths import FIGURES_ROOT

RESULTS = Path(__file__).resolve().parent.parent / "results"
OFFICIAL = "record_confidence/random"
MS = (10, 25, 50)
# (public records, prior) -> (colour, legend label, result set per m under results/)
PUBLIC_DATA = {
    (2000, "plain"): ("#9e9e9e", "2,000 records, plain", f"{OFFICIAL}/m{{m:03d}}"),
    (2000, "dirichlet"): (
        "#7fb8d9",
        "2,000 records, Dirichlet",
        {
            10: "20261020_234026/record_confidence/m010",
            25: "20261023_130000/record_confidence/m025",
            50: "20261023_130000/record_confidence/m050",
        },
    ),
    (200, "plain"): (
        "#555555",
        "200 records, plain",
        {
            10: "20261020_210941/record_confidence/m010",
            25: "20261022_080000/record_confidence/m025",
            50: "20261022_080000/record_confidence/m050",
        },
    ),
    (200, "dirichlet"): (
        "#0072B2",
        "200 records, Dirichlet",
        {
            10: "20261020_224806/record_confidence/m010",
            25: "20261022_080000_01/record_confidence/m025",
            50: "20261022_080000_01/record_confidence/m050",
        },
    ),
}


def rows(directory: Path, method: str = "prior_guided") -> pd.DataFrame:
    files = sorted(directory.glob("seed*.csv"))
    if not files:
        raise FileNotFoundError(f"No seed CSVs in {directory}")
    df = pd.concat(pd.read_csv(f) for f in files)
    return df[df.method == method]


def overconfidence(directory: Path) -> tuple[float, float]:
    by_seed = rows(directory).groupby("seed")
    v = (by_seed.confidence.mean() - by_seed.correct.mean()).to_numpy() * 100
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * stats.sem(v)


def reliability(out: Path) -> None:
    edges = np.linspace(0, 1, 11)
    edges[-1] = 1.0001  # confidence 1.0 belongs to the last bin
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, ls="--", label="Perfect calibration")
    for method, color, marker, label in [
        ("prior_only", "#555555", "s", "Public statistics only"),
        ("prior_guided", "#0072B2", "o", "Prior-guided, m = 50"),
    ]:
        d = rows(RESULTS / OFFICIAL / "m050", method)
        g = d.groupby(pd.cut(d.confidence, edges, right=False), observed=True).agg(
            conf=("confidence", "mean"), acc=("correct", "mean"), n=("correct", "size")
        )
        ax.plot(g.conf, g.acc, color=color, lw=1.8, label=label)
        ax.scatter(
            g.conf,
            g.acc,
            s=10 + 900 * g.n / len(d),
            color=color,
            marker=marker,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Stated confidence", fontsize=12)
    ax.set_ylabel("Share of rows correct", fontsize=12)
    ax.grid(color="#e4e3df", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def public_data(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    width = 0.19
    for (color, label, sets), offset in zip(
        PUBLIC_DATA.values(), (-1.5, -0.5, 0.5, 1.5)
    ):
        bars = [
            overconfidence(
                RESULTS / (sets.format(m=m) if isinstance(sets, str) else sets[m])
            )
            for m in MS
        ]
        means, halves = zip(*bars)
        ax.bar(
            np.arange(len(MS)) + offset * width,
            means,
            width * 0.92,
            yerr=halves,
            color=color,
            ecolor="#333333",
            capsize=2.5,
            error_kw={"lw": 1},
            label=label,
        )
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(range(len(MS)), [f"m = {m}" for m in MS], fontsize=11)
    ax.set_ylabel("Overconfidence (percentage points)", fontsize=12)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-1.2, 13.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=2, columnspacing=1)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    reliability(FIGURES_ROOT / "confidence_reliability_m50.png")
    public_data(FIGURES_ROOT / "confidence_public_data.png")
    print(f"wrote confidence figures to {FIGURES_ROOT}")
