"""Figures for the A2 poster, drawn from the committed result sets.

Usage: .venv/bin/python poster/make_figures.py   (writes poster/figures/*.pdf)

Every number is computed here from per-seed CSVs; error bars and bands are 95%
t-intervals over seeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29b", "#e6e5e0"
WIDTH = 6.9  # inches: one poster column minus the box padding

plt.rcParams.update(
    {
        "font.family": "Inter",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
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
        "lines.linewidth": 2.5,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def ci(values) -> tuple[float, float]:
    """Mean and half-width of a 95% t-interval."""
    v = np.asarray(values, dtype=float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * stats.sem(v)


def record_rows(directory: Path) -> pd.DataFrame:
    return pd.concat(pd.read_csv(f) for f in sorted(directory.glob("seed*.csv")))


def per_seed(rows: pd.DataFrame, method: str) -> pd.DataFrame:
    d = rows[rows.method == method]
    g = d.groupby("seed")
    return pd.DataFrame(
        {
            "accuracy": g.correct.mean(),
            "vouched": g.confidence.apply(lambda c: (c >= 0.9).mean()),
            "overconfidence": g.confidence.mean() - g.correct.mean(),
        }
    )


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(
        text,
        (x, y),
        xytext=(8, dy),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=14,
        color=INK,
        fontweight="medium",
    )
    ax.plot([x], [y], "o", ms=8, color=color, zorder=5)


def leak_vs_m():
    """Rows correct and rows vouched for at >= 0.9, by number of released trees."""
    base = RESULTS / "record_confidence" / "random"
    ms = [10, 25, 50, 100]
    public = per_seed(record_rows(base / "m010"), "prior_only")
    series = {"accuracy": [ci(public.accuracy)], "vouched": [ci(public.vouched)]}
    vouched_correct = []
    for m in ms:
        rows = record_rows(base / f"m{m:03d}")
        s = per_seed(rows, "prior_guided")
        series["accuracy"].append(ci(s.accuracy))
        series["vouched"].append(ci(s.vouched))
        pg = rows[(rows.method == "prior_guided") & (rows.confidence >= 0.9)]
        vouched_correct.append(pg.groupby("seed").correct.mean().mean())

    x = [0, *ms]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    names = {
        "accuracy": ("rows correct", BLUE, (30, 81)),
        "vouched": (
            "rows the attacker vouches for (confidence ≥ 0.9)",
            ORANGE,
            (6, 55),
        ),
    }
    for key, (name, color, at) in names.items():
        mean = np.array([m for m, _ in series[key]]) * 100
        half = np.array([h for _, h in series[key]]) * 100
        ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.12, lw=0)
        ax.plot(x, mean, color=color)
        ax.plot(x, mean, "o", ms=8, color=color, markeredgecolor="white", mew=1.5)
        ax.text(*at, name, fontsize=14, color=INK, fontweight="medium")
        for xi, yi, side in [(x[0], mean[0], -1), (x[-1], mean[-1], 1)]:
            ax.annotate(
                f"{yi:.0f}%",
                (xi, yi),
                xytext=(10 * side, 0),
                textcoords="offset points",
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=13,
                color=INK,
            )
    lo, hi = min(vouched_correct), max(vouched_correct)
    ax.text(
        100,
        18,
        f"of which {lo:.0%}–{hi:.0%} are correct",
        ha="right",
        fontsize=13,
        color=INK_2,
    )
    ax.set_xticks(x)
    ax.set_xlim(-14, 114)
    ax.set_ylim(0, 92)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("released trees m   (m = 0: public statistics only)")
    ax.set_ylabel("share of the 100 private rows")
    fig.savefig(OUT / "leak_vs_m.pdf")
    plt.close(fig)


def reliability():
    """Stated confidence against observed accuracy, pooled rows at m = 50."""
    rows = record_rows(RESULTS / "record_confidence" / "random" / "m050")
    edges = np.linspace(0, 1, 11)
    edges[-1] = 1.0001
    fig, ax = plt.subplots(figsize=(WIDTH, 2.45))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    handles = []
    for method, color, name in [
        ("prior_only", MUTED, "public statistics only"),
        ("prior_guided", BLUE, "our attack, m = 50 trees"),
    ]:
        d = rows[rows.method == method]
        b = pd.cut(d.confidence, edges, right=False)
        g = d.groupby(b, observed=True).agg(
            conf=("confidence", "mean"), acc=("correct", "mean"), n=("correct", "size")
        )
        ax.plot(g.conf, g.acc, color=color, lw=2, zorder=2)
        ax.scatter(
            g.conf,
            g.acc,
            s=g.n / len(d) * 2500,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        handles.append(
            plt.Line2D([], [], color=color, marker="o", ms=10, lw=2, label=name)
        )
    ax.legend(
        handles=handles[::-1],
        loc="upper left",
        frameon=False,
        fontsize=13,
        borderaxespad=0.2,
    )
    ax.text(
        0.98,
        0.05,
        "dashed: perfect calibration\ndot area: number of rows",
        ha="right",
        fontsize=12,
        color=INK_2,
        transform=ax.transAxes,
    )
    ax.set_xlim(0, 1.06)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("stated confidence (no fitting, no ground truth)")
    ax.set_ylabel("rows actually correct")
    ax.grid(axis="x", color=GRID, lw=1)
    fig.savefig(OUT / "reliability_m50.pdf")
    plt.close(fig)


def final_cells(timestamp: str) -> pd.DataFrame:
    d = record_rows(RESULTS / timestamp / "final_experiment")
    wide = d.pivot_table(
        index=["seed", "m"], columns="condition", values="reconstruction_error"
    )
    return wide.dropna()  # cells where all four conditions found a table


def background_knowledge():
    """Error on sex against m, with and without background knowledge."""
    edu, age = final_cells("20261020_195325"), final_cells("20261020_195325_01")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    ax.axhline(0.445, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax.text(1, 0.452, "guess from population shares", fontsize=12, color=INK_2)
    lines = [
        (edu, "none", BLUE, "no background knowledge", 0),
        (edu, "A+B", ORANGE, "knows marital status + education", -14),
        (age, "A+B", AQUA, "knows marital status + age", 0),
    ]
    for cells, condition, color, name, dy in lines:
        g = cells[condition].groupby("m")
        stats_m = g.agg(lambda v: ci(v)[0]), g.agg(lambda v: ci(v)[1])
        mean, half = stats_m
        if condition == "none":
            ax.fill_between(
                mean.index, mean - half, mean + half, color=color, alpha=0.12, lw=0
            )
        ax.plot(mean.index, mean, color=color, lw=2.5 if condition == "none" else 2)
    for cells, condition, color, name, dy in lines:
        mean = cells[condition].groupby("m").mean()
        ax.plot([mean.index[-1]], [mean.iloc[-1]], "o", ms=8, color=color)
    ax.legend(
        handles=[plt.Line2D([], [], color=c, lw=3, label=n) for _, _, c, n, _ in lines],
        loc="lower left",
        frameon=False,
        fontsize=13,
        borderaxespad=0.2,
    )
    start = edu["none"].groupby("m").mean()
    ax.annotate(
        f"{start.iloc[0]:.2f}",
        (1, start.iloc[0]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="left",
        fontsize=13,
        color=INK,
    )
    ax.annotate(
        f"{start.iloc[-1]:.2f}",
        (200, start.iloc[-1]),
        xytext=(8, 4),
        textcoords="offset points",
        ha="left",
        fontsize=13,
        color=INK,
    )
    ax.set_xscale("log")
    ticks = [1, 3, 10, 30, 100, 200]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0.1, 0.47)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_xlim(0.85, 260)
    ax.set_xlabel("released trees m (log scale)")
    ax.set_ylabel("wrong sex cells")
    fig.savefig(OUT / "background_knowledge.pdf")
    plt.close(fig)


def bars(ax, groups, pairs, colors, names, fmt):
    width = 0.36
    for i, (name, color) in enumerate(zip(names, colors)):
        xs = np.arange(len(groups)) + (i - 0.5) * width
        vals = [p[i] for p in pairs]
        ax.bar(
            xs,
            [v for v, _ in vals],
            width * 0.92,
            color=color,
            yerr=[h for _, h in vals],
            ecolor=INK_2,
            capsize=3,
            error_kw={"lw": 1.2},
            label=name,
        )
        for x, (v, h) in zip(xs, vals):
            ax.text(x, v + h + 0.6, fmt(v), ha="center", fontsize=12, color=INK)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.grid(axis="y")


def robustness():
    """Left: which trees are released. Right: scarce public data."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"wspace": 0.38}
    )
    rc = RESULTS / "record_confidence"
    pairs = []
    for m in (10, 25):
        first = per_seed(record_rows(rc / "first" / f"m{m:03d}"), "prior_guided")
        rand = per_seed(record_rows(rc / "random" / f"m{m:03d}"), "prior_guided")
        pairs.append((ci(first.accuracy * 100), ci(rand.accuracy * 100)))
    bars(
        left,
        ["m = 10", "m = 25"],
        pairs,
        [MUTED, BLUE],
        ["first m trees", "m random trees"],
        lambda v: f"{v:.0f}%",
    )
    left.set_ylim(0, 118)
    left.set_yticks([0, 20, 40, 60, 80])
    left.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    left.set_ylabel("rows correct")
    left.set_title("Which trees are released", loc="left", color=INK)
    left.legend(frameon=False, fontsize=12, loc="upper left", ncol=1, borderaxespad=0)

    sets = {
        10: ("20261020_210941", "20261020_224806"),
        25: ("20261022_080000", "20261022_080000_01"),
        50: ("20261022_080000", "20261022_080000_01"),
    }
    pairs = []
    for m, (plain, dirichlet) in sets.items():
        p = per_seed(
            record_rows(RESULTS / plain / "record_confidence" / f"m{m:03d}"),
            "prior_guided",
        )
        d = per_seed(
            record_rows(RESULTS / dirichlet / "record_confidence" / f"m{m:03d}"),
            "prior_guided",
        )
        pairs.append((ci(p.overconfidence * 100), ci(d.overconfidence * 100)))
    bars(
        right,
        ["m = 10", "m = 25", "m = 50"],
        pairs,
        [MUTED, BLUE],
        ["plain", "Dirichlet"],
        lambda v: f"+{v:.1f}",
    )
    right.set_ylim(0, 16.5)
    right.set_yticks([0, 3, 6, 9, 12])
    right.set_ylabel("overconfidence (points)")
    right.set_title("Only 200 public records", loc="left", color=INK)
    right.legend(frameon=False, fontsize=12, loc="upper right", borderaxespad=0)
    fig.savefig(OUT / "robustness.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    leak_vs_m()
    reliability()
    background_knowledge()
    robustness()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.pdf'))}")
