"""Score the record-level confidence attackers from run_record_confidence.py.

Every attacker outputs a 100-row table with a confidence per row; nothing
is fitted, so a score is calibrated "out of the box" only if, among rows
it gives confidence c, about a fraction c are correct. Reports per m and
attacker: table accuracy (1 - global error), mean confidence, the gap
between the two, ECE, Brier score and AUROC (does confidence separate the
correct rows from the wrong ones), plus paired per-seed tests of
prior-guided sampling against the no-trees baseline, separately for each
tree selection (first m in TreeFARMS order vs. m at random).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_output import run_analysis
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "record_confidence"
RESULTS_DIR = ROOT / "results"
N_BINS = 10
MIN_BIN_ROWS = 20  # bins with fewer rows are too noisy to plot

# Probability-valued scores; per_tree_absolute is a count, so it gets no calibration metrics.
PROBABILITY_METHODS = ["prior_guided", "prior_only", "per_tree_occurrence", "per_tree_presence", "seed_only"]
ALL_METHODS = PROBABILITY_METHODS + ["per_tree_absolute"]
PLOTTED = {  # fixed categorical order; marker shape is the non-color identity cue
    "prior_guided": ("Prior-guided sampling", "#2a78d6", "o"),
    "prior_only": ("Public statistics only (no trees)", "#eb6834", "s"),
    "per_tree_occurrence": ("Per-tree counting (per copy)", "#1baf7a", "^"),
    "seed_only": ("Joint solves, solver seed only", "#eda100", "D"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    return float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    if correct.min() == correct.max() or conf.min() == conf.max():
        return float("nan")
    return float(roc_auc_score(correct, conf))


def per_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (selection, m, method, seed), g in df.groupby(["selection", "m", "method", "seed"]):
        conf, ok = g["confidence"].to_numpy(), g["correct"].to_numpy().astype(float)
        rows.append({"selection": selection, "m": m, "method": method, "seed": seed, "accuracy": ok.mean(), "auroc": auroc(conf, ok),
                     "brier": float(np.mean((conf - ok) ** 2)) if method in PROBABILITY_METHODS else np.nan})
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (selection, m, method), g in df.groupby(["selection", "m", "method"]):
        conf, ok = g["confidence"].to_numpy(), g["correct"].to_numpy().astype(float)
        s = seeds[(seeds["selection"] == selection) & (seeds["m"] == m) & (seeds["method"] == method)]
        is_prob = method in PROBABILITY_METHODS
        rows.append({
            "selection": selection, "m": m, "method": method, "seeds": g["seed"].nunique(),
            "accuracy": ok.mean(), "acc_sem": s["accuracy"].sem(),
            "mean_conf": conf.mean() if is_prob else np.nan,
            "gap": conf.mean() - ok.mean() if is_prob else np.nan,
            "ece": ece(conf, ok) if is_prob else np.nan,
            "brier": float(np.mean((conf - ok) ** 2)) if is_prob else np.nan,
            "auroc_pooled": auroc(conf, ok), "auroc_seed_mean": s["auroc"].mean(),
        })
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], ALL_METHODS, ordered=True)
    return out.sort_values(["selection", "m", "method"])


def paired_tests(seeds: pd.DataFrame, a: str = "prior_guided", b: str = "prior_only") -> None:
    print(f"\nPaired per-seed comparison, {a} minus {b} (Wilcoxon signed-rank):")
    for (selection, m), g in seeds.groupby(["selection", "m"]):
        wide = g.pivot(index="seed", columns="method")
        for metric in ["accuracy", "auroc", "brier"]:
            diff = (wide[(metric, a)] - wide[(metric, b)]).dropna()
            p = wilcoxon(diff).pvalue if len(diff) >= 5 and diff.abs().sum() > 0 else float("nan")
            print(f"  {selection:<6} m={m:<4} {metric:<9} mean diff {diff.mean():+.4f} (n={len(diff)}, "
                  f"{(diff > 0).sum()} seeds up, {(diff < 0).sum()} down) p={p:.3g}")


def selection_effect(seeds: pd.DataFrame, method: str = "prior_guided") -> None:
    """Same seeds, same m: how much does releasing random instead of the
    first m trees change the attacker's table accuracy?"""
    g = seeds[seeds["method"] == method]
    if g["selection"].nunique() < 2:
        return
    print(f"\nTree selection effect on {method} accuracy, random minus first (paired by seed):")
    for m, sub in g.groupby("m"):
        wide = sub.pivot(index="seed", columns="selection", values="accuracy").dropna()
        if {"first", "random"} <= set(wide.columns) and len(wide) >= 5:
            diff = wide["random"] - wide["first"]
            p = wilcoxon(diff).pvalue if diff.abs().sum() > 0 else float("nan")
            print(f"  m={m:<4} first {wide['first'].mean():.3f}  random {wide['random'].mean():.3f}  "
                  f"diff {diff.mean():+.4f} (n={len(diff)}) p={p:.3g}")


def reliability_figure(df: pd.DataFrame) -> Path:
    m_values = sorted(df["m"].unique())
    selections = sorted(df["selection"].unique())
    fig, axes = plt.subplots(len(selections), len(m_values), figsize=(4.2 * len(m_values), 4.0 * len(selections)),
                             sharex=True, sharey=True, squeeze=False)
    for (row, selection), (col, m) in ((r, c) for r in enumerate(selections) for c in enumerate(m_values)):
        ax = axes[row][col]
        ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Perfectly calibrated")
        for method, (label, color, marker) in PLOTTED.items():
            g = df[(df["selection"] == selection) & (df["m"] == m) & (df["method"] == method)]
            if g.empty:
                continue
            bins = np.clip((g["confidence"].to_numpy() * N_BINS).astype(int), 0, N_BINS - 1)
            agg = g.assign(bin=bins).groupby("bin").agg(conf=("confidence", "mean"), acc=("correct", "mean"),
                                                        n=("correct", "size"))
            agg = agg[agg["n"] >= MIN_BIN_ROWS]
            ax.plot(agg["conf"], agg["acc"], color=color, marker=marker, markersize=6, linewidth=2,
                    markeredgecolor="white", markeredgewidth=1, label=label)
        ax.set_title(f"m = {m} trees, {selection} selection", color=INK, fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Stated confidence (no calibration fitted)", color=MUTED, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for row in axes:
        row[0].set_ylabel("Share of rows actually correct", color=MUTED, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1 / len(selections), 1, 1))
    out = RESULTS_DIR / "record_confidence_reliability.pdf"
    fig.savefig(out)
    return out


def main() -> None:
    files = sorted(RAW_DIR.glob("*/m*/seed*.csv"))  # <selection>/m<m>/seed<seed>.csv
    if not files:
        raise SystemExit(f"no results under {RAW_DIR}; run run_record_confidence.py first")
    df = pd.concat([pd.read_csv(f).assign(selection=f.parent.parent.name) for f in files], ignore_index=True)
    skipped = sorted(p.relative_to(RAW_DIR) for p in RAW_DIR.glob("*/m*/seed*.skip"))
    print(f"Loaded {len(files)} (selection, m, seed) files, {len(df)} output rows; skipped (Rashomon set < m): "
          f"{[str(s) for s in skipped] or 'none'}")

    seeds = per_seed_metrics(df)
    table = summary_table(df, seeds)
    print("\nPer attacker: accuracy = 1 - global error of its 100-row table; gap = mean confidence - accuracy")
    print("(positive gap = overconfident). per_tree_absolute is a count, not a probability.")
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))

    paired_tests(seeds)
    selection_effect(seeds)
    print(f"\nWrote {reliability_figure(df)}")


if __name__ == "__main__":
    run_analysis(main, RESULTS_DIR / "summaries" / "analyze_record_confidence_summary.txt")
