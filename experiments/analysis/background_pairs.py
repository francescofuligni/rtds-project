"""Compare main-experiment result sets that differ only in the background
attributes A/B (e.g. marital-status/education vs. marital-status/age).

Every set uses the same seeds, so the same private rows and the same released
trees per (seed, m); conditions are compared within a seed. Reports, per set:
how many solves found a table (a timed-out solve has no error, so means over
the solved seeds alone would be biased), each condition's error and its gain
over None, and the benefit of releasing more trees (error at the smallest m
minus error at m). Across sets: the difference in gain per condition, paired
by seed, and a check that conditions with identical knowledge (None always,
A when both sets share A) reproduce the same error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from analysis_output import condition_labels, run_analysis
from provenance import describe, load_results, require_comparable
from scipy.stats import wilcoxon

EXPERIMENT = "final_experiment"
CONDITIONS = ["none", "A", "B", "A+B"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # one per result set
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _p(diff: pd.Series) -> float:
    diff = diff.dropna()
    if len(diff) < 5 or diff.abs().sum() == 0:
        return float("nan")
    return wilcoxon(diff).pvalue


def pair_name(protocol: dict) -> str:
    return f"{protocol['attr_a']} / {protocol['attr_b']}"


def solved_share(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(solved=df["reconstruction_error"].notna())
        .pivot_table(index="m", columns="condition", values="solved", aggfunc="mean")
        .reindex(columns=CONDITIONS)
    )


def complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    """(seed, m) x condition errors, keeping cells where all four solved."""
    wide = df.pivot_table(
        index=["seed", "m"], columns="condition", values="reconstruction_error"
    ).reindex(columns=CONDITIONS)
    return wide.dropna()


def report_set(name: str, df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    print(f"\n=== {name} ===")
    print("Share of solves that found a table (per m and condition):")
    print(solved_share(df).round(2).rename(columns=labels).to_string())
    wide = complete_cases(df)
    print(
        "\nMean error over (seed, m) cells where all four conditions solved; "
        "gain = None error - condition error, Wilcoxon paired by seed:"
    )
    rows = []
    for m, g in wide.groupby(level="m"):
        row = {"m": m, "cells": len(g)}
        for c in CONDITIONS:
            row[labels[c]] = g[c].mean()
        for c in CONDITIONS[1:]:
            gain = g["none"] - g[c]
            row[f"gain {c}"] = gain.mean()
            row[f"p {c}"] = _p(gain)
        rows.append(row)
    with pd.option_context("display.width", 220):
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    m_min = wide.index.get_level_values("m").min()
    print(
        f"\nBenefit of more trees: error at m={m_min} minus error at m "
        "(paired by seed, cells solved in both):"
    )
    first = wide.xs(m_min, level="m")
    for m, g in wide.groupby(level="m"):
        if m == m_min:
            continue
        g = g.droplevel("m")
        cells = []
        for c in CONDITIONS:
            diff = (first[c] - g[c]).dropna()
            cells.append(f"{c} {diff.mean():+.3f} (p={_p(diff):.2g})")
        print(f"  m={m:<4} n={len(g):<4} " + "  ".join(cells))
    return wide


def compare_sets(names: list[str], wides: list[pd.DataFrame], protocols: list[dict]):
    ref_name, ref = names[0], wides[0]
    for name, wide, protocol in zip(names[1:], wides[1:], protocols[1:]):
        print(f"\n=== {name} minus {ref_name} (paired by seed and m) ===")
        joined = ref.join(wide, lsuffix="_ref", rsuffix="_other", how="inner")
        same = ["none"] + (
            ["A"] if protocol["attr_a"] == protocols[0]["attr_a"] else []
        )
        for c in same:
            diff = (joined[f"{c}_other"] - joined[f"{c}_ref"]).abs()
            print(
                f"  identical knowledge, condition {c}: {(diff < 1e-12).mean():.0%} of "
                f"{len(diff)} cells reproduce the same error (max |diff| {diff.max():.3f})"
            )
        print("  Difference in gain over None (other - reference), per m:")
        for m, g in joined.groupby(level="m"):
            cells = []
            for c in CONDITIONS[1:]:
                gain_other = g["none_other"] - g[f"{c}_other"]
                gain_ref = g["none_ref"] - g[f"{c}_ref"]
                diff = gain_other - gain_ref
                cells.append(f"{c} {diff.mean():+.4f} (p={_p(diff):.2g})")
            print(f"    m={m:<4} n={len(g):<4} " + "  ".join(cells))


def figure(names, wides, all_labels, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True)
    for ax, c in zip(axes, CONDITIONS):
        for i, (name, wide, labels) in enumerate(zip(names, wides, all_labels)):
            agg = wide[c].groupby(level="m").agg(["mean", "sem"])
            ax.errorbar(
                agg.index,
                agg["mean"],
                yerr=agg["sem"],
                color=PALETTE[i],
                marker=MARKERS[i],
                markersize=5,
                linewidth=2,
                capsize=2,
                markeredgecolor="white",
                label=name,
            )
        title = all_labels[0][c] if c in ("none", "A") else c
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("Released trees m (log scale)", color=MUTED, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel(
        "Reconstruction error on T = sex\n(mean ± SEM, cells solved in all conditions)",
        color=MUTED,
        fontsize=9,
    )
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="lower center", ncol=len(names), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = out_dir / "background_pairs_error_vs_m.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main(result_dirs: list[Path], out_dir: Path) -> None:
    names, wides, protocols, all_labels = [], [], [], []
    for result_dir in result_dirs:
        df, manifest = load_results(result_dir, EXPERIMENT)
        print(f"{describe(manifest)}\n  directory: {result_dir}\n")
        protocols.append(manifest["protocol"])
    require_comparable(
        [{"protocol": p} for p in protocols], may_differ={"attr_a", "attr_b"}
    )
    for result_dir, protocol in zip(result_dirs, protocols):
        df, _ = load_results(result_dir, EXPERIMENT)
        labels = condition_labels(protocol)
        names.append(pair_name(protocol))
        all_labels.append(labels)
        wides.append(report_set(names[-1], df, labels))
    compare_sets(names, wides, protocols)
    print(f"\nWrote {figure(names, wides, all_labels, out_dir)}")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="final_experiment result sets differing only in attr_a/attr_b",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where the summary and figure go (default: the last result set)",
    )
    args = parser.parse_args()
    dirs = args.results_dirs
    out = args.out_dir or dirs[-1]
    run_analysis(lambda: main(dirs, out), out / "summary_compare_background_pairs.txt")


if __name__ == "__main__":
    cli()
