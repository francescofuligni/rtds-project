"""Analyze target errors, paired comparisons and model-free controls."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_output import run_analysis
from experiment_io import (
    analysis_arguments,
    load_results,
    print_availability,
)
from scipy.stats import wilcoxon


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Holm correction and return significance flags in input order."""
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
    "none": {"color": "#555555", "marker": "o", "label": "None known"},
    "A": {"color": "#1b9e77", "marker": "s", "label": "A known (marital-status)"},
    "B": {"color": "#d95f02", "marker": "^", "label": "B known (education)"},
    "A+B": {"color": "#7570b3", "marker": "D", "label": "A+B known"},
}


def paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each condition with None using available pairs."""
    paired = df.pivot(
        index=["seed", "m"], columns="condition", values="reconstruction_error"
    ).reindex(columns=["none", "A", "B", "A+B"])
    records = []
    eligible = []
    pvalues: list[float] = []
    for m in sorted(df["m"].unique()):
        for condition in ["A", "B", "A+B"]:
            sub = paired.xs(m, level="m")[["none", condition]].dropna()
            diff = sub[condition] - sub["none"]
            n = len(diff)
            p = None
            if n >= 5:
                if np.all(diff == 0):
                    p = 1.0
                else:
                    _stat, p_value = wilcoxon(diff)
                    p = float(p_value)
            records.append(
                {
                    "m": m,
                    "condition": condition,
                    "n_pairs": n,
                    "beat_none_fraction": None if not n else float((diff < 0).mean()),
                    "mean_difference": None if not n else float(diff.mean()),
                    "p_value": p,
                    "holm_significant": None,
                }
            )
            if p is not None:
                eligible.append(len(records) - 1)
                pvalues.append(p)
    flags = holm_bonferroni(pvalues)
    for i, flag in zip(eligible, flags):
        records[i]["holm_significant"] = flag
    return pd.DataFrame(records)


def main(input_paths: list[Path], output_dir: Path):
    df = load_results(input_paths)
    print_availability(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg = (
        df.groupby(["m", "condition"])["reconstruction_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["attempts"] = df.groupby(["m", "condition"]).size().to_numpy()
    agg["sem"] = agg["std"] / agg["count"] ** 0.5
    print(
        "Reconstruction error conditional on available candidates, with denominators:"
    )
    print(agg.to_string(index=False))
    evaluated = df.dropna(subset=["reconstruction_error"])
    if evaluated.empty:
        raise ValueError("no evaluated candidates available for plotting")
    baselines = evaluated[["uniform_baseline_error", "marginal_baseline_error"]].agg(
        ["mean", "count"]
    )
    print("Controls scored with the same attack permutation; denominators:")
    print(baselines.to_string())
    print(
        "Paired per-m comparisons; Holm correction over tests with >=5 available pairs:"
    )
    print(paired_comparisons(df).to_string(index=False))
    print("Pooled across-m hypothesis tests are omitted because seeds repeat across m.")
    if "matching_ambiguous" in evaluated:
        ambiguous = evaluated["matching_ambiguous"].astype(str).str.lower().eq("true")
        print(
            f"Ambiguous optimal assignments: {ambiguous.sum()}/{len(evaluated)} evaluated attempts"
        )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for condition, style in CONDITION_STYLE.items():
        sub = agg[(agg["condition"] == condition) & (agg["count"] > 0)].sort_values("m")
        if len(sub):
            ax.errorbar(
                sub["m"],
                sub["mean"],
                yerr=sub["sem"].fillna(0),
                capsize=3,
                linewidth=1.5,
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
            )
    for column, label, linestyle in [
        ("uniform_baseline_error", "Uniform random", ":"),
        ("marginal_baseline_error", "Holdout-marginal", "--"),
    ]:
        value = baselines.loc["mean", column]
        if pd.notna(value):
            ax.axhline(
                value,
                color="#999999",
                linestyle=linestyle,
                linewidth=1.2,
                label=f"{label} control ({value:.2f})",
            )
    ax.set_xscale("log")
    ax.set_xticks(sorted(df["m"].unique()))
    ax.set_xticklabels([str(m) for m in sorted(df["m"].unique())])
    ax.set_xlabel("Number of released trees (m)")
    ax.set_ylabel("Reconstruction error on T = sex\n(mean ± SEM over available seeds)")
    ax.set_title("No-known-Y reconstruction by background knowledge", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = output_dir / "reconstruction_error_vs_m.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    args = analysis_arguments()
    run_analysis(
        lambda: main(args.input, args.output_dir),
        args.output_dir / "analyze_results_summary.txt",
    )
