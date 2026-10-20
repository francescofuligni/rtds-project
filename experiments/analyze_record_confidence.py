"""Score the record-level confidence attackers from run_record_confidence.py.

Every attacker outputs a 100-row table with a confidence per row; nothing
is fitted, so a score is calibrated "out of the box" only if, among rows
it gives confidence c, about a fraction c are correct. Reports per m and
attacker: table accuracy (1 - global error), mean confidence, the gap
between the two, ECE, Brier score and AUROC (does confidence separate the
correct rows from the wrong ones), plus paired per-seed tests of
prior-guided sampling against the no-trees baseline and of each
prior-guided variant against the plain sampler, and selective accuracy (how
right are the rows an attacker keeps when it keeps only its most confident
ones). Several result sets may differ in one compared setting -- tree
selection (sets stamped before random selection became the only option),
public sample size or records vs. marginals -- and are then reported per
setting and compared with the first set, paired by seed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_output import run_analysis
from output_paths import latest_result_dir
from provenance import describe, load_results, require_comparable
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

EXPERIMENT = "record_confidence"
N_BINS = 10
MIN_BIN_ROWS = 20  # bins with fewer rows are too noisy to plot
SELECTIVE_THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]
TOP_K = [10, 25, 50]
# Protocol settings result sets may differ in when compared; older sets lack the
# public-data keys and used the defaults below.
COMPARED_SETTINGS = {"tree_selection", "public_n", "public_prior"}
DEFAULT_SETTINGS = {
    "tree_selection": "random",
    "public_n": 2000,
    "public_prior": "records",
}

# Display order; other attackers (e.g. prior-guided variants) follow alphabetically.
METHOD_ORDER = [
    "prior_guided",
    "prior_only",
    "per_tree_occurrence",
    "per_tree_presence",
    "seed_only",
    "per_tree_absolute",
]
NON_PROBABILITY_METHODS = {"per_tree_absolute"}  # a count: no calibration metrics
PLOTTED = {  # fixed categorical order; marker shape is the non-color identity cue
    "prior_guided": ("Prior-guided sampling", "#2a78d6", "o"),
    "prior_only": ("Public statistics only (no trees)", "#eb6834", "s"),
    "per_tree_occurrence": ("Per-tree counting (per copy)", "#1baf7a", "^"),
    "seed_only": ("Joint solves, solver seed only", "#eda100", "D"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    return float(
        sum(
            abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
            for b in range(n_bins)
            if (bins == b).any()
        )
    )


def auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    if correct.min() == correct.max() or conf.min() == conf.max():
        return float("nan")
    return float(roc_auc_score(correct, conf))


def setting_labels(protocols: list[dict]) -> list[str]:
    """Name each result set by the compared settings in which the sets differ
    (the tree selection alone keeps its plain name, e.g. 'random')."""
    keys = [
        k
        for k in sorted(COMPARED_SETTINGS)
        if len({json.dumps(p.get(k, DEFAULT_SETTINGS.get(k))) for p in protocols}) > 1
    ]
    if not keys:
        return [str(protocols[0].get("tree_selection", "random"))] * len(protocols)
    if keys == ["tree_selection"]:
        return [str(p["tree_selection"]) for p in protocols]
    return [
        ", ".join(f"{k}={p.get(k, DEFAULT_SETTINGS.get(k))}" for k in keys)
        for p in protocols
    ]


def per_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, m, method, seed), g in df.groupby(["setting", "m", "method", "seed"]):
        conf, ok = g["confidence"].to_numpy(), g["correct"].to_numpy().astype(float)
        is_prob = method not in NON_PROBABILITY_METHODS
        rows.append(
            {
                "setting": setting,
                "m": m,
                "method": method,
                "seed": seed,
                "accuracy": ok.mean(),
                "auroc": auroc(conf, ok),
                "brier": float(np.mean((conf - ok) ** 2)) if is_prob else np.nan,
                # per-seed calibration error: |mean stated confidence - share correct|
                "abs_gap": abs(conf.mean() - ok.mean()) if is_prob else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, m, method), g in df.groupby(["setting", "m", "method"]):
        conf, ok = g["confidence"].to_numpy(), g["correct"].to_numpy().astype(float)
        s = seeds[
            (seeds["setting"] == setting)
            & (seeds["m"] == m)
            & (seeds["method"] == method)
        ]
        is_prob = method not in NON_PROBABILITY_METHODS
        rows.append(
            {
                "setting": setting,
                "m": m,
                "method": method,
                "seeds": g["seed"].nunique(),
                "accuracy": ok.mean(),
                "acc_sem": s["accuracy"].sem(),
                "mean_conf": conf.mean() if is_prob else np.nan,
                "gap": conf.mean() - ok.mean() if is_prob else np.nan,
                "ece": ece(conf, ok) if is_prob else np.nan,
                "brier": float(np.mean((conf - ok) ** 2)) if is_prob else np.nan,
                "auroc_pooled": auroc(conf, ok),
                "auroc_seed_mean": s["auroc"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    order = [m for m in METHOD_ORDER if m in set(out["method"])] + sorted(
        set(out["method"]) - set(METHOD_ORDER)
    )
    out["method"] = pd.Categorical(out["method"], order, ordered=True)
    return out.sort_values(["setting", "m", "method"])


def paired_tests(
    seeds: pd.DataFrame, a: str = "prior_guided", b: str = "prior_only"
) -> None:
    if not {a, b} <= set(seeds["method"]):
        return
    print(
        f"\nPaired per-seed comparison, {a} minus {b} (Wilcoxon signed-rank; abs_gap = calibration error):"
    )
    for (setting, m), g in seeds.groupby(["setting", "m"]):
        wide = g.pivot(index="seed", columns="method")
        for metric in ["accuracy", "auroc", "brier", "abs_gap"]:
            diff = (wide[(metric, a)] - wide[(metric, b)]).dropna()
            p = (
                wilcoxon(diff).pvalue
                if len(diff) >= 5 and diff.abs().sum() > 0
                else float("nan")
            )
            print(
                f"  {setting:<6} m={m:<4} {metric:<9} mean diff {diff.mean():+.4f} (n={len(diff)}, "
                f"{(diff > 0).sum()} seeds up, {(diff < 0).sum()} down) p={p:.3g}"
            )


def selective_accuracy(
    df: pd.DataFrame,
    seeds: pd.DataFrame,
    a: str = "prior_guided",
    b: str = "prior_only",
    threshold: float = 0.9,
) -> None:
    """Local error in practice: if the attacker keeps only the rows it is
    most confident about, how often are those right, and how many are left?"""
    print(
        "\nSelective accuracy: keep rows with confidence >= t -> share correct / share of the table kept"
    )
    for (setting, m), g in df.groupby(["setting", "m"]):
        print(f"  {setting} m={m}")
        for method, h in g.groupby("method"):
            if method in NON_PROBABILITY_METHODS:
                continue
            cells = []
            for t in SELECTIVE_THRESHOLDS:
                kept = h[h["confidence"] >= t]
                cells.append(
                    f">={t:.2f} {kept['correct'].mean():.2f}/{len(kept) / len(h):3.0%}"
                    if len(kept)
                    else f">={t:.2f}  -  /  0%"
                )
            ranked = h.sort_values(["seed", "confidence"], ascending=[True, False])
            tops = [
                f"top{k} {ranked.groupby('seed').head(k)['correct'].mean():.2f}"
                for k in TOP_K
            ]
            print(
                f"    {method:22s} "
                + " | ".join(cells)
                + "   most confident: "
                + ", ".join(tops)
            )
    print(
        f"\nRows an attacker can vouch for at confidence >= {threshold} (per seed), {a} vs {b} (Wilcoxon):"
    )
    per_seed = (
        df[df["method"].isin([a, b])]
        .assign(kept=lambda d: d["confidence"] >= threshold)
        .groupby(["setting", "m", "method", "seed"])
        .agg(
            coverage=("kept", "mean"),
            kept_correct=(
                "correct",
                lambda c: c[df.loc[c.index, "confidence"] >= threshold].mean(),
            ),
        )
        .reset_index()
    )
    for (setting, m), g in per_seed.groupby(["setting", "m"]):
        wide = g.pivot(index="seed", columns="method", values="coverage").dropna()
        if {a, b} <= set(wide.columns) and len(wide) >= 5:
            diff = wide[a] - wide[b]
            p = wilcoxon(diff).pvalue if diff.abs().sum() > 0 else float("nan")
            acc = g.groupby("method")["kept_correct"].mean()
            print(
                f"  {setting:<6} m={m:<4} coverage {a} {wide[a].mean():.0%} (correct {acc.get(a, np.nan):.2f}) vs "
                f"{b} {wide[b].mean():.0%} (correct {acc.get(b, np.nan):.2f}); diff {diff.mean():+.0%} p={p:.3g}"
            )


def selective_accuracy_figure(df: pd.DataFrame, out_dir: Path) -> Path:
    """Accuracy of the rows an attacker keeps when it keeps its most confident
    share of the table, as that share grows (averaged over seeds)."""
    m_values = sorted(df["m"].unique())
    settings = sorted(df["setting"].unique())
    shares = np.linspace(0.05, 1.0, 20)
    fig, axes = plt.subplots(
        len(settings),
        len(m_values),
        figsize=(4.2 * len(m_values), 4.0 * len(settings)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for (row, setting), (col, m) in (
        (r, c) for r in enumerate(settings) for c in enumerate(m_values)
    ):
        ax = axes[row][col]
        for method, (label, color, marker) in PLOTTED.items():
            g = df[
                (df["setting"] == setting) & (df["m"] == m) & (df["method"] == method)
            ]
            if g.empty:
                continue
            ranked = g.sort_values(
                ["seed", "confidence", "record", "k"],
                ascending=[True, False, True, True],
            )
            curve = []
            for share in shares:
                kept = ranked.groupby("seed", group_keys=False).apply(
                    lambda s: s.head(max(1, int(round(share * len(s))))),
                    include_groups=False,
                )
                curve.append(kept["correct"].mean())
            ax.plot(
                shares,
                curve,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=2,
                markevery=3,
                markeredgecolor="white",
                markeredgewidth=1,
                label=label,
            )
        title = f"m = {m} trees" + (f", {setting}" if len(settings) > 1 else "")
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(
            "Share of its table the attacker keeps (most confident first)",
            color=MUTED,
            fontsize=9,
        )
        ax.grid(color=GRID, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for row in axes:
        row[0].set_ylabel(
            "Share of kept rows that are correct", color=MUTED, fontsize=9
        )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    fig.tight_layout(rect=(0, 0.1 / len(settings), 1, 1))
    out = out_dir / "record_confidence_selective_accuracy.pdf"
    fig.savefig(out)
    return out


def setting_effect(
    seeds: pd.DataFrame,
    reference: str,
    methods: tuple[str, ...] = ("prior_guided", "prior_only"),
) -> None:
    """Same seeds, same m, result sets that differ in one setting (tree
    setting, public sample size, records vs. marginals): how much does
    each setting change accuracy, calibration error and ranking, relative
    to the reference set?"""
    if seeds["setting"].nunique() < 2:
        return
    print(
        f"\nSetting effects, each setting minus {reference!r} (paired by seed, Wilcoxon):"
    )
    for method in methods:
        g = seeds[seeds["method"] == method]
        for m, sub in g.groupby("m"):
            for setting in sorted(set(sub["setting"]) - {reference}):
                cells = []
                for metric in ["accuracy", "abs_gap", "auroc"]:
                    wide = sub.pivot(index="seed", columns="setting", values=metric)
                    if not {reference, setting} <= set(wide.columns):
                        continue
                    diff = (wide[setting] - wide[reference]).dropna()
                    if len(diff) < 5:
                        continue
                    p = wilcoxon(diff).pvalue if diff.abs().sum() > 0 else float("nan")
                    cells.append(
                        f"{metric} {wide[reference].mean():.3f}->{wide[setting].mean():.3f} "
                        f"({diff.mean():+.3f}, p={p:.2g})"
                    )
                if cells:
                    print(f"  {method:<13} m={m:<4} {setting:<22} " + "  ".join(cells))


def reliability_figure(df: pd.DataFrame, out_dir: Path) -> Path:
    m_values = sorted(df["m"].unique())
    settings = sorted(df["setting"].unique())
    fig, axes = plt.subplots(
        len(settings),
        len(m_values),
        figsize=(4.2 * len(m_values), 4.0 * len(settings)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for (row, setting), (col, m) in (
        (r, c) for r in enumerate(settings) for c in enumerate(m_values)
    ):
        ax = axes[row][col]
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color=MUTED,
            linewidth=1,
            label="Perfectly calibrated",
        )
        for method, (label, color, marker) in PLOTTED.items():
            g = df[
                (df["setting"] == setting) & (df["m"] == m) & (df["method"] == method)
            ]
            if g.empty:
                continue
            bins = np.clip(
                (g["confidence"].to_numpy() * N_BINS).astype(int), 0, N_BINS - 1
            )
            agg = (
                g.assign(bin=bins)
                .groupby("bin")
                .agg(
                    conf=("confidence", "mean"),
                    acc=("correct", "mean"),
                    n=("correct", "size"),
                )
            )
            agg = agg[agg["n"] >= MIN_BIN_ROWS]
            ax.plot(
                agg["conf"],
                agg["acc"],
                color=color,
                marker=marker,
                markersize=6,
                linewidth=2,
                markeredgecolor="white",
                markeredgewidth=1,
                label=label,
            )
        title = f"m = {m} trees" + (f", {setting}" if len(settings) > 1 else "")
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(
            "Stated confidence (no calibration fitted)", color=MUTED, fontsize=9
        )
        ax.grid(color=GRID, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for row in axes:
        row[0].set_ylabel("Share of rows actually correct", color=MUTED, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    fig.tight_layout(rect=(0, 0.1 / len(settings), 1, 1))
    out = out_dir / "record_confidence_reliability.pdf"
    fig.savefig(out)
    return out


def main(result_dirs: list[Path], out_dir: Path) -> None:
    frames, manifests = [], []
    for result_dir in result_dirs:
        data, manifest = load_results(result_dir, EXPERIMENT)
        frames.append(data)
        manifests.append(manifest)
        skipped = sorted(
            str(p.relative_to(result_dir)) for p in result_dir.glob("m*/seed*.skip")
        )
        print(
            f"{describe(manifest)}\n  directory: {result_dir}\n  {len(data)} output rows; "
            f"skipped (Rashomon set < m): {skipped or 'none'}\n"
        )
    # Result sets may differ in these settings (that is what is compared), in nothing else.
    require_comparable(manifests, may_differ=COMPARED_SETTINGS)
    labels = setting_labels([m["protocol"] for m in manifests])
    df = pd.concat(
        [f.assign(setting=label) for f, label in zip(frames, labels)],
        ignore_index=True,
    )

    seeds = per_seed_metrics(df)
    table = summary_table(df, seeds)
    print(
        "\nPer attacker: accuracy = 1 - global error of its 100-row table; gap = mean confidence - accuracy"
    )
    print(
        "(positive gap = overconfident). per_tree_absolute is a count, not a probability."
    )
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))

    paired_tests(seeds)
    # Prior-guided variants (split scoring, random tie-breaks, ...) against the plain sampler.
    for variant in sorted(
        m for m in seeds["method"].unique() if m.startswith("prior_guided_")
    ):
        paired_tests(seeds, variant, "prior_guided")
    selective_accuracy(df, seeds)
    setting_effect(seeds, reference=labels[0])
    print(f"\nWrote {reliability_figure(df, out_dir)}")
    print(f"Wrote {selective_accuracy_figure(df, out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="stamped result sets to compare (default: latest record_confidence run)",
    )
    args = parser.parse_args()
    dirs = args.results_dirs or [latest_result_dir(EXPERIMENT)]
    # One result set: outputs go next to it. Several: into their common parent.
    out = (
        dirs[0]
        if len(dirs) == 1
        else Path(os.path.commonpath([d.resolve() for d in dirs]))
    )
    run_analysis(lambda: main(dirs, out), out / "summary_analyze_record_confidence.txt")
