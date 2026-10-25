"""Plot meeting attack figures from saved results, without running any solves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.attack_figures import aggregate, load_accuracy, paired_selection
from output_paths import FIGURE_DATA_ROOT, FIGURES_ROOT, require_saved_result_dir
from provenance import load_results, require_comparable
from scipy.interpolate import PchipInterpolator

from plotting.style import SELECTION_STYLES, STYLES


def draw(
    stats: pd.DataFrame,
    labels: dict[str, str],
    stem: Path,
    styles: dict | None = None,
    accuracy: pd.DataFrame | None = None,
    accuracy_label: str = "Accuracy",
    uncertainty: bool = False,
    height: float = 4.9,
    gap: tuple[str, str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, height))
    curves = {}
    for condition, (color, marker, _) in (styles or STYLES).items():
        sub = (
            stats[stats.condition == condition].dropna(subset=["mean"]).sort_values("m")
        )
        if sub.empty:
            raise ValueError(f"Missing condition {condition}")
        x, mean, sem = (sub[col].to_numpy() for col in ["m", "mean", "sem"])
        curves[condition] = sub.set_index("m")["mean"]
        # 95% confidence band of the mean; markers sit on the measured tree counts.
        if uncertainty:
            ax.fill_between(
                x, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.15, lw=0
            )
        # A gap figure emphasises its two conditions and keeps the rest as context.
        context = gap is not None and condition not in gap
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=1.1 if context else 2.4 if gap else 1.8,
            alpha=0.55 if context else 1.0,
            zorder=2 if context else 3,
            marker=marker if uncertainty else None,
            markersize=4.5,
            label=labels[condition],
        )
    if gap:
        pair = pd.concat([curves[gap[0]], curves[gap[1]]], axis=1, join="inner")
        ax.fill_between(
            pair.index,
            pair.iloc[:, 0],
            pair.iloc[:, 1],
            color=(styles or STYLES)[gap[1]][0],
            alpha=0.18,
            lw=0,
            label="Gain from both",
        )
    grid = sorted(stats.m.unique())
    # Round 1-2-5 ticks keep the log axis readable for any measured grid.
    decades = 10.0 ** np.arange(np.floor(np.log10(grid[0])), np.log10(grid[-1]) + 1)
    ticks = [
        t for t in np.outer(decades, [1, 2, 5]).ravel() if grid[0] <= t <= grid[-1]
    ]
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.tick_params(axis="x", which="major", length=3, direction="out")
    ax.set_xlabel("Released trees (m)", fontsize=12)
    ax.set_ylabel("Sex reconstruction error ↓", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = ax.get_legend_handles_labels()
    if accuracy is not None:
        utility = (
            accuracy.groupby("m").accuracy.agg(["mean", "std", "count"]).reset_index()
        )
        right = ax.twinx()
        x = utility.m.to_numpy()
        mean = utility["mean"].to_numpy()
        dense = np.geomspace(x.min(), x.max(), 400) if len(x) > 1 else x
        center = (
            PchipInterpolator(np.log(x), mean)(np.log(dense)) if len(x) > 1 else mean
        )
        right.plot(dense, center, color="#CC79A7", linewidth=1.2, label=accuracy_label)
        right.set_ylabel("Income accuracy ↑", color="#a6568e")
        right.set_ylim(0, 1)
        other, other_labels = right.get_legend_handles_labels()
        handles += other
        legend_labels += other_labels
    ax.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=len(handles),
        frameon=False,
        # The gap legend has a fifth entry, so it packs tighter to stay on one row.
        **(
            {"fontsize": 10.5, "handlelength": 1.6, "columnspacing": 1.2}
            if gap
            else {"fontsize": 11}
        ),
    )
    # Margins in inches keep the plot area the same size at any figure height.
    top, plot, footnote = 0.147, 3.332, 0.1225
    fig.text(
        0.5,
        footnote / height,
        f"Mean{' ± 95% CI' if uncertainty else ''} · "
        f"n={int(stats['count'].min())}–{int(stats['count'].max())} per point",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(
        bottom=1 - (top + plot) / height,
        left=0.12,
        right=0.87 if accuracy is not None else 0.98,
        top=1 - top / height,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--education-dir", type=Path, required=True)
    parser.add_argument("--age-dir", type=Path, required=True)
    parser.add_argument("--sequential-dir", type=Path)
    parser.add_argument("--education-accuracy-dir", type=Path)
    parser.add_argument("--age-accuracy-dir", type=Path)
    args = parser.parse_args()
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".render-", dir=FIGURES_ROOT) as temporary:
        output_dir = Path(temporary)
        data_dir = output_dir / "data"
        render(args, output_dir, data_dir)
        FIGURE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
        for path in sorted(output_dir.glob("*.png")):
            path.replace(FIGURES_ROOT / path.name)
        for path in sorted(data_dir.iterdir()):
            path.replace(FIGURE_DATA_ROOT / path.name)
    print(f"Published figures to {FIGURES_ROOT}; aggregates to {FIGURE_DATA_ROOT}")


def render(args, output_dir: Path, data_dir: Path) -> None:
    """Validate every archive before rendering into a temporary publication folder."""
    renders = []
    tables = []
    input_dirs = [args.education_dir, args.age_dir]
    if args.sequential_dir:
        input_dirs.append(args.sequential_dir)
    input_dirs.extend(
        path
        for path in (args.education_accuracy_dir, args.age_accuracy_dir)
        if path is not None
    )
    for path in input_dirs:
        require_saved_result_dir(path)
    loaded = [
        load_results(path, "final_experiment")
        for path in (args.education_dir, args.age_dir)
    ]
    require_comparable([manifest for _, manifest in loaded], may_differ=["attr_b"])
    data_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "bands": "selection comparison only: 95% CI of the mean (mean ± 1.96 SEM)",
        "gap_figures": "attack_marital_*_gap: same aggregates; shading spans None to A+B",
        "curves": "reconstruction: straight segments; accuracy: PCHIP in log(m); no extrapolation",
        "metric": "categorical target error after evaluator row matching",
        "implementation_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "analysis" / "attack_figures.py",
                Path(__file__).resolve().parents[1] / "output_paths.py",
                Path(__file__).resolve().parents[1] / "provenance.py",
                Path(__file__).resolve().parent / "style.py",
                Path(__file__).resolve().parents[2] / "pyproject.toml",
                Path(__file__).resolve().parents[2] / "uv.lock",
            )
        },
        "sources": [],
    }
    for name, path, (df, manifest) in zip(
        ("education", "age"), (args.education_dir, args.age_dir), loaded, strict=True
    ):
        protocol = manifest["protocol"]
        if (protocol["attr_a"], protocol["attr_b"]) != ("marital-status", name):
            raise ValueError(f"Unexpected A/B roles for {name}")
        if protocol["tree_selection"] != "random":
            raise ValueError("These background figures expect random releases")
        stats = aggregate(df)
        stem = output_dir / f"attack_marital_{name}"
        labels = {key: style[2] for key, style in STYLES.items()}
        labels.update(
            {
                "none": "None",
                "A": "Marital",
                "B": name.capitalize(),
                "A+B": f"Marital + {name}",
            }
        )
        accuracy_path = getattr(args, f"{name}_accuracy_dir")
        accuracy, accuracy_label = (
            load_accuracy(accuracy_path, manifest, df)
            if accuracy_path
            else (None, "Predictive accuracy on Y")
        )
        renders.append(
            (stats, labels, stem, None, accuracy, accuracy_label, {"height": 4.7})
        )
        # Same aggregates, with the None to A+B gap shaded as the effect to read.
        renders.append(
            (
                stats,
                labels,
                stem.with_name(f"{stem.name}_gap"),
                None,
                None,
                accuracy_label,
                {"height": 4.7, "gap": ("none", "A+B")},
            )
        )
        data_path = data_dir / f"{stem.name}.csv"
        tables.append((stats, data_path))
        files = sorted(path.glob(manifest["data"])) + [path / "manifest.json"]
        accuracy_files = (
            [accuracy_path / "accuracy.csv", accuracy_path / "manifest.json"]
            if accuracy_path
            else []
        )
        provenance["sources"].append(
            {
                "directory": str(path.resolve()),
                "accuracy_directory": str(accuracy_path.resolve())
                if accuracy_path
                else None,
                "accuracy_sha256": {
                    str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in accuracy_files
                },
                "protocol": protocol,
                "seed_count": int(df.seed.nunique()),
                "missing_errors": df.loc[
                    df.reconstruction_error.isna(), ["seed", "m", "condition", "status"]
                ].to_dict("records"),
                "sha256": {
                    str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
            }
        )
    if args.sequential_dir:
        sequential, seq_manifest = load_results(args.sequential_dir, "final_experiment")
        random, random_manifest = loaded[0]
        if seq_manifest["protocol"]["tree_selection"] not in ("first", "sequential"):
            raise ValueError("Expected sequential tree selection")
        ignored = [
            "tree_selection",
            "tree_selection_order",
            "tree_selection_pool",
            "tree_selection_seed_offset",
            "m_values",
            "m_schedule",
            "m_min",
            "m_max",
            "k",
            "k_effective",
            "k_steps",
            "m_rounding",
            "include_all_trees",
        ]
        require_comparable([random_manifest, seq_manifest], may_differ=ignored)
        paired = paired_selection(random, sequential)
        comparison = aggregate(paired)
        styles = SELECTION_STYLES
        labels = {k: v[2] for k, v in styles.items()}
        stem = output_dir / "attack_selection_comparison"
        renders.append(
            (
                comparison,
                labels,
                stem,
                styles,
                None,
                "Predictive accuracy on Y",
                {"uncertainty": True},
            )
        )
        data_path = data_dir / f"{stem.name}.csv"
        tables.append((comparison, data_path))
        provenance["selection_comparison"] = {
            "sequential_directory": str(args.sequential_dir.resolve()),
            "sequential_protocol": seq_manifest["protocol"],
            "paired_keys": paired[["seed", "m"]].drop_duplicates().to_dict("records"),
            "maximum_m": int(paired.m.max()),
            "no_extrapolation": True,
            "source_sha256": {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.sequential_dir.glob(seq_manifest["data"]))
                + [args.sequential_dir / "manifest.json"]
            },
        }
    for stats, *_ in renders:
        if stats["mean"].notna().sum() == 0:
            raise ValueError("No valid reconstruction outcomes to plot")
    for stats, labels, stem, styles, accuracy, accuracy_label, options in renders:
        draw(
            stats,
            labels,
            stem,
            styles=styles or STYLES,
            accuracy=accuracy,
            accuracy_label=accuracy_label,
            **options,
        )
    for table, path in tables:
        table.to_csv(path, index=False)
    (data_dir / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    cli()
