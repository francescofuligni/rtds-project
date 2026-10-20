"""Read and write experiment CSVs with shared protocol metadata."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import ortools
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attack.metrics import MATCHING_POLICY, TIE_POLICY
from attack.reconstruct import PROTOCOL_VERSION, RELEASE_POLICY, PublicRelease

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DIAGNOSTIC_KIND = "truth_aligned_solver_agreement_diagnostic"


def metadata(
    seed: int, m: int, condition: str, release: PublicRelease, timeout: float
) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "release_policy": RELEASE_POLICY,
        "matching_policy": MATCHING_POLICY,
        "tie_policy": TIE_POLICY,
        "scipy_version": scipy.__version__,
        "ortools_version": ortools.__version__,
        "seed": seed,
        "m": m,
        "condition": condition,
        "release_id": release.release_id,
        "n_rows": release.n_rows,
        "timeout_s": timeout,
        "solver_workers": 1,
    }


def write_rows(rows: list[dict], path: Path) -> None:
    """Overwrite the requested CSV with sorted experiment results."""
    if not rows:
        raise ValueError("no results: none of the requested releases was available")
    rows.sort(
        key=lambda row: (row["seed"], row["m"], row["condition"], row.get("row", 0))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analysis_arguments(*, diagnostic: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze no-known-Y results.")
    name = "calibration_experiment.csv" if diagnostic else "final_experiment.csv"
    parser.add_argument("--input", nargs="+", type=Path, default=[RESULTS_DIR / name])
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def load_results(paths: list[Path], *, diagnostic: bool = False) -> pd.DataFrame:
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if data.empty or "protocol_version" not in data:
        raise ValueError(
            "input is empty or from the old known-Y protocol; rerun the experiment"
        )
    for key, expected in [
        ("protocol_version", PROTOCOL_VERSION),
        ("release_policy", RELEASE_POLICY),
        ("matching_policy", MATCHING_POLICY),
        ("tie_policy", TIE_POLICY),
    ]:
        if key not in data or not data[key].eq(expected).all():
            raise ValueError(f"mixed or unsupported protocol: {key}")
    for key in ["scipy_version", "ortools_version"]:
        if key not in data or data[key].isna().any() or data[key].nunique() != 1:
            raise ValueError(f"mixed or missing library version: {key}")
    keys = ["seed", "m", "condition"]
    if diagnostic:
        if (
            "confidence_kind" not in data
            or not data["confidence_kind"].eq(DIAGNOSTIC_KIND).all()
        ):
            raise ValueError("expected truth-aligned diagnostic scores")
        keys.append("row")
    if data.duplicated(keys).any():
        raise ValueError("duplicate results across inputs")
    return data


def print_availability(data: pd.DataFrame) -> None:
    print("Available results by release size and condition:")
    print(data.groupby(["m", "condition"]).size().rename("rows").to_string())


def diagnostic_scores(data: pd.DataFrame) -> pd.DataFrame:
    print("DIAGNOSTIC ONLY: agreement is aligned using evaluator truth.")
    scores = data.dropna(subset=["agreement", "correct"]).copy()
    if scores.empty:
        raise ValueError("no diagnostic scores available")
    values = (
        scores["correct"]
        .astype(str)
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1.0": 1.0, "0.0": 0.0})
    )
    if values.isna().any():
        raise ValueError("invalid correctness values")
    scores["correct"] = values
    print(data.groupby(["m", "condition"])["n_solved"].agg(["min", "max"]).to_string())
    return scores
