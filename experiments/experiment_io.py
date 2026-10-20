"""Read and write experiment CSVs with shared protocol metadata."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import ortools
import pandas as pd
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attack.metrics import MATCHING_POLICY, TIE_POLICY
from attack.reconstruct import PROTOCOL_VERSION, RELEASE_POLICY, PublicRelease

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DIAGNOSTIC_KIND = "truth_aligned_solver_agreement_diagnostic"


def experiment_output(filename: str, output: Path | None = None) -> Path:
    """Reserve a timestamped directory under the requested CSV's parent."""
    root = RESULTS_DIR if output is None else output.parent
    filename = filename if output is None else output.name
    stamp = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        directory = root / (stamp if suffix == 0 else f"{stamp}-{suffix}")
        try:
            directory.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return directory / filename


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
    """Write sorted experiment results without replacing an existing CSV."""
    if not rows:
        raise ValueError("no results: none of the requested releases was available")
    rows.sort(
        key=lambda row: (row["seed"], row["m"], row["condition"], row.get("row", 0))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_order(path: Path) -> tuple[float, int, str] | None:
    """Order compact local timestamps and legacy UTC run directories."""
    stamp, _, suffix = path.parent.name.partition("-")
    compact = "_" in stamp
    pattern = "%Y%m%d_%H%M%S" if compact else "%Y%m%dT%H%M%S.%fZ"
    try:
        started = datetime.strptime(stamp, pattern).replace(
            tzinfo=ZoneInfo("Europe/Stockholm") if compact else UTC
        )
        return started.timestamp(), int(suffix or 0), path.name
    except ValueError:
        return None


def analysis_arguments(*, diagnostic: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze no-known-Y results.")
    name = "calibration_experiment.csv" if diagnostic else "final_experiment.csv"
    parser.add_argument("--input", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.input is None:
        names = [name, name.replace(".csv", "_part2.csv")]
        candidates = [
            (order, path)
            for filename in names
            for path in RESULTS_DIR.glob(f"*/{filename}")
            if path.is_file() and (order := _run_order(path)) is not None
        ]
        if not candidates:
            parser.error(
                "no timestamped experiment found; supply --input CSV explicitly"
            )
        args.input = [max(candidates)[1]]
    if args.output_dir is None:
        parents = {path.resolve().parent for path in args.input}
        if len(parents) != 1:
            parser.error("inputs from different runs require an explicit --output-dir")
        args.output_dir = parents.pop()
    print(f"Analysis input: {', '.join(str(path) for path in args.input)}")
    print(f"Analysis output: {args.output_dir}")
    return args


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
