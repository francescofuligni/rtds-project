"""Resolve timestamped experiment result directories."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"
TIMESTAMP_PATTERN = re.compile(r"\d{8}_\d{6}(?:_\d+)?\Z")


def new_timestamp(results_root: Path | None = None) -> str:
    """Return a Stockholm-local timestamp not already used under results."""
    results_root = RESULTS_ROOT if results_root is None else results_root
    base = datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y%m%d_%H%M%S")
    timestamp = base
    suffix = 1
    while (results_root / timestamp).exists():
        timestamp = f"{base}_{suffix:02d}"
        suffix += 1
    return timestamp


def experiment_result_dir(experiment: str, timestamp: str | None = None) -> Path:
    """Return and create one experiment directory in a shared run folder."""
    run_timestamp = timestamp or os.environ.get("RTDS_RUN_TIMESTAMP") or new_timestamp()
    if not TIMESTAMP_PATTERN.fullmatch(run_timestamp):
        raise ValueError(f"Invalid RTDS run timestamp: {run_timestamp!r}")
    path = RESULTS_ROOT / run_timestamp / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_result_dir(experiment: str, results_root: Path = RESULTS_ROOT) -> Path:
    """Find the newest timestamped result directory for an experiment."""
    candidates = []
    if results_root.is_dir():
        for timestamp_dir in sorted(results_root.iterdir()):
            result_dir = timestamp_dir / experiment
            manifest_path = result_dir / "manifest.json"
            if not TIMESTAMP_PATTERN.fullmatch(timestamp_dir.name):
                continue
            if not timestamp_dir.is_dir() or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("experiment") != experiment:
                continue
            if any(result_dir.glob(manifest["data"])):
                candidates.append(result_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No timestamped {experiment!r} results found in {results_root}"
        )
    return candidates[-1]


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("usage: python experiments/output_paths.py")
    print(new_timestamp())
