"""Stamp every result set with the protocol that produced it.

Each timestamped experiment directory contains a manifest.json:

  experiment  which experiment wrote it
  protocol    every setting that changes what the numbers mean: tree
              selection, Rashomon parameters, sample sizes, attacker
              knowledge, ... Result sets are comparable only if these match.
  data        glob (relative to the directory) matching the data files
  runs        one entry per invocation: git commit, whether the code had
              uncommitted changes, command, start time, host, package versions

A run refuses to write into a directory whose protocol differs from its
own, and analyses load data only through the manifest and print the
protocol at the top of their summaries, so results produced under
different settings are never pooled by accident. Changing a setting means
writing to a new directory.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "manifest.json"
TRACKED_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "ortools", "treefarms"]


class ProtocolMismatch(RuntimeError):
    """A result directory holds results produced under different settings."""


def available_cpu_count() -> int:
    """Return CPUs available to this process, with a portable fallback."""
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            return len(get_affinity(0))
        except OSError:
            pass
    return os.cpu_count() or 1


def _normalise(protocol: dict) -> dict:
    return json.loads(json.dumps(protocol, sort_keys=True))


def _diff(old: dict, new: dict) -> str:
    keys = sorted(set(old) | set(new))
    return ", ".join(
        f"{k}: {old.get(k)!r} -> {new.get(k)!r}"
        for k in keys
        if old.get(k) != new.get(k)
    )


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def run_info(argv: list[str] | None = None) -> dict:
    """How this invocation was produced."""
    dirty = _git("status", "--porcelain", "--", "src", "experiments")
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "uncommitted_code_changes": None if dirty is None else bool(dirty),
        "command": [Path(sys.argv[0]).name, *(sys.argv[1:] if argv is None else argv)],
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "packages": versions,
    }


def _write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def stamp(
    result_dir: Path,
    experiment: str,
    protocol: dict,
    data: str,
    argv: list[str] | None = None,
) -> dict:
    """Create result_dir's manifest, or add this run to it if the protocol
    matches; raise ProtocolMismatch if it does not."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / MANIFEST
    protocol = _normalise(protocol)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["experiment"] != experiment or manifest["data"] != data:
            raise ProtocolMismatch(
                f"{result_dir} holds {manifest['experiment']} results, not {experiment}"
            )
        if manifest["protocol"] != protocol:
            raise ProtocolMismatch(
                f"{result_dir} already holds results under a different protocol "
                f"({_diff(manifest['protocol'], protocol)}); write to a new directory instead"
            )
        manifest["runs"].append(run_info(argv))
    elif any(result_dir.iterdir()):
        # Resuming would treat these files as finished seeds of this protocol.
        raise ProtocolMismatch(
            f"{result_dir} already contains unstamped files; move them away first"
        )
    else:
        manifest = {
            "experiment": experiment,
            "protocol": protocol,
            "data": data,
            "runs": [run_info(argv)],
        }
    _write_manifest(path, manifest)
    return manifest


def read_manifest(result_dir: Path, experiment: str) -> dict:
    path = Path(result_dir) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"{result_dir} has no {MANIFEST}; unstamped results cannot be analysed"
        )
    manifest = json.loads(path.read_text())
    if manifest["experiment"] != experiment:
        raise ProtocolMismatch(
            f"{result_dir} holds {manifest['experiment']} results, not {experiment}"
        )
    return manifest


def load_results(result_dir: Path, experiment: str) -> tuple[pd.DataFrame, dict]:
    """All data files of a stamped result set, concatenated, plus its manifest."""
    result_dir = Path(result_dir)
    manifest = read_manifest(result_dir, experiment)
    files = sorted(result_dir.glob(manifest["data"]))
    if not files:
        raise FileNotFoundError(
            f"no data files matching {manifest['data']!r} in {result_dir}"
        )
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True), manifest


def require_comparable(manifests: list[dict], may_differ: Iterable[str] = ()) -> None:
    """Refuse to compare result sets whose protocols differ beyond `may_differ`."""
    may_differ = set(may_differ)
    reference = {
        k: v for k, v in manifests[0]["protocol"].items() if k not in may_differ
    }
    for other in manifests[1:]:
        candidate = {k: v for k, v in other["protocol"].items() if k not in may_differ}
        if candidate != reference:
            raise ProtocolMismatch(
                f"result sets are not comparable: {_diff(reference, candidate)}"
            )


def describe(manifest: dict) -> str:
    """Protocol header for the top of an analysis summary."""
    lines = [f"Result set: {manifest['experiment']}"]
    lines += [
        f"  {key} = {value}" for key, value in sorted(manifest["protocol"].items())
    ]
    if manifest["runs"]:
        commits = sorted(
            {run.get("git_commit") or "unknown" for run in manifest["runs"]}
        )
        lines.append(
            f"  produced by {len(manifest['runs'])} run(s) at commit(s) {', '.join(c[:8] for c in commits)}"
        )
    else:
        lines.append("  no recorded runs (stamped after the fact)")
    if any(run.get("uncommitted_code_changes") for run in manifest["runs"]):
        lines.append("  WARNING: at least one run had uncommitted code changes")
    if manifest.get("notes"):
        lines.append(f"  notes: {manifest['notes']}")
    return "\n".join(lines)


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write a CSV atomically, so an interrupted run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def seed_path(result_dir: Path, seed: int) -> Path:
    return Path(result_dir) / f"seed{seed:03d}.csv"


def run_seeds(
    result_dir: Path,
    seeds: Iterable[int],
    run_seed: Callable[[int], list[dict]],
    *,
    workers: int,
) -> None:
    """Run `run_seed` for every seed without a result file yet, in parallel,
    writing one CSV per seed (a seed with no rows gets a .skip marker).
    Rerunning the same command resumes where an interrupted run stopped.
    Worker exceptions propagate and abort the runner.
    """
    seeds = list(seeds)
    todo = [
        s
        for s in seeds
        if not (
            seed_path(result_dir, s).exists()
            or seed_path(result_dir, s).with_suffix(".skip").exists()
        )
    ]
    print(
        f"{len(seeds) - len(todo)} of {len(seeds)} seeds already done in {result_dir}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_seed, seed): seed for seed in todo}
        for future in as_completed(futures):
            seed = futures[future]
            rows = future.result()
            if rows:
                write_rows(seed_path(result_dir, seed), rows)
                print(f"[seed {seed}] done: {len(rows)} rows", flush=True)
            else:
                seed_path(result_dir, seed).with_suffix(".skip").write_text(
                    "no rows (Rashomon set too small)\n"
                )
                print(f"[seed {seed}] skipped: no rows", flush=True)
