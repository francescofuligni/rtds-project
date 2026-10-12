"""Per-row confidence / calibration experiment.

For a subset of seeds and two representative tree-release sizes (a
sparse m=10 and a rich m=100), re-solve each (seed, m, condition) cell
N_REPEATS times with different solver seeds and record, per true row,
the row-matched agreement rate across repeats (the confidence score) and
whether the majority-vote reconstruction is actually correct. This is
the data a calibration curve (agreement rate vs. empirical accuracy) is
built from in analyze_calibration.py.

Usage: run from the repository root with the project venv, e.g.
    .venv/bin/python experiments/run_calibration_experiment.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attack.confidence import repeated_reconstruct_with_confidence
from attack.conditions import CONDITIONS
from attack.reconstruct import extract_branches
from rashomon.build import build_rashomon_set, select_trees
from rashomon.data import binarize_adult_curated, load_adult_raw, sample_rows

TARGET, ATTR_A, ATTR_B = "sex", "marital-status", "education"
CALIBRATION_SEEDS = 20
M_VALUES = [10, 100]
PILOT_N = 100
REGULARIZATION = 0.01
RASHOMON_BOUND_MULTIPLIER = 0.1
N_REPEATS = 7
SOLVE_TIMEOUT = 60.0
N_WORKERS = min(6, os.cpu_count() or 1)  # lower than the 12 used elsewhere: each worker
# does 2 m-values x 4 conditions x N_REPEATS solves sequentially (more memory held per
# worker over a longer-lived process than the single-solve experiments), so fewer
# concurrent workers trades wall-time for headroom after repeated OOM-adjacent kills

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_seed(seed: int) -> list[dict]:
    df = load_adult_raw()
    data_full = binarize_adult_curated(df)
    pilot = sample_rows(data_full, n=PILOT_N, seed=seed)

    t0 = time.time()
    model = build_rashomon_set(pilot.X, pilot.y, regularization=REGULARIZATION,
                                rashomon_bound_multiplier=RASHOMON_BOUND_MULTIPLIER)
    n_available = model.get_tree_count()
    build_time = time.time() - t0

    rows_out = []
    for m in M_VALUES:
        if m > n_available:
            continue
        trees = select_trees(model, m)
        X = pilot.X.to_numpy()
        y = pilot.y.to_numpy()
        classes = sorted(pilot.y.unique().tolist())
        trees_branches = [extract_branches(t, X, y, classes) for t in trees]

        for condition in CONDITIONS:
            rows, statuses = repeated_reconstruct_with_confidence(
                trees_branches, pilot, ATTR_A, ATTR_B, TARGET, condition,
                n_repeats=N_REPEATS, timeout=SOLVE_TIMEOUT, base_seed=seed * 1000, n_threads=1,
            )
            n_solved = sum(1 for s in statuses if s in ("OPTIMAL", "FEASIBLE"))
            for r in rows:
                rows_out.append({
                    "seed": seed, "m": m, "condition": condition,
                    "row": r.row, "n_votes": r.n_votes, "n_repeats": N_REPEATS,
                    "n_solved": n_solved, "agreement": round(r.agreement, 4),
                    "correct": r.correct,
                })

    print(f"[seed {seed}] done: {n_available} trees (build {build_time:.1f}s), {len(rows_out)} rows", flush=True)
    return rows_out


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    all_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_seed, seed): seed for seed in range(CALIBRATION_SEEDS)}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as e:
                print(f"[seed {seed}] FAILED: {e}", flush=True)

    out_path = RESULTS_DIR / "calibration_experiment.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {out_path} in {time.time() - t_start:.1f}s "
          f"({N_WORKERS} parallel workers, {CALIBRATION_SEEDS} seeds)")


if __name__ == "__main__":
    main()
