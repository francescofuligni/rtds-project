"""Final experiment: the None/A/B/A+B background-knowledge ablation on
Adult, repeated across seeds and swept over the number of released trees m.

For each seed: sample a fresh n=100 pilot (disjoint from a holdout used
only for the baselines), build one Rashomon set, and for each m in
M_VALUES select the first m trees and run all four conditions against
the *same* sample and *same* trees -- this is the "paired" part: within
a repetition, condition differences are directly comparable because
everything else is held fixed (meeting notes Sec. 3/6).

Usage: run from the repository root with the project venv, e.g.
    .venv/bin/python experiments/run_final_experiment.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attack.baselines import marginal_baseline, uniform_random_guess
from attack.conditions import CONDITIONS, run_condition
from attack.reconstruct import extract_branches
from rashomon.build import build_rashomon_set, select_trees
from rashomon.data import binarize_adult_curated, known_columns_for, load_adult_raw, sample_rows

TARGET, ATTR_A, ATTR_B = "sex", "marital-status", "education"
N_SEEDS = 10
M_VALUES = [1, 5, 10, 25, 50]
PILOT_N = 100
HOLDOUT_N = 2000
REGULARIZATION = 0.02
RASHOMON_BOUND_MULTIPLIER = 0.2
SOLVE_TIMEOUT = 60.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_adult_raw()
    data_full = binarize_adult_curated(df)

    rows = []
    for seed in range(N_SEEDS):
        pilot_idx = data_full.X.sample(n=PILOT_N, random_state=seed).index
        pilot = sample_rows(data_full, n=PILOT_N, seed=seed)
        holdout_df = data_full.X.drop(index=pilot_idx).sample(n=HOLDOUT_N, random_state=seed + 10_000)

        t0 = time.time()
        model = build_rashomon_set(pilot.X, pilot.y, regularization=REGULARIZATION,
                                    rashomon_bound_multiplier=RASHOMON_BOUND_MULTIPLIER)
        n_available = model.get_tree_count()
        build_time = time.time() - t0
        print(f"[seed {seed}] Rashomon set: {n_available} trees in {build_time:.1f}s", flush=True)

        target_idx = [pilot.X.columns.get_loc(c) for c in known_columns_for(pilot, TARGET)]
        X = pilot.X.to_numpy()
        uniform = uniform_random_guess(PILOT_N, target_idx, pilot.X.shape[1], seed=seed)
        marginal = marginal_baseline(pilot, TARGET, PILOT_N, holdout_df, seed=seed)
        uniform_error = float((uniform[:, target_idx] != X[:, target_idx]).any(axis=1).mean())
        marginal_error = float((marginal[:, target_idx] != X[:, target_idx]).any(axis=1).mean())

        for m in M_VALUES:
            if m > n_available:
                print(f"[seed {seed}] skipping m={m}: only {n_available} trees available", flush=True)
                continue

            trees = select_trees(model, m)
            y = pilot.y.to_numpy()
            classes = sorted(pilot.y.unique().tolist())
            trees_branches = [extract_branches(t, X, y, classes) for t in trees]

            for condition in CONDITIONS:
                t0 = time.time()
                result = run_condition(condition, trees_branches, pilot, ATTR_A, ATTR_B, TARGET,
                                        timeout=SOLVE_TIMEOUT, seed=seed)
                duration = time.time() - t0
                rows.append({
                    "seed": seed, "m": m, "n_trees_available": n_available,
                    "condition": condition, "status": result.status,
                    "duration_s": round(duration, 3),
                    "reconstruction_error": result.reconstruction_error,
                    "uniform_baseline_error": uniform_error,
                    "marginal_baseline_error": marginal_error,
                })
                print(f"[seed {seed}] m={m:3d} {condition:6s} status={result.status:10s} "
                      f"error={result.reconstruction_error} time={duration:.1f}s", flush=True)

    out_path = RESULTS_DIR / "final_experiment.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
