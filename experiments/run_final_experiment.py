"""Run paired None/A/B/A+B reconstructions on Adult without known Y."""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment_io import experiment_output, metadata, write_rows

from attack.baselines import marginal_baseline, uniform_random_guess
from attack.conditions import CONDITIONS, run_condition
from attack.metrics import target_error
from attack.reconstruct import _walk_leaves, prepare_release
from rashomon.build import build_rashomon_set, select_trees
from rashomon.data import (
    binarize_adult_curated,
    known_columns_for,
    load_adult_raw,
    sample_rows,
)

TARGET, ATTR_A, ATTR_B = "sex", "marital-status", "education"
N_SEEDS = 200
M_VALUES = [5, 10, 25, 50, 100]
PILOT_N = 100
HOLDOUT_N = 2000
REGULARIZATION = 0.01
RASHOMON_BOUND_MULTIPLIER = 0.1
SOLVE_TIMEOUT = 120.0
N_WORKERS = min(12, os.cpu_count() or 1)


def joint_split_fraction(
    trees: list[dict], target_cols: set[int], other_cols: set[int]
) -> float:
    """Measure the fraction of trees with a path splitting on both groups."""
    if not trees:
        return 0.0
    n_joint = 0
    for t in trees:
        for path, _pred in _walk_leaves(t, []):
            feats = {f for f, _ in path}
            if feats & target_cols and feats & other_cols:
                n_joint += 1
                break
    return n_joint / len(trees)


def run_seed(seed: int) -> list[dict]:
    data_full = binarize_adult_curated(load_adult_raw())
    pilot_idx = data_full.X.sample(n=PILOT_N, random_state=seed).index
    pilot = sample_rows(data_full, n=PILOT_N, seed=seed)
    holdout = data_full.X.drop(index=pilot_idx).sample(
        n=HOLDOUT_N, random_state=seed + 10_000
    )
    model = build_rashomon_set(
        pilot.X,
        pilot.y,
        regularization=REGULARIZATION,
        rashomon_bound_multiplier=RASHOMON_BOUND_MULTIPLIER,
    )
    n_available = model.get_tree_count()
    names = pilot.X.columns.to_list()
    target_cols = [names.index(c) for c in known_columns_for(pilot, TARGET)]
    a_cols = {names.index(c) for c in known_columns_for(pilot, ATTR_A)}
    b_cols = {names.index(c) for c in known_columns_for(pilot, ATTR_B)}
    groups = [[names.index(c) for c in cols] for cols in pilot.one_hot_groups.values()]
    X = pilot.X.to_numpy()
    uniform = uniform_random_guess(PILOT_N, target_cols, len(names), seed=seed)
    marginal = marginal_baseline(pilot, TARGET, PILOT_N, holdout, seed=seed)
    rows = []
    for m in M_VALUES:
        if m > n_available:
            print(f"[seed {seed}] skip m={m}: only {n_available} trees", flush=True)
            continue
        trees = select_trees(model, m)
        release = prepare_release(trees, X, pilot.y.to_numpy(), names, groups)
        joint_a = joint_split_fraction(trees, set(target_cols), a_cols)
        joint_b = joint_split_fraction(trees, set(target_cols), b_cols)
        for condition in CONDITIONS:
            start = time.perf_counter()
            result = run_condition(
                condition,
                release,
                pilot,
                ATTR_A,
                ATTR_B,
                TARGET,
                timeout=SOLVE_TIMEOUT,
                seed=seed,
                n_threads=1,
            )
            if result.matching is None:
                raise RuntimeError(
                    f"seed={seed}, m={m}, condition={condition}: solver {result.status}, no candidate"
                )
            match = result.matching
            row = metadata(seed, m, condition, release, SOLVE_TIMEOUT)
            row.update(
                n_trees_available=n_available,
                status=result.status,
                duration_s=round(time.perf_counter() - start, 3),
                reconstruction_error=result.reconstruction_error,
                matching_cost=match.total_cost,
                matching_ambiguous=match.ambiguous,
                uniform_baseline_error=target_error(
                    X, uniform, target_cols, match.permutation
                ),
                marginal_baseline_error=target_error(
                    X, marginal, target_cols, match.permutation
                ),
                joint_split_frac_A=round(joint_a, 4),
                joint_split_frac_B=round(joint_b, 4),
            )
            rows.append(row)
    print(f"[seed {seed}] done: {n_available} trees, {len(rows)} results", flush=True)
    return rows


def main(argv: list[str] | None = None, *, output_name: str = "final_experiment.csv"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=N_SEEDS)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV parent/name; a fresh timestamp directory is inserted before the filename",
    )
    args = parser.parse_args(argv)
    if args.seed_start < 0 or args.seed_end <= args.seed_start or args.workers <= 0:
        parser.error("require 0 <= seed-start < seed-end and positive workers")
    start = time.perf_counter()
    output = experiment_output(output_name, args.output)
    print(f"Experiment output: {output}", flush=True)
    rows = []
    with Pool(processes=args.workers) as pool:
        for batch in pool.imap_unordered(
            run_seed, range(args.seed_start, args.seed_end)
        ):
            rows.extend(batch)
    write_rows(rows, output)
    print(
        f"Wrote {len(rows)} results to {output} in {time.perf_counter() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
