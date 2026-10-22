"""Run the paired None/A/B/A+B background-knowledge experiment on Adult.

For each seed, sample a private pilot and a disjoint holdout used only by the
baselines, then build one Rashomon set. For every configured geometric or linear
release size, use
the same sample and selected trees in all four knowledge conditions. This
pairing makes condition differences comparable within each seed and release.

Tree selection uses one random ordering without replacement. Releasing its
first m trees creates nested prefixes; sequential TreeFARMS enumeration is not
used.

Seeds run independently in a process pool. Each worker builds one Rashomon set
and processes all release sizes and conditions. Each CP-SAT solve uses
--threads search workers (a portfolio of strategies; see configuration.py).

Results are written to results/{timestamp}/final_experiment/, with one CSV per
seed and a protocol manifest. Each invocation creates a fresh directory.

Usage: .venv/bin/python experiments/run_final_experiment.py --n-seeds 350
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from configuration import (
    ExperimentConfig,
    add_configuration_arguments,
    DEFAULT_SOLVER_THREADS,
    add_threads_argument,
    configuration_from_args,
)
from output_paths import experiment_result_dir
from provenance import available_cpu_count, run_seeds, seed_indices, stamp

from attack.baselines import marginal_baseline, uniform_random_guess
from attack.conditions import CONDITIONS, run_condition
from attack.reconstruct import _walk_leaves, extract_branches
from rashomon.build import (
    build_rashomon_set,
    select_random_trees,
)
from rashomon.data import (
    binarize_adult_curated,
    known_columns_for,
    load_adult_raw,
    sample_rows,
)

TARGET = "sex"
HOLDOUT_N = 2000
REGULARIZATION = 0.01
RASHOMON_BOUND_MULTIPLIER = 0.1
SOLVE_TIMEOUT = 180.0
TREE_SELECTION_SEED_OFFSET = 20_000  # keeps tree choice independent of the row sample

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "final_experiment"


def protocol(
    config: ExperimentConfig | None = None,
    timeout: float = SOLVE_TIMEOUT,
    workers: int | None = None,
    threads: int = DEFAULT_SOLVER_THREADS,
) -> dict:
    """Every setting that changes what this experiment's numbers mean."""
    config = config or ExperimentConfig()
    return {
        "dataset": "adult_curated_13col",
        "target": TARGET,
        "attr_a": config.attr_a,
        "attr_b": config.attr_b,
        "pilot_n": config.n,
        "holdout_n": HOLDOUT_N,
        **config.release_protocol(),
        "regularization": REGULARIZATION,
        "rashomon_bound_multiplier": RASHOMON_BOUND_MULTIPLIER,
        "tree_selection": "random",
        "solve_timeout": timeout,
        "solver_threads": threads,
        "seed_workers": workers,
        "tree_selection_seed_offset": TREE_SELECTION_SEED_OFFSET,
        "tree_selection_pool": (
            "all_available_per_seed" if config.include_all_trees else config.tree_pool
        ),
        "tree_selection_order": "uniform_without_replacement_nested_prefix",
        "conditions": CONDITIONS,
        "labels": "given_per_row",
    }


def joint_split_fraction(
    trees: list[dict], target_cols: set[int], other_cols: set[int]
) -> float:
    """Fraction of trees with at least one leaf whose root-to-leaf path
    splits on a target column AND an "other" column together -- the
    structural precondition for background knowledge of "other" to
    propagate into a tighter constraint on "target" at that leaf."""
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


def run_seed(
    seed: int,
    *,
    config: ExperimentConfig | None = None,
    timeout: float = SOLVE_TIMEOUT,
    threads: int = DEFAULT_SOLVER_THREADS,
) -> list[dict]:
    config = config or ExperimentConfig()
    df = load_adult_raw(ROOT / "data")
    data_full = binarize_adult_curated(df)

    pilot_idx = data_full.X.sample(n=config.n, random_state=seed).index
    pilot = sample_rows(data_full, n=config.n, seed=seed)
    holdout_df = data_full.X.drop(index=pilot_idx).sample(
        n=HOLDOUT_N, random_state=seed + 10_000
    )

    t0 = time.time()
    model = build_rashomon_set(
        pilot.X,
        pilot.y,
        regularization=REGULARIZATION,
        rashomon_bound_multiplier=RASHOMON_BOUND_MULTIPLIER,
    )
    n_available = model.get_tree_count()
    build_time = time.time() - t0

    target_idx = [pilot.X.columns.get_loc(c) for c in known_columns_for(pilot, TARGET)]
    a_idx = {
        pilot.X.columns.get_loc(c) for c in known_columns_for(pilot, config.attr_a)
    }
    b_idx = {
        pilot.X.columns.get_loc(c) for c in known_columns_for(pilot, config.attr_b)
    }
    X = pilot.X.to_numpy()
    uniform = uniform_random_guess(config.n, target_idx, pilot.X.shape[1], seed=seed)
    marginal = marginal_baseline(pilot, TARGET, config.n, holdout_df, seed=seed)
    uniform_error = float(
        (uniform[:, target_idx] != X[:, target_idx]).any(axis=1).mean()
    )
    marginal_error = float(
        (marginal[:, target_idx] != X[:, target_idx]).any(axis=1).mean()
    )

    rows = []
    m_values = sorted(
        set(config.m_values)
        | ({n_available} if config.include_all_trees and n_available else set())
    )
    selection_pool = (
        max(config.tree_pool, n_available)
        if config.include_all_trees
        else config.tree_pool
    )
    for m in m_values:
        if m > n_available:
            continue

        trees = select_random_trees(
            model, m, seed=seed + TREE_SELECTION_SEED_OFFSET, pool=selection_pool
        )
        y = pilot.y.to_numpy()
        classes = sorted(pilot.y.unique().tolist())
        trees_branches = [extract_branches(t, X, y, classes) for t in trees]
        joint_a_frac = joint_split_fraction(trees, set(target_idx), a_idx)
        joint_b_frac = joint_split_fraction(trees, set(target_idx), b_idx)

        for condition in CONDITIONS:
            t0 = time.time()
            result = run_condition(
                condition,
                trees_branches,
                pilot,
                config.attr_a,
                config.attr_b,
                TARGET,
                timeout=timeout,
                seed=seed,
                n_threads=threads,
            )
            duration = time.time() - t0
            rows.append(
                {
                    "seed": seed,
                    "m": m,
                    "m_is_all_trees": m == n_available,
                    "n_trees_available": n_available,
                    "condition": condition,
                    "status": result.status,
                    "duration_s": round(duration, 3),
                    "reconstruction_error": result.reconstruction_error,
                    "uniform_baseline_error": uniform_error,
                    "marginal_baseline_error": marginal_error,
                    "joint_split_frac_A": round(joint_a_frac, 4),
                    "joint_split_frac_B": round(joint_b_frac, 4),
                }
            )

    print(
        f"[seed {seed}] done: {n_available} trees (build {build_time:.1f}s), {len(rows)} rows",
        flush=True,
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed-start", type=int, default=0, help="first seed index")
    parser.add_argument("--n-seeds", type=int, default=350, help="number of seeds")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="seeds solved in parallel (default: available CPUs // threads)",
    )
    add_threads_argument(parser)
    parser.add_argument(
        "--timeout", type=float, default=SOLVE_TIMEOUT, help="seconds per solve"
    )
    add_configuration_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = configuration_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.workers is None:
        args.workers = max(1, available_cpu_count() // args.threads)
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        seeds = seed_indices(args.seed_start, args.n_seeds)
    except ValueError as exc:
        parser.error(str(exc))

    result_dir = experiment_result_dir(EXPERIMENT)
    stamp(
        result_dir,
        EXPERIMENT,
        protocol(config, args.timeout, args.workers, args.threads),
        data="seed*.csv",
        argv=argv,
        invocation={
            "seed_start": args.seed_start,
            "n_seeds": args.n_seeds,
            "seed_end_exclusive": seeds.stop,
            "workers": args.workers,
        },
    )
    t_start = time.time()
    run_seeds(
        result_dir,
        seeds,
        partial(run_seed, config=config, timeout=args.timeout, threads=args.threads),
        workers=args.workers,
    )
    print(f"\nFinished in {time.time() - t_start:.0f}s -> {result_dir}")


if __name__ == "__main__":
    main()
