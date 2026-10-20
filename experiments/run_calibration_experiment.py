"""Measure evaluator-side agreement across repeated reconstructions.

For each seed, build a Rashomon set from the private pilot and release the
configured release grid of tree counts (geometric by default; m=1 and m=500 for
the default calibration settings). For each
release and knowledge condition, solve N_REPEATS times with different solver
seeds. Record each row's agreement across candidates and majority-vote accuracy.

Candidates are aligned to the true table, so these are evaluator diagnostics,
not attacker-visible confidence scores. See run_record_confidence.py for the
attacker-side experiment.

Tree selection uses the same seeded random nested prefixes as the main runner.
Results are written to results/{timestamp}/calibration_experiment/, with one
CSV per seed and a protocol manifest. Each invocation creates a fresh directory.

Usage: .venv/bin/python experiments/run_calibration_experiment.py --n-seeds 35
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from configuration import (
    ExperimentConfig,
    add_configuration_arguments,
    configuration_from_args,
)
from output_paths import experiment_result_dir
from provenance import available_cpu_count, run_seeds, seed_indices, stamp

from attack.conditions import CONDITIONS
from attack.confidence import repeated_reconstruct_with_confidence
from attack.reconstruct import extract_branches
from rashomon.build import (
    build_rashomon_set,
    select_random_trees,
)
from rashomon.data import binarize_adult_curated, load_adult_raw, sample_rows

TARGET = "sex"
PILOT_N = 100
K = 2
REGULARIZATION = 0.01
RASHOMON_BOUND_MULTIPLIER = 0.1
N_REPEATS = 15  # odd, avoids ties in the binary-T majority vote; gives 8 distinct
# agreement levels (8/15..15/15) instead of 7's 4 (4/7..7/7), for a finer isotonic fit
SOLVE_TIMEOUT = 60.0
TREE_SELECTION_SEED_OFFSET = (
    20_000  # same tree choice as run_final_experiment.py for a seed
)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "calibration_experiment"


def protocol(
    config: ExperimentConfig | None = None,
    timeout: float = SOLVE_TIMEOUT,
    workers: int | None = None,
) -> dict:
    """Every setting that changes what this experiment's numbers mean."""
    config = config or ExperimentConfig(
        n=PILOT_N,
        k=K,
    )
    return {
        "dataset": "adult_curated_13col",
        "target": TARGET,
        "attr_a": config.attr_a,
        "attr_b": config.attr_b,
        "pilot_n": config.n,
        **config.release_protocol(),
        "n_repeats": N_REPEATS,
        "regularization": REGULARIZATION,
        "rashomon_bound_multiplier": RASHOMON_BOUND_MULTIPLIER,
        "tree_selection": "random",
        "solve_timeout": timeout,
        "seed_workers": workers,
        "tree_selection_seed_offset": TREE_SELECTION_SEED_OFFSET,
        "tree_selection_pool": (
            "all_available_per_seed" if config.include_all_trees else config.tree_pool
        ),
        "tree_selection_order": "uniform_without_replacement_nested_prefix",
        "conditions": CONDITIONS,
        "labels": "given_per_row",
    }


def run_seed(
    seed: int,
    *,
    config: ExperimentConfig | None = None,
    timeout: float = SOLVE_TIMEOUT,
) -> list[dict]:
    config = config or ExperimentConfig(
        n=PILOT_N,
        k=K,
    )
    df = load_adult_raw(ROOT / "data")
    data_full = binarize_adult_curated(df)
    pilot = sample_rows(data_full, n=config.n, seed=seed)

    t0 = time.time()
    model = build_rashomon_set(
        pilot.X,
        pilot.y,
        regularization=REGULARIZATION,
        rashomon_bound_multiplier=RASHOMON_BOUND_MULTIPLIER,
    )
    n_available = model.get_tree_count()
    build_time = time.time() - t0

    rows_out = []
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
        X = pilot.X.to_numpy()
        y = pilot.y.to_numpy()
        classes = sorted(pilot.y.unique().tolist())
        trees_branches = [extract_branches(t, X, y, classes) for t in trees]

        for condition in CONDITIONS:
            rows, statuses = repeated_reconstruct_with_confidence(
                trees_branches,
                pilot,
                config.attr_a,
                config.attr_b,
                TARGET,
                condition,
                n_repeats=N_REPEATS,
                timeout=timeout,
                base_seed=seed * 1000,
                n_threads=1,
            )
            n_solved = sum(1 for s in statuses if s in ("OPTIMAL", "FEASIBLE"))
            for r in rows:
                rows_out.append(
                    {
                        "seed": seed,
                        "m": m,
                        "m_is_all_trees": m == n_available,
                        "condition": condition,
                        "row": r.row,
                        "n_votes": r.n_votes,
                        "n_repeats": N_REPEATS,
                        "n_solved": n_solved,
                        "agreement": round(r.agreement, 4),
                        "correct": r.correct,
                    }
                )
            # Release solver models before this worker processes the next cell.
            gc.collect()

    print(
        f"[seed {seed}] done: {n_available} trees (build {build_time:.1f}s), {len(rows_out)} rows",
        flush=True,
    )
    return rows_out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed-start", type=int, default=0, help="first seed index")
    parser.add_argument("--n-seeds", type=int, default=35, help="number of seeds")
    # Workers retain solver state across the configured grid, conditions, and repeats.
    # Limiting the default to half the available CPUs reduces memory pressure.
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, available_cpu_count() // 2),
        help="seeds solved in parallel (each solve uses one thread)",
    )
    parser.add_argument(
        "--timeout", type=float, default=SOLVE_TIMEOUT, help="seconds per solve"
    )
    add_configuration_arguments(parser, default_n=PILOT_N, default_k=K)
    args = parser.parse_args(argv)
    try:
        config = configuration_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
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
        protocol(config, args.timeout, args.workers),
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
        partial(run_seed, config=config, timeout=args.timeout),
        workers=args.workers,
    )
    print(f"\nFinished in {time.time() - t_start:.0f}s -> {result_dir}")


if __name__ == "__main__":
    main()
