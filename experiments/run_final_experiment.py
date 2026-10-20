"""Final experiment: the None/A/B/A+B background-knowledge ablation on
Adult, repeated across seeds and swept over the number of released trees m.

For each seed: sample a fresh n=100 pilot (disjoint from a holdout used
only for the baselines), build one Rashomon set, and for each m in
M_VALUES release m of its trees and run all four conditions against
the *same* sample and *same* trees -- this is the "paired" part: within
a repetition, condition differences are directly comparable because
everything else is held fixed (meeting notes Sec. 3/6).

Which m trees are released is part of the protocol. The runner samples one
random ordering without replacement and releases its first m trees, nested
across m. Sequential TreeFARMS enumeration is not used.

Results go to results/final_experiment/random/, one CSV per seed
next to a manifest.json recording the protocol (see provenance.py), so
rerunning the same command resumes an interrupted run and results from
different settings never end up in the same directory.

Seeds are independent of each other, so they're farmed out across a
process pool (one seed's full pipeline -- Rashomon-set build + every m +
every condition -- per worker). Each worker pins its own CP-SAT solves
to a single thread (n_threads=1): parallelism happens across seeds, not
within a single solve, to avoid oversubscribing the machine's cores.

Usage: .venv/bin/python experiments/run_final_experiment.py --seed-end 350
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from provenance import available_cpu_count, run_seeds, stamp

from attack.baselines import marginal_baseline, uniform_random_guess
from attack.conditions import CONDITIONS, run_condition
from attack.reconstruct import _walk_leaves, extract_branches
from rashomon.build import (
    DEFAULT_RANDOM_TREE_POOL,
    build_rashomon_set,
    select_random_trees,
)
from rashomon.data import (
    binarize_adult_curated,
    known_columns_for,
    load_adult_raw,
    sample_rows,
)

TARGET, ATTR_A, ATTR_B = "sex", "marital-status", "education"
M_VALUES = [5, 10, 25, 50, 100]
PILOT_N = 100
HOLDOUT_N = 2000
REGULARIZATION = 0.01
RASHOMON_BOUND_MULTIPLIER = 0.1
SOLVE_TIMEOUT = 120.0
TREE_SELECTION_SEED_OFFSET = 20_000  # keeps tree choice independent of the row sample

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "final_experiment"


def protocol() -> dict:
    """Every setting that changes what this experiment's numbers mean."""
    return {
        "dataset": "adult_curated_13col",
        "target": TARGET,
        "attr_a": ATTR_A,
        "attr_b": ATTR_B,
        "pilot_n": PILOT_N,
        "holdout_n": HOLDOUT_N,
        "m_values": M_VALUES,
        "regularization": REGULARIZATION,
        "rashomon_bound_multiplier": RASHOMON_BOUND_MULTIPLIER,
        "tree_selection": "random",
        "solve_timeout": SOLVE_TIMEOUT,
        "tree_selection_seed_offset": TREE_SELECTION_SEED_OFFSET,
        "tree_selection_pool": DEFAULT_RANDOM_TREE_POOL,
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


def run_seed(seed: int) -> list[dict]:
    df = load_adult_raw(ROOT / "data")
    data_full = binarize_adult_curated(df)

    pilot_idx = data_full.X.sample(n=PILOT_N, random_state=seed).index
    pilot = sample_rows(data_full, n=PILOT_N, seed=seed)
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
    a_idx = {pilot.X.columns.get_loc(c) for c in known_columns_for(pilot, ATTR_A)}
    b_idx = {pilot.X.columns.get_loc(c) for c in known_columns_for(pilot, ATTR_B)}
    X = pilot.X.to_numpy()
    uniform = uniform_random_guess(PILOT_N, target_idx, pilot.X.shape[1], seed=seed)
    marginal = marginal_baseline(pilot, TARGET, PILOT_N, holdout_df, seed=seed)
    uniform_error = float(
        (uniform[:, target_idx] != X[:, target_idx]).any(axis=1).mean()
    )
    marginal_error = float(
        (marginal[:, target_idx] != X[:, target_idx]).any(axis=1).mean()
    )

    rows = []
    for m in M_VALUES:
        if m > n_available:
            continue

        trees = select_random_trees(model, m, seed=seed + TREE_SELECTION_SEED_OFFSET)
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
                ATTR_A,
                ATTR_B,
                TARGET,
                timeout=SOLVE_TIMEOUT,
                seed=seed,
                n_threads=1,
            )
            duration = time.time() - t0
            rows.append(
                {
                    "seed": seed,
                    "m": m,
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
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=350, help="exclusive")
    parser.add_argument(
        "--workers",
        type=int,
        default=available_cpu_count(),
        help="seeds solved in parallel (each solve uses one thread)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"defaults to results/{EXPERIMENT}/random",
    )
    args = parser.parse_args(argv)

    result_dir = args.out_dir or ROOT / "results" / EXPERIMENT / "random"
    stamp(result_dir, EXPERIMENT, protocol(), data="seed*.csv", argv=argv)
    t_start = time.time()
    run_seeds(
        result_dir,
        range(args.seed_start, args.seed_end),
        run_seed,
        workers=args.workers,
    )
    print(f"\nFinished in {time.time() - t_start:.0f}s -> {result_dir}")


if __name__ == "__main__":
    main()
