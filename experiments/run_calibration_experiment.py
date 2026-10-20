"""Per-row confidence / calibration experiment.

For a subset of seeds and two representative tree-release sizes (a
sparse m=10 and a rich m=100), re-solve each (seed, m, condition) cell
N_REPEATS times with different solver seeds and record, per true row,
the row-matched agreement rate across repeats (the confidence score) and
whether the majority-vote reconstruction is actually correct. This is
the data a calibration curve (agreement rate vs. empirical accuracy) is
built from in analyze_calibration.py.

The agreement is aligned to the true table, so it is an evaluator-side
diagnostic, not a confidence an attacker could compute (see
run_record_confidence.py for that).

Tree selection is part of the protocol (--selection, default random; see
run_final_experiment.py). Results go to
results/calibration_experiment/<selection>/, one CSV per seed next to a
manifest.json (see provenance.py), so reruns resume and settings never mix.

Usage: .venv/bin/python experiments/run_calibration_experiment.py --seed-end 35
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attack.confidence import repeated_reconstruct_with_confidence
from attack.conditions import CONDITIONS
from attack.reconstruct import extract_branches
from rashomon.build import build_rashomon_set, select_random_trees, select_trees
from rashomon.data import binarize_adult_curated, load_adult_raw, sample_rows

from provenance import run_seeds, stamp

TARGET, ATTR_A, ATTR_B = "sex", "marital-status", "education"
M_VALUES = [10, 100]
PILOT_N = 100
REGULARIZATION = 0.01
RASHOMON_BOUND_MULTIPLIER = 0.1
N_REPEATS = 15  # odd, avoids ties in the binary-T majority vote; gives 8 distinct
# agreement levels (8/15..15/15) instead of 7's 4 (4/7..7/7), for a finer isotonic fit
SOLVE_TIMEOUT = 60.0
TREE_SELECTION_SEED_OFFSET = 20_000  # same tree choice as run_final_experiment.py for a seed

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "calibration_experiment"


def protocol(selection: str) -> dict:
    """Every setting that changes what this experiment's numbers mean."""
    return {
        "dataset": "adult_curated_13col", "target": TARGET, "attr_a": ATTR_A, "attr_b": ATTR_B,
        "pilot_n": PILOT_N, "m_values": M_VALUES, "n_repeats": N_REPEATS,
        "regularization": REGULARIZATION, "rashomon_bound_multiplier": RASHOMON_BOUND_MULTIPLIER,
        "tree_selection": selection, "solve_timeout": SOLVE_TIMEOUT,
        "conditions": CONDITIONS, "labels": "given_per_row",
    }


def run_seed(seed: int, selection: str = "random") -> list[dict]:
    df = load_adult_raw(ROOT / "data")
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
        trees = (select_trees(model, m) if selection == "first"
                 else select_random_trees(model, m, seed=seed + TREE_SELECTION_SEED_OFFSET))
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
            gc.collect()  # each repeated_reconstruct_with_confidence call builds and
            # solves N_REPEATS fresh CP-SAT models in this same long-lived worker
            # process; force collection between cells rather than relying on it
            # happening before memory builds up across all 2*4*N_REPEATS solves

    print(f"[seed {seed}] done: {n_available} trees (build {build_time:.1f}s), {len(rows_out)} rows", flush=True)
    return rows_out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selection", choices=["random", "first"], default="random",
                        help="which m trees of the Rashomon set are released")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=35, help="exclusive")
    # Each worker runs 2 m-values x 4 conditions x N_REPEATS solves and holds more
    # memory than the single-solve experiment; half the cores kept it out of swap.
    parser.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) // 2),
                        help="seeds solved in parallel (each solve uses one thread)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"defaults to results/{EXPERIMENT}/<selection>")
    args = parser.parse_args(argv)

    result_dir = args.out_dir or ROOT / "results" / EXPERIMENT / args.selection
    stamp(result_dir, EXPERIMENT, protocol(args.selection), data="seed*.csv", argv=argv)
    t_start = time.time()
    run_seeds(result_dir, range(args.seed_start, args.seed_end), partial(run_seed, selection=args.selection),
              workers=args.workers)
    print(f"\nFinished in {time.time() - t_start:.0f}s -> {result_dir}")


if __name__ == "__main__":
    main()
