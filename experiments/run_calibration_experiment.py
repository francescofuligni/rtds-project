"""Run repeated reconstructions for truth-aligned diagnostic agreement."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment_io import DIAGNOSTIC_KIND, experiment_output, metadata, write_rows

from attack.conditions import CONDITIONS
from attack.confidence import repeated_reconstruct_with_confidence
from attack.reconstruct import prepare_release
from rashomon.build import build_rashomon_set, select_trees
from rashomon.data import binarize_adult_curated, load_adult_raw, sample_rows

TARGET, ATTR_A, ATTR_B = "sex", "marital-status", "education"
CALIBRATION_SEEDS = 20
M_VALUES = [10, 100]
PILOT_N = 100
REGULARIZATION = 0.01
RASHOMON_BOUND_MULTIPLIER = 0.1
N_REPEATS = 15
SOLVE_TIMEOUT = 60.0
N_WORKERS = min(6, os.cpu_count() or 1)


def run_seed(seed: int) -> list[dict]:
    pilot = sample_rows(binarize_adult_curated(load_adult_raw()), n=PILOT_N, seed=seed)
    model = build_rashomon_set(
        pilot.X,
        pilot.y,
        regularization=REGULARIZATION,
        rashomon_bound_multiplier=RASHOMON_BOUND_MULTIPLIER,
    )
    names = pilot.X.columns.to_list()
    groups = [[names.index(c) for c in cols] for cols in pilot.one_hot_groups.values()]
    rows = []
    for m in M_VALUES:
        if m > model.get_tree_count():
            print(f"[seed {seed}] skip m={m}: insufficient trees", flush=True)
            continue
        release = prepare_release(
            select_trees(model, m),
            pilot.X.to_numpy(),
            pilot.y.to_numpy(),
            names,
            groups,
        )
        for condition in CONDITIONS:
            scores, statuses = repeated_reconstruct_with_confidence(
                release,
                pilot,
                ATTR_A,
                ATTR_B,
                TARGET,
                condition,
                n_repeats=N_REPEATS,
                timeout=SOLVE_TIMEOUT,
                base_seed=seed * 1000,
                n_threads=1,
            )
            if not scores:
                raise RuntimeError(
                    f"seed={seed}, m={m}, condition={condition}: no successful diagnostic repeat; statuses={statuses}"
                )
            solved = sum(s in ("OPTIMAL", "FEASIBLE") for s in statuses)
            for score in scores:
                row = metadata(seed, m, condition, release, SOLVE_TIMEOUT)
                row.update(
                    confidence_kind=DIAGNOSTIC_KIND,
                    row=score.row,
                    n_votes=score.n_votes,
                    n_repeats=N_REPEATS,
                    n_solved=solved,
                    agreement=round(score.agreement, 4),
                    correct=score.correct,
                )
                rows.append(row)
            gc.collect()
    return rows


def main(
    argv: list[str] | None = None, *, output_name: str = "calibration_experiment.csv"
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=CALIBRATION_SEEDS)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV parent/name; a fresh timestamp directory is inserted before the filename",
    )
    args = parser.parse_args(argv)
    if args.seed_start < 0 or args.seed_end <= args.seed_start or args.workers <= 0:
        parser.error("require 0 <= seed-start < seed-end and positive workers")
    output = experiment_output(output_name, args.output)
    print(f"Experiment output: {output}", flush=True)
    rows = []
    with Pool(processes=args.workers) as pool:
        for batch in pool.imap_unordered(
            run_seed, range(args.seed_start, args.seed_end)
        ):
            rows.extend(batch)
    write_rows(rows, output)
    print(f"Wrote {len(rows)} diagnostic scores to {output}")


if __name__ == "__main__":
    main()
