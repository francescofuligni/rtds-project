"""Record-level, attacker-side confidence: which attacker can say how much
to trust each record it outputs, without ever seeing the ground truth?

For each seed, sample a private table of 100 rows and a disjoint public sample
of 2,000 rows (--public-n) from the same population. With --public-prior
marginals the attacker keeps only per-attribute shares within each income
class from that sample (published tables) instead of whole records, for both
prior-guided sampling and the no-trees baseline. Build a Rashomon set and release m
trees selected uniformly without replacement using nested prefixes. Each
attacker produces a 100-row candidate table whose label counts match the
private sample, then scores claims about record multiplicities. A claim is
correct when the private table contains at least the claimed number of copies
of that record (features and label). The attackers are:

  per_tree_presence    one reconstruction per tree; score = share of trees
                       whose reconstruction contains the record
  per_tree_absolute    same reconstructions; score = total copies across
                       them (a ranking, not a probability)
  per_tree_occurrence  same reconstructions; per copy k, share of trees
                       with at least k copies (absolute count split per copy)
  seed_only            K joint reconstructions differing only in solver seed
  prior_guided         K joint reconstructions, each the consistent table
                       closest to a random table drawn from the public sample
  prior_only           no trees: P(at least k copies) from public frequencies

Results go to results/{timestamp}/record_confidence/m<m>/seed<seed>.csv
next to a manifest.json recording the protocol (see provenance.py). Each
invocation creates a fresh timestamp directory.

Usage: .venv/bin/python experiments/run_record_confidence.py --m 10 --n-seeds 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from configuration import (
    RECORD_CONFIDENCE_SOLVER_THREADS,
    add_threads_argument,
    add_tree_selection_argument,
)
from output_paths import experiment_result_dir
from provenance import seed_indices, stamp, write_outcome_summary, write_rows

from attack.reconstruct import extract_branches
from attack.record_confidence import (
    DirichletRecordPrior,
    MarginalPrior,
    RecordPrior,
    best_table_claims,
    enumerate_feature_vectors,
    per_tree_candidates,
    record_counts,
    sampled_candidates,
    support_matrix,
)
from rashomon.build import build_rashomon_set
from rashomon.data import binarize_adult_curated, load_adult_raw
from rashomon.publisher import sample_private_public_indices
from rashomon.selection import (
    DEFAULT_RANDOM_TREE_POOL,
    select_random_trees,
    select_trees,
    selection_protocol,
)

PILOT_N = 100
PUBLIC_N = 2000
# What the attacker knows about the population: a sample of other people's
# whole records, or only per-attribute shares within each income class
# (published tables), estimated from that sample (see MarginalPrior).
# records_dirichlet: whole records, with the uncertainty of a finite sample
# about the population carried into the confidence (see DirichletRecordPrior).
PUBLIC_PRIORS = ["records", "records_dirichlet", "marginals"]
REGULARIZATION = 0.01
RASHOMON_BOUND_MULTIPLIER = 0.1
PRIOR_ALPHA = (
    0.1  # add-alpha smoothing of the public record frequencies (see RecordPrior)
)
PRIOR_MIN_PROBABILITY = 1e-4  # prior_only claims below this cannot reach any table
TREE_SELECTION_SEED_OFFSET = 20_000  # keeps tree choice independent of the row sample

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "record_confidence"


def protocol(
    samples: int,
    timeout: float,
    methods: list[str] | None = None,
    m_values: list[int] | None = None,
    workers: int = 2,
    threads: int = RECORD_CONFIDENCE_SOLVER_THREADS,
    public_n: int = PUBLIC_N,
    public_prior: str = "records",
    tree_selection: str = "random",
) -> dict:
    """Every setting that changes what this experiment's numbers mean."""
    return {
        "dataset": "adult_curated_13col",
        "pilot_n": PILOT_N,
        "public_n": public_n,
        "public_prior": public_prior,
        "regularization": REGULARIZATION,
        "rashomon_bound_multiplier": RASHOMON_BOUND_MULTIPLIER,
        **selection_protocol(
            tree_selection,
            seed_offset=TREE_SELECTION_SEED_OFFSET,
            pool=DEFAULT_RANDOM_TREE_POOL,
        ),
        "samples": samples,
        "m_values": sorted(m_values or [10]),
        "solve_timeout": timeout,
        # Too few search workers find no prior-guided table, or never prove the
        # closest one, within the timeout; see configuration.py.
        "solver_threads": threads,
        "seed_workers": workers,
        "prior_alpha": PRIOR_ALPHA,
        "methods": [m for m in METHOD_CHOICES if m in (methods or METHODS)],
        "background_knowledge": "none",
        "labels": "given_per_row",
    }


PER_TREE_SCORES = {
    "per_tree_presence": lambda c, k: float(np.mean(c >= 1)),
    "per_tree_absolute": lambda c, k: float(c.sum()),
    "per_tree_occurrence": lambda c, k: float(np.mean(c >= k)),
}
# Prior-guided sampling variants: (random tie-break, samples that pick the table,
# samples that score it), as [start, stop) in units of --samples K. Scoring the
# table with the same samples that picked it favours claims whose frequency came
# out high by chance (winner's curse); a split scores with independent samples.
PRIOR_GUIDED_VARIANTS = {
    "prior_guided": (False, (0, 1), (0, 1)),
    "prior_guided_2k": (False, (0, 2), (0, 2)),
    "prior_guided_split": (False, (0, 1), (1, 2)),
    "prior_guided_tb": (True, (0, 1), (0, 1)),
    "prior_guided_tb_2k": (True, (0, 2), (0, 2)),
    "prior_guided_tb_split": (True, (0, 1), (1, 2)),
}
METHODS = [*PER_TREE_SCORES, "seed_only", "prior_guided", "prior_only"]  # default set
METHOD_CHOICES = METHODS + [
    name for name in PRIOR_GUIDED_VARIANTS if name not in METHODS
]


def _scores_from_support(support: dict, score) -> dict:
    """Score every claim (record, k) for k up to the record's largest multiplicity."""
    return {
        (r, k): score(mult, k)
        for r, mult in support.items()
        for k in range(1, int(mult.max()) + 1)
    }


def _n_solved(support: dict) -> int:
    return len(next(iter(support.values()))) if support else 0


def run_seed(
    seed: int,
    m: int,
    n_samples: int,
    timeout: float,
    methods: set[str] = frozenset(METHODS),
    threads: int = RECORD_CONFIDENCE_SOLVER_THREADS,
    public_n: int = PUBLIC_N,
    public_prior: str = "records",
    tree_selection: str = "random",
    coverage: dict | None = None,
) -> list[dict] | None:
    if coverage is None:
        coverage = {"methods": [], "solver_statuses": []}

    def record_coverage(name, pick_statuses, score_statuses):
        def counts(statuses):
            return {
                "requested": len(statuses),
                "successful": sum(s in ("FEASIBLE", "OPTIMAL") for s in statuses),
                "statuses": dict(Counter(statuses)),
            }

        coverage["methods"].append(
            {
                "method": name,
                "selection": counts(pick_statuses),
                "scoring": counts(score_statuses),
            }
        )

    data_full = binarize_adult_curated(load_adult_raw(ROOT / "data"))
    cols = list(data_full.X.columns)
    groups = [[cols.index(c) for c in g] for g in data_full.one_hot_groups.values()]

    idx, public = sample_private_public_indices(
        data_full, n=PILOT_N, public_n=public_n, seed=seed
    )
    X, y = data_full.X.loc[idx].to_numpy(), data_full.y.loc[idx].to_numpy()
    public_X, public_y = (
        data_full.X.loc[public].to_numpy(),
        data_full.y.loc[public].to_numpy(),
    )

    # Attacker-side quantities: the release reveals how many rows carry each label.
    classes = sorted(np.unique(y).tolist())
    class_totals = {int(c): int((y == c).sum()) for c in classes}
    vectors = enumerate_feature_vectors(len(cols), groups)
    if public_prior == "marginals":
        prior = MarginalPrior(
            public_X, public_y, class_totals, groups, alpha=PRIOR_ALPHA
        )
        draw_table = prior.draw_table
    elif public_prior == "records_dirichlet":
        prior = DirichletRecordPrior(
            public_X, public_y, class_totals, vectors, alpha=PRIOR_ALPHA
        )
        draw_table = prior.draw_table
    else:
        prior = RecordPrior(
            public_X, public_y, class_totals, len(vectors), alpha=PRIOR_ALPHA
        )
        draw_table = None

    # name -> (scores that pick the table, number of candidates, scores reported as confidence)
    scored: dict[str, tuple[dict, int, dict]] = {}
    # name -> solves proven optimal (for a feasibility solve: a table was found;
    # for prior-guided sampling: the closest consistent table was proven closest)
    optimal: dict[str, int] = {}
    if "prior_only" in methods:  # needs no trees, so also no Rashomon set
        prior_scores = {}
        for v in vectors:
            for c, total in class_totals.items():
                for k in range(1, total + 1):
                    p = prior.probability(v + (c,), k)
                    if p < PRIOR_MIN_PROBABILITY:
                        break
                    prior_scores[(v + (c,), k)] = p
        scored["prior_only"] = (prior_scores, 0, prior_scores)
        record_coverage("prior_only", [], [])

    if methods - {"prior_only"}:
        model = build_rashomon_set(
            data_full.X.loc[idx].reset_index(drop=True),
            data_full.y.loc[idx].reset_index(drop=True),
            regularization=REGULARIZATION,
            rashomon_bound_multiplier=RASHOMON_BOUND_MULTIPLIER,
        )
        if model.get_tree_count() < m:
            return None
        trees = (
            select_trees(model, m)
            if tree_selection == "sequential"
            else select_random_trees(model, m, seed=seed + TREE_SELECTION_SEED_OFFSET)
        )
        branches = [extract_branches(t, X, y, classes) for t in trees]
        solve = {
            "timeout": timeout,
            "seed": seed,
            "n_threads": threads,
        }

        if methods & PER_TREE_SCORES.keys():
            per_tree, statuses = per_tree_candidates(
                branches, y, classes, len(cols), groups, **solve
            )
            coverage["solver_statuses"].extend(statuses)
            support = support_matrix(per_tree)
            for name in methods & PER_TREE_SCORES.keys():
                scores = _scores_from_support(support, PER_TREE_SCORES[name])
                scored[name] = (scores, _n_solved(support), scores)
                optimal[name] = statuses.count("OPTIMAL")
                record_coverage(name, statuses, statuses)

        occurrence = lambda c, k: float(np.mean(c >= k))
        if "seed_only" in methods:
            candidates, statuses = sampled_candidates(
                branches, y, classes, len(cols), groups, n_samples=n_samples, **solve
            )
            coverage["solver_statuses"].extend(statuses)
            support = support_matrix(candidates)
            scores = _scores_from_support(support, occurrence)
            scored["seed_only"] = (scores, _n_solved(support), scores)
            optimal["seed_only"] = statuses.count("OPTIMAL")
            record_coverage("seed_only", statuses, statuses)

        variants = [name for name in methods if name in PRIOR_GUIDED_VARIANTS]
        for tie_break in (False, True):
            names = [n for n in variants if PRIOR_GUIDED_VARIANTS[n][0] == tie_break]
            if not names:
                continue
            # One pool of samples per tie-break setting; variants use slices of it.
            blocks = max(
                max(PRIOR_GUIDED_VARIANTS[n][1][1], PRIOR_GUIDED_VARIANTS[n][2][1])
                for n in names
            )
            candidates, statuses = sampled_candidates(
                branches,
                y,
                classes,
                len(cols),
                groups,
                n_samples=blocks * n_samples,
                public_X=public_X,
                public_y=public_y,
                random_tie_break=tie_break,
                draw_table=draw_table,
                **solve,
            )
            coverage["solver_statuses"].extend(statuses)
            for name in names:
                _, (p0, p1), (s0, s1) = PRIOR_GUIDED_VARIANTS[name]
                record_coverage(
                    name,
                    statuses[p0 * n_samples : p1 * n_samples],
                    statuses[s0 * n_samples : s1 * n_samples],
                )
                pick = support_matrix(candidates[p0 * n_samples : p1 * n_samples])
                score = support_matrix(candidates[s0 * n_samples : s1 * n_samples])
                scored[name] = (
                    _scores_from_support(pick, occurrence),
                    _n_solved(score),
                    _scores_from_support(score, occurrence),
                )
                optimal[name] = statuses[s0 * n_samples : s1 * n_samples].count(
                    "OPTIMAL"
                )

    # Ground truth is used only from here on, to score the tables.
    truth = record_counts(X, y)
    rows = []
    for name, (scores, n_candidates, reported) in scored.items():
        if name != "prior_only" and (not scores or n_candidates == 0):
            continue  # No selection/scoring candidate means no supported claim.
        tiebreak = {key: prior.probability(*key) for key in scores}
        for record, k, _ in best_table_claims(scores, class_totals, tiebreak):
            # a split scores claims its half never saw as 0
            confidence = reported.get((record, k), 0.0)
            rows.append(
                {
                    "seed": seed,
                    "m": m,
                    "method": name,
                    "record": "".join(map(str, record[:-1])) + f"|{record[-1]}",
                    "k": k,
                    "confidence": round(confidence, 6),
                    "correct": int(truth[record] >= k),
                    "true_count": truth[record],
                    "n_candidates": n_candidates,
                    "n_optimal": optimal.get(name, 0),
                }
            )
    return rows


def _run_and_write(seed: int, m: int, args) -> dict:
    out = args.out_dir / f"m{m:03d}" / f"seed{seed:03d}.csv"
    if out.with_suffix(".skip").exists():
        return {"seed": seed, "m": m, "outcome": "skipped", "existing": True}
    only = set(args.only or [])
    if out.exists() and not only:
        return {
            "seed": seed,
            "m": m,
            "outcome": "completed",
            "existing": True,
            "coverage": "not available",
        }
    t0 = time.time()
    coverage = {"methods": [], "solver_statuses": []}
    rows = run_seed(
        seed,
        m,
        args.samples,
        args.timeout,
        methods=only or set(args.methods),
        threads=args.threads,
        public_n=args.public_n,
        public_prior=args.public_prior,
        tree_selection=args.tree_selection,
        coverage=coverage,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows is None:
        out.with_suffix(".skip").write_text("Rashomon set smaller than m\n")
        return {"seed": seed, "m": m, "outcome": "skipped"}
    if only and out.exists():  # preserve methods already present in this run
        with out.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["method"] not in only] + rows
    if rows:
        write_rows(out, rows)
    return {
        "seed": seed,
        "m": m,
        "outcome": "completed",
        "methods": coverage["methods"],
        "solver_statuses": dict(Counter(coverage["solver_statuses"])),
        "duration_s": time.time() - t0,
        "no_candidate": sum(
            s not in ("FEASIBLE", "OPTIMAL") for s in coverage["solver_statuses"]
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_tree_selection_argument(parser)
    parser.add_argument(
        "--m", type=int, nargs="+", default=[10], help="numbers of released trees"
    )
    parser.add_argument("--seed-start", type=int, default=0, help="first seed index")
    parser.add_argument("--n-seeds", type=int, default=20, help="number of seeds")
    parser.add_argument(
        "--samples", type=int, default=20, help="K for the sampling attackers"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="seeds solved in parallel"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per solve")
    add_threads_argument(parser, default=RECORD_CONFIDENCE_SOLVER_THREADS)
    parser.add_argument(
        "--public-n",
        type=int,
        default=PUBLIC_N,
        help=f"public sample size (default {PUBLIC_N})",
    )
    parser.add_argument(
        "--public-prior",
        choices=PUBLIC_PRIORS,
        default="records",
        help="records: the attacker holds the public records; marginals: only "
        "per-attribute shares within each label, estimated from them",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=METHODS,
        help="attackers to run (part of the protocol)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=METHOD_CHOICES,
        help="recompute just these attackers inside existing per-seed files",
    )
    args = parser.parse_args(argv)
    try:
        seeds = seed_indices(args.seed_start, args.n_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if any(m < 1 for m in args.m) or len(set(args.m)) != len(args.m):
        parser.error("--m values must be positive and unique")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.public_n < 1:
        parser.error("--public-n must be positive")
    if args.only and not set(args.only) <= set(args.methods):
        parser.error("--only must name attackers that are part of --methods")

    args.out_dir = experiment_result_dir(EXPERIMENT)
    stamp(
        args.out_dir,
        EXPERIMENT,
        protocol(
            args.samples,
            args.timeout,
            args.methods,
            args.m,
            args.workers,
            args.threads,
            args.public_n,
            args.public_prior,
            args.tree_selection,
        ),
        data="m*/seed*.csv",
        argv=argv,
        invocation={
            "seed_start": args.seed_start,
            "n_seeds": args.n_seeds,
            "seed_end_exclusive": seeds.stop,
            "m_values": args.m,
            "workers": args.workers,
        },
    )
    jobs = [(seed, m) for m in args.m for seed in seeds]
    t_start = time.time()
    units = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_and_write, seed, m, args): (seed, m) for seed, m in jobs
        }
        for future in as_completed(futures):
            seed, m = futures[future]
            try:
                units.append(future.result())
            except Exception as exc:  # noqa: BLE001 -- isolate failed work units, report below
                units.append(
                    {
                        "seed": seed,
                        "m": m,
                        "outcome": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            print(units[-1], flush=True)
    summary = write_outcome_summary(args.out_dir, units)
    print(
        f"\nFinished {len(jobs)} jobs in {time.time() - t_start:.0f}s -> {args.out_dir}",
        flush=True,
    )

    if summary["incomplete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
