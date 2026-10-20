"""Record-level, attacker-side confidence: which attacker can say how much
to trust each record it outputs, without ever seeing the ground truth?

Per seed: sample n=100 rows (the private table) and a disjoint 2000-row
public sample from the same population, build a Rashomon set, release m of
its trees (--selection: the first m in TreeFARMS order, which are
near-copies, or m drawn at random, nested across m), and let each attacker output a 100-row table (per label,
as many rows as the release says carry that label) with a confidence per
row. A row is correct if the private table really contains at least that
many copies of the record (features + label). Attackers:

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

One CSV per (selection, m, seed) so an interrupted run keeps every finished
seed and resumes where it stopped.

Usage: .venv/bin/python experiments/run_record_confidence.py --m 10 --seed-end 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from attack.reconstruct import extract_branches
from attack.record_confidence import (
    RecordPrior,
    best_table_claims,
    enumerate_feature_vectors,
    per_tree_candidates,
    record_counts,
    sampled_candidates,
    support_matrix,
)
from rashomon.build import build_rashomon_set, select_random_trees, select_trees
from rashomon.data import binarize_adult_curated, load_adult_raw

PILOT_N = 100
PUBLIC_N = 2000
REGULARIZATION = 0.01
RASHOMON_BOUND_MULTIPLIER = 0.1
PRIOR_MIN_PROBABILITY = 1e-4  # prior_only claims below this cannot reach any table
TREE_SELECTION_SEED_OFFSET = 20_000  # keeps tree choice independent of the row sample

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "record_confidence"
FIELDS = ["seed", "m", "method", "record", "k", "confidence", "correct", "true_count", "n_candidates"]


PER_TREE_SCORES = {
    "per_tree_presence": lambda c, k: float(np.mean(c >= 1)),
    "per_tree_absolute": lambda c, k: float(c.sum()),
    "per_tree_occurrence": lambda c, k: float(np.mean(c >= k)),
}
METHODS = [*PER_TREE_SCORES, "seed_only", "prior_guided", "prior_only"]


def _scores_from_support(support: dict, score) -> dict:
    """Score every claim (record, k) for k up to the record's largest multiplicity."""
    return {(r, k): score(mult, k) for r, mult in support.items() for k in range(1, int(mult.max()) + 1)}


def _n_solved(support: dict) -> int:
    return len(next(iter(support.values()))) if support else 0


def run_seed(seed: int, m: int, n_samples: int, n_threads: int, timeout: float,
             methods: set[str] = frozenset(METHODS), selection: str = "first") -> list[dict] | None:
    data_full = binarize_adult_curated(load_adult_raw(ROOT / "data"))
    cols = list(data_full.X.columns)
    groups = [[cols.index(c) for c in g] for g in data_full.one_hot_groups.values()]

    idx = data_full.X.sample(n=PILOT_N, random_state=seed).index
    X, y = data_full.X.loc[idx].to_numpy(), data_full.y.loc[idx].to_numpy()
    public = data_full.X.drop(index=idx).sample(n=PUBLIC_N, random_state=seed + 10_000).index
    public_X, public_y = data_full.X.loc[public].to_numpy(), data_full.y.loc[public].to_numpy()

    # Attacker-side quantities: the release reveals how many rows carry each label.
    classes = sorted(np.unique(y).tolist())
    class_totals = {int(c): int((y == c).sum()) for c in classes}
    vectors = enumerate_feature_vectors(len(cols), groups)
    prior = RecordPrior(public_X, public_y, class_totals, len(vectors))

    scored: dict[str, tuple[dict, int]] = {}
    if "prior_only" in methods:  # needs no trees, so also no Rashomon set
        prior_scores = {}
        for v in vectors:
            for c in class_totals:
                for k in range(1, class_totals[c] + 1):
                    p = prior.probability(v + (c,), k)
                    if p < PRIOR_MIN_PROBABILITY:
                        break
                    prior_scores[(v + (c,), k)] = p
        scored["prior_only"] = (prior_scores, 0)

    if methods - {"prior_only"}:
        model = build_rashomon_set(data_full.X.loc[idx].reset_index(drop=True),
                                   data_full.y.loc[idx].reset_index(drop=True),
                                   regularization=REGULARIZATION,
                                   rashomon_bound_multiplier=RASHOMON_BOUND_MULTIPLIER)
        if model.get_tree_count() < m:
            return None
        trees = (select_trees(model, m) if selection == "first"
                 else select_random_trees(model, m, seed=seed + TREE_SELECTION_SEED_OFFSET))
        branches = [extract_branches(t, X, y, classes) for t in trees]
        solve = dict(timeout=timeout, seed=seed, n_threads=n_threads)

        if methods & PER_TREE_SCORES.keys():
            per_tree, _ = per_tree_candidates(branches, y, classes, len(cols), groups, **solve)
            support = support_matrix(per_tree)
            for name in methods & PER_TREE_SCORES.keys():
                scored[name] = (_scores_from_support(support, PER_TREE_SCORES[name]), _n_solved(support))

        for name, public_data in [("seed_only", (None, None)), ("prior_guided", (public_X, public_y))]:
            if name not in methods:
                continue
            candidates, _ = sampled_candidates(branches, y, classes, len(cols), groups, n_samples=n_samples,
                                               public_X=public_data[0], public_y=public_data[1], **solve)
            support = support_matrix(candidates)
            scored[name] = (_scores_from_support(support, lambda c, k: float(np.mean(c >= k))),
                            _n_solved(support))

    # Ground truth is used only from here on, to score the tables.
    truth = record_counts(X, y)
    rows = []
    for name, (scores, n_candidates) in scored.items():
        tiebreak = {key: prior.probability(*key) for key in scores}
        for record, k, confidence in best_table_claims(scores, class_totals, tiebreak):
            rows.append({
                "seed": seed, "m": m, "method": name,
                "record": "".join(map(str, record[:-1])) + f"|{record[-1]}", "k": k,
                "confidence": round(confidence, 6), "correct": int(truth[record] >= k),
                "true_count": truth[record], "n_candidates": n_candidates,
            })
    return rows


def _run_and_write(seed: int, m: int, args) -> str:
    out = args.out_dir / args.selection / f"m{m:03d}" / f"seed{seed:03d}.csv"
    if out.with_suffix(".skip").exists():
        return f"[m={m} seed {seed}] skipped earlier: Rashomon set smaller than m"
    only = set(args.only or [])
    if only and not out.exists():
        return f"[m={m} seed {seed}] nothing to update (no existing file)"
    if out.exists() and not only:
        return f"[m={m} seed {seed}] already done"
    t0 = time.time()
    rows = run_seed(seed, m, args.samples, args.threads, args.timeout, methods=only or set(METHODS),
                    selection=args.selection)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows is None:
        out.with_suffix(".skip").write_text("Rashomon set smaller than m\n")
        return f"[m={m} seed {seed}] skipped: Rashomon set smaller than m"
    if only:  # keep the other attackers' rows, replace only the recomputed ones
        with out.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["method"] not in only] + rows
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)
    return f"[m={m} seed {seed}] done in {time.time() - t0:.0f}s"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--m", type=int, nargs="+", default=[10], help="numbers of released trees")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=20, help="exclusive")
    parser.add_argument("--samples", type=int, default=20, help="K for the sampling attackers")
    parser.add_argument("--workers", type=int, default=2, help="seeds solved in parallel")
    # sched_getaffinity respects the cores a cluster scheduler actually allocated
    parser.add_argument("--threads", type=int, default=max(1, len(os.sched_getaffinity(0)) // 2),
                        help="CP-SAT threads per solve")
    parser.add_argument("--selection", choices=["first", "random"], default="random",
                        help="which m trees of the Rashomon set are released")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per solve")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="+", choices=METHODS,
                        help="recompute just these attackers inside existing per-seed files")
    args = parser.parse_args(argv)

    jobs = [(seed, m) for m in args.m for seed in range(args.seed_start, args.seed_end)]
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_and_write, seed, m, args): (seed, m) for seed, m in jobs}
        for future in as_completed(futures):
            seed, m = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as e:
                print(f"[m={m} seed {seed}] FAILED: {e!r}", flush=True)
    print(f"\nFinished {len(jobs)} jobs in {time.time() - t_start:.0f}s -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
