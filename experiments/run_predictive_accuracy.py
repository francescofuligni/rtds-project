"""Rebuild saved main releases and measure held-out majority-vote income accuracy.

No reconstruction solves run. Historical releases have no tree hashes: count
and structural checks detect drift but cannot certify historical tree identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from analysis.attack_figures import aggregate
from output_paths import experiment_result_dir
from provenance import load_results, run_info
from run_final_experiment import joint_split_fraction

from rashomon.build import build_rashomon_set
from rashomon.data import binarize_adult_curated, known_columns_for, load_adult_raw
from rashomon.publisher import sample_private_public_indices
from rashomon.selection import select_random_trees, select_trees, selection_protocol
from rashomon.utility import prefix_accuracy

ROOT = Path(__file__).resolve().parent.parent


def validate_protocol(protocol: dict) -> None:
    if protocol["dataset"] != "adult_curated_13col" or protocol["target"] != "sex":
        raise ValueError("Expected the curated Adult sex-reconstruction protocol")
    mode = protocol["tree_selection"]
    expected = selection_protocol(
        mode,
        seed_offset=protocol["tree_selection_seed_offset"],
        pool=protocol["tree_selection_pool"],
    )
    if any(protocol[key] != value for key, value in expected.items()):
        raise ValueError("Unsupported tree selection protocol")
    if protocol["pilot_n"] < 1 or protocol["holdout_n"] < 1:
        raise ValueError("Sample sizes must be positive")


def score_seed(seed: int, original: pd.DataFrame, protocol: dict, data) -> list[dict]:
    private, public = sample_private_public_indices(
        data, n=protocol["pilot_n"], public_n=protocol["holdout_n"], seed=seed
    )
    X, y = data.X.loc[private], data.y.loc[private]
    model = build_rashomon_set(
        X,
        y,
        regularization=protocol["regularization"],
        rashomon_bound_multiplier=protocol["rashomon_bound_multiplier"],
    )
    n = model.get_tree_count()
    if not original.n_trees_available.eq(n).all():
        raise ValueError(f"seed {seed}: rebuilt tree count differs from saved run")
    m_values = sorted(original.m.unique().tolist())
    if protocol["tree_selection"] == "sequential":
        trees = select_trees(model, max(m_values))
    else:
        pool = protocol["tree_selection_pool"]
        pool = n if pool == "all_available_per_seed" else pool
        trees = select_random_trees(
            model,
            max(m_values),
            seed=seed + protocol["tree_selection_seed_offset"],
            pool=pool,
        )
    target = {X.columns.get_loc(c) for c in known_columns_for(data, protocol["target"])}
    for role in ("A", "B"):
        columns = {
            X.columns.get_loc(c)
            for c in known_columns_for(data, protocol[f"attr_{role.lower()}"])
        }
        for m in m_values:
            observed = original.loc[original.m == m, f"joint_split_frac_{role}"]
            rebuilt = round(joint_split_fraction(trees[:m], target, columns), 4)
            if not np.allclose(observed, rebuilt, rtol=0, atol=1e-8):
                raise ValueError(
                    f"seed {seed}, m {m}: structural statistic {role} differs"
                )
    scores = prefix_accuracy(trees, data.X.loc[public], data.y.loc[public], m_values)
    return [
        {
            "seed": seed,
            "m": m,
            "accuracy": scores[m],
            "release_sha256": hashlib.sha256(
                json.dumps(trees[:m], sort_keys=True).encode()
            ).hexdigest(),
        }
        for m in m_values
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--attack-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--n-seeds",
        type=int,
        help="limit to a seed interval for a pilot; default all saved seeds",
    )
    args = parser.parse_args(argv)
    if args.seed_start < 0 or (args.n_seeds is not None and args.n_seeds < 1):
        parser.error("seed-start must be nonnegative and n-seeds positive")
    original, source = load_results(args.attack_dir, "final_experiment")
    aggregate(original)
    protocol = source["protocol"]
    validate_protocol(protocol)
    seeds = sorted(int(s) for s in original.seed.unique() if s >= args.seed_start)
    if args.n_seeds is not None:
        requested = list(range(args.seed_start, args.seed_start + args.n_seeds))
        if not set(requested).issubset(seeds):
            parser.error("Requested seeds are absent from the saved attack")
        seeds = requested
    if not seeds:
        parser.error("No saved seeds selected")
    data = binarize_adult_curated(load_adult_raw(ROOT / "data"))
    result_dir = experiment_result_dir("predictive_accuracy")
    if any(result_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {result_dir}")
    source_files = [
        args.attack_dir / "manifest.json",
        *sorted(args.attack_dir.glob(source["data"])),
    ]
    metadata = {
        "experiment": "predictive_accuracy",
        "data": "accuracy.csv",
        "attack_protocol": protocol,
        "utility_metric": "majority_vote",
        "tie_policy": "smallest_class",
        "attack_directory": str(args.attack_dir.resolve()),
        "seeds": seeds,
        "run": run_info(argv),
        "preprocessing": "historical full-table curated binarization (not training-only)",
        "release_identity": "reconstructed; tree count and joint-split statistics checked; historical hashes unavailable",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
        },
        "implementation_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "experiments/run_predictive_accuracy.py",
                "experiments/run_final_experiment.py",
                "src/rashomon/data.py",
                "src/rashomon/build.py",
                "src/rashomon/selection.py",
                "src/rashomon/publisher.py",
                "src/rashomon/utility.py",
                "uv.lock",
            )
        },
        "encoded_data_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(data.X, index=True).values.tobytes()
            + pd.util.hash_pandas_object(data.y, index=True).values.tobytes()
            + json.dumps(list(data.X.columns)).encode()
        ).hexdigest(),
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    rows, failures = [], []
    print(f"Writing utility results to {result_dir}", flush=True)
    for seed in seeds:
        try:
            print(
                f"[seed {seed}] rebuilding releases and evaluating holdout", flush=True
            )
            rows.extend(
                score_seed(seed, original[original.seed == seed], protocol, data)
            )
            pd.DataFrame(rows).to_csv(result_dir / "accuracy.csv", index=False)
            print(f"[seed {seed}] accuracy saved", flush=True)
        except Exception as exc:  # noqa: BLE001 -- isolate seed errors and exit nonzero below
            failures.append(
                {"seed": seed, "error": str(exc), "type": type(exc).__name__}
            )
            print(f"[seed {seed}] failed: {exc}", file=sys.stderr, flush=True)
    (result_dir / "outcome.json").write_text(
        json.dumps(
            {
                "failures": failures,
                "completed_seeds": sorted({r["seed"] for r in rows}),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Accuracy results -> {result_dir}", flush=True)
    print("Historical tree identity is not certified; see manifest.json.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
