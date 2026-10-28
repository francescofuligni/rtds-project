"""Shared release-grid configuration for paired experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from rashomon.selection import DEFAULT_RANDOM_TREE_POOL

# CP-SAT workers run different search strategies in parallel.
# Record their count in each protocol to keep solve budgets comparable.
DEFAULT_SOLVER_THREADS = 4
# Prior-guided distance objectives use a larger search portfolio.
RECORD_CONFIDENCE_SOLVER_THREADS = 8


def add_tree_selection_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tree-selection",
        choices=("random", "sequential"),
        default="random",
        help="release random nested prefixes (default) or TreeFARMS enumeration prefixes",
    )


def add_threads_argument(
    parser: argparse.ArgumentParser, default: int = DEFAULT_SOLVER_THREADS
) -> None:
    parser.add_argument(
        "--threads",
        type=int,
        default=default,
        help=f"CP-SAT search workers per solve (default {default})",
    )


# Keep the default sample small enough for local reconstruction runs.
@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 100
    k: int = 15
    k_steps: int | None = None
    m_max: int = 200
    attr_a: str = "marital-status"
    attr_b: str = "education"
    include_all_trees: bool = False
    tree_selection: str = "random"

    def __post_init__(self):
        if self.tree_selection not in ("random", "sequential"):
            raise ValueError("tree-selection must be random or sequential")
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.m_max < 1:
            raise ValueError("m-max must be positive")
        if self.k_steps is not None and not 1 <= self.k_steps <= self.m_max:
            raise ValueError("k-steps must be between 1 and m-max")
        if self.k_steps is None and self.k < 2:
            raise ValueError("k must be at least 2 to include both endpoints")
        allowed = {
            "marital-status",
            "education",
            "age",
            "hours-per-week",
            "capital-gain",
        }
        if self.attr_a not in allowed or self.attr_b not in allowed:
            raise ValueError(
                f"A/B must be curated attributes other than income/sex: {sorted(allowed)}"
            )
        if self.attr_a == self.attr_b:
            raise ValueError("A and B must be distinct")

    @property
    def m_values(self) -> tuple[int, ...]:
        """Return the selected integer grid, retaining both endpoints."""
        if self.k_steps is not None:
            values = list(range(1, self.m_max + 1, self.k_steps))
            if values[-1] != self.m_max:
                values.append(self.m_max)
            return tuple(values)
        return tuple(
            int(m) for m in np.unique(np.rint(np.geomspace(1, self.m_max, self.k)))
        )

    @property
    def tree_pool(self) -> int:
        return max(DEFAULT_RANDOM_TREE_POOL, self.m_max)

    def release_protocol(self) -> dict:
        return {
            "m_schedule": "linear" if self.k_steps is not None else "geometric",
            "m_min": 1,
            "m_max": self.m_max,
            "k": self.k if self.k_steps is None else None,
            "k_steps": self.k_steps,
            "m_values": list(self.m_values),
            "k_effective": len(self.m_values),
            "m_rounding": (
                "nearest_ties_to_even_unique"
                if self.k_steps is None
                else "integer_step_with_max_endpoint"
            ),
            "include_all_trees": self.include_all_trees,
        }


def add_configuration_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_n: int = ExperimentConfig.n,
    default_k: int = ExperimentConfig.k,
) -> None:
    add_tree_selection_argument(parser)
    parser.add_argument(
        "--n", type=int, default=default_n, help="private training rows"
    )
    parser.add_argument("--attr-a", default="marital-status")
    parser.add_argument("--attr-b", default="education")
    all_trees = parser.add_mutually_exclusive_group()
    all_trees.add_argument(
        "--all-trees",
        dest="all_trees",
        action="store_true",
        help="append the full per-seed Rashomon set, beyond m-max",
    )
    all_trees.add_argument("--no-all-trees", dest="all_trees", action="store_false")
    parser.set_defaults(all_trees=False)
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument(
        "--k",
        type=int,
        help="number of geometric release points before rounding/deduplication (>=2)",
    )
    grid.add_argument(
        "--k-steps",
        type=int,
        help="linear release increment; must be in [1, m-max]",
    )
    parser.add_argument(
        "--m-max",
        type=int,
        default=ExperimentConfig.m_max,
        help=f"maximum release size (default: {ExperimentConfig.m_max})",
    )
    parser.set_defaults(default_k=default_k)


def configuration_from_args(args) -> ExperimentConfig:
    m_max = args.m_max
    return ExperimentConfig(
        n=args.n,
        k=args.k if args.k is not None else args.default_k,
        k_steps=args.k_steps,
        m_max=m_max,
        attr_a=args.attr_a,
        attr_b=args.attr_b,
        include_all_trees=args.all_trees,
        tree_selection=args.tree_selection,
    )
