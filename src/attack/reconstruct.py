"""Reconstruct binary tables from public trees and exact class counts.

CP-SAT feasibility adaptation of DRAFT (Ferry et al., 2024).
The publisher adds aggregate counts; true per-row Y is not attacker input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

PROTOCOL_VERSION = "no_y_v1"
RELEASE_POLICY = "selected_trees_with_publisher_exact_class_counts"
Branches = list[tuple[list[tuple[int, float]], list[int]]]


def _walk_leaves(node: dict, path: list[tuple[int, bool]]):
    if "prediction" in node:
        yield path, node["prediction"]
    else:
        feat = node["feature"]
        yield from _walk_leaves(node["true"], path + [(feat, True)])
        yield from _walk_leaves(node["false"], path + [(feat, False)])


def extract_branches(
    tree_source: dict, X: np.ndarray, y: np.ndarray, classes: list
) -> Branches:
    """Compute leaf paths and class counts on the publisher side."""
    branches = []
    for path, _prediction in _walk_leaves(tree_source, []):
        mask = np.ones(len(X), dtype=bool)
        splits = []
        for feat, taken_true in path:
            mask &= (X[:, feat] == 1) if taken_true else (X[:, feat] == 0)
            splits.append(((feat + 1) if taken_true else -(feat + 1), 0.5))
        counts = [int(np.sum((y == c) & mask)) for c in classes]
        branches.append((splits, counts))
    return branches


@dataclass(frozen=True)
class PublicRelease:
    """Public tree sources, aggregate counts and feature schema."""

    trees: list[dict]
    trees_branches: list[Branches]
    classes: tuple[int | str, ...]
    n_rows: int
    feature_names: tuple[str, ...]
    one_hot_groups: tuple[tuple[int, ...], ...]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def release_id(self) -> str:
        payload = {
            "trees": self.trees,
            "branches": self.trees_branches,
            "classes": self.classes,
            "n_rows": self.n_rows,
            "features": self.feature_names,
            "groups": self.one_hot_groups,
            "policy": RELEASE_POLICY,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda value: value.item(),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def prepare_release(
    trees: list[dict],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    one_hot_groups: list[list[int]],
) -> PublicRelease:
    """Copy selected trees and publish their exact class counts."""
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or X.shape[1] != len(feature_names):
        raise ValueError("training arrays and feature schema must have matching shapes")
    if not np.isin(X, [0, 1]).all():
        raise ValueError("features must be binary")
    classes = tuple(np.unique(y).tolist())
    public_trees = copy.deepcopy(trees)
    release = PublicRelease(
        public_trees,
        [extract_branches(tree, X, y, list(classes)) for tree in public_trees],
        classes,
        len(X),
        tuple(feature_names),
        tuple(tuple(group) for group in one_hot_groups),
    )
    validate_release(release)
    for group in release.one_hot_groups:
        if not (X[:, group].sum(axis=1) == 1).all():
            raise ValueError("training rows violate the one-hot schema")
    return release


def validate_release(release: PublicRelease) -> None:
    if not isinstance(release, PublicRelease):
        raise TypeError("reconstruct requires a PublicRelease, not training labels")
    if release.n_rows <= 0 or not release.feature_names or not release.classes:
        raise ValueError("release must have positive dimensions and a class domain")
    if len(set(release.feature_names)) != release.n_features or len(
        set(release.classes)
    ) != len(release.classes):
        raise ValueError("feature names and classes must be unique")
    if not release.trees or len(release.trees) != len(release.trees_branches):
        raise ValueError("release must contain matching trees and branch lists")
    grouped = set()
    for group in release.one_hot_groups:
        if not group or len(set(group)) != len(group):
            raise ValueError("one-hot groups must be nonempty with unique columns")
        if any(i < 0 or i >= release.n_features or i in grouped for i in group):
            raise ValueError("one-hot groups contain overlapping or invalid columns")
        grouped.update(group)
    reference_totals = None
    for tree, branches in zip(release.trees, release.trees_branches):
        paths = [
            [((f + 1) if taken else -(f + 1), 0.5) for f, taken in path]
            for path, _ in _walk_leaves(tree, [])
        ]
        if not branches or paths != [splits for splits, _ in branches]:
            raise ValueError("released paths must match the selected tree structure")
        totals = np.zeros(len(release.classes), dtype=int)
        for splits, counts in branches:
            if any(
                abs(f) < 1 or abs(f) > release.n_features or t != 0.5 for f, t in splits
            ):
                raise ValueError("invalid binary split")
            if len(counts) != len(release.classes) or any(
                not isinstance(c, (int, np.integer)) or c < 0 for c in counts
            ):
                raise ValueError(
                    "class counts must be nonnegative integers in class order"
                )
            totals += counts
        if totals.sum() != release.n_rows:
            raise ValueError("each tree must count every row exactly once")
        if reference_totals is not None and not np.array_equal(
            totals, reference_totals
        ):
            raise ValueError("class totals must agree across non-bootstrap trees")
        reference_totals = totals


@dataclass
class ReconstructionResult:
    status: str
    duration: float
    reconstructed: np.ndarray | None
    reconstructed_labels: np.ndarray | None = None


def reconstruct(
    release: PublicRelease,
    *,
    known_mask: np.ndarray | None = None,
    known_values: np.ndarray | None = None,
    timeout: float = 60.0,
    n_threads: int = 0,
    seed: int = 0,
    verbosity: bool = False,
) -> ReconstructionResult:
    """Find feature rows and unknown classes consistent with the release.

    Known cells constrain row slots; other slots remain exchangeable.
    """
    start = time.perf_counter()
    validate_release(release)
    n, m, n_classes = release.n_rows, release.n_features, len(release.classes)
    if timeout <= 0 or n_threads < 0:
        raise ValueError("timeout must be positive and worker count nonnegative")
    if known_mask is not None:
        if known_values is None:
            raise ValueError("known_values is required when known_mask is provided")
        if known_mask.shape != (n, m) or known_mask.dtype != bool:
            raise ValueError(
                "known_mask must be a boolean array matching release dimensions"
            )
        if (
            known_values.shape != (n, m)
            or not np.isin(known_values[known_mask], [0, 1]).all()
        ):
            raise ValueError(
                "known_values must match dimensions and contain binary known cells"
            )
    elif known_values is not None:
        raise ValueError("known_mask is required when known_values is provided")
    model = cp_model.CpModel()
    x = [[model.new_bool_var(f"x_{i}_{j}") for j in range(m)] for i in range(n)]
    z = [
        [model.new_bool_var(f"class_{i}_{c}") for c in range(n_classes)]
        for i in range(n)
    ]
    for i in range(n):
        model.add(sum(z[i]) == 1)
        for group in release.one_hot_groups:
            model.add(sum(x[i][j] for j in group) == 1)
        if known_mask is not None:
            assert known_values is not None
            for j in np.flatnonzero(known_mask[i]):
                model.add(x[i][j] == int(known_values[i, j]))
    for tree_index, branches in enumerate(release.trees_branches):
        assignments = [[] for _ in range(n)]
        for leaf_index, (splits, counts) in enumerate(branches):
            for c, count in enumerate(counts):
                captured = []
                if count:
                    for i in range(n):
                        a = model.new_bool_var(
                            f"capture_{tree_index}_{leaf_index}_{c}_{i}"
                        )
                        assignments[i].append(a)
                        captured.append(a)
                        model.add(z[i][c] == 1).OnlyEnforceIf(a)
                        for signed_feature, _threshold in splits:
                            model.add(
                                x[i][abs(signed_feature) - 1] == int(signed_feature > 0)
                            ).OnlyEnforceIf(a)
                model.add(sum(captured) == count)
        for captures in assignments:
            model.add(sum(captures) == 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_workers = n_threads
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = verbosity
    status = solver.Solve(model)
    candidate = labels = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        candidate = np.array([[solver.Value(v) for v in row] for row in x], dtype=int)
        labels = np.array(
            [
                release.classes[
                    next(c for c in range(n_classes) if solver.Value(z[i][c]))
                ]
                for i in range(n)
            ]
        )
    return ReconstructionResult(
        solver.StatusName(status), time.perf_counter() - start, candidate, labels
    )
