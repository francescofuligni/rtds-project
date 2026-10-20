"""Dataset reconstruction attack against a released set of TreeFARMS trees.

This is a from-scratch reimplementation of the CP-SAT formulation in
Ferry et al. (2024), "Trained Random Forests Completely Reveal your
Dataset" (DRAFT, https://github.com/vidalt/DRAFT, MIT licensed), adapted
to two things upstream DRAFT doesn't support out of the box:

  1. Trees come from TreeFARMS's native recursive-dict representation
     (`{"feature": i, "true": ..., "false": ...}` / `{"prediction": c}`)
     instead of sklearn RandomForestClassifier.estimators_. Since
     TreeFARMS only operates on pre-binarized data, every feature is a
     plain boolean, so the ordinal/numerical-attribute machinery
     upstream DRAFT needs is not required here.
  2. The attacker can be given exact values for a subset of feature
     columns, for every training row (background knowledge A/B in the
     None/A/B/A+B ablation), by fixing those CP-SAT variables before
     solving.

Threat model: the attacker is assumed to know the training labels y (as
in all reviewed reconstruction papers -- the released trees predict y,
so y is public) and the released trees' exact per-leaf, per-class
training-example counts (the "leaf counts" that make a published tree a
small table of statistics). What the attacker does NOT get is the
ground-truth feature matrix; `extract_branches` is the one place that
touches it, to compute what the released artifact would contain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model


def _walk_leaves(node: dict, path: list[tuple[int, bool]]):
    """Yield (path, prediction) for every leaf of a TreeFARMS tree.source dict.
    path is a list of (feature_index, taken_true_branch)."""
    if "prediction" in node:
        yield path, node["prediction"]
    else:
        feat = node["feature"]
        yield from _walk_leaves(node["true"], path + [(feat, True)])
        yield from _walk_leaves(node["false"], path + [(feat, False)])


def extract_branches(tree_source: dict, X: np.ndarray, y: np.ndarray, classes: list) -> list:
    """Compute, for one tree, the list of (splits, per_class_counts)
    branches a released tree-with-leaf-counts would expose.

    splits: list of (signed_feature_id, threshold) pairs along the leaf's
        root path, 1-indexed and signed as in upstream DRAFT: positive
        means the feature's "true" (==1) branch was taken, negative means
        "false" (==0). threshold is always 0.5 since every feature is boolean.
    per_class_counts: number of rows of (X, y) landing in that leaf, per
        class in `classes`.

    This is the only function that touches ground-truth X -- it plays
    the role of "the model's author publishing the tree with its leaf
    statistics", not the attacker.
    """
    branches = []
    for path, _prediction in _walk_leaves(tree_source, []):
        mask = np.ones(len(X), dtype=bool)
        splits = []
        for feat, taken_true in path:
            mask &= (X[:, feat] == 1) if taken_true else (X[:, feat] == 0)
            signed = (feat + 1) if taken_true else -(feat + 1)
            splits.append((signed, 0.5))
        counts = [int(np.sum((y == c) & mask)) for c in classes]
        branches.append((splits, counts))
    return branches


@dataclass
class ReconstructionResult:
    status: str
    duration: float
    reconstructed: np.ndarray | None  # shape (N, M), None if infeasible/no solution found


def reconstruct(
    trees_branches: list[list[tuple[list[tuple[int, float]], list[int]]]],
    y: np.ndarray,
    classes: list,
    n_features: int,
    one_hot_groups: list[list[int]],
    *,
    known_mask: np.ndarray | None = None,
    known_values: np.ndarray | None = None,
    timeout: float = 60.0,
    n_threads: int = 0,
    seed: int = 0,
    verbosity: bool = False,
    target: np.ndarray | None = None,
) -> ReconstructionResult:
    """Solve the CP-SAT reconstruction model.

    trees_branches: per-tree list of branches, as produced by extract_branches.
    y: shape (N,) true labels (assumed known to the attacker).
    classes: ordered list of class labels matching the per-class count order
        used in trees_branches.
    n_features: M, total number of (boolean) feature columns.
    one_hot_groups: list of column-index groups that must each sum to 1
        (categorical attributes' one-hot encoding).
    known_mask / known_values: optional (N, M) arrays. Where known_mask is
        True, x_vars[k][i] is fixed to known_values[k][i] -- this is the
        attacker's background knowledge (None/A/B/A+B conditions).
    target: optional (N, M) 0/1 array. If given, the solver returns the
        consistent table with minimum Hamming distance to it (within the
        timeout) instead of an arbitrary one, so cells the release leaves
        free take the target's values rather than solver defaults.
    """
    start = time.time()
    N = len(y)
    M = n_features

    model = cp_model.CpModel()

    x_vars = [[model.NewBoolVar(f"x_{k}_{i}") for i in range(M)] for k in range(N)]

    if known_mask is not None:
        for k in range(N):
            for i in range(M):
                if known_mask[k, i]:
                    model.Add(x_vars[k][i] == int(known_values[k, i]))

    for group in one_hot_groups:
        for k in range(N):
            model.Add(cp_model.LinearExpr.Sum([x_vars[k][i] for i in group]) == 1)

    for all_branches_t in trees_branches:
        examples_capts = [[] for _ in range(N)]
        for branch_nb, (splits, class_counts) in enumerate(all_branches_t):
            branch_vars_c = [[] for _ in classes]
            for k in range(N):
                c_idx = classes.index(y[k])
                if class_counts[c_idx] > 0:
                    capt_var = model.NewBoolVar(f"branch_capt_{c_idx}_{branch_nb}_{k}")
                    branch_vars_c[c_idx].append(capt_var)
                    examples_capts[k].append(capt_var)
                    for signed_feat, _threshold in splits:
                        feature_id = abs(signed_feat) - 1
                        if signed_feat > 0:
                            model.Add(x_vars[k][feature_id] == 1).OnlyEnforceIf(capt_var)
                        else:
                            model.Add(x_vars[k][feature_id] == 0).OnlyEnforceIf(capt_var)
            for c_idx in range(len(classes)):
                model.Add(cp_model.LinearExpr.Sum(branch_vars_c[c_idx]) == int(class_counts[c_idx]))
        for k in range(N):
            if examples_capts[k]:
                model.Add(cp_model.LinearExpr.Sum(examples_capts[k]) == 1)

    if target is not None:
        mismatches = []
        for k in range(N):
            for i in range(M):
                model.AddHint(x_vars[k][i], int(target[k, i]))
                mismatches.append(1 - x_vars[k][i] if target[k, i] else x_vars[k][i])
        model.Minimize(cp_model.LinearExpr.Sum(mismatches))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_workers = n_threads
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = verbosity

    status = solver.Solve(model)
    duration = time.time() - start

    status_name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        reconstructed = np.array([[solver.Value(x_vars[k][i]) for i in range(M)] for k in range(N)])
    else:
        reconstructed = None

    return ReconstructionResult(status=status_name, duration=duration, reconstructed=reconstructed)
