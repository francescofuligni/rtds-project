"""The None/A/B/A+B background-knowledge ablation.

For a fixed data sample, fixed released trees, and fixed target T, build
the known-column mask for each of the four conditions and run the same
reconstruction attack under each -- the controlled ablation described in
the project's meeting notes (Sec. 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attack.metrics import reconstruction_error
from attack.reconstruct import ReconstructionResult, reconstruct
from rashomon.data import BinarizedDataset, known_columns_for

CONDITIONS = ["none", "A", "B", "A+B"]


def _group_columns(data: BinarizedDataset, attribute: str, col_index: dict[str, int]) -> list[int]:
    return [col_index[c] for c in known_columns_for(data, attribute)]


def known_mask_for_condition(data: BinarizedDataset, condition: str, attr_a: str, attr_b: str) -> np.ndarray:
    """(N, M) boolean mask: True where the attacker is given that
    column's true value for every row, under the given condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}, expected one of {CONDITIONS}")

    n, m = data.X.shape
    col_index = {c: i for i, c in enumerate(data.X.columns)}
    mask = np.zeros((n, m), dtype=bool)

    if condition in ("A", "A+B"):
        mask[:, _group_columns(data, attr_a, col_index)] = True
    if condition in ("B", "A+B"):
        mask[:, _group_columns(data, attr_b, col_index)] = True

    return mask


@dataclass
class ConditionResult:
    condition: str
    status: str
    duration: float
    reconstruction_error: float | None
    reconstructed: np.ndarray | None


def run_condition(
    condition: str,
    trees_branches,
    data: BinarizedDataset,
    attr_a: str,
    attr_b: str,
    target: str,
    *,
    timeout: float = 120.0,
    seed: int = 0,
    n_threads: int = 0,
) -> ConditionResult:
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    classes = sorted(data.y.unique().tolist())
    col_index = {c: i for i, c in enumerate(data.X.columns)}
    target_cols = _group_columns(data, target, col_index)

    known_mask = known_mask_for_condition(data, condition, attr_a, attr_b)
    known_values = np.where(known_mask, X, 0)

    result: ReconstructionResult = reconstruct(
        trees_branches, y, classes, n_features=X.shape[1],
        one_hot_groups=[_group_columns(data, g, col_index) for g in data.one_hot_groups],
        known_mask=known_mask, known_values=known_values,
        timeout=timeout, seed=seed, n_threads=n_threads,
    )

    if result.reconstructed is None:
        return ConditionResult(condition, result.status, result.duration, None, None)

    error, _perm = reconstruction_error(X, result.reconstructed, y, target_cols)
    return ConditionResult(condition, result.status, result.duration, error, result.reconstructed)
