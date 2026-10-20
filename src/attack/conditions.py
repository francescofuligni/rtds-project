"""Build None/A/B/A+B knowledge and evaluate reconstructed features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attack.metrics import MatchingResult, reconstruction_error
from attack.reconstruct import PublicRelease, reconstruct
from rashomon.data import BinarizedDataset, known_columns_for

CONDITIONS = ["none", "A", "B", "A+B"]


def _group_columns(data: BinarizedDataset, attribute: str) -> list[int]:
    names = data.X.columns.to_list()
    return [names.index(c) for c in known_columns_for(data, attribute)]


def known_mask_for_condition(
    data: BinarizedDataset, condition: str, attr_a: str, attr_b: str, *, target: str
) -> np.ndarray:
    """Select known feature groups, excluding T and overlapping attributes."""
    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition {condition!r}, expected one of {CONDITIONS}"
        )
    if len({attr_a, attr_b, target}) != 3:
        raise ValueError("A, B and T must be distinct attributes")
    a, b, t = (set(_group_columns(data, attr)) for attr in (attr_a, attr_b, target))
    if a & b or a & t or b & t:
        raise ValueError("A, B and T encoding groups must be disjoint")
    mask = np.zeros(data.X.shape, dtype=bool)
    if condition in ("A", "A+B"):
        mask[:, sorted(a)] = True
    if condition in ("B", "A+B"):
        mask[:, sorted(b)] = True
    return mask


def condition_knowledge(
    release: PublicRelease,
    data: BinarizedDataset,
    condition: str,
    attr_a: str,
    attr_b: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Prepare known cells and zero all undisclosed values."""
    if (
        data.X.shape != (release.n_rows, release.n_features)
        or tuple(data.X.columns) != release.feature_names
    ):
        raise ValueError("evaluation/knowledge schema must match the public release")
    mask = known_mask_for_condition(data, condition, attr_a, attr_b, target=target)
    values = np.where(mask, data.X.to_numpy(), 0)
    return mask, values, np.flatnonzero(mask.any(axis=0)).tolist()


@dataclass
class ConditionResult:
    condition: str
    status: str
    duration: float
    reconstruction_error: float | None
    reconstructed: np.ndarray | None
    matching: MatchingResult | None = None


def run_condition(
    condition: str,
    release: PublicRelease,
    data: BinarizedDataset,
    attr_a: str,
    attr_b: str,
    target: str,
    *,
    timeout: float = 120.0,
    seed: int = 0,
    n_threads: int = 0,
) -> ConditionResult:
    """Solve from public inputs, then evaluate against private truth."""
    mask, values, known_cols = condition_knowledge(
        release, data, condition, attr_a, attr_b, target
    )
    result = reconstruct(
        release,
        known_mask=mask,
        known_values=values,
        timeout=timeout,
        seed=seed,
        n_threads=n_threads,
    )
    if result.reconstructed is None:
        return ConditionResult(condition, result.status, result.duration, None, None)
    if not np.array_equal(result.reconstructed[mask], values[mask]):
        raise RuntimeError("candidate violates supplied knowledge")
    error, matching = reconstruction_error(
        data.X.to_numpy(),
        result.reconstructed,
        _group_columns(data, target),
        known_cols=known_cols,
    )
    return ConditionResult(
        condition, result.status, result.duration, error, result.reconstructed, matching
    )
