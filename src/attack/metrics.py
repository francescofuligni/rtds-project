"""Match reconstructed rows and score T using evaluator ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

MATCHING_POLICY = "global_non_target_bit_hamming_known_groups"
TIE_POLICY = "scipy_assignment_fixed_input_order_no_target_tiebreak"


@dataclass
class MatchingResult:
    permutation: np.ndarray
    total_cost: int
    ambiguous: bool


def match_rows(
    true_X: np.ndarray,
    reconstructed_X: np.ndarray,
    exclude_cols: set[int],
    *,
    known_cols: list[int] | None = None,
) -> MatchingResult:
    """Find a minimum-cost bijection compatible with known profiles.

    Excluded columns do not affect assignment or tie-breaking. Ambiguity
    means that another assignment has the same optimal total cost.
    """
    true_X, reconstructed_X = np.asarray(true_X), np.asarray(reconstructed_X)
    if true_X.ndim != 2 or true_X.shape != reconstructed_X.shape or not len(true_X):
        raise ValueError("matching requires equal, nonempty feature tables")
    n, m = true_X.shape
    known_cols = [] if known_cols is None else list(known_cols)
    if any(c < 0 or c >= m for c in set(known_cols) | exclude_cols):
        raise ValueError("matching columns are outside the schema")
    if set(known_cols) & exclude_cols:
        raise ValueError("target columns cannot be known matching columns")
    cols = [j for j in range(m) if j not in exclude_cols]
    cost = (
        (true_X[:, None, cols] != reconstructed_X[None, :, cols])
        .sum(axis=2)
        .astype(float)
    )
    if known_cols:
        compatible = (
            true_X[:, None, known_cols] == reconstructed_X[None, :, known_cols]
        ).all(axis=2)
        cost[~compatible] = np.inf
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ValueError as error:
        raise ValueError(
            "no bijection is compatible with the known feature profiles"
        ) from error
    perm = np.empty(n, dtype=int)
    perm[row_ind] = col_ind
    optimum = int(cost[row_ind, col_ind].sum())
    ambiguous = False
    for row, col in zip(row_ind, col_ind):
        original = cost[row, col]
        cost[row, col] = np.inf
        try:
            alt_rows, alt_cols = linear_sum_assignment(cost)
            if cost[alt_rows, alt_cols].sum() == optimum:
                ambiguous = True
        except ValueError:
            pass  # No complete alternative assignment after removing this edge.
        finally:
            cost[row, col] = original
        if ambiguous:
            break
    return MatchingResult(perm, optimum, ambiguous)


def target_error(
    true_X: np.ndarray,
    candidate: np.ndarray,
    target_cols: list[int],
    permutation: np.ndarray,
) -> float:
    """Score categorical T under an existing matching permutation."""
    if not target_cols or true_X.shape != candidate.shape or not len(true_X):
        raise ValueError(
            "target scoring requires equal nonempty tables and target columns"
        )
    if any(c < 0 or c >= true_X.shape[1] for c in target_cols):
        raise ValueError("target columns are outside the schema")
    if permutation.shape != (len(true_X),) or not np.array_equal(
        np.sort(permutation), np.arange(len(true_X))
    ):
        raise ValueError("scoring requires a bijective permutation")
    return float(
        np.any(
            candidate[permutation][:, target_cols] != true_X[:, target_cols], axis=1
        ).mean()
    )


def reconstruction_error(
    true_X: np.ndarray,
    reconstructed_X: np.ndarray,
    target_cols: list[int],
    *,
    known_cols: list[int] | None = None,
) -> tuple[float, MatchingResult]:
    matching = match_rows(
        true_X, reconstructed_X, set(target_cols), known_cols=known_cols
    )
    return target_error(
        true_X, reconstructed_X, target_cols, matching.permutation
    ), matching
