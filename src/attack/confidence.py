"""Per-row confidence for the reconstruction attack.

The CP-SAT solver has no notion of probability -- it returns *a* table
consistent with the released leaf counts, not a calibrated distribution
over compatible tables the way Ferry et al.'s (2024a) entropy-based
method does. We approximate a confidence signal by re-solving the same
(seed, m, condition) cell multiple times with different solver seeds
(different search order -> potentially different feasible solution when
more than one exists) and measuring, per true row, how often the
row-matched reconstruction agrees on T across repeats. A row where every
repeat agrees is one the model's constraints pin down regardless of
search order; a row that flips is one where multiple values remain
compatible with the released trees.

This needs the row-matching to be done *within* each repeat (matching a
single solve's output to the truth), not once across repeats -- each
solve's reconstructed table has its own, independent row ordering (see
metrics.py), so "row k of solve 3" and "row k of solve 7" are not
comparable without first aligning each to the true rows separately.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from attack.conditions import known_mask_for_condition, validate_knowledge_roles
from attack.metrics import match_rows
from attack.reconstruct import reconstruct
from rashomon.data import BinarizedDataset, known_columns_for


@dataclass
class ConfidenceRow:
    row: int
    n_votes: int
    agreement: float
    correct: bool


def repeated_reconstruct_with_confidence(
    trees_branches,
    data: BinarizedDataset,
    attr_a: str,
    attr_b: str,
    target: str,
    condition: str,
    *,
    n_repeats: int = 7,
    timeout: float = 120.0,
    base_seed: int = 0,
    n_threads: int = 1,
) -> tuple[list[ConfidenceRow], list[str]]:
    """Solve the same condition n_repeats times with different solver
    seeds; for each true row, collect the row-matched reconstructed T
    value from every repeat that solved, and return the majority value's
    agreement rate (the row's confidence score) and whether that majority
    value is actually correct. Also returns the list of solve statuses.
    """
    validate_knowledge_roles(data, attr_a, attr_b, target)
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    classes = sorted(data.y.unique().tolist())
    col_index = {c: i for i, c in enumerate(data.X.columns)}
    target_cols = [col_index[c] for c in known_columns_for(data, target)]

    known_mask = known_mask_for_condition(data, condition, attr_a, attr_b)
    known_values = np.where(known_mask, X, 0)
    one_hot_groups = [
        [col_index[c] for c in cols] for cols in data.one_hot_groups.values()
    ]

    n_rows = len(X)
    votes: list[list[tuple]] = [[] for _ in range(n_rows)]
    statuses = []

    for r in range(n_repeats):
        result = reconstruct(
            trees_branches,
            y,
            classes,
            n_features=X.shape[1],
            one_hot_groups=one_hot_groups,
            known_mask=known_mask,
            known_values=known_values,
            timeout=timeout,
            seed=base_seed + r,
            n_threads=n_threads,
        )
        statuses.append(result.status)
        if result.reconstructed is None:
            continue
        perm = match_rows(X, result.reconstructed, y, exclude_cols=set(target_cols))
        matched_target = result.reconstructed[perm][:, target_cols]
        for j in range(n_rows):
            votes[j].append(tuple(matched_target[j]))

    rows = []
    for j in range(n_rows):
        if not votes[j]:
            continue
        counts = Counter(votes[j])
        mode_value, mode_count = counts.most_common(1)[0]
        agreement = mode_count / len(votes[j])
        true_value = tuple(X[j, target_cols])
        rows.append(
            ConfidenceRow(
                row=j,
                n_votes=len(votes[j]),
                agreement=agreement,
                correct=(mode_value == true_value),
            )
        )

    return rows, statuses
