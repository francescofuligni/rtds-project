"""Compute truth-aligned agreement across repeated solves.

This diagnostic uses evaluator ground truth, not attacker confidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from attack.conditions import condition_knowledge
from attack.metrics import match_rows
from attack.reconstruct import PublicRelease, reconstruct
from rashomon.data import BinarizedDataset, known_columns_for


@dataclass
class ConfidenceRow:
    row: int
    n_votes: int
    agreement: float
    correct: bool


def repeated_reconstruct_with_confidence(
    release: PublicRelease,
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
    """Return majority T agreement after evaluator matching.

    Each repeat must produce a candidate; this score does not establish
    uniqueness or calibrated person-level confidence.
    """
    X = data.X.to_numpy()
    col_index = {c: i for i, c in enumerate(data.X.columns)}
    target_cols = [col_index[c] for c in known_columns_for(data, target)]

    known_mask, known_values, known_cols = condition_knowledge(
        release, data, condition, attr_a, attr_b, target
    )

    n_rows = len(X)
    votes: list[list[tuple]] = [[] for _ in range(n_rows)]
    statuses = []

    for r in range(n_repeats):
        result = reconstruct(
            release,
            known_mask=known_mask,
            known_values=known_values,
            timeout=timeout,
            seed=base_seed + r,
            n_threads=n_threads,
        )
        statuses.append(result.status)
        if result.reconstructed is None:
            raise RuntimeError(
                f"diagnostic repeat {r}: solver {result.status}, no candidate"
            )
        matching = match_rows(
            X,
            result.reconstructed,
            exclude_cols=set(target_cols),
            known_cols=known_cols,
        )
        matched_target = result.reconstructed[matching.permutation][:, target_cols]
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
