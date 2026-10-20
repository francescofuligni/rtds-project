"""Attacker-side, record-level confidence from per-tree reconstructions.

Each released tree is reconstructed on its own (its leaf counts only),
giving one candidate table per tree. A record is identified by its
content -- every feature value plus the label -- so candidates from
different trees can be compared without aligning them to the true table.
Everything here except `record_counts` applied to the true data uses only
what the attacker has: the released trees and (for the prior baseline) a
public sample from the same population.

For a record r with multiplicities c_1..c_m across the m candidates:
  presence(r)       fraction of candidates containing r at least once
  absolute(r)       sum_j c_j, total copies across all candidates
  occurrence(r, k)  fraction of candidates containing r at least k times

The absolute count decomposes into occurrence scores:
sum_j c_j = m * sum_k occurrence(r, k). Splitting it per copy gives one
claim per (record, k) -- "the data contains at least k copies of r" --
whose score lies in [0, 1] and can be read as a probability directly.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from attack.reconstruct import reconstruct

RecordKey = tuple[int, ...]


def record_counts(X: np.ndarray, y: np.ndarray) -> Counter:
    """Multiset of records, each record being (features..., label)."""
    return Counter(tuple(int(v) for v in row) + (int(label),) for row, label in zip(X, y))


def per_tree_candidates(
    trees_branches,
    y: np.ndarray,
    classes: list,
    n_features: int,
    one_hot_groups: list[list[int]],
    *,
    known_mask: np.ndarray | None = None,
    known_values: np.ndarray | None = None,
    timeout: float = 30.0,
    seed: int = 0,
    n_threads: int = 1,
) -> tuple[list[Counter | None], list[str]]:
    """Reconstruct one candidate table per tree, each from that tree alone.

    Returns each candidate as a record multiset (None where the solver
    found no table) and the solver status per tree.
    """
    candidates, statuses = [], []
    for branches in trees_branches:
        result = reconstruct(
            [branches], y, classes, n_features, one_hot_groups,
            known_mask=known_mask, known_values=known_values,
            timeout=timeout, seed=seed, n_threads=n_threads,
        )
        statuses.append(result.status)
        candidates.append(None if result.reconstructed is None else record_counts(result.reconstructed, y))
    return candidates, statuses


def draw_prior_table(public_X: np.ndarray, public_y: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A plausible random table: for each row, a public record with the same label."""
    table = np.empty((len(y), public_X.shape[1]), dtype=int)
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        pool = np.flatnonzero(public_y == label)
        table[rows] = public_X[rng.choice(pool, size=len(rows), replace=True)]
    return table


def sampled_candidates(
    trees_branches,
    y: np.ndarray,
    classes: list,
    n_features: int,
    one_hot_groups: list[list[int]],
    *,
    n_samples: int,
    public_X: np.ndarray | None = None,
    public_y: np.ndarray | None = None,
    known_mask: np.ndarray | None = None,
    known_values: np.ndarray | None = None,
    timeout: float = 30.0,
    seed: int = 0,
    n_threads: int = 1,
) -> tuple[list[Counter | None], list[str]]:
    """Several candidates from the joint release (all trees at once).

    With public data: prior-guided sampling. Each sample is the
    release-consistent table closest to a fresh random table drawn from the
    public sample, so cells the release pins down agree across samples and
    free cells vary the way the population does (perturb-and-MAP style).
    Without public data: only the solver's random seed varies.
    """
    rng = np.random.default_rng(seed)
    candidates, statuses = [], []
    for s in range(n_samples):
        target = None if public_X is None else draw_prior_table(public_X, public_y, y, rng)
        result = reconstruct(
            trees_branches, y, classes, n_features, one_hot_groups,
            known_mask=known_mask, known_values=known_values,
            timeout=timeout, seed=seed * 1000 + s, n_threads=n_threads, target=target,
        )
        statuses.append(result.status)
        candidates.append(None if result.reconstructed is None else record_counts(result.reconstructed, y))
    return candidates, statuses


def support_matrix(candidates: list[Counter | None]) -> dict[RecordKey, np.ndarray]:
    """Per record, its multiplicity in each successful candidate (0 if absent)."""
    solved = [c for c in candidates if c is not None]
    records = sorted(set().union(*solved)) if solved else []
    return {r: np.array([c[r] for c in solved], dtype=int) for r in records}


@dataclass
class RecordClaim:
    """'The training data contains at least k copies of record.'"""

    record: RecordKey
    k: int
    presence: float
    absolute: int
    occurrence: float


def record_claims(support: dict[RecordKey, np.ndarray]) -> list[RecordClaim]:
    """One claim per (record, k) for k up to the record's largest multiplicity."""
    claims = []
    for record, mult in support.items():
        presence = float(np.mean(mult >= 1))
        absolute = int(mult.sum())
        for k in range(1, int(mult.max()) + 1):
            claims.append(RecordClaim(record, k, presence, absolute, float(np.mean(mult >= k))))
    return claims


def enumerate_feature_vectors(n_features: int, one_hot_groups: list[list[int]]) -> list[tuple[int, ...]]:
    """Every feature vector the schema allows (each one-hot group picks one column)."""
    grouped = {i for group in one_hot_groups for i in group}
    free = [i for i in range(n_features) if i not in grouped]
    vectors = []
    for free_bits in itertools.product([0, 1], repeat=len(free)):
        for picks in itertools.product(*one_hot_groups):
            v = [0] * n_features
            for i, bit in zip(free, free_bits):
                v[i] = bit
            for i in picks:
                v[i] = 1
            vectors.append(tuple(v))
    return vectors


class RecordPrior:
    """No-trees baseline. Records with label c are drawn i.i.d. from the
    public sample's smoothed feature distribution given c, and the attacker
    knows how many of the n rows carry each label (the trees reveal this),
    so P(at least k copies of r) = P(Binomial(n_c, p(r | c)) >= k).

    alpha is the add-alpha smoothing per possible feature vector. With a few
    hundred public rows per label and 320 possible vectors, alpha=1 moves so
    much mass to unseen records that the baseline turns underconfident (ECE
    0.063 over 20 seeds vs 0.010 at alpha=0.1), which would handicap it.
    """

    def __init__(self, public_X, public_y, class_totals: dict, n_feature_vectors: int, alpha: float = 0.1):
        self.class_totals = class_totals
        self.public_counts = record_counts(public_X, public_y)
        self.label_totals = Counter(int(v) for v in public_y)
        self.n_feature_vectors = n_feature_vectors
        self.alpha = alpha

    def probability(self, record: RecordKey, k: int) -> float:
        label = record[-1]
        n_c = self.class_totals.get(label, 0)
        p = (self.public_counts[record] + self.alpha) / (self.label_totals[label] + self.alpha * self.n_feature_vectors)
        return float(binom.sf(k - 1, n_c, p))


def best_table_claims(
    scores: dict[tuple[RecordKey, int], float],
    class_totals: dict,
    tiebreak: dict[tuple[RecordKey, int], float],
) -> list[tuple[RecordKey, int, float]]:
    """The attacker's output table: per label c, the n_c highest-scoring
    claims (record, k) with that label, ties broken by `tiebreak`.

    Each selected claim is one output row; its score is that row's
    confidence. Scores must not increase with k for the table to be valid,
    which holds for every score here.
    """
    table = []
    for label, n_c in class_totals.items():
        options = [key for key in scores if key[0][-1] == label]
        options.sort(key=lambda key: (scores[key], tiebreak.get(key, 0.0)), reverse=True)
        table.extend((record, k, scores[(record, k)]) for record, k in options[:n_c])
    return table
