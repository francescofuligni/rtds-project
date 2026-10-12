"""Row matching and reconstruction-error scoring.

The CP-SAT model in reconstruct.py has no notion of row identity beyond
whatever is pinned via `known_mask`/`known_values` (plus the label y,
which is fixed per slot in every condition). Under partial background
knowledge, many reconstructed rows sharing a label are interchangeable
as far as the solver's constraints are concerned, so before scoring the
sensitive target T we match reconstructed rows to true rows via the
Hungarian algorithm, using Hamming distance over all columns *except* T.

Excluding T from the matching cost is deliberate (see Francesco's
review, Sec. 3.3): matching on T itself would let the evaluator's
knowledge of the true values improve T's measured accuracy in a way the
attacker has no access to. This also means the "None" condition's match
is often close to arbitrary within a label group -- that's the honest
reflection of having no identifying information, not a bug.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_rows(true_X: np.ndarray, reconstructed_X: np.ndarray, y: np.ndarray,
                exclude_cols: set[int]) -> np.ndarray:
    """Return `perm` of length N such that reconstructed_X[perm[j]] is the
    row matched to true_X[j]. Matching is done separately within each
    y-label group (y is always known/fixed per slot) using Hamming
    distance over all columns not in exclude_cols.
    """
    n = len(true_X)
    cols = [i for i in range(true_X.shape[1]) if i not in exclude_cols]
    perm = np.full(n, -1, dtype=int)

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        true_sub = true_X[np.ix_(idx, cols)]
        recon_sub = reconstructed_X[np.ix_(idx, cols)]
        cost = (true_sub[:, None, :] != recon_sub[None, :, :]).sum(axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            perm[idx[r]] = idx[c]

    assert (perm >= 0).all(), "every row must be matched"
    return perm


def reconstruction_error(true_X: np.ndarray, reconstructed_X: np.ndarray, y: np.ndarray,
                          target_cols: list[int]) -> tuple[float, np.ndarray]:
    """Match rows (excluding target_cols from the matching cost), then
    return the fraction of rows where the matched reconstruction disagrees
    with the truth on target_cols (a one-hot group collapses to one
    categorical comparison), plus the matching permutation used.
    """
    perm = match_rows(true_X, reconstructed_X, y, exclude_cols=set(target_cols))
    matched = reconstructed_X[perm]
    mismatches = np.any(matched[:, target_cols] != true_X[:, target_cols], axis=1)
    return float(mismatches.mean()), perm
