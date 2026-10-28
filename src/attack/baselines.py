"""Model-free guesses for the sensitive target T.

Uniform guesses use the target's categories. Marginal guesses use target
frequencies from a disjoint holdout, without released trees. Both are
unconditional controls; they do not exploit A/B background knowledge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rashomon.data import BinarizedDataset, known_columns_for


def _group_columns(data: BinarizedDataset, attribute: str) -> list[int]:
    cols = known_columns_for(data, attribute)
    index = {c: i for i, c in enumerate(data.X.columns)}
    return [index[c] for c in cols]


def _fill_guesses(out: np.ndarray, group_cols: list[int], choice: np.ndarray) -> None:
    """Set out[row, group_cols[choice[row]]] = 1 for every row. For a
    two-valued attribute stored as a single indicator column (len(group_cols)
    == 1), `choice` must already be a 0/1 Bernoulli draw, not a column index
    -- there is no second column to pick between.
    """
    n_rows = out.shape[0]
    if len(group_cols) == 1:
        out[:, group_cols[0]] = choice
    else:
        out[np.arange(n_rows), np.array(group_cols)[choice]] = 1


def uniform_random_guess(
    n_rows: int, group_cols: list[int], n_features: int, seed: int
) -> np.ndarray:
    """Guess T uniformly at random among its possible categories, for
    every row; all other columns are left at 0 (unused by the metric)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_rows, n_features), dtype=int)
    n_choices = 2 if len(group_cols) == 1 else len(group_cols)
    choice = rng.integers(0, n_choices, size=n_rows)
    _fill_guesses(out, group_cols, choice)
    return out


def marginal_baseline(
    data: BinarizedDataset,
    attribute: str,
    n_rows: int,
    holdout_df: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """Guess T by sampling from its marginal distribution estimated on a
    holdout sample disjoint from the attacked rows (never passed to the
    Rashomon-set construction or the attack).
    """
    group_cols = _group_columns(data, attribute)
    group_names = [data.X.columns[i] for i in group_cols]
    probs = holdout_df[group_names].mean(axis=0).to_numpy()

    rng = np.random.default_rng(seed)
    out = np.zeros((n_rows, data.X.shape[1]), dtype=int)
    if len(group_cols) == 1:
        p = probs[0]  # P(indicator == 1), e.g. P(sex=Female)
        choice = (rng.random(n_rows) < p).astype(int)
    else:
        probs = probs / probs.sum()
        choice = rng.choice(len(group_cols), size=n_rows, p=probs)
    _fill_guesses(out, group_cols, choice)
    return out
