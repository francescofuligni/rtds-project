"""Generate uniform and holdout-marginal target guesses without trees."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rashomon.data import BinarizedDataset, known_columns_for


def _group_columns(data: BinarizedDataset, attribute: str) -> list[int]:
    cols = known_columns_for(data, attribute)
    index = {c: i for i, c in enumerate(data.X.columns)}
    return [index[c] for c in cols]


def _fill_guesses(out: np.ndarray, group_cols: list[int], choice: np.ndarray) -> None:
    """Fill a binary indicator or one-hot target group with the draws."""
    n_rows = out.shape[0]
    if len(group_cols) == 1:
        out[:, group_cols[0]] = choice
    else:
        out[np.arange(n_rows), np.array(group_cols)[choice]] = 1


def uniform_random_guess(
    n_rows: int, group_cols: list[int], n_features: int, seed: int
) -> np.ndarray:
    """Draw T uniformly; leave unused feature columns at zero."""
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
    """Draw T from marginal frequencies in a disjoint holdout."""
    group_cols = _group_columns(data, attribute)
    group_names = [data.X.columns[i] for i in group_cols]
    probs = holdout_df[group_names].mean(axis=0).to_numpy()

    rng = np.random.default_rng(seed)
    out = np.zeros((n_rows, data.X.shape[1]), dtype=int)
    if len(group_cols) == 1:
        p = probs[0]  # Positive rate for a single binary indicator.
        choice = (rng.random(n_rows) < p).astype(int)
    else:
        probs = probs / probs.sum()
        choice = rng.choice(len(group_cols), size=n_rows, p=probs)
    _fill_guesses(out, group_cols, choice)
    return out
