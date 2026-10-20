"""Rashomon set construction via TreeFARMS, and tree-selection strategies
for growing a released ensemble of size m from it."""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
from treefarms import TREEFARMS

DEFAULT_RANDOM_TREE_POOL = 1000


@contextlib.contextmanager
def _suppress_native_stdout():
    """TreeFARMS's underlying C++ extension prints search progress and
    every found tree as raw JSON directly to file descriptor 1, bypassing
    Python's sys.stdout -- silence it at the fd level for the duration of
    the fit() call."""
    fd = 1
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def build_rashomon_set(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    regularization: float = 0.01,
    rashomon_bound_multiplier: float = 0.05,
    verbose: bool = False,
):
    """Fit TreeFARMS and return the model (supports get_tree_count() and
    indexing to access individual TreeClassifier objects)."""
    config = {
        "regularization": regularization,
        "rashomon_bound_multiplier": rashomon_bound_multiplier,
    }
    model = TREEFARMS(config)
    if verbose:
        model.fit(X, y)
    else:
        with _suppress_native_stdout():
            model.fit(X, y)
    return model


def select_random_trees(
    model, m: int, seed: int, pool: int = DEFAULT_RANDOM_TREE_POOL
) -> list[dict]:
    """Select m trees uniformly at random (without replacement) from the
    Rashomon set.

    A random ordering of up to `pool` indices is drawn uniformly from the
    full Rashomon set and its first m are returned, so for one seed the
    selections are nested:
    the m=10 trees are among the m=50 trees, which are among the m=100
    trees. Increasing m therefore adds trees instead of swapping them.
    """
    n = model.get_tree_count()
    if m > n:
        raise ValueError(f"Requested {m} trees but the Rashomon set only has {n}")
    if m > pool:
        raise ValueError(f"Requested {m} trees but the nested pool only has {pool}")
    order = np.random.default_rng(seed).choice(n, size=min(pool, n), replace=False)
    return [model[int(i)].source for i in order[:m]]
