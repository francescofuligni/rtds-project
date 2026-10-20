"""Rashomon set construction via TreeFARMS, and tree-selection strategies
for growing a released ensemble of size m from it."""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
from treefarms import TREEFARMS


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


def build_rashomon_set(X: pd.DataFrame, y: pd.Series, *, regularization: float = 0.01,
                        rashomon_bound_multiplier: float = 0.05, verbose: bool = False):
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


def select_trees(model, m: int) -> list[dict]:
    """Select the first m trees from the Rashomon set (index 0 is the
    optimal tree; this is TreeFARMS's default enumeration order).
    Returns each tree's raw recursive-dict representation (`.source`).

    Consecutive indices are near-copies: TreeFARMS decodes an index like an
    odometer over its stored subtree choices (the false-branch subtree
    changes fastest), so the first m trees mostly share their root and most
    branches. Use `select_random_trees` to measure leakage as a function of m.
    """
    n = model.get_tree_count()
    if m > n:
        raise ValueError(f"Requested {m} trees but the Rashomon set only has {n}")
    return [model[i].source for i in range(m)]


def select_random_trees(model, m: int, seed: int, pool: int = 1000) -> list[dict]:
    """Select m trees uniformly at random (without replacement) from the
    Rashomon set.

    A fixed-size random ordering of `pool` indices is drawn per seed and
    its first m are returned, so for one seed the selections are nested:
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
