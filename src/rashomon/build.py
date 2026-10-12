"""Rashomon set construction via TreeFARMS, and tree-selection strategies
for growing a released ensemble of size m from it."""

from __future__ import annotations

import contextlib
import os

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
    """
    n = model.get_tree_count()
    if m > n:
        raise ValueError(f"Requested {m} trees but the Rashomon set only has {n}")
    return [model[i].source for i in range(m)]
