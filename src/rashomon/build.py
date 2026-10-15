"""Build a TreeFARMS Rashomon set and select tree sources."""

from __future__ import annotations

import contextlib
import os

import pandas as pd
from treefarms import TREEFARMS


@contextlib.contextmanager
def _suppress_native_stdout():
    """Silence native TreeFARMS output written to file descriptor 1."""
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
    """Fit TreeFARMS with the requested regularization and bound."""
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
    """Return the first m raw tree sources in enumeration order."""
    n = model.get_tree_count()
    if m > n:
        raise ValueError(f"Requested {m} trees but the Rashomon set only has {n}")
    return [model[i].source for i in range(m)]
