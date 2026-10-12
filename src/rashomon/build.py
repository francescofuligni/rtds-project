"""Rashomon set construction via TreeFARMS, and tree-selection strategies
for growing a released ensemble of size m from it."""

from __future__ import annotations

import pandas as pd
from treefarms import TREEFARMS


def build_rashomon_set(X: pd.DataFrame, y: pd.Series, *, regularization: float = 0.01,
                        rashomon_bound_multiplier: float = 0.05):
    """Fit TreeFARMS and return the model (supports len() and indexing to
    access individual TreeClassifier objects)."""
    config = {
        "regularization": regularization,
        "rashomon_bound_multiplier": rashomon_bound_multiplier,
    }
    model = TREEFARMS(config)
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
