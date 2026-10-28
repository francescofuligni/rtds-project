"""Nested random and sequential tree releases and their protocol metadata."""

import numpy as np

DEFAULT_RANDOM_TREE_POOL = 1000


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


def select_trees(model, m: int) -> list[dict]:
    """Return the first m sources in TreeFARMS enumeration order.

    Increasing m retains the previously released prefix.
    """
    n = model.get_tree_count()
    if m < 1 or m > n:
        raise ValueError(f"Requested {m} trees but the Rashomon set has {n}")
    return [model[i].source for i in range(m)]


def selection_protocol(mode: str, *, seed_offset: int, pool: int | str) -> dict:
    """Keep release metadata consistent across experiment runners."""
    if mode not in ("random", "sequential"):
        raise ValueError("tree-selection must be random or sequential")
    return {
        "tree_selection": mode,
        "tree_selection_seed_offset": seed_offset if mode == "random" else None,
        "tree_selection_pool": pool if mode == "random" else None,
        "tree_selection_order": "uniform_without_replacement_nested_prefix"
        if mode == "random"
        else "treefarms_enumeration_nested_prefix",
    }
