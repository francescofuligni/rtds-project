import numpy as np
import pandas as pd
import pytest

from attack.reconstruct import extract_branches, reconstruct
from rashomon.build import build_rashomon_set
from rashomon.selection import select_random_trees


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(0)
    n, m = 30, 5
    X = rng.integers(0, 2, size=(n, m))
    # make y depend on a couple of features so trees actually split on them
    y = ((X[:, 0] == 1) & (X[:, 1] == 0)).astype(int)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(m)])
    y_s = pd.Series(y, name="label")
    return X_df, y_s, X, y


@pytest.fixture(scope="module")
def toy_trees(toy_data):
    X_df, y_s, _, _ = toy_data
    model = build_rashomon_set(
        X_df, y_s, regularization=0.05, rashomon_bound_multiplier=0.1
    )
    assert model.get_tree_count() >= 1
    return select_random_trees(model, min(3, model.get_tree_count()), seed=0)


def _branches_for(trees, X, y, classes):
    return [extract_branches(t, X, y, classes) for t in trees]


def test_no_background_knowledge_is_feasible_and_respects_leaf_counts(
    toy_data, toy_trees
):
    _, _, X, y = toy_data
    classes = [0, 1]
    trees_branches = _branches_for(toy_trees, X, y, classes)

    result = reconstruct(
        trees_branches, y, classes, n_features=X.shape[1], one_hot_groups=[], timeout=20
    )

    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert result.reconstructed is not None
    assert result.reconstructed.shape == X.shape

    # re-extracting branches from the reconstructed table against the same
    # trees must reproduce identical per-leaf per-class counts: the
    # solution must be a dataset the released trees are actually
    # consistent with, even if it isn't bit-identical to the original.
    for tree, orig_branches in zip(toy_trees, trees_branches):
        recon_branches = extract_branches(tree, result.reconstructed, y, classes)
        orig_counts = sorted(b[1] for b in orig_branches)
        recon_counts = sorted(b[1] for b in recon_branches)
        assert orig_counts == recon_counts


def test_fully_known_background_reconstructs_exactly(toy_data, toy_trees):
    _, _, X, y = toy_data
    classes = [0, 1]
    trees_branches = _branches_for(toy_trees, X, y, classes)

    known_mask = np.ones_like(X, dtype=bool)
    result = reconstruct(
        trees_branches,
        y,
        classes,
        n_features=X.shape[1],
        one_hot_groups=[],
        known_mask=known_mask,
        known_values=X,
        timeout=20,
    )

    assert result.status in ("OPTIMAL", "FEASIBLE")
    np.testing.assert_array_equal(result.reconstructed, X)


def test_single_known_column_is_respected(toy_data, toy_trees):
    _, _, X, y = toy_data
    classes = [0, 1]
    trees_branches = _branches_for(toy_trees, X, y, classes)

    known_mask = np.zeros_like(X, dtype=bool)
    known_mask[:, 0] = True  # attacker knows column 0 (e.g. "A") for every row
    known_values = np.zeros_like(X)
    known_values[:, 0] = X[:, 0]

    result = reconstruct(
        trees_branches,
        y,
        classes,
        n_features=X.shape[1],
        one_hot_groups=[],
        known_mask=known_mask,
        known_values=known_values,
        timeout=20,
    )

    assert result.status in ("OPTIMAL", "FEASIBLE")
    np.testing.assert_array_equal(result.reconstructed[:, 0], X[:, 0])
