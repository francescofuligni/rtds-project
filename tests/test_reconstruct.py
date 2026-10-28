import numpy as np
import pytest

from attack.reconstruct import extract_branches, reconstruct


@pytest.fixture
def release():
    X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]])
    y = np.array([0, 1, 0, 1])
    tree = {
        "feature": 0,
        "true": {
            "feature": 1,
            "true": {"prediction": 0},
            "false": {"prediction": 1},
        },
        "false": {"prediction": 0},
    }
    branches = extract_branches(tree, X, y, [0, 1])
    assert branches == [
        ([(1, 0.5), (2, 0.5)], [1, 0]),
        ([(1, 0.5), (-2, 0.5)], [0, 1]),
        ([(-1, 0.5)], [1, 1]),
    ]
    return X, y, branches


def test_reconstruction_respects_leaf_counts_and_schema(release):
    X, y, branches = release
    result = reconstruct(
        [branches],
        y,
        [0, 1],
        n_features=3,
        one_hot_groups=[[1, 2]],
        timeout=5,
        n_threads=1,
    )
    assert result.status in ("OPTIMAL", "FEASIBLE")
    candidate = result.reconstructed
    assert candidate is not None and candidate.shape == X.shape
    assert set(candidate.ravel()) <= {0, 1}
    np.testing.assert_array_equal(candidate[:, [1, 2]].sum(axis=1), 1)

    # Route directly through the handmade tree, independently of extraction.
    leaves = np.where(candidate[:, 0] == 0, 2, np.where(candidate[:, 1] == 1, 0, 1))
    counts = [
        [int(((leaves == leaf) & (y == c)).sum()) for c in [0, 1]] for leaf in range(3)
    ]
    assert counts == [[1, 0], [0, 1], [1, 1]]


@pytest.mark.parametrize("columns", [[0], [0, 1, 2]], ids=["partial", "fully-fixed"])
def test_known_cells_are_respected(release, columns):
    X, y, branches = release
    mask = np.zeros_like(X, dtype=bool)
    mask[:, columns] = True
    result = reconstruct(
        [branches],
        y,
        [0, 1],
        n_features=3,
        one_hot_groups=[[1, 2]],
        known_mask=mask,
        known_values=np.where(mask, X, 0),
        timeout=5,
        n_threads=1,
    )
    assert result.status in ("OPTIMAL", "FEASIBLE")
    np.testing.assert_array_equal(result.reconstructed[:, columns], X[:, columns])


def test_incompatible_counts_are_proven_infeasible():
    # Both trees have the correct class totals but force opposite feature values.
    branches = [
        [([(1, 0.5)], [1, 1]), ([(-1, 0.5)], [0, 0])],
        [([(1, 0.5)], [0, 0]), ([(-1, 0.5)], [1, 1])],
    ]
    result = reconstruct(
        branches,
        np.array([0, 1]),
        [0, 1],
        n_features=1,
        one_hot_groups=[],
        timeout=5,
        n_threads=1,
    )
    assert result.status == "INFEASIBLE"
    assert result.reconstructed is None
