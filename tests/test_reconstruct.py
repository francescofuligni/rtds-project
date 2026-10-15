"""Check the no-Y formulation using a separate raw-tree routing oracle."""

import copy
from dataclasses import fields, replace

import numpy as np
import pytest

from attack.reconstruct import PublicRelease, prepare_release, reconstruct


@pytest.fixture
def toy_release():
    X = np.array([[1, 0, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
    y = np.array([0, 1, 1, 0])
    trees = [
        {"feature": 0, "true": {"prediction": 1}, "false": {"prediction": 0}},
        {"feature": 3, "true": {"prediction": 1}, "false": {"prediction": 0}},
    ]
    return prepare_release(trees, X, y, ["a0", "a1", "b", "T"], [[0, 1]]), X, y


def raw_counts(tree, X, y, classes):
    """Independent oracle: route each row directly through raw child pointers."""
    counts = {}
    for row, label in zip(X, y):
        node, path = tree, []
        while "prediction" not in node:
            branch = "true" if row[node["feature"]] else "false"
            path.append(branch)
            node = node[branch]
        key = tuple(path)
        counts.setdefault(key, [0] * len(classes))[classes.index(label)] += 1
    return counts


def assert_candidate(release, X, y, result):
    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert result.reconstructed is not None and result.reconstructed_labels is not None
    assert result.reconstructed.shape == X.shape
    assert np.isin(result.reconstructed, [0, 1]).all()
    for group in release.one_hot_groups:
        assert (result.reconstructed[:, group].sum(axis=1) == 1).all()
    for tree in release.trees:
        assert raw_counts(tree, X, y, list(release.classes)) == raw_counts(
            tree,
            result.reconstructed,
            result.reconstructed_labels,
            list(release.classes),
        )


def test_no_background_respects_counts_with_inferred_labels(toy_release):
    release, X, y = toy_release
    assert_candidate(release, X, y, reconstruct(release, timeout=5, n_threads=1))


def test_fully_fixed_feature_probe_respects_features(toy_release):
    # Low-level constraint probe, not a research condition with T as background.
    release, X, y = toy_release
    result = reconstruct(
        release,
        known_mask=np.ones_like(X, dtype=bool),
        known_values=X,
        timeout=5,
        n_threads=1,
    )
    assert_candidate(release, X, y, result)
    np.testing.assert_array_equal(result.reconstructed, X)


def test_known_column_is_respected(toy_release):
    release, X, y = toy_release
    mask = np.zeros_like(X, dtype=bool)
    mask[:, 2] = True
    values = np.where(mask, X, 0)
    result = reconstruct(
        release, known_mask=mask, known_values=values, timeout=5, n_threads=1
    )
    assert_candidate(release, X, y, result)
    assert result.reconstructed is not None
    np.testing.assert_array_equal(result.reconstructed[:, 2], X[:, 2])


def test_release_has_no_private_rows_and_is_detached(toy_release):
    release, X, y = toy_release
    assert {f.name for f in fields(PublicRelease)} == {
        "trees",
        "trees_branches",
        "classes",
        "n_rows",
        "feature_names",
        "one_hot_groups",
    }
    identifier = release.release_id
    X[:] = 0
    y[:] = 0
    assert release.release_id == identifier
    with pytest.raises(TypeError, match="PublicRelease"):
        reconstruct(y)


def test_publisher_does_not_share_tree_references():
    tree = {"prediction": 0}
    release = prepare_release([tree], np.array([[0]]), np.array([0]), ["T"], [])
    tree["prediction"] = 99
    assert release.trees[0]["prediction"] == 0


def test_unknown_classes_are_shared_across_trees():
    tree = {"feature": 0, "true": {"prediction": 1}, "false": {"prediction": 0}}
    release = prepare_release([tree], np.array([[0], [1]]), np.array([0, 1]), ["f"], [])
    opposite = copy.deepcopy(release.trees_branches[0])
    opposite[0][1][:] = [1, 0]
    opposite[1][1][:] = [0, 1]
    contradiction = replace(
        release,
        trees=[tree, copy.deepcopy(tree)],
        trees_branches=[release.trees_branches[0], opposite],
    )
    result = reconstruct(contradiction, timeout=5, n_threads=1)
    assert result.status == "INFEASIBLE"
    assert result.reconstructed is None and result.reconstructed_labels is None


def test_underdetermined_target_has_two_feasible_values():
    release = prepare_release(
        [{"prediction": 0}], np.array([[0], [1]]), np.array([0, 1]), ["T"], []
    )
    # Force alternative values only to prove modeled ambiguity, not as knowledge.
    for target_value in [0, 1]:
        result = reconstruct(
            release,
            known_mask=np.ones((2, 1), dtype=bool),
            known_values=np.full((2, 1), target_value),
            timeout=5,
            n_threads=1,
        )
        assert result.status in ("OPTIMAL", "FEASIBLE")


def test_known_mask_requires_values(toy_release):
    release, X, _ = toy_release
    with pytest.raises(ValueError, match="known_values is required"):
        reconstruct(release, known_mask=np.ones_like(X, dtype=bool))


def test_schema_and_counts_are_validated(toy_release):
    release, _, _ = toy_release
    with pytest.raises(ValueError, match="overlapping"):
        reconstruct(replace(release, one_hot_groups=((0, 1), (1, 2))))
    with pytest.raises(ValueError, match="each tree"):
        reconstruct(replace(release, n_rows=5))
    counts = copy.deepcopy(release.trees_branches)
    counts[0][0][1][0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        reconstruct(replace(release, trees_branches=counts))


def test_single_class_and_empty_inputs():
    release = prepare_release(
        [{"prediction": 0}], np.array([[0], [1]]), np.array([0, 0]), ["f"], []
    )
    result = reconstruct(release, timeout=5, n_threads=1)
    assert result.reconstructed_labels is not None
    np.testing.assert_array_equal(result.reconstructed_labels, [0, 0])
    with pytest.raises(ValueError, match="positive dimensions"):
        prepare_release([{"prediction": 0}], np.empty((0, 1)), np.array([]), ["f"], [])
