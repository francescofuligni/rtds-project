import numpy as np
import pytest

from attack.metrics import match_rows, reconstruction_error, target_error


def test_global_matching_recovers_permutation_without_label_groups():
    true = np.array([[0, 0], [1, 1]])
    recon = true[::-1].copy()
    match = match_rows(true, recon, exclude_cols={0})
    np.testing.assert_array_equal(match.permutation, [1, 0])
    assert match.total_cost == 0 and not match.ambiguous


def test_target_values_cannot_influence_assignment():
    true = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]])
    recon = true[[2, 0, 3, 1]].copy()
    before = match_rows(true, recon, {0})
    true[:, 0] = 1 - true[:, 0]
    recon[:, 0] = 0
    after = match_rows(true, recon, {0})
    np.testing.assert_array_equal(before.permutation, after.permutation)
    assert before.total_cost == after.total_cost


def test_perfect_and_flipped_targets_with_unique_non_target_profiles():
    true = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]])
    error, _ = reconstruction_error(true, true.copy(), [0])
    assert error == 0
    recon = true.copy()
    recon[:, 0] = 1 - recon[:, 0]
    error, _ = reconstruction_error(true, recon, [0])
    assert error == 1


def test_duplicate_profiles_and_global_ties_are_detected():
    duplicate = np.array([[0, 1], [1, 1]])
    match = match_rows(duplicate, duplicate, {0})
    assert match.ambiguous and match.total_cost == 0
    # Equal-cost global assignments despite no duplicate profiles.
    true = np.array([[0, 0, 0], [0, 1, 0]])
    recon = np.array([[1, 0, 0], [1, 0, 1]])
    match = match_rows(true, recon, {0})
    assert match.ambiguous and match.total_cost == 2


def test_known_profiles_are_hard_constraints():
    true = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    recon = np.array([[1, 0, 1, 1, 1], [0, 1, 0, 0, 0]])
    unconstrained = match_rows(true, recon, {0})
    known = match_rows(true, recon, {0}, known_cols=[1])
    np.testing.assert_array_equal(unconstrained.permutation, [1, 0])
    np.testing.assert_array_equal(known.permutation, [0, 1])
    with pytest.raises(ValueError, match="compatible"):
        match_rows(true, np.zeros_like(true), {0}, known_cols=[1])


def test_categorical_target_counts_one_error_per_row():
    true = np.array([[1, 0, 0], [0, 1, 1]])
    recon = np.array([[0, 1, 0], [0, 1, 1]])
    error, _ = reconstruction_error(true, recon, [0, 1])
    assert error == 0.5


def test_target_only_baselines_use_the_attack_permutation():
    true = np.array([[0, 0], [1, 1]])
    recon = true[::-1]
    match = match_rows(true, recon, {0})
    baseline = np.array([[1, 0], [0, 0]])
    assert target_error(true, baseline, [0], match.permutation) == 0
    assert target_error(true, baseline, [0], np.arange(2)) == 1


def test_invalid_matching_and_permutations():
    true = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError, match="nonempty"):
        match_rows(true, true[:1], {0})
    with pytest.raises(ValueError, match="cannot be known"):
        match_rows(true, true, {0}, known_cols=[0])
    with pytest.raises(ValueError, match="bijective"):
        target_error(true, true, [0], np.array([0, 0]))
