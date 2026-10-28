import numpy as np

from attack.metrics import match_rows, reconstruction_error


def test_match_rows_recovers_known_permutation():
    rng = np.random.default_rng(1)
    n, m = 8, 4
    true_X = rng.integers(0, 2, size=(n, m))
    y = rng.integers(0, 2, size=n)

    # Permutations within Y groups must be recoverable without using T.
    recon = true_X.copy()
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        shuffled = rng.permutation(idx)
        recon[idx] = true_X[shuffled]

    perm = match_rows(true_X, recon, y, exclude_cols={0})
    matched = recon[perm]
    assert np.array_equal(matched[:, 1:], true_X[:, 1:])


def test_reconstruction_error_is_zero_for_perfect_reconstruction():
    rng = np.random.default_rng(2)
    n, m = 6, 3
    true_X = rng.integers(0, 2, size=(n, m))
    y = rng.integers(0, 2, size=n)
    err, _perm = reconstruction_error(true_X, true_X.copy(), y, target_cols=[0])
    assert err == 0.0


def test_reconstruction_error_detects_flipped_target():
    true_X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([0, 0, 1, 1])
    recon = true_X.copy()
    recon[:, 0] = 1 - recon[:, 0]
    err, _perm = reconstruction_error(true_X, recon, y, target_cols=[0])
    assert err == 1.0


def test_matching_is_target_blind_even_when_costs_tie():
    true_X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    recon = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    y = np.array([0, 0, 1, 1])
    original = match_rows(true_X, recon, y, exclude_cols={0})
    for change_truth, change_candidate in [(True, False), (False, True), (True, True)]:
        truth, candidate = true_X.copy(), recon.copy()
        if change_truth:
            truth[:, 0] = 1 - truth[:, 0]
        if change_candidate:
            candidate[:, 0] = 1 - candidate[:, 0]
        perm = match_rows(truth, candidate, y, exclude_cols={0})
        np.testing.assert_array_equal(perm, original)
        np.testing.assert_array_equal(y[perm], y)


def test_categorical_target_error_counts_rows_instead_of_changed_bits():
    true_X = np.array([[1, 0, 0], [0, 1, 1]])
    recon = np.array([[0, 1, 0], [0, 1, 1]])
    error, perm = reconstruction_error(true_X, recon, np.array([0, 0]), [0, 1])
    np.testing.assert_array_equal(perm, [0, 1])
    assert error == 0.5
