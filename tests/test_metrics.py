import numpy as np

from attack.metrics import match_rows, reconstruction_error


def test_match_rows_recovers_known_permutation():
    rng = np.random.default_rng(1)
    n, m = 8, 4
    true_X = rng.integers(0, 2, size=(n, m))
    y = rng.integers(0, 2, size=n)

    # shuffle rows within each label group to build "reconstructed" data
    # with a known best match, excluding column 0 (pretend T) from the cost
    recon = true_X.copy()
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        shuffled = rng.permutation(idx)
        recon[idx] = true_X[shuffled]

    perm = match_rows(true_X, recon, y, exclude_cols={0})
    # every matched row (outside T) should be identical after matching
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
    recon[:, 0] = 1 - recon[:, 0]  # flip T for every row
    err, _perm = reconstruction_error(true_X, recon, y, target_cols=[0])
    assert err == 1.0
