from collections import Counter

import numpy as np
import pandas as pd
import pytest

from attack.reconstruct import extract_branches, reconstruct
from attack.record_confidence import (
    RecordPrior,
    best_table_claims,
    draw_prior_table,
    enumerate_feature_vectors,
    record_counts,
    sampled_candidates,
    support_matrix,
)
from rashomon.build import build_rashomon_set, select_trees


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(30, 5))
    y = ((X[:, 0] == 1) & (X[:, 1] == 0)).astype(int)
    model = build_rashomon_set(pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y),
                               regularization=0.05, rashomon_bound_multiplier=0.1)
    trees = select_trees(model, min(3, model.get_tree_count()))
    branches = [extract_branches(t, X, y, [0, 1]) for t in trees]
    return X, y, trees, branches


def test_record_counts_include_label_and_duplicates():
    X = np.array([[1, 0], [1, 0], [1, 0]])
    y = np.array([0, 0, 1])
    assert record_counts(X, y) == Counter({(1, 0, 0): 2, (1, 0, 1): 1})


def test_support_matrix_fills_absent_records_and_skips_failed_solves():
    support = support_matrix([Counter({(1,): 2}), None, Counter({(0,): 1})])
    assert support[(1,)].tolist() == [2, 0]
    assert support[(0,)].tolist() == [0, 1]


def test_enumerated_vectors_respect_one_hot_groups():
    vectors = enumerate_feature_vectors(4, [[1, 2]])
    assert len(vectors) == 2 * 2 * 2  # two free bits, one two-way group
    assert len(set(vectors)) == len(vectors)
    assert all(v[1] + v[2] == 1 for v in vectors)


def test_prior_favours_common_public_records_and_decreases_with_k():
    public_X = np.array([[1, 0]] * 9 + [[0, 1]])
    public_y = np.zeros(10, dtype=int)
    prior = RecordPrior(public_X, public_y, class_totals={0: 5}, n_feature_vectors=4)
    common, rare = (1, 0, 0), (1, 1, 0)
    assert prior.probability(common, 1) > prior.probability(rare, 1)
    assert prior.probability(common, 1) >= prior.probability(common, 2) >= prior.probability(common, 3)
    assert prior.probability(common, 6) == 0.0  # only 5 rows carry label 0


def test_best_table_respects_class_totals_and_score_order():
    scores = {((0, 0), 1): 0.9, ((0, 0), 2): 0.5, ((1, 0), 1): 0.7, ((0, 1), 1): 0.8}
    tiebreak = {}
    table = best_table_claims(scores, {0: 2, 1: 1}, tiebreak)
    assert [(r, k) for r, k, _ in table] == [((0, 0), 1), ((1, 0), 1), ((0, 1), 1)]


def test_target_equal_to_truth_is_returned_exactly(toy):
    X, y, _, branches = toy
    result = reconstruct(branches, y, [0, 1], X.shape[1], [], timeout=20, n_threads=4, target=X)
    assert result.status == "OPTIMAL"
    np.testing.assert_array_equal(result.reconstructed, X)


def test_prior_guided_samples_stay_consistent_with_the_release(toy):
    X, y, trees, branches = toy
    rng = np.random.default_rng(1)
    public_X = rng.integers(0, 2, size=(50, 5))
    public_y = rng.integers(0, 2, size=50)
    candidates, statuses = sampled_candidates(branches, y, [0, 1], X.shape[1], [], n_samples=3,
                                              public_X=public_X, public_y=public_y, timeout=20, n_threads=4)
    assert len(candidates) == 3 and all(s == "OPTIMAL" for s in statuses)
    for candidate in candidates:
        rows = np.array([r[:-1] for r, c in candidate.items() for _ in range(c)])
        labels = np.array([r[-1] for r, c in candidate.items() for _ in range(c)])
        for tree, original in zip(trees, branches):
            assert sorted(b[1] for b in extract_branches(tree, rows, labels, [0, 1])) == sorted(b[1] for b in original)


def test_prior_table_keeps_each_rows_label():
    public_X = np.array([[1, 1], [0, 0]])
    public_y = np.array([1, 0])
    y = np.array([0, 1, 0])
    table = draw_prior_table(public_X, public_y, y, np.random.default_rng(0))
    np.testing.assert_array_equal(table, [[0, 0], [1, 1], [0, 0]])
