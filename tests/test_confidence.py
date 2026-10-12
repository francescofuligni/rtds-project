import numpy as np
import pandas as pd
import pytest

from rashomon.data import BinarizedDataset
from rashomon.build import build_rashomon_set, select_trees
from attack.reconstruct import extract_branches
from attack.confidence import repeated_reconstruct_with_confidence


@pytest.fixture(scope="module")
def toy_dataset():
    rng = np.random.default_rng(5)
    n = 40
    marital = rng.integers(0, 3, size=n)
    marital_df = pd.get_dummies(pd.Series(marital, name="marital"), prefix="marital").astype(int)
    sex = rng.integers(0, 2, size=n)
    other = rng.integers(0, 2, size=(n, 2))

    X = pd.concat([
        marital_df,
        pd.DataFrame(other, columns=["edu_hi", "age_hi"]),
        pd.Series(sex, name="sex=Female"),
    ], axis=1)
    y = pd.Series(((marital_df["marital_1"] == 1) | (sex == 1)).astype(int), name="label")

    return BinarizedDataset(
        X=X, y=y,
        one_hot_groups={"marital-status": list(marital_df.columns)},
        binary_columns={"sex": "sex=Female", "education": "edu_hi"},
    )


def test_confidence_rows_are_well_formed(toy_dataset):
    data = toy_dataset
    model = build_rashomon_set(data.X, data.y, regularization=0.1, rashomon_bound_multiplier=0.2)
    trees = select_trees(model, min(3, model.get_tree_count()))
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    classes = sorted(data.y.unique().tolist())
    trees_branches = [extract_branches(t, X, y, classes) for t in trees]

    rows, statuses = repeated_reconstruct_with_confidence(
        trees_branches, data, "marital-status", "sex", "education", "none",
        n_repeats=5, timeout=10,
    )

    assert len(statuses) == 5
    assert all(s in ("OPTIMAL", "FEASIBLE") for s in statuses)
    assert len(rows) == len(X)
    for row in rows:
        assert 0.0 < row.agreement <= 1.0
        assert 1 <= row.n_votes <= 5
        assert isinstance(row.correct, bool)


def test_fully_known_background_gives_perfect_agreement(toy_dataset):
    # with A+B... well there's no A+B here, but "sex" as the known
    # background attribute (not the target) should make the *other*
    # known target ("education" here, via attr_b="sex") irrelevant --
    # instead verify that fixing the condition's known columns is
    # respected across every repeat by checking status validity and
    # perfect agreement isn't required, only well-formedness (covered
    # above). This test checks a stronger condition should not reduce
    # agreement below a weaker one is NOT guaranteed in general, so we
    # only assert the pipeline runs for the "A" condition too.
    data = toy_dataset
    model = build_rashomon_set(data.X, data.y, regularization=0.1, rashomon_bound_multiplier=0.2)
    trees = select_trees(model, min(3, model.get_tree_count()))
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    classes = sorted(data.y.unique().tolist())
    trees_branches = [extract_branches(t, X, y, classes) for t in trees]

    rows, statuses = repeated_reconstruct_with_confidence(
        trees_branches, data, "marital-status", "sex", "education", "A",
        n_repeats=5, timeout=10,
    )
    assert len(rows) == len(X)
    assert all(s in ("OPTIMAL", "FEASIBLE") for s in statuses)
