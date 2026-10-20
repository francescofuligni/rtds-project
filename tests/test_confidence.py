import numpy as np
import pandas as pd
import pytest

from attack.confidence import repeated_reconstruct_with_confidence
from attack.reconstruct import prepare_release
from rashomon.build import build_rashomon_set, select_trees
from rashomon.data import BinarizedDataset


@pytest.fixture(scope="module")
def toy_dataset():
    rng = np.random.default_rng(5)
    n = 40
    marital = rng.integers(0, 3, size=n)
    marital_df = pd.get_dummies(
        pd.Series(marital, name="marital"), prefix="marital"
    ).astype(int)
    sex = rng.integers(0, 2, size=n)
    other = rng.integers(0, 2, size=(n, 2))

    X = pd.concat(
        [
            marital_df,
            pd.DataFrame(other, columns=["edu_hi", "age_hi"]),
            pd.Series(sex, name="sex=Female"),
        ],
        axis=1,
    )
    y = pd.Series(
        ((marital_df["marital_1"] == 1) | (sex == 1)).astype(int), name="label"
    )

    return BinarizedDataset(
        X=X,
        y=y,
        one_hot_groups={"marital-status": list(marital_df.columns)},
        binary_columns={"sex": "sex=Female", "education": "edu_hi"},
    )


def test_confidence_rows_are_well_formed(toy_dataset):
    data = toy_dataset
    model = build_rashomon_set(
        data.X, data.y, regularization=0.1, rashomon_bound_multiplier=0.2
    )
    trees = select_trees(model, min(3, model.get_tree_count()))
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    names = data.X.columns.to_list()
    release = prepare_release(
        trees,
        X,
        y,
        names,
        [[names.index(c) for c in group] for group in data.one_hot_groups.values()],
    )

    rows, statuses = repeated_reconstruct_with_confidence(
        release,
        data,
        "marital-status",
        "sex",
        "education",
        "none",
        n_repeats=5,
        timeout=10,
    )

    assert len(statuses) == 5
    assert all(s in ("OPTIMAL", "FEASIBLE") for s in statuses)
    assert len(rows) == len(X)
    for row in rows:
        assert 0.0 < row.agreement <= 1.0
        assert 1 <= row.n_votes <= 5
        assert isinstance(row.correct, bool)


def test_diagnostic_runs_with_known_a(toy_dataset):
    data = toy_dataset
    model = build_rashomon_set(
        data.X, data.y, regularization=0.1, rashomon_bound_multiplier=0.2
    )
    trees = select_trees(model, min(3, model.get_tree_count()))
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    names = data.X.columns.to_list()
    release = prepare_release(
        trees,
        X,
        y,
        names,
        [[names.index(c) for c in group] for group in data.one_hot_groups.values()],
    )

    rows, statuses = repeated_reconstruct_with_confidence(
        release,
        data,
        "marital-status",
        "sex",
        "education",
        "A",
        n_repeats=5,
        timeout=10,
    )
    assert len(rows) == len(X)
    assert all(s in ("OPTIMAL", "FEASIBLE") for s in statuses)


def test_diagnostic_repeat_without_candidate_stops(toy_dataset, monkeypatch):
    import pytest

    from attack import confidence
    from attack.reconstruct import ReconstructionResult

    data = toy_dataset
    names = data.X.columns.to_list()
    release = prepare_release(
        [{"prediction": 0}],
        data.X.to_numpy(),
        data.y.to_numpy(),
        names,
        [[names.index(c) for c in cols] for cols in data.one_hot_groups.values()],
    )
    monkeypatch.setattr(
        confidence,
        "reconstruct",
        lambda *a, **kw: ReconstructionResult("UNKNOWN", 0, None),
    )
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        repeated_reconstruct_with_confidence(
            release, data, "marital-status", "education", "sex", "none", n_repeats=3
        )
