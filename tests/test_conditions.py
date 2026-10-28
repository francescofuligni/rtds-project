import numpy as np
import pandas as pd
import pytest

from attack.baselines import marginal_baseline, uniform_random_guess
from attack.conditions import CONDITIONS, known_mask_for_condition, run_condition
from attack.reconstruct import extract_branches
from rashomon.build import build_rashomon_set
from rashomon.data import BinarizedDataset
from rashomon.selection import select_random_trees


@pytest.fixture(scope="module")
def toy_dataset():
    rng = np.random.default_rng(3)
    n = 40
    marital = rng.integers(0, 3, size=n)  # 3-category one-hot group "A"
    marital_df = pd.get_dummies(
        pd.Series(marital, name="marital"), prefix="marital"
    ).astype(int)
    sex = rng.integers(0, 2, size=n)  # single-column binary "T"
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

    data = BinarizedDataset(
        X=X,
        y=y,
        one_hot_groups={"marital-status": list(marital_df.columns)},
        binary_columns={"sex": "sex=Female", "education": "edu_hi"},
    )
    return data


def test_known_mask_matches_condition(toy_dataset):
    a_columns = ["marital_0", "marital_1", "marital_2"]
    b_columns = ["edu_hi"]
    for condition, known_columns in [
        ("none", []),
        ("A", a_columns),
        ("B", b_columns),
        ("A+B", a_columns + b_columns),
    ]:
        expected = np.tile(
            toy_dataset.X.columns.isin(known_columns), (len(toy_dataset.X), 1)
        )
        mask = known_mask_for_condition(
            toy_dataset, condition, "marital-status", "education"
        )
        np.testing.assert_array_equal(mask, expected)


def test_run_all_conditions_is_feasible(toy_dataset):
    data = toy_dataset
    model = build_rashomon_set(
        data.X, data.y, regularization=0.1, rashomon_bound_multiplier=0.2
    )
    trees = select_random_trees(model, min(3, model.get_tree_count()), seed=3)
    X = data.X.to_numpy()
    y = data.y.to_numpy()
    classes = sorted(data.y.unique().tolist())
    trees_branches = [extract_branches(t, X, y, classes) for t in trees]

    for condition in CONDITIONS:
        result = run_condition(
            condition,
            trees_branches,
            data,
            "marital-status",
            "education",
            "sex",
            timeout=10,
        )
        assert result.status in ("OPTIMAL", "FEASIBLE")
        assert result.reconstruction_error is not None
        assert 0.0 <= result.reconstruction_error <= 1.0


def test_baselines_are_not_constant_for_binary_attribute(toy_dataset):
    data = toy_dataset
    holdout = pd.DataFrame({"sex=Female": [0, 0, 0, 1, 1]})  # 40% positive rate
    uniform = uniform_random_guess(
        200, [data.X.columns.get_loc("sex=Female")], data.X.shape[1], seed=0
    )
    marginal = marginal_baseline(data, "sex", 200, holdout, seed=0)
    # A binary attribute has two possible values despite using one column.
    assert 0 < uniform[:, data.X.columns.get_loc("sex=Female")].mean() < 1
    assert 0 < marginal[:, data.X.columns.get_loc("sex=Female")].mean() < 1
