from typing import cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from attack.baselines import marginal_baseline, uniform_random_guess
from attack.conditions import CONDITIONS, known_mask_for_condition, run_condition
from attack.reconstruct import prepare_release
from rashomon.build import build_rashomon_set, select_trees
from rashomon.data import BinarizedDataset


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
    # Use correlated labels so the fitted trees contain meaningful splits.
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
    names = toy_dataset.X.columns.to_list()
    a = [names.index(c) for c in toy_dataset.one_hot_groups["marital-status"]]
    b = names.index("edu_hi")
    target = names.index("sex=Female")
    for condition, expect_a, expect_b in [
        ("none", False, False),
        ("A", True, False),
        ("B", False, True),
        ("A+B", True, True),
    ]:
        mask = known_mask_for_condition(
            toy_dataset, condition, "marital-status", "education", target="sex"
        )
        assert mask[:, a].all() == expect_a
        assert mask[:, b].all() == expect_b
        assert not mask[:, target].any()


def test_rejects_known_target_and_unknown_attribute(toy_dataset):
    with pytest.raises(ValueError, match="distinct"):
        known_mask_for_condition(toy_dataset, "A", "sex", "education", target="sex")
    with pytest.raises(KeyError):
        known_mask_for_condition(toy_dataset, "A", "income", "education", target="sex")
    with pytest.raises(ValueError, match="Unknown condition"):
        known_mask_for_condition(
            toy_dataset, "Y", "marital-status", "education", target="sex"
        )


def test_run_all_conditions_is_feasible(toy_dataset):
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

    for condition in CONDITIONS:
        result = run_condition(
            condition,
            release,
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
        200, [data.X.columns.to_list().index("sex=Female")], data.X.shape[1], seed=0
    )
    marginal = marginal_baseline(data, "sex", 200, holdout, seed=0)
    # Binary target draws must include both categories.
    assert 0 < uniform[:, data.X.columns.to_list().index("sex=Female")].mean() < 1
    assert 0 < marginal[:, data.X.columns.to_list().index("sex=Female")].mean() < 1


def test_coordinator_sanitizes_cells_and_never_passes_private_labels(
    toy_dataset, monkeypatch
):
    from attack import conditions
    from attack.reconstruct import reconstruct

    data = toy_dataset
    tree = {"feature": 5, "true": {"prediction": 1}, "false": {"prediction": 0}}
    names = data.X.columns.to_list()
    release = prepare_release(
        [tree],
        data.X.to_numpy(),
        data.y.to_numpy(),
        names,
        [[names.index(c) for c in g] for g in data.one_hot_groups.values()],
    )
    seen = []

    def capture(public, **kwargs):
        seen.append(kwargs)
        mask, values = kwargs["known_mask"], kwargs["known_values"]
        assert np.all(values[~mask] == 0)
        assert not mask[:, names.index("sex=Female")].any()
        assert "y" not in kwargs
        return reconstruct(public, **kwargs)

    monkeypatch.setattr(conditions, "reconstruct", capture)
    # Block private-label access after publication.
    original = data.y
    blocked_labels = Mock(spec=pd.Series)
    blocked_labels.to_numpy.side_effect = AssertionError("private labels accessed")
    blocked_labels.unique.side_effect = AssertionError("private labels accessed")
    data.y = cast(pd.Series, blocked_labels)
    try:
        result = run_condition(
            "A+B",
            release,
            data,
            "marital-status",
            "education",
            "sex",
            timeout=5,
            n_threads=1,
        )
    finally:
        data.y = original
    assert blocked_labels.mock_calls == []
    assert result.matching is not None and len(seen) == 1
