import pandas as pd
import pytest

from rashomon.data import (
    COLUMNS,
    binarize_adult_curated,
    known_columns_for,
    load_adult_raw,
    sample_rows,
)


@pytest.fixture
def adult_data(tmp_path, monkeypatch):
    cached = pd.DataFrame(
        {
            "age": [25, 50, 35, 45],
            "workclass": "Private",
            "fnlwgt": 1,
            "education": ["Bachelors", "HS-grad", "Masters", "Some-college"],
            "education-num": 9,
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Widowed",
            ],
            "occupation": "Sales",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": ["Female", "Male", "Female", "Male"],
            "capital-gain": [0, 100, 0, 50],
            "capital-loss": 0,
            "hours-per-week": [30, 50, 40, 45],
            "native-country": "United-States",
            "income": ["<=50K", ">50K", "<=50K", ">50K"],
        },
        columns=COLUMNS,
    ).rename(columns=lambda c: f" {c} ")
    cached.loc[0, " income "] = " <=50K. "
    cached.loc[0, " sex "] = " Female "
    cached.loc[4] = cached.iloc[0]
    cached.loc[4, " education "] = None
    path = tmp_path / "adult.data.csv"
    cached.to_csv(path, index=False)
    original = path.read_bytes()
    read_csv = pd.read_csv

    def read_cache(source, *args, **kwargs):
        assert source == path, "Tests must use the local fixture, never download Adult"
        return read_csv(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_cache)
    raw = load_adult_raw(tmp_path)
    assert list(raw.columns) == COLUMNS
    assert len(raw) == 4
    assert raw.loc[0, "income"] == "<=50K"
    assert raw.loc[0, "sex"] == "Female"
    assert path.read_bytes() == original
    return binarize_adult_curated(raw)


def test_curated_encoding_and_knowledge_columns(adult_data):
    data = adult_data
    assert set(data.X.values.ravel()) <= {0, 1}
    assert set(data.y.unique()) == {0, 1}
    for cols in data.one_hot_groups.values():
        assert (data.X[cols].sum(axis=1) == 1).all()
    assert known_columns_for(data, "sex") == ["sex=Female"]
    assert len(known_columns_for(data, "marital-status")) == 4
    assert len(known_columns_for(data, "education")) == 4
    assert known_columns_for(data, "age") == ["age>=40"]


def test_sample_rows_is_reproducible(adult_data):
    a = sample_rows(adult_data, n=3, seed=0)
    b = sample_rows(adult_data, n=3, seed=0)
    pd.testing.assert_frame_equal(a.X, b.X)
    pd.testing.assert_series_equal(a.y, b.y)
    assert len(a.X) == 3
