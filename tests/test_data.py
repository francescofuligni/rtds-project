from rashomon.data import binarize_adult, known_columns_for, load_adult_raw, sample_rows


def test_binarize_adult_is_all_binary():
    df = load_adult_raw()
    data = binarize_adult(df)
    assert set(data.X.values.ravel()).issubset({0, 1})
    assert set(data.y.unique()).issubset({0, 1})


def test_one_hot_groups_sum_to_one():
    df = load_adult_raw()
    data = binarize_adult(df)
    for cols in data.one_hot_groups.values():
        assert (data.X[cols].sum(axis=1) == 1).all()


def test_known_columns_for_target_and_background_attrs():
    df = load_adult_raw()
    data = binarize_adult(df)
    assert known_columns_for(data, "sex") == ["sex=Female"]
    assert len(known_columns_for(data, "marital-status")) > 1
    assert len(known_columns_for(data, "education")) > 1


def test_sample_rows_is_reproducible():
    df = load_adult_raw()
    data = binarize_adult(df)
    a = sample_rows(data, n=100, seed=0)
    b = sample_rows(data, n=100, seed=0)
    assert a.X.equals(b.X)
    assert len(a.X) == 100


def test_cached_column_padding_is_normalized_without_rewriting_data(tmp_path):
    import pandas as pd

    from rashomon.data import COLUMNS

    cache = tmp_path / "adult.data.csv"
    frame = pd.DataFrame({f" {name} ": [1] for name in COLUMNS})
    frame.to_csv(cache, index=False)
    before = cache.read_bytes()
    data = load_adult_raw(tmp_path)
    assert data.columns.to_list() == COLUMNS
    assert cache.read_bytes() == before
