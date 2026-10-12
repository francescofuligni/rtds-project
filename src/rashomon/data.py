"""Loading and binarization for the UCI Adult dataset.

Produces an all-binary feature table so TreeFARMS can build a Rashomon
set, and keeps track of which binarized columns came from which original
attribute so the reconstruction attack can treat a one-hot group as a
single known/unknown unit (sex, marital-status, education, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

CONTINUOUS_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "native-country",
]

# sex is binary in the raw data, so it is kept as a single 0/1 column
# instead of a one-hot group.
BINARY_COLUMNS = ["sex"]

LABEL_COLUMN = "income"


@dataclass
class BinarizedDataset:
    X: pd.DataFrame  # all-binary feature columns
    y: pd.Series  # binary label (income >50K = 1)
    one_hot_groups: dict[str, list[str]]  # original categorical col -> binarized columns
    binary_columns: dict[str, str]  # original binary col -> its single binarized column


def load_adult_raw(cache_dir: str | Path = "data") -> pd.DataFrame:
    """Download (and cache) the raw UCI Adult training split."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "adult.data.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        df = pd.read_csv(ADULT_TRAIN_URL, header=None, names=COLUMNS,
                          skipinitialspace=True, na_values="?")
        df.to_csv(cache_path, index=False)

    df = df.dropna().reset_index(drop=True)
    for col in CATEGORICAL_COLUMNS + BINARY_COLUMNS + [LABEL_COLUMN]:
        df[col] = df[col].astype(str).str.strip().str.rstrip(".")
    return df


def binarize_adult(df: pd.DataFrame, n_quantile_bins: int = 4) -> BinarizedDataset:
    """Binarize every feature column; continuous columns become
    `col >= q_k` threshold indicators at quantile cut points, categorical
    columns become one-hot indicators, and the (already binary) sex
    column is mapped to a single `sex=Female` indicator.
    """
    columns = []
    one_hot_groups: dict[str, list[str]] = {}
    binary_columns: dict[str, str] = {}

    for col in CONTINUOUS_COLUMNS:
        quantiles = np.unique(
            df[col].quantile(np.linspace(0, 1, n_quantile_bins + 1)[1:-1])
        )
        for q in quantiles:
            name = f"{col}>={q:g}"
            columns.append(pd.Series((df[col] >= q).astype(int), name=name))

    for col in CATEGORICAL_COLUMNS:
        dummies = pd.get_dummies(df[col], prefix=col).astype(int)
        one_hot_groups[col] = list(dummies.columns)
        columns.append(dummies)

    # sex: Female -> 1, Male -> 0 (two categories, one indicator column)
    sex_name = "sex=Female"
    columns.append(pd.Series((df["sex"] == "Female").astype(int), name=sex_name))
    binary_columns["sex"] = sex_name

    X = pd.concat(columns, axis=1)
    y = (df[LABEL_COLUMN] == ">50K").astype(int)
    y.name = LABEL_COLUMN

    return BinarizedDataset(X=X, y=y, one_hot_groups=one_hot_groups, binary_columns=binary_columns)


def sample_rows(data: BinarizedDataset, n: int = 100, seed: int = 0) -> BinarizedDataset:
    """Draw a fixed-seed subsample of n rows (used for the n=100 pilot/final setup)."""
    idx = data.X.sample(n=n, random_state=seed).index
    return BinarizedDataset(
        X=data.X.loc[idx].reset_index(drop=True),
        y=data.y.loc[idx].reset_index(drop=True),
        one_hot_groups=data.one_hot_groups,
        binary_columns=data.binary_columns,
    )


def known_columns_for(data: BinarizedDataset, attribute: str) -> list[str]:
    """Return the binarized column name(s) backing an original attribute
    (e.g. 'sex' -> ['sex=Female'], 'marital-status' -> its one-hot group).
    """
    if attribute in data.binary_columns:
        return [data.binary_columns[attribute]]
    if attribute in data.one_hot_groups:
        return data.one_hot_groups[attribute]
    raise KeyError(f"Unknown attribute: {attribute}")
