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

ADULT_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

CONTINUOUS_COLUMNS = [
    "age",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
# fnlwgt (census sampling weight) and education-num (a redundant integer
# encoding of `education`) are dropped: they add columns without adding
# modelling signal, and for n=100 laptop-scale Rashomon sets, feature
# count matters a lot for tractability.

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
]

# sex is binary in the raw data, kept as a single 0/1 column.
# native-country is collapsed to US-vs-other before binarization: in
# raw form it has ~41 categories (mostly near-empty), which is a large,
# low-signal contributor to feature count relative to its predictive
# value -- a standard simplification for this dataset.
BINARY_COLUMNS = ["sex", "native-country"]

LABEL_COLUMN = "income"


@dataclass
class BinarizedDataset:
    X: pd.DataFrame  # all-binary feature columns
    y: pd.Series  # binary label (income >50K = 1)
    one_hot_groups: dict[
        str, list[str]
    ]  # original categorical col -> binarized columns
    binary_columns: dict[str, str]  # original binary col -> its single binarized column


def load_adult_raw(cache_dir: str | Path = "data") -> pd.DataFrame:
    """Download (and cache) the raw UCI Adult training split."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "adult.data.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        df = pd.read_csv(
            ADULT_TRAIN_URL,
            header=None,
            names=COLUMNS,
            skipinitialspace=True,
            na_values="?",
        )
        df.to_csv(cache_path, index=False)

    df = df.dropna().reset_index(drop=True)
    for col in CATEGORICAL_COLUMNS + BINARY_COLUMNS + [LABEL_COLUMN]:
        df[col] = df[col].astype(str).str.strip().str.rstrip(".")
    return df


# The "positive" (==1) category used for each two-valued attribute's
# single indicator column.
_BINARY_POSITIVE_CATEGORY = {
    "sex": "Female",
    "native-country": None,  # handled specially: "not United-States" -> 1
}


def binarize_adult(df: pd.DataFrame, n_quantile_bins: int = 3) -> BinarizedDataset:
    """Binarize every feature column; continuous columns become
    `col >= q_k` threshold indicators at quantile cut points, categorical
    columns become one-hot indicators, and each two-valued attribute
    (sex, native-country) becomes a single indicator column.
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

    for col in BINARY_COLUMNS:
        if col == "native-country":
            name = "native-country=Other"
            indicator = (df[col] != "United-States").astype(int)
        else:
            positive = _BINARY_POSITIVE_CATEGORY[col]
            name = f"{col}={positive}"
            indicator = (df[col] == positive).astype(int)
        columns.append(pd.Series(indicator, name=name))
        binary_columns[col] = name

    X = pd.concat(columns, axis=1)
    y = (df[LABEL_COLUMN] == ">50K").astype(int)
    y.name = LABEL_COLUMN

    return BinarizedDataset(
        X=X, y=y, one_hot_groups=one_hot_groups, binary_columns=binary_columns
    )


# Collapsed categories for the curated, laptop-tractable feature set.
# TreeFARMS's exhaustive Rashomon-set search does not scale to the full
# ~60-column binarization on n=100 rows within a reasonable time on a
# laptop (see commit message / project notes): at a regularization loose
# enough to produce a non-trivial Rashomon set, trees never got deep
# enough to split on T=sex at all among 62 competing features, making
# every background-knowledge condition identical by construction. This
# curated set keeps T, A, B plus a handful of the next-most-predictive
# attributes, trading completeness for trees that actually use T.
_MARITAL_STATUS_GROUPS = {
    "Married-civ-spouse": "Married",
    "Married-AF-spouse": "Married",
    "Married-spouse-absent": "Married",
    "Never-married": "Never-married",
    "Divorced": "Divorced",
    "Separated": "Widowed-Separated",
    "Widowed": "Widowed-Separated",
}
_EDUCATION_GROUPS = {
    "Preschool": "Below-HS",
    "1st-4th": "Below-HS",
    "5th-6th": "Below-HS",
    "7th-8th": "Below-HS",
    "9th": "Below-HS",
    "10th": "Below-HS",
    "11th": "Below-HS",
    "12th": "Below-HS",
    "HS-grad": "HS-grad",
    "Some-college": "Some-college",
    "Assoc-acdm": "Some-college",
    "Assoc-voc": "Some-college",
    "Bachelors": "Bachelors",
    "Masters": "Grad+",
    "Doctorate": "Grad+",
    "Prof-school": "Grad+",
}

CURATED_CONTINUOUS_COLUMNS = ["age", "hours-per-week"]


def binarize_adult_curated(
    df: pd.DataFrame, n_quantile_bins: int = 2
) -> BinarizedDataset:
    """A small, laptop-tractable binarization: T=sex, A=marital-status
    (collapsed to 4 categories), B=education (collapsed to 5 categories),
    plus age/hours-per-week thresholds and a capital-gain>0 indicator as
    the other predictive features available for the trees to split on.
    """
    columns = []
    one_hot_groups: dict[str, list[str]] = {}
    binary_columns: dict[str, str] = {}

    for col in CURATED_CONTINUOUS_COLUMNS:
        quantiles = np.unique(
            df[col].quantile(np.linspace(0, 1, n_quantile_bins + 1)[1:-1])
        )
        for q in quantiles:
            name = f"{col}>={q:g}"
            columns.append(pd.Series((df[col] >= q).astype(int), name=name))

    cap_gain_name = "capital-gain>0"
    columns.append(pd.Series((df["capital-gain"] > 0).astype(int), name=cap_gain_name))
    binary_columns["capital-gain"] = cap_gain_name

    for col, groups in [
        ("marital-status", _MARITAL_STATUS_GROUPS),
        ("education", _EDUCATION_GROUPS),
    ]:
        collapsed = df[col].map(groups)
        dummies = pd.get_dummies(collapsed, prefix=col).astype(int)
        one_hot_groups[col] = list(dummies.columns)
        columns.append(dummies)

    sex_name = "sex=Female"
    columns.append(pd.Series((df["sex"] == "Female").astype(int), name=sex_name))
    binary_columns["sex"] = sex_name

    X = pd.concat(columns, axis=1)
    y = (df[LABEL_COLUMN] == ">50K").astype(int)
    y.name = LABEL_COLUMN

    return BinarizedDataset(
        X=X, y=y, one_hot_groups=one_hot_groups, binary_columns=binary_columns
    )


def sample_rows(
    data: BinarizedDataset, n: int = 100, seed: int = 0
) -> BinarizedDataset:
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
