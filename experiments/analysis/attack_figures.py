"""Validated inputs and aggregation for the poster attack figures."""

import json
from pathlib import Path

import pandas as pd

from rashomon.utility import UTILITY_PROTOCOL


def embedded_accuracy(
    attack_manifest: dict, attack: pd.DataFrame
) -> pd.DataFrame | None:
    """Read utility measured on the original in-memory release, once per seed/m."""
    declared = attack_manifest["protocol"].get("predictive_utility")
    if declared is None and "predictive_accuracy" not in attack:
        return None  # Historical reconstruction-only archive.
    if declared != UTILITY_PROTOCOL:
        raise ValueError("Unsupported embedded predictive utility protocol")
    if (
        "predictive_accuracy" not in attack
        or not attack.predictive_accuracy.between(0, 1).all()
    ):
        raise ValueError("Missing or invalid embedded predictive accuracy")
    grouped = attack.groupby(["seed", "m"]).predictive_accuracy.nunique()
    if not grouped.eq(1).all():
        raise ValueError("Predictive accuracy differs across knowledge conditions")
    return (
        attack[["seed", "m", "predictive_accuracy"]]
        .drop_duplicates()
        .rename(columns={"predictive_accuracy": "accuracy"})
    )


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise valid errors; missing solves are never assigned an error."""
    if df.duplicated(["seed", "m", "condition"]).any():
        raise ValueError("Duplicate seed/m/condition keys")
    errors = df["reconstruction_error"].dropna()
    if not errors.between(0, 1).all():
        raise ValueError("Reconstruction errors must be in [0, 1]")
    valid = df.dropna(subset=["reconstruction_error"])
    if not valid["status"].isin(["OPTIMAL", "FEASIBLE"]).all():
        raise ValueError("An error is attached to an unsuccessful solve")
    stats = (
        df.groupby(["m", "condition"])["reconstruction_error"]
        .agg(mean="mean", std="std", count="count", attempted="size")
        .reset_index()
    )
    stats["sem"] = stats["std"] / stats["count"] ** 0.5
    return stats


def paired_selection(random: pd.DataFrame, sequential: pd.DataFrame) -> pd.DataFrame:
    """Keep identical measured seed/m pairs with valid errors in both policies."""
    selected = []
    for frame in (random, sequential):
        sub = frame[frame.condition == "A+B"].copy()
        aggregate(sub)  # validate duplicate keys and solver outcomes before joining
        selected.append(sub.dropna(subset=["reconstruction_error"]))
    keys = selected[0][["seed", "m"]].merge(
        selected[1][["seed", "m"]], on=["seed", "m"]
    )
    if keys.empty:
        raise ValueError("No shared valid seed/m measurements for selection comparison")
    frames = []
    for name, frame in zip(("random", "sequential"), selected, strict=True):
        sub = frame.merge(keys, on=["seed", "m"])
        sub["condition"] = name
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def load_accuracy(
    path: Path, attack_manifest: dict, attack: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Read a utility-only output; never infer utility from T error."""
    metadata = json.loads((path / "manifest.json").read_text())
    if metadata["experiment"] != "predictive_accuracy":
        raise ValueError("Expected predictive_accuracy results")
    if metadata["attack_protocol"] != attack_manifest["protocol"]:
        raise ValueError("Accuracy and attack protocols differ")
    if metadata.get("utility_metric") != "majority_vote":
        raise ValueError(
            "Accuracy must measure majority voting over the released trees"
        )
    if metadata.get("tie_policy") != "smallest_class":
        raise ValueError("Expected deterministic smallest-class tie policy")
    label = "Accuracy (vote)"
    df = pd.read_csv(path / "accuracy.csv")
    if df.duplicated(["seed", "m"]).any() or not df.accuracy.between(0, 1).all():
        raise ValueError("Invalid accuracy rows")
    expected = attack[["seed", "m"]].drop_duplicates()
    if set(map(tuple, df[["seed", "m"]].to_numpy())) != set(
        map(tuple, expected.to_numpy())
    ):
        raise ValueError("Accuracy must cover exactly the measured attack seed/m keys")
    return df, label
