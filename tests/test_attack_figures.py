import pandas as pd
import pytest
from analysis.attack_figures import aggregate


def test_missing_solves_do_not_enter_mean_or_sem():
    df = pd.DataFrame(
        {
            "seed": [0, 1, 2],
            "m": [1, 1, 1],
            "condition": ["none"] * 3,
            "status": ["OPTIMAL", "FEASIBLE", "UNKNOWN"],
            "reconstruction_error": [0.2, 0.4, float("nan")],
        }
    )
    row = aggregate(df).iloc[0]
    assert row["count"] == 2 and row["attempted"] == 3
    assert row["mean"] == pytest.approx(0.3)
    assert row["sem"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(pd.concat([df, df.iloc[:1]]))


def test_selection_uses_only_shared_valid_measurements():
    from analysis.attack_figures import paired_selection

    random = pd.DataFrame(
        {
            "seed": [0, 1, 0],
            "m": [10, 10, 200],
            "condition": ["A+B"] * 3,
            "status": ["OPTIMAL"] * 3,
            "reconstruction_error": [0.2, 0.3, 0.1],
        }
    )
    sequential = random.copy()
    sequential.loc[1, "reconstruction_error"] = float("nan")
    sequential.loc[1, "status"] = "UNKNOWN"
    result = paired_selection(random, sequential)
    assert len(result) == 4
    assert set(result.seed) == {0}
    assert set(result.m) == {10, 200}


def test_accuracy_contract_and_overlay(tmp_path):
    import json

    from analysis.attack_figures import load_accuracy
    from plotting.attack_figures import draw

    protocol = {"target": "sex"}
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "predictive_accuracy",
                "attack_protocol": protocol,
                "utility_metric": "majority_vote",
                "tie_policy": "smallest_class",
            }
        )
    )
    pd.DataFrame({"seed": [0, 1], "m": [1, 1], "accuracy": [0.7, 0.9]}).to_csv(
        tmp_path / "accuracy.csv", index=False
    )
    attack = pd.DataFrame({"seed": [0, 1], "m": [1, 1]})
    utility, label = load_accuracy(tmp_path, {"protocol": protocol}, attack)
    stats = pd.DataFrame(
        {"condition": ["none"], "m": [1], "mean": [0.3], "sem": [0.1], "count": [2]}
    )
    draw(
        stats,
        {"none": "None"},
        tmp_path / "overlay",
        {"none": ("black", "o", "None")},
        accuracy=utility,
        accuracy_label=label,
    )
    assert (tmp_path / "overlay.png").exists()
    assert not (tmp_path / "overlay.pdf").exists()
    with pytest.raises(ValueError, match="protocols"):
        load_accuracy(tmp_path, {"protocol": {}}, attack)
