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


def test_embedded_accuracy_is_counted_once_per_release():
    from analysis.attack_figures import embedded_accuracy

    from rashomon.utility import UTILITY_PROTOCOL

    df = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "m": [1] * 4,
            "predictive_accuracy": [0.8, 0.8, 0.9, 0.9],
        }
    )
    manifest = {"protocol": {"predictive_utility": UTILITY_PROTOCOL}}
    result = embedded_accuracy(manifest, df)
    assert len(result) == 2
    assert result.accuracy.mean() == pytest.approx(0.85)
    df.loc[0, "predictive_accuracy"] = 0.5
    with pytest.raises(ValueError, match="differs"):
        embedded_accuracy(manifest, df)
    df.loc[0, "predictive_accuracy"] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        embedded_accuracy(manifest, df)
    with pytest.raises(ValueError, match="Missing"):
        embedded_accuracy(manifest, df.drop(columns="predictive_accuracy"))
    assert (
        embedded_accuracy({"protocol": {}}, df.drop(columns="predictive_accuracy"))
        is None
    )


def test_background_summary_deduplicates_controls_and_declares_none_reference():
    from analysis.attack_figures import background_summary

    education = pd.DataFrame(
        [
            {
                "seed": seed,
                "m": 1,
                "condition": condition,
                "status": "OPTIMAL",
                "reconstruction_error": error,
                "uniform_baseline_error": 0.4 + seed * 0.1,
                "marginal_baseline_error": 0.35,
            }
            for seed in [0, 1]
            for condition, error in [("none", 0.3), ("A+B", 0.2)]
        ]
    )
    age = education.copy()
    age.loc[age.condition == "none", "reconstruction_error"] = 0.4
    age.loc[age.condition == "A+B", "reconstruction_error"] = 0.1
    result = background_summary(education, age).set_index("condition")
    assert result.loc["none", "mean"] == pytest.approx(0.3)
    assert result.loc["age", "mean"] == pytest.approx(0.1)
    assert result.loc["education", "mean"] == pytest.approx(0.2)
    assert result.loc["uniform", "count"] == 2
    assert result.loc["uniform", "mean"] == pytest.approx(0.45)
    assert result.loc["marginal", "mean"] == pytest.approx(0.35)
    age.loc[0, "uniform_baseline_error"] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        background_summary(education, age)
    age.loc[age.seed == 0, "uniform_baseline_error"] = 0.6
    with pytest.raises(ValueError, match="between"):
        background_summary(education, age)


def test_none_never_has_uncertainty_shading(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from plotting.attack_figures import draw

    figures = []
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    stats = pd.DataFrame(
        {"condition": ["none"], "m": [1], "mean": [0.3], "sem": [0.1], "count": [2]}
    )
    draw(
        stats,
        {"none": "None"},
        tmp_path / "none",
        {"none": ("black", "o", "None")},
        uncertainty=True,
    )
    assert not figures[0].axes[0].collections
    assert all("Mean" not in text.get_text() for text in figures[0].texts)
    figures[0].clear()
