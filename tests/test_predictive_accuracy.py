import json

import numpy as np
import pandas as pd
import pytest
import run_predictive_accuracy as utility
from analysis.attack_figures import load_accuracy
from run_final_experiment import protocol

from rashomon.build import build_rashomon_set
from rashomon.data import BinarizedDataset
from rashomon.utility import predict_tree, prefix_accuracy


def test_majority_vote_prefixes_and_ties():
    split = {"feature": 0, "true": {"prediction": 1}, "false": {"prediction": 0}}
    trees = [split, {"prediction": 0}, {"prediction": 1}]
    X, y = np.array([[0], [1]]), np.array([0, 1])
    assert prefix_accuracy(trees, X, y, [1, 2, 3]) == {
        1: 1.0,
        2: 0.5,
        3: 1.0,
    }
    with pytest.raises(ValueError, match="release sizes"):
        prefix_accuracy(trees, X, y, [4])


def test_routing_matches_native_treefarms_on_synthetic_data():
    X = pd.DataFrame({"a": [0, 0, 1, 1] * 8, "b": [0, 1, 0, 1] * 8})
    y = X.a ^ X.b
    model = build_rashomon_set(X, y, regularization=0.01, rashomon_bound_multiplier=0.1)
    for i in range(min(model.get_tree_count(), 10)):
        np.testing.assert_array_equal(
            predict_tree(model[i].source, X.to_numpy()), model[i].predict(X)
        )


@pytest.fixture
def saved_attack(tmp_path, monkeypatch):
    p = protocol()
    p.update(pilot_n=2, holdout_n=2)
    X = pd.DataFrame(
        {"sex": [0, 1] * 6, "marital": [0, 1] * 6, "education": [1, 0] * 6}
    )
    data = BinarizedDataset(
        X,
        X.sex.copy(),
        {},
        {"sex": "sex", "marital-status": "marital", "education": "education"},
    )
    tree = {"prediction": 0}

    class Model:
        def get_tree_count(self):
            return 2

        def __getitem__(self, index):
            from types import SimpleNamespace

            return SimpleNamespace(source=tree)

    monkeypatch.setattr(utility, "build_rashomon_set", lambda *a, **k: Model())
    monkeypatch.setattr(utility, "load_adult_raw", lambda *a: None)
    monkeypatch.setattr(utility, "binarize_adult_curated", lambda *a: data)
    attack = tmp_path / "attack"
    attack.mkdir()
    (attack / "manifest.json").write_text(
        json.dumps(
            {"experiment": "final_experiment", "protocol": p, "data": "seed*.csv"}
        )
    )
    for seed in [0, 1]:
        pd.DataFrame(
            [
                {
                    "seed": seed,
                    "m": m,
                    "n_trees_available": 2,
                    "condition": c,
                    "status": "UNKNOWN" if c == "none" else "OPTIMAL",
                    "reconstruction_error": float("nan") if c == "none" else 0.25,
                    "joint_split_frac_A": 0.0,
                    "joint_split_frac_B": 0.0,
                }
                for m in [1, 2]
                for c in ["none", "A", "B", "A+B"]
            ]
        ).to_csv(attack / f"seed{seed:03}.csv", index=False)
    out = tmp_path / "utility"

    def output(_):
        out.mkdir(exist_ok=True)
        return out

    monkeypatch.setattr(utility, "experiment_result_dir", output)
    return attack, out, p, data


def test_cli_output_overlays_exact_saved_keys(saved_attack):
    attack, out, p, _ = saved_attack
    utility.main(["--attack-dir", str(attack)])
    original, _ = utility.load_results(attack, "final_experiment")
    result, _ = load_accuracy(out, {"protocol": p}, original)
    assert len(result) == 4  # UNKNOWN attack errors do not remove utility measurements
    assert result.accuracy.between(0, 1).all()
    assert result.release_sha256.str.len().eq(64).all()
    assert json.loads((out / "outcome.json").read_text())["failures"] == []
    with pytest.raises(FileExistsError):
        utility.main(["--attack-dir", str(attack)])


def test_partial_pilot_is_rejected_for_full_overlay(saved_attack):
    attack, out, p, _ = saved_attack
    utility.main(["--attack-dir", str(attack), "--n-seeds", "1"])
    original, _ = utility.load_results(attack, "final_experiment")
    with pytest.raises(ValueError, match="exactly"):
        load_accuracy(out, {"protocol": p}, original)


@pytest.mark.parametrize("drift", ["count", "structure"])
def test_replay_drift_is_rejected(saved_attack, drift):
    attack, _, p, data = saved_attack
    original, _ = utility.load_results(attack, "final_experiment")
    original = original[original.seed == 0].copy()
    original["n_trees_available" if drift == "count" else "joint_split_frac_A"] = (
        3 if drift == "count" else 0.5
    )
    with pytest.raises(ValueError, match="differs"):
        utility.score_seed(0, original, p, data)


def test_seed_failure_does_not_prevent_other_seed(saved_attack, monkeypatch):
    attack, out, _, _ = saved_attack
    score = utility.score_seed

    def fail_first(seed, *args):
        if seed == 0:
            raise RuntimeError("synthetic failure")
        return score(seed, *args)

    monkeypatch.setattr(utility, "score_seed", fail_first)
    with pytest.raises(SystemExit) as exc:
        utility.main(["--attack-dir", str(attack)])
    assert exc.value.code == 1
    assert set(pd.read_csv(out / "accuracy.csv").seed) == {1}
    outcome = json.loads((out / "outcome.json").read_text())
    assert outcome["failures"][0]["seed"] == 0
