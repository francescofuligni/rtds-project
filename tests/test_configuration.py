import argparse
from functools import partial

import numpy as np
import pandas as pd
import pytest
from configuration import (
    ExperimentConfig,
    add_configuration_arguments,
    configuration_from_args,
)
from run_final_experiment import main as final_main

from attack.conditions import run_condition, validate_knowledge_roles
from rashomon.data import BinarizedDataset, known_columns_for


def parse(*argv):
    parser = argparse.ArgumentParser()
    add_configuration_arguments(parser)
    return configuration_from_args(parser.parse_args(argv))


def test_release_grids():
    grid = parse().m_values
    assert grid == (1, 2, 3, 5, 7, 10, 14, 21, 30, 44, 64, 94, 137, 200)
    assert grid == tuple(sorted(set(grid)))
    assert parse("--k", "4", "--m-max", "1000").m_values == (1, 10, 100, 1000)
    assert parse("--k", "40", "--m-max", "1").m_values == (1,)
    assert parse("--k", "40", "--m-max", "2").m_values == (1, 2)
    assert parse("--n", "5000", "--attr-b", "age").n == 5000


def test_linear_release_grid_and_protocol():
    config = parse("--k-steps", "10", "--m-max", "25")
    assert config.m_values == (1, 11, 21, 25)
    assert config.release_protocol() == {
        "m_schedule": "linear",
        "m_min": 1,
        "m_max": 25,
        "k": None,
        "k_steps": 10,
        "m_values": [1, 11, 21, 25],
        "k_effective": 4,
        "m_rounding": "integer_step_with_max_endpoint",
        "include_all_trees": False,
    }


@pytest.mark.parametrize(
    "argv",
    [
        ("--n", "0"),
        ("--k", "0"),
        ("--k", "1"),
        ("--m-max", "0"),
        ("--k-steps", "0"),
        ("--k-steps", "501"),
        ("--attr-a", "sex"),
        ("--attr-b", "income"),
        ("--attr-b", "marital-status"),
        ("--attr-b", "occupation"),
    ],
)
def test_invalid_configuration(argv):
    with pytest.raises(ValueError):
        parse(*argv)


@pytest.mark.parametrize("option", ["--m-values", "--m-step", "--m-picks"])
def test_removed_grid_options_are_rejected(option):
    with pytest.raises(SystemExit):
        parse(option, "5")


def test_geometric_and_linear_grid_options_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse("--k", "10", "--k-steps", "5")


@pytest.mark.parametrize("module,entry", [("run_final_experiment", final_main)])
def test_cli_protocol_and_worker_share_configuration(
    monkeypatch, tmp_path, module, entry
):
    captured = {}

    def stamp(path, experiment, protocol, **kwargs):
        captured["protocol"] = protocol
        captured["path"] = path
        captured["invocation"] = kwargs["invocation"]

    def run_seeds(path, seeds, worker, **kwargs):
        assert isinstance(worker, partial)
        captured["config"] = worker.keywords["config"]
        captured["timeout"] = worker.keywords["timeout"]
        captured["workers"] = kwargs["workers"]
        captured["seeds"] = list(seeds)

    monkeypatch.setattr(f"{module}.stamp", stamp)
    monkeypatch.setattr(f"{module}.run_seeds", run_seeds)
    monkeypatch.setattr(
        f"{module}.experiment_result_dir", lambda experiment: tmp_path / experiment
    )
    entry(
        [
            "--n",
            "200",
            "--k",
            "3",
            "--m-max",
            "25",
            "--attr-b",
            "age",
            "--timeout",
            "17",
            "--seed-start",
            "2",
            "--n-seeds",
            "2",
        ]
    )
    assert captured["config"].n == 200
    assert captured["config"].m_values == (1, 5, 25)
    assert captured["config"].attr_a == "marital-status"
    assert captured["config"].attr_b == "age"
    assert captured["config"].include_all_trees is False
    assert captured["protocol"]["pilot_n"] == 200
    assert captured["protocol"]["m_values"] == [1, 5, 25]
    assert captured["protocol"]["m_schedule"] == "geometric"
    assert captured["protocol"]["k"] == 3
    assert captured["protocol"]["m_max"] == 25
    assert captured["protocol"]["k_effective"] == 3
    assert captured["protocol"]["attr_b"] == "age"
    assert captured["protocol"]["labels"] == "given_per_row"
    assert captured["protocol"]["solve_timeout"] == captured["timeout"] == 17
    assert captured["protocol"]["seed_workers"] == captured["workers"]
    assert captured["seeds"] == [2, 3]
    assert captured["invocation"] == {
        "seed_start": 2,
        "n_seeds": 2,
        "seed_end_exclusive": 4,
        "workers": captured["workers"],
    }
    assert captured["workers"] >= 1
    assert captured["path"] == tmp_path / module.removeprefix("run_")


@pytest.mark.parametrize(
    "module",
    ["run_final_experiment", "run_record_confidence"],
)
def test_output_directory_override_is_rejected(module):
    import importlib

    with pytest.raises(SystemExit) as exc:
        importlib.import_module(module).main(["--out-dir", "/tmp/custom-results"])
    assert exc.value.code == 2


def test_numeric_knowledge_and_role_boundary():
    data = BinarizedDataset(
        pd.DataFrame({"sex=Female": [0, 1], "age>=37": [1, 0], "edu": [0, 1]}),
        pd.Series([0, 1], name="income"),
        {},
        {"sex": "sex=Female", "education": "edu"},
    )
    assert known_columns_for(data, "age") == ["age>=37"]
    validate_knowledge_roles(data, "age", "education", "sex")
    for a, b in [("sex", "education"), ("age", "age"), ("income", "education")]:
        # Invalid knowledge must fail before touching trees or solving, even for None.
        with pytest.raises(ValueError):
            run_condition("none", None, data, a, b, "sex")
    data.binary_columns["alias"] = "sex=Female"
    with pytest.raises(ValueError):
        validate_knowledge_roles(data, "alias", "education", "sex")
    mask_cols = [data.X.columns.get_loc(c) for c in known_columns_for(data, "age")]
    assert np.array_equal(data.X.iloc[:, mask_cols].to_numpy().ravel(), [1, 0])


@pytest.mark.parametrize("all_trees", [False, True])
def test_configured_seed_with_real_solver(monkeypatch, all_trees):
    from types import SimpleNamespace

    import run_final_experiment as module

    data = BinarizedDataset(
        pd.DataFrame(
            {
                "sex=Female": [0, 1, 0, 1, 0, 1, 0, 1],
                "age>=37": [0, 0, 1, 1, 0, 0, 1, 1],
                "edu": [0, 1, 1, 0, 1, 0, 0, 1],
            }
        ),
        pd.Series([0, 1, 0, 1, 1, 0, 1, 0], name="income"),
        {},
        {"sex": "sex=Female", "education": "edu"},
    )
    tree = {"feature": 0, "true": {"prediction": 1}, "false": {"prediction": 0}}
    selected = []

    def build(X, y, **kwargs):
        assert len(X) == len(y) == 4
        return SimpleNamespace(get_tree_count=lambda: 3)

    def select(model, m, seed, pool):
        selected.append((m, seed, pool))
        return [tree] * m

    monkeypatch.setattr(module, "load_adult_raw", lambda path: None)
    monkeypatch.setattr(module, "binarize_adult_curated", lambda raw: data)
    monkeypatch.setattr(module, "build_rashomon_set", build)
    monkeypatch.setattr(module, "select_random_trees", select)
    monkeypatch.setattr(module, "HOLDOUT_N", 3)
    config = ExperimentConfig(
        4,
        k=2,
        m_max=2,
        attr_a="education",
        attr_b="age",
        include_all_trees=all_trees,
    )
    rows = module.run_seed(0, config=config)
    expected = [1, 2, 3] if all_trees else [1, 2]
    assert [s[0] for s in selected] == expected
    assert all(s[1:] == (20_000, 1000) for s in selected)
    assert {r["m"] for r in rows} == set(expected)
    assert {r["condition"] for r in rows} == {"none", "A", "B", "A+B"}
    assert len(rows) == 4 * len(expected)
    assert all(r["n_trees_available"] == 3 for r in rows)
    assert all(r["m_is_all_trees"] == (r["m"] == 3) for r in rows)
    if all_trees:
        protocol = module.protocol(config)
        assert protocol["include_all_trees"] is True
        assert protocol["tree_selection_pool"] == "all_available_per_seed"


def test_large_release_expands_declared_pool():
    import run_final_experiment as runner

    config = parse("--m-max", "1500", "--no-all-trees")
    assert runner.protocol(config)["tree_selection_pool"] == 1500


def test_runner_default_protocols():
    import run_final_experiment as main
    import run_record_confidence as record

    protocol = main.protocol()
    assert protocol["pilot_n"] == 100
    assert protocol["m_values"] == list(ExperimentConfig().m_values)
    assert protocol["m_schedule"] == "geometric"
    assert protocol["k"] == 15
    assert protocol["m_max"] == 200
    assert protocol["k_effective"] == 14
    assert protocol["solve_timeout"] == 180
    assert protocol["attr_a"] == "marital-status"
    assert protocol["attr_b"] == "education"
    assert protocol["include_all_trees"] is False
    record_protocol = record.protocol(20, 30, ["prior_only"], [10], 2)
    assert record_protocol["solver_threads"] == 8
    assert record_protocol["methods"] == ["prior_only"]
    assert record_protocol["m_values"] == [10]
    assert record_protocol["seed_workers"] == 2


def test_solver_threads_are_part_of_every_protocol():
    import run_final_experiment as main
    import run_record_confidence as record

    assert main.protocol(threads=2)["solver_threads"] == 2
    assert record.protocol(20, 30, threads=2)["solver_threads"] == 2
    with pytest.raises(SystemExit) as exc:
        record.main(["--threads", "0"])
    assert exc.value.code == 2


def test_record_confidence_result_sets_are_named_by_their_differences():
    from analyze_record_confidence import setting_labels

    old = {"tree_selection": "random"}  # stamped before the public-data keys
    small = {"tree_selection": "random", "public_n": 200, "public_prior": "records"}
    assert setting_labels([old, old]) == ["random", "random"]
    assert setting_labels([{"tree_selection": "first"}, old]) == ["first", "random"]
    assert setting_labels([old, small]) == ["public_n=2000", "public_n=200"]


def test_record_confidence_public_options_are_part_of_the_protocol():
    import run_record_confidence as record

    protocol = record.protocol(20, 30, public_n=200, public_prior="marginals")
    assert protocol["public_n"] == 200 and protocol["public_prior"] == "marginals"
    with pytest.raises(SystemExit):
        record.main(["--public-prior", "census"])
