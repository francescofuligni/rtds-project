"""Bounded integration checks; no Adult experiment is run."""

from datetime import UTC

import numpy as np
import pandas as pd
import pytest
import run_final_experiment as runner
from analyze_results import paired_comparisons
from experiment_io import load_results, metadata, write_rows

from attack.metrics import target_error
from attack.reconstruct import prepare_release
from rashomon.data import BinarizedDataset


@pytest.fixture
def runner_fixture(monkeypatch):
    X = pd.DataFrame(
        np.tile([[1, 0, 0, 0], [1, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], (3, 1)),
        columns=["marital_0", "marital_1", "edu_hi", "sex=Female"],
    )
    data = BinarizedDataset(
        X,
        X["sex=Female"].rename("income"),
        {"marital-status": ["marital_0", "marital_1"]},
        {"education": "edu_hi", "sex": "sex=Female"},
    )
    trees = [
        {"feature": 0, "true": {"prediction": 1}, "false": {"prediction": 0}},
        {"feature": 3, "true": {"prediction": 1}, "false": {"prediction": 0}},
    ]

    class Model:
        def get_tree_count(self):
            return len(trees)

    monkeypatch.setattr(runner, "load_adult_raw", lambda: data)
    monkeypatch.setattr(runner, "binarize_adult_curated", lambda raw: raw)
    monkeypatch.setattr(runner, "build_rashomon_set", lambda *a, **kw: Model())
    monkeypatch.setattr(runner, "select_trees", lambda model, m: trees[:m])
    monkeypatch.setattr(runner, "PILOT_N", 4)
    monkeypatch.setattr(runner, "HOLDOUT_N", 3)
    monkeypatch.setattr(runner, "M_VALUES", [1, 2])
    monkeypatch.setattr(runner, "SOLVE_TIMEOUT", 5)
    return data


def test_runner_pairs_release_and_scores_controls_with_same_permutation(
    runner_fixture, monkeypatch
):
    calls = []
    original = runner.run_condition

    def capture(condition, release, data, *args, **kwargs):
        result = original(condition, release, data, *args, **kwargs)
        calls.append((condition, release, data, result))
        return result

    monkeypatch.setattr(runner, "run_condition", capture)
    guess = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    monkeypatch.setattr(runner, "uniform_random_guess", lambda *args, **kwargs: guess)
    monkeypatch.setattr(runner, "marginal_baseline", lambda *args, **kwargs: guess)
    rows = runner.run_seed(0)
    assert len(rows) == 8
    for batch in [calls[:4], calls[4:]]:
        assert len({id(call[1]) for call in batch}) == 1
    for row, (_, release, data, result) in zip(rows, calls):
        assert result.matching is not None
        expected = target_error(
            data.X.to_numpy(), guess, [3], result.matching.permutation
        )
        assert (
            row["uniform_baseline_error"] == row["marginal_baseline_error"] == expected
        )
        assert row["release_id"] == release.release_id
        assert row["timeout_s"] == 5 and row["solver_workers"] == 1
        assert row["protocol_version"] == "no_y_v1"


def test_runner_propagates_errors(runner_fixture, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic solver crash")

    monkeypatch.setattr(runner, "run_condition", fail)
    with pytest.raises(RuntimeError, match="synthetic solver crash"):
        runner.run_seed(0)


def test_runner_stops_without_a_candidate(runner_fixture, monkeypatch):
    from attack.conditions import ConditionResult

    monkeypatch.setattr(
        runner,
        "run_condition",
        lambda *a, **kw: ConditionResult("none", "UNKNOWN", 0, None, None),
    )
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        runner.run_seed(0)


def test_no_available_release_is_not_an_empty_success(
    runner_fixture, monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "M_VALUES", [3])
    rows = runner.run_seed(0)
    assert rows == []
    with pytest.raises(ValueError, match="no results"):
        write_rows(rows, tmp_path / "empty.csv")


def test_csv_preserves_history_and_input_rejects_protocol_mixing(tmp_path):
    release = prepare_release(
        [{"prediction": 0}], np.array([[0], [1]]), np.array([0, 1]), ["T"], []
    )
    row = metadata(0, 1, "none", release, 5)
    row.update(reconstruction_error=0.5)
    path = tmp_path / "final_experiment.csv"
    write_rows([row], path)
    row["reconstruction_error"] = 0.25
    with pytest.raises(FileExistsError):
        write_rows([row], path)
    assert load_results([path])["reconstruction_error"].tolist() == [0.5]
    other = row.copy()
    other.update(seed=1, protocol_version="known_y_legacy")
    second = tmp_path / "other.csv"
    write_rows([other], second)
    with pytest.raises(ValueError, match="mixed"):
        load_results([path, second])
    with pytest.raises(ValueError, match="duplicate"):
        load_results([path, path])
    legacy = tmp_path / "legacy.csv"
    pd.DataFrame([{"seed": 0, "reconstruction_error": 0.5}]).to_csv(legacy, index=False)
    with pytest.raises(ValueError, match="old known-Y"):
        load_results([legacy])


def test_unavailable_comparisons_are_not_counted_as_losses():
    rows = [
        {"seed": 0, "m": 5, "condition": "none", "reconstruction_error": 0.4},
        {"seed": 0, "m": 5, "condition": "A", "reconstruction_error": 0.2},
        {"seed": 1, "m": 5, "condition": "none", "reconstruction_error": 0.6},
        {"seed": 1, "m": 5, "condition": "A", "reconstruction_error": None},
    ]
    result = paired_comparisons(pd.DataFrame(rows)).set_index("condition")
    assert result.loc["A", "n_pairs"] == 1
    assert result.loc["A", "beat_none_fraction"] == 1
    assert result.loc["B", "n_pairs"] == 0
    assert pd.isna(result.loc["B", "beat_none_fraction"])
    assert pd.isna(result.loc["A", "p_value"])


def test_timestamped_runs_preserve_previous_csv_even_with_clock_collision(
    tmp_path, monkeypatch
):
    from datetime import datetime

    import experiment_io

    class FixedClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 20, 12, 30, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(experiment_io, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(experiment_io, "datetime", FixedClock)
    first = experiment_io.experiment_output("final_experiment.csv")
    first.write_text("original")
    second = experiment_io.experiment_output("final_experiment.csv")
    assert first.parent != second.parent
    assert first.read_text() == "original"
    assert second.parent.is_dir()
    assert first.parent.name == "20261020_143000"
    custom = experiment_io.experiment_output("ignored.csv", tmp_path / "custom.csv")
    assert custom.name == "custom.csv" and custom.parent.parent == tmp_path
    assert custom.parent not in (first.parent, second.parent)


@pytest.mark.parametrize(
    ("module_name", "filename"),
    [
        ("run_final_experiment", "final_experiment.csv"),
        ("run_calibration_experiment", "calibration_experiment.csv"),
    ],
)
def test_runner_main_creates_separate_run_directories(
    tmp_path, monkeypatch, module_name, filename
):
    import importlib

    import experiment_io

    module = importlib.import_module(module_name)

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def imap_unordered(self, function, seeds):
            return [[{"seed": seed, "m": 1, "condition": "none"}] for seed in seeds]

    monkeypatch.setattr(experiment_io, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(module, "Pool", Pool)
    for _ in range(2):
        module.main(["--seed-end", "1", "--workers", "1"])
    files = sorted(tmp_path.glob(f"*/{filename}"))
    assert len(files) == 2
    assert all(pd.read_csv(path)["seed"].tolist() == [0] for path in files)
    module.main(["--seed-end", "1", "--workers", "1"], output_name="part2.csv")
    assert len(list(tmp_path.glob("*/part2.csv"))) == 1


@pytest.mark.parametrize("diagnostic", [False, True])
def test_analysis_defaults_to_latest_matching_run(tmp_path, monkeypatch, diagnostic):
    import sys

    import experiment_io

    name = "calibration_experiment.csv" if diagnostic else "final_experiment.csv"
    other = "final_experiment.csv" if diagnostic else "calibration_experiment.csv"
    for stamp, filename in [
        ("20261020T120000.000000Z", name),
        ("20261020T130000.000000Z", name.replace(".csv", "_part2.csv")),
        ("20261020T140000.000000Z", other),
        ("20261020_150000-9", name),
        ("20261020_150000-10", name),
    ]:
        folder = tmp_path / stamp
        folder.mkdir()
        (folder / filename).touch()
    (tmp_path / name).write_text("legacy")
    monkeypatch.setattr(experiment_io, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["analysis"])
    args = experiment_io.analysis_arguments(diagnostic=diagnostic)
    assert args.input[0].parent.name == "20261020_150000-10"
    assert args.output_dir == args.input[0].parent


def test_analysis_explicit_inputs_route_outputs_and_require_merge_destination(
    tmp_path, monkeypatch
):
    import sys

    import experiment_io

    first = tmp_path / "one" / "final_experiment.csv"
    second = tmp_path / "two" / "final_experiment.csv"
    monkeypatch.setattr(sys, "argv", ["analysis", "--input", str(first)])
    assert experiment_io.analysis_arguments().output_dir == first.parent
    monkeypatch.setattr(sys, "argv", ["analysis", "--input", str(first), str(second)])
    with pytest.raises(SystemExit):
        experiment_io.analysis_arguments()
    merged = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        ["analysis", "--input", str(first), str(second), "--output-dir", str(merged)],
    )
    assert experiment_io.analysis_arguments().output_dir == merged


def test_analysis_without_new_runs_requires_explicit_legacy_input(
    tmp_path, monkeypatch
):
    import sys

    import experiment_io

    (tmp_path / "final_experiment.csv").touch()
    monkeypatch.setattr(experiment_io, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["analysis"])
    with pytest.raises(SystemExit):
        experiment_io.analysis_arguments()
