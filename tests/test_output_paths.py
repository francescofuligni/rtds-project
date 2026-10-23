import json
from datetime import datetime

import pytest
from output_paths import (
    FIGURES_ROOT,
    RESULTS_ROOT,
    experiment_result_dir,
    figure_output_dir,
    latest_result_dir,
    new_timestamp,
    require_saved_result_dir,
)


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 20, 14, 16, 41, tzinfo=tz)


def test_new_timestamp_adds_suffix_when_timestamp_folder_exists(tmp_path, monkeypatch):
    import output_paths

    monkeypatch.setattr(output_paths, "datetime", FixedDateTime)
    (tmp_path / "20261020_141641").mkdir()
    assert new_timestamp(tmp_path) == "20261020_141641_01"


def test_figure_output_dir_is_shared_repo_folder():
    assert figure_output_dir() == FIGURES_ROOT


def test_figure_sources_must_be_saved_under_results(tmp_path):
    saved = require_saved_result_dir(
        RESULTS_ROOT / "20261023_120000" / "final_experiment"
    )
    assert saved == RESULTS_ROOT / "20261023_120000" / "final_experiment"
    with pytest.raises(ValueError, match="stored under"):
        require_saved_result_dir(tmp_path)


def test_experiment_result_dir_uses_supplied_timestamp(tmp_path, monkeypatch):
    import output_paths

    monkeypatch.setattr(output_paths, "RESULTS_ROOT", tmp_path)
    assert experiment_result_dir("final_experiment", "20261020_141641") == (
        tmp_path / "20261020_141641" / "final_experiment"
    )


def test_experiment_result_dir_rejects_invalid_timestamp(tmp_path, monkeypatch):
    import output_paths

    monkeypatch.setattr(output_paths, "RESULTS_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Invalid RTDS run timestamp"):
        experiment_result_dir("final_experiment", timestamp="latest")


def test_latest_result_dir_uses_latest_timestamp_and_ignores_legacy(tmp_path):
    old = tmp_path / "20261019_141641" / "final_experiment"
    new = tmp_path / "20261020_141641" / "final_experiment"
    legacy = tmp_path / "legacy" / "final_experiment"
    for path in (old, new, legacy):
        path.mkdir(parents=True)
        (path / "seed000.csv").touch()
        (path / "manifest.json").write_text(
            json.dumps({"experiment": "final_experiment", "data": "seed*.csv"})
        )

    assert latest_result_dir("final_experiment", tmp_path) == new


def test_latest_result_dir_fails_without_timestamped_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="No timestamped"):
        latest_result_dir("final_experiment", tmp_path)


def test_internal_timestamp_keeps_families_in_one_fixed_root(tmp_path, monkeypatch):
    import output_paths

    monkeypatch.setattr(output_paths, "RESULTS_ROOT", tmp_path)
    monkeypatch.setenv("RTDS_RUN_TIMESTAMP", "20261020_141641")
    for family in ("final_experiment", "record_confidence"):
        assert experiment_result_dir(family) == tmp_path / "20261020_141641" / family
    monkeypatch.setenv("RTDS_RUN_TIMESTAMP", "../../outside")
    with pytest.raises(ValueError, match="Invalid RTDS run timestamp"):
        experiment_result_dir("final_experiment")


def test_make_pipeline_keeps_timestamp_and_rejects_removed_overrides():
    import os
    import re
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k != "RTDS_RUN_TIMESTAMP"}
    dry_run = subprocess.run(
        ["make", "-n", "run"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    timestamps = re.findall(r"results/(\d{8}_\d{6}(?:_\d+)?)/", dry_run.stdout)
    assert len(timestamps) == 3
    assert len(set(timestamps)) == 1
    assert "--out-dir" not in dry_run.stdout
    assert "--k 15 --m-max 200" in dry_run.stdout
    assert "--seed-start 0 --n-seeds 350" in dry_run.stdout
    assert "--seed-start 0 --n-seeds 20" in dry_run.stdout
    linear_run = subprocess.run(
        ["make", "-n", "run-final", "K_STEPS=10", "M_MAX=25"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--k-steps 10 --m-max 25" in linear_run.stdout
    override_run = subprocess.run(
        ["make", "-n", "run-final", "N_SEEDS=3", "SEED_START=20"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--seed-start 20 --n-seeds 3" in override_run.stdout
    for override in (
        "RUN_DIR=/tmp/custom",
        "M_STEP=5",
        "M_VALUES=1 10",
        "M_PICKS=40",
        "SEED_END=20",
        "RECORD_THREADS=2",
    ):
        result = subprocess.run(
            ["make", "-n", "run-final", override],
            check=False,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
    for overrides in (("K=10", "K_STEPS=5"),):
        result = subprocess.run(
            ["make", "-n", "run-final", *overrides],
            check=False,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


def test_standalone_invocations_allocate_distinct_runs(tmp_path, monkeypatch):
    import output_paths

    monkeypatch.delenv("RTDS_RUN_TIMESTAMP", raising=False)
    monkeypatch.setattr(output_paths, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(output_paths, "datetime", FixedDateTime)
    first = experiment_result_dir("final_experiment")
    second = experiment_result_dir("final_experiment")
    assert first.parent.name == "20261020_141641"
    assert second.parent.name == "20261020_141641_01"
    assert first != second


def test_record_method_subset_writes_a_fresh_run(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import run_record_confidence as runner

    captured = {}

    def run_seed(seed, m, samples, timeout, methods, **settings):
        captured["methods"] = methods
        captured["tree_selection"] = settings["tree_selection"]
        return [{"seed": seed, "m": m, "method": "prior_only"}]

    monkeypatch.setattr(runner, "run_seed", run_seed)
    args = SimpleNamespace(
        out_dir=tmp_path,
        only=["prior_only"],
        samples=2,
        timeout=1,
        threads=1,
        public_n=2000,
        public_prior="records",
        tree_selection="sequential",
    )
    runner._run_and_write(0, 10, args)
    assert captured["tree_selection"] == "sequential"
    assert captured["methods"] == {"prior_only"}
    assert (tmp_path / "m010" / "seed000.csv").is_file()
