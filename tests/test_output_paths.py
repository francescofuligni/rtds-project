import json
from datetime import datetime

import pytest
from output_paths import (
    experiment_result_dir,
    latest_result_dir,
    new_timestamp,
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
