import json

import pytest
from provenance import (
    MANIFEST,
    ProtocolMismatch,
    available_cpu_count,
    describe,
    load_results,
    require_comparable,
    run_seeds,
    seed_indices,
    stamp,
    write_rows,
)


def test_available_cpu_count_falls_back_when_affinity_is_unavailable(monkeypatch):
    monkeypatch.delattr("provenance.os.sched_getaffinity", raising=False)
    monkeypatch.setattr("provenance.os.cpu_count", lambda: 6)
    assert available_cpu_count() == 6


def test_stamp_creates_manifest_and_records_each_run(tmp_path):
    stamp(
        tmp_path,
        "exp",
        {"tree_selection": "random", "m_values": (10, 50)},
        data="seed*.csv",
        argv=["--a"],
    )
    manifest = stamp(
        tmp_path,
        "exp",
        {"tree_selection": "random", "m_values": [10, 50]},
        data="seed*.csv",
    )
    assert manifest["protocol"] == {"tree_selection": "random", "m_values": [10, 50]}
    assert len(manifest["runs"]) == 2
    assert manifest["runs"][0]["command"][1:] == ["--a"]
    assert manifest["runs"][1]["invocation"] == {}
    assert json.loads((tmp_path / MANIFEST).read_text()) == manifest


def test_stamp_refuses_a_different_protocol(tmp_path):
    stamp(tmp_path, "exp", {"setting": "old"}, data="seed*.csv")
    with pytest.raises(ProtocolMismatch, match="setting"):
        stamp(tmp_path, "exp", {"setting": "new"}, data="seed*.csv")
    with pytest.raises(ProtocolMismatch):
        stamp(tmp_path, "other", {"setting": "old"}, data="seed*.csv")


def test_stamp_refuses_a_directory_with_unstamped_files(tmp_path):
    write_rows(tmp_path / "seed000.csv", [{"seed": 0}])
    with pytest.raises(ProtocolMismatch, match="unstamped"):
        stamp(tmp_path, "exp", {"k": 1}, data="seed*.csv")


def test_load_results_reads_only_stamped_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path, "exp")
    stamp(tmp_path, "exp", {"k": 1}, data="seed*.csv")
    write_rows(tmp_path / "seed000.csv", [{"seed": 0, "v": 1}])
    write_rows(tmp_path / "seed001.csv", [{"seed": 1, "v": 2}])
    write_rows(tmp_path / "notes.csv", [{"seed": 9, "v": 9}])  # outside the data glob
    df, manifest = load_results(tmp_path, "exp")
    assert sorted(df["seed"]) == [0, 1]
    assert "k = 1" in describe(manifest)
    with pytest.raises(ProtocolMismatch):
        load_results(tmp_path, "other")


def test_require_comparable_allows_only_named_differences():
    a = {"protocol": {"setting": "uniform", "samples": 20}}
    b = {"protocol": {"setting": "weighted", "samples": 20}}
    c = {"protocol": {"setting": "weighted", "samples": 10}}
    require_comparable([a, b], may_differ={"setting"})
    with pytest.raises(ProtocolMismatch, match="samples"):
        require_comparable([a, c], may_differ={"setting"})


def _rows_for(seed):
    return [] if seed == 2 else [{"seed": seed}]


def _raising_rows(seed):
    raise ValueError(f"seed {seed} failed")


def test_run_seeds_writes_one_file_per_seed_and_resumes(tmp_path, capsys):
    run_seeds(tmp_path, range(3), _rows_for, workers=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "seed000.csv",
        "seed001.csv",
        "seed002.skip",
    ]
    capsys.readouterr()
    run_seeds(tmp_path, range(3), _rows_for, workers=1)
    assert "3 of 3 seeds already done" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.tmp"))


def test_run_seeds_propagates_worker_errors(tmp_path):
    with pytest.raises(ValueError, match="seed 0 failed"):
        run_seeds(tmp_path, [0], _raising_rows, workers=1)


def test_seed_indices_are_contiguous_and_validated():
    assert list(seed_indices(0, 3)) == [0, 1, 2]
    assert list(seed_indices(350, 2)) == [350, 351]
    with pytest.raises(ValueError, match="seed-start"):
        seed_indices(-1, 3)
    with pytest.raises(ValueError, match="n-seeds"):
        seed_indices(0, 0)
