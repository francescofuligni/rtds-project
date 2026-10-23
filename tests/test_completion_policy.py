"""Bounded regression checks for incomplete experiments and the public pipeline."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from analysis.reconstruction import paired_statistics
from analysis_output import condition_labels
from provenance import read_manifest, run_seeds, stamp, write_outcome_summary

ROOT = Path(__file__).resolve().parents[1]


def _mixed_seed(seed):
    if seed == 0:
        raise ValueError("synthetic failure")
    return [
        {
            "seed": seed,
            "m": 1,
            "condition": "none",
            "status": "UNKNOWN" if seed == 1 else "FEASIBLE",
            "reconstruction_error": None if seed == 1 else 0.0,
        }
    ]


def test_workers_finish_independently_and_outcomes_do_not_pollute_provenance(tmp_path):
    stamp(tmp_path, "exp", {}, data="seed*.csv")
    summary = run_seeds(tmp_path, range(3), _mixed_seed, workers=2)
    assert summary["requested"] == 3
    assert summary["completed"] == 2 and summary["failed"] == 1
    assert summary["no_candidate"] == 1 and summary["incomplete"]
    assert not (tmp_path / "seed000.csv").exists()
    assert (tmp_path / "seed002.csv").exists()
    assert len(read_manifest(tmp_path, "exp")["runs"]) == 1
    again = run_seeds(tmp_path, [1, 2], _mixed_seed, workers=1)
    assert again["no_candidate"] == 1
    assert len(list((tmp_path / "runs" / "outcomes").glob("*.json"))) == 2


def test_pairwise_statistics_exclude_only_missing_pairs_and_never_pool_m(capsys):
    rows = []
    for seed in range(6):
        for m in (1, 2):
            for condition, error in [("none", 0.5), ("A", 0.4), ("B", 0.5)]:
                if condition == "B" and seed < 5:
                    error = None
                rows.append(
                    {
                        "seed": seed,
                        "m": m,
                        "condition": condition,
                        "reconstruction_error": error,
                    }
                )
    df = pd.DataFrame(rows)
    paired_statistics(
        df, condition_labels({"attr_a": "marital-status", "attr_b": "age"})
    )
    output = capsys.readouterr().out
    assert output.count("wins=6, paired=6, missing=0") == 2
    assert "wins=0, paired=1, missing=5" in output
    assert "no valid pairs" in output and "fewer than 5" in output
    assert "n=12" not in output and "pooled" not in output
    df.loc[df.condition == "A", "reconstruction_error"] = 0.5
    paired_statistics(
        df, condition_labels({"attr_a": "marital-status", "attr_b": "age"})
    )
    assert "all differences zero; p=1" in capsys.readouterr().out


@pytest.fixture
def fake_confidence(monkeypatch):
    import run_record_confidence as runner

    from rashomon.data import BinarizedDataset

    data = BinarizedDataset(
        pd.DataFrame({"bit": [0, 1, 0, 1, 0, 1]}), pd.Series([0, 1, 0, 1, 0, 1]), {}, {}
    )
    monkeypatch.setattr(runner, "PILOT_N", 2)
    monkeypatch.setattr(runner, "load_adult_raw", lambda _: None)
    monkeypatch.setattr(runner, "binarize_adult_curated", lambda _: data)
    monkeypatch.setattr(
        runner,
        "build_rashomon_set",
        lambda *a, **k: SimpleNamespace(get_tree_count=lambda: 1),
    )
    monkeypatch.setattr(runner, "select_random_trees", lambda *a, **k: [None])
    monkeypatch.setattr(runner, "extract_branches", lambda *a: [])
    return runner


@pytest.mark.parametrize("statuses", [["UNKNOWN", "UNKNOWN"], ["OPTIMAL", "UNKNOWN"]])
def test_confidence_reports_all_attempts_without_inventing_claims(
    fake_confidence, monkeypatch, statuses
):
    runner = fake_confidence
    monkeypatch.setattr(
        runner,
        "sampled_candidates",
        lambda *a, **k: (
            [
                Counter({(0, 0): 1, (1, 1): 1}) if s == "OPTIMAL" else None
                for s in statuses
            ],
            statuses,
        ),
    )
    coverage = {"methods": [], "solver_statuses": []}
    rows = runner.run_seed(
        0, 1, 2, 1, methods={"seed_only"}, public_n=2, coverage=coverage
    )
    method = coverage["methods"][0]
    assert method["selection"]["requested"] == 2
    assert method["scoring"]["successful"] == statuses.count("OPTIMAL")
    assert coverage["solver_statuses"] == statuses
    assert bool(rows) == ("OPTIMAL" in statuses)


def test_confidence_split_without_scoring_candidate_has_no_claims(
    fake_confidence, monkeypatch
):
    runner = fake_confidence
    monkeypatch.setattr(
        runner,
        "sampled_candidates",
        lambda *a, **k: (
            [Counter({(0, 0): 1, (1, 1): 1}), None],
            ["FEASIBLE", "UNKNOWN"],
        ),
    )
    coverage = {"methods": [], "solver_statuses": []}
    assert (
        runner.run_seed(
            0, 1, 1, 1, methods={"prior_guided_split"}, public_n=2, coverage=coverage
        )
        == []
    )
    method = coverage["methods"][0]
    assert method["selection"]["successful"] == 1
    assert method["scoring"]["successful"] == 0


def test_confidence_empty_method_is_saved_in_outcomes(
    fake_confidence, monkeypatch, tmp_path
):
    runner = fake_confidence
    monkeypatch.setattr(
        runner, "sampled_candidates", lambda *a, **k: ([None], ["INFEASIBLE"])
    )
    args = SimpleNamespace(
        out_dir=tmp_path,
        only=None,
        methods=["seed_only"],
        samples=1,
        timeout=1,
        threads=1,
        public_n=2,
        public_prior="records",
        tree_selection="random",
    )
    unit = runner._run_and_write(0, 1, args)
    summary = write_outcome_summary(tmp_path, [unit])
    assert summary["no_candidate"] == 1
    assert unit["methods"][0]["method"] == "seed_only"
    assert not list(tmp_path.glob("m*/seed*.csv"))


def test_make_continues_all_families_and_analyses_after_failure(tmp_path):
    log = tmp_path / "calls.txt"
    fake = tmp_path / "python"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\nexit 1\n')
    fake.chmod(0o755)
    result = subprocess.run(
        ["make", "run", f"PYTHON={fake}"],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    calls = log.read_text().splitlines()
    assert len(calls) == 5
    assert "run_record_confidence.py" in calls[3]
    assert all("--no-plots" in call for call in calls if "analyze_" in call)
    dry = subprocess.run(
        ["make", "-n", "figures", "EDUCATION_DIR=x", "AGE_DIR=y"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "plot_attack_figures.py" in dry.stdout
    old = subprocess.run(
        ["make", "-n", "plot-attack-figures"],
        check=False,
        cwd=ROOT,
        capture_output=True,
    )
    assert old.returncode != 0


def test_main_cli_returns_nonzero_for_incomplete_summary(monkeypatch, tmp_path):
    import run_final_experiment as runner

    monkeypatch.setattr(runner, "stamp", lambda *a, **k: None)
    monkeypatch.setattr(runner, "experiment_result_dir", lambda _: tmp_path)
    monkeypatch.setattr(runner, "run_seeds", lambda *a, **k: {"incomplete": True})
    with pytest.raises(SystemExit) as exc:
        runner.main(["--n-seeds", "1"])
    assert exc.value.code == 1


def test_latest_incomplete_archive_does_not_silently_select_older_data(tmp_path):
    from output_paths import latest_result_dir
    from provenance import print_outcome_coverage

    directory = tmp_path / "20261023_120000" / "record_confidence"
    stamp(directory, "record_confidence", {}, data="m*/seed*.csv")
    write_outcome_summary(
        directory, [{"seed": 0, "m": 1, "outcome": "failed", "error": "x"}]
    )
    assert latest_result_dir("record_confidence", tmp_path) == directory
    print_outcome_coverage(directory)


def _mixed_confidence_job(seed, m, args):
    from provenance import write_rows

    if seed == 0:
        raise RuntimeError("confidence job failed")
    write_rows(args.out_dir / f"m{m:03d}" / f"seed{seed:03d}.csv", [{"seed": seed}])
    return {
        "seed": seed,
        "m": m,
        "outcome": "completed",
        "no_candidate": int(seed == 1),
    }


def test_confidence_coordinator_finishes_other_jobs_and_exits_nonzero(
    tmp_path, monkeypatch
):
    import run_record_confidence as runner

    monkeypatch.setattr(runner, "experiment_result_dir", lambda _: tmp_path)
    monkeypatch.setattr(runner, "_run_and_write", _mixed_confidence_job)
    with pytest.raises(SystemExit) as exc:
        runner.main(["--n-seeds", "3", "--workers", "2", "--m", "1"])
    assert exc.value.code == 1
    summary = json.loads(
        next((tmp_path / "runs" / "outcomes").glob("*.json")).read_text()
    )
    assert summary["failed"] == 1 and summary["completed"] == 2
    assert summary["no_candidate"] == 1
    assert not (tmp_path / "m001" / "seed000.csv").exists()
    assert (tmp_path / "m001" / "seed002.csv").exists()


@pytest.mark.parametrize("failure", ["protocol", "render", None])
def test_figure_cli_stages_all_outputs_before_publication(
    tmp_path, monkeypatch, failure
):
    import output_paths
    from plotting import attack_figures as plotter
    from provenance import write_rows

    results = tmp_path / "results"
    monkeypatch.setattr(output_paths, "RESULTS_ROOT", results)
    output = tmp_path / "figures"
    data = output / "data"
    data.mkdir(parents=True)
    for name in [
        "attack_marital_education",
        "attack_marital_age",
        "attack_selection_comparison",
    ]:
        (output / f"{name}.png").write_bytes(b"previous figure")
        (data / f"{name}.csv").write_text("previous table")
    (data / "sources.json").write_text("previous provenance")
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr(plotter, "FIGURES_ROOT", output)
    monkeypatch.setattr(plotter, "FIGURE_DATA_ROOT", data)
    inputs = []
    for index, (feature, selection) in enumerate(
        [("education", "random"), ("age", "random"), ("education", "sequential")]
    ):
        directory = results / f"20261023_12000{index}" / "final_experiment"
        protocol = {
            "attr_a": "marital-status",
            "attr_b": feature,
            "tree_selection": selection,
            "solve_timeout": 2 if index == 2 and failure == "protocol" else 1,
        }
        stamp(directory, "final_experiment", protocol, data="seed*.csv")
        write_rows(
            directory / "seed000.csv",
            [
                {
                    "seed": 0,
                    "m": 1,
                    "condition": c,
                    "status": "FEASIBLE",
                    "reconstruction_error": 0.25,
                }
                for c in ["none", "A", "B", "A+B"]
            ],
        )
        inputs.append(directory)
    renders = []

    def draw(stats, labels, stem, **kwargs):
        renders.append(stem)
        stem.with_suffix(".png").write_bytes(b"new figure")
        if failure == "render" and len(renders) == 2:
            raise RuntimeError("render failed")

    monkeypatch.setattr(plotter, "draw", draw)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_attack_figures.py",
            "--education-dir",
            str(inputs[0]),
            "--age-dir",
            str(inputs[1]),
            "--sequential-dir",
            str(inputs[2]),
        ],
    )
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            plotter.cli()
        assert all(p.read_bytes() == contents for p, contents in before.items())
        if failure == "protocol":
            assert not renders
    else:
        plotter.cli()
        assert len(renders) == 3
        provenance = json.loads((data / "sources.json").read_text())
        assert provenance["selection_comparison"]["paired_keys"] == [
            {"seed": 0, "m": 1}
        ]
        assert (output / "attack_marital_age.png").read_bytes() == b"new figure"
    assert not list(output.glob(".render-*"))


def test_summary_only_analyses_create_no_figures(tmp_path, monkeypatch, capsys):
    import matplotlib.pyplot as plt
    from analysis.mechanism import main as mechanism
    from analysis.reconstruction import main as reconstruction
    from analysis.record_confidence import main as confidence
    from provenance import write_rows

    def forbidden(*args, **kwargs):
        pytest.fail("Summary-only analysis attempted rendering")

    monkeypatch.setattr(plt, "subplots", forbidden)
    directory = tmp_path / "main"
    stamp(
        directory,
        "final_experiment",
        {"attr_a": "marital-status", "attr_b": "age"},
        data="seed*.csv",
    )
    write_rows(
        directory / "seed000.csv",
        [
            {
                "seed": 0,
                "m": 1,
                "condition": "none",
                "status": "FEASIBLE",
                "reconstruction_error": 0.2,
                "uniform_baseline_error": 0.5,
                "marginal_baseline_error": 0.4,
                "joint_split_frac_A": 0,
                "joint_split_frac_B": 0,
            }
        ],
    )
    figure_dir = tmp_path / "figures"
    reconstruction(directory, figure_dir, plots=False)
    mechanism(directory, figure_dir, plots=False)
    records = tmp_path / "records"
    stamp(
        records, "record_confidence", {"tree_selection": "random"}, data="m*/seed*.csv"
    )
    write_rows(
        records / "m001" / "seed000.csv",
        [
            {
                "seed": 0,
                "m": 1,
                "method": "prior_only",
                "confidence": 0.5,
                "correct": 1,
                "k": 1,
            }
        ],
    )
    confidence([records], figure_dir, plots=False)
    assert "not available (historical archive)" in capsys.readouterr().out
    assert not figure_dir.exists()


def test_all_failed_confidence_prints_coverage_without_data(tmp_path, capsys):
    from analysis.record_confidence import main

    stamp(tmp_path, "record_confidence", {}, data="m*/seed*.csv")
    write_outcome_summary(
        tmp_path,
        [
            {
                "seed": 0,
                "m": 1,
                "outcome": "completed",
                "no_candidate": 1,
                "methods": [
                    {
                        "method": "seed_only",
                        "selection": {"requested": 1, "successful": 0},
                        "scoring": {"requested": 1, "successful": 0},
                    }
                ],
            }
        ],
    )
    main([tmp_path], tmp_path / "figures", plots=False)
    output = capsys.readouterr().out
    assert "seed_only" in output and "No valid claim outcomes" in output
    assert not (tmp_path / "figures").exists()
