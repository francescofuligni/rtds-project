import json

import pandas as pd
from analysis.mechanism import main as analyze_mechanism
from analysis.reconstruction import main as analyze_results
from analysis_output import condition_labels, run_analysis


def test_condition_labels_include_configured_features():
    labels = condition_labels({"attr_a": "marital-status", "attr_b": "capital-gain"})
    assert labels == {
        "none": "No background",
        "A": "A: marital",
        "B": "B: capital gain",
        "A+B": "A+B: marital + capital gain",
    }
    assert max(map(len, labels.values())) <= 30


def test_main_analysis_outputs_use_manifest_feature_names(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    result_dir = tmp_path / "final_experiment"
    result_dir.mkdir()
    protocol = {"attr_a": "marital-status", "attr_b": "age"}
    (result_dir / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "final_experiment",
                "protocol": protocol,
                "data": "seed*.csv",
                "runs": [],
            }
        )
    )
    rows = []
    for seed in range(5):
        for condition, error in (
            ("none", 0.5),
            ("A", 0.45 - seed * 0.01),
            ("B", 0.48 - seed * 0.01),
            ("A+B", 0.4 - seed * 0.01),
        ):
            rows.append(
                {
                    "seed": seed,
                    "m": 5,
                    "condition": condition,
                    "reconstruction_error": error,
                    "uniform_baseline_error": 0.5,
                    "marginal_baseline_error": 0.45,
                    "joint_split_frac_A": seed / 5,
                    "joint_split_frac_B": (seed + 1) / 6,
                }
            )
    pd.DataFrame(rows).to_csv(result_dir / "seed000.csv", index=False)

    labels_seen = []
    original_errorbar = Axes.errorbar
    legend_layouts = []
    original_legend = Figure.legend

    def capture_errorbar(self, *args, **kwargs):
        labels_seen.append(kwargs.get("label"))
        return original_errorbar(self, *args, **kwargs)

    def capture_legend(self, *args, **kwargs):
        legend_layouts.append((kwargs.get("loc"), kwargs.get("ncol")))
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture_errorbar)
    monkeypatch.setattr(Figure, "legend", capture_legend)
    summary_path = tmp_path / "summary.txt"
    run_analysis(lambda: analyze_results(result_dir), summary_path)
    assert "A: marital" in labels_seen
    assert "B: age" in labels_seen
    assert "A+B: marital + age" in labels_seen
    assert legend_layouts == [("lower center", 3)]
    assert (result_dir / "reconstruction_error_vs_m.pdf").exists()
    assert "attr_a = marital-status" in summary_path.read_text()
    assert "attr_b = age" in summary_path.read_text()

    titles_seen = []
    original_set_title = Axes.set_title

    def capture_title(self, label, *args, **kwargs):
        titles_seen.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", capture_title)
    analyze_mechanism(result_dir)
    assert "A: marital" in titles_seen
    assert "B: age" in titles_seen
    assert (result_dir / "mechanism_joint_split_correlation.pdf").exists()
