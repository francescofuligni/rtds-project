# RTDS Project

RTDS project for the Research Topics in Data Science course: sensitive attribute reconstruction from decision trees in a Rashomon set, comparing attacks under different background knowledge conditions.

Each experiment invocation creates a new directory under `results/`, named with
its Stockholm local start time (`YYYYMMDD_HHMMSS`), for example
`results/20261020_153045/`. Collisions receive a numeric suffix.
CSV filenames stay unchanged inside that directory. Supplementary batches also
create their own run directory. Existing results are preserved.

```sh
.venv/bin/python experiments/run_final_experiment.py
.venv/bin/python experiments/run_calibration_experiment.py
```

Analysis scripts default to the latest timestamped run containing the relevant
CSV (reconstruction or calibration). They print the selected paths, save PDF
figures beside that CSV, and save text summaries in its `summaries/` subdirectory.
Rerunning an analysis refreshes those derived files within the selected run.

```sh
.venv/bin/python experiments/analyze_results.py
.venv/bin/python experiments/analyze_mechanism.py
.venv/bin/python experiments/analyze_calibration.py
.venv/bin/python experiments/calibrate_confidence.py
```

To analyze a particular run, supply its CSV explicitly:

```sh
.venv/bin/python experiments/analyze_results.py --input results/20261020_153045/final_experiment.csv
```

`--output-dir` overrides the analysis destination. Combining CSVs from different
run directories requires an explicit `--output-dir` for the combined analysis.
Earlier UTC timestamp directories remain discoverable.
Legacy CSVs directly under `results/` are not selected automatically; the analysis
still rejects data from incompatible scientific protocols.

Runner `--output path/name.csv` selects the parent and filename, producing
`path/{timestamp}/name.csv`. CSV writing refuses to replace an existing file.
A failed experiment can leave an empty run directory; errors still stop execution.
Figures and tables can be selected and copied manually when writing up the study.

The existing historical outputs are grouped in [20261020_124802](results/20261020_124802/README.md).
Its timestamp marks archival time; these files retain the older scientific protocol.
