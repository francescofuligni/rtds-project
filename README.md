# RTDS Project

RTDS project for the Research Topics in Data Science course: sensitive attribute reconstruction from decision trees in a Rashomon set, comparing attacks under different background knowledge conditions.

Experiment runners write CSV files directly to `results/`. Analysis scripts save
PDF figures in `results/` and text summaries in `results/summaries/`,
overwriting the established filenames on subsequent runs. Figures and tables can be selected and copied
manually when writing up the study.

To analyze the existing CSVs without rerunning experiments:

```sh
.venv/bin/python experiments/analyze_results.py
.venv/bin/python experiments/analyze_mechanism.py
.venv/bin/python experiments/analyze_calibration.py
.venv/bin/python experiments/calibrate_confidence.py
```
