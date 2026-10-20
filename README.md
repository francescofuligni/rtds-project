# RTDS Project

RTDS project for the Research Topics in Data Science course: reconstructing
training data from a released Rashomon set of decision trees (TreeFARMS), and
giving the attacker a per-record confidence it can compute without ground truth.

## Experiments

| Runner (`experiments/`) | Question | Analysis |
|---|---|---|
| `run_record_confidence.py` | Can an attacker attach a calibrated confidence to each reconstructed record? Compares per-tree counting, solver-seed sampling, prior-guided sampling and a no-trees public-statistics baseline. | `analyze_record_confidence.py` |
| `run_final_experiment.py` | Does background knowledge of A (marital-status) / B (education) lower reconstruction error on T (sex), as a function of the number of released trees m? | `analyze_results.py`, `analyze_mechanism.py` |
| `run_calibration_experiment.py` | Evaluator-side diagnostic: does solver agreement (aligned to the true table) track correctness? Not usable by an attacker. | `analyze_calibration.py`, `calibrate_confidence.py` |

Which m trees are released matters. TreeFARMS decodes tree indices like an
odometer over shared subtrees, so sequential indices are near-copies and leak
far less than m distinct trees would. All runners therefore use one seeded
random ordering without replacement, releasing nested prefixes for each m.
The random prefix uses `seed + 20000` and a nested ordering of up to 1000
indices sampled without replacement from the full set. The manifest records
the selection policy, offset and pool size; package versions and run seed are
also recorded for reproducibility.

## Results and protocol stamping

Every `make run` creates one Stockholm-local timestamp directory, shared by the
experiment families in that run. Results are organized by timestamp, then type:

```
results/<timestamp>/
  final_experiment/
    manifest.json, seed000.csv ...
    summary_*.txt, *.pdf
  record_confidence/
    manifest.json, m010/seed000.csv ...
    summary_*.txt, *.pdf
  calibration_experiment/
    manifest.json, seed000.csv ...
    summary_*.txt, *.pdf
```

- Timestamps use Stockholm local time in `YYYYMMDD_HHMMSS` format. Direct runner
  calls create their own timestamp; `make run` shares one across its full pipeline.
- A manifest records the protocol and each run's Git state, command, host and package versions.
- A runner refuses to write into a directory whose protocol differs from its own,
  or that holds unstamped files. Changing a setting means a new directory
  (`--out-dir`).
- Each default invocation creates a new timestamp, preserving earlier results.
  To resume or split seed ranges across machines, pass the same `--out-dir`.
- Analyses only read data through the manifest and refuse to compare result
  sets whose protocols differ beyond what is being compared.
- Analyses default to the latest timestamp for their experiment and save summaries
  and figures beside its CSVs. Their tables also print to the terminal.
- Before a new run exists, that latest folder can be an archived legacy run; check
  its manifest and protocol summary before interpreting the output.
- Stamped result sets are committed; scratch runs go in `results/dev/` (ignored).

Archived legacy results are kept in the same layout under their archive timestamp.
Their manifests preserve the original protocol, including sequential tree
selection and per-row labels where applicable.

## Running

The Makefile provides the agreed project commands:

```sh
make help
make setup
make test
make run   # full experiments, analyses, and generated artifacts
```

`make run` executes each pipeline stage in sequence and stops if a command fails.
It uses the experiment defaults below, so the full run can take a long time.

For individual or bounded runs, call the scripts directly:

```sh
uv sync                                                           # or: pip install -e . into .venv
.venv/bin/python -m pytest -q

.venv/bin/python experiments/run_record_confidence.py --m 10 50 100 --seed-end 20
.venv/bin/python experiments/analyze_record_confidence.py         # analyzes latest set

.venv/bin/python experiments/run_final_experiment.py --seed-end 350
.venv/bin/python experiments/analyze_results.py                   # --results-dir picks another set
.venv/bin/python experiments/analyze_mechanism.py

# Re-analyse archived results
.venv/bin/python experiments/analyze_results.py --results-dir results/<archive-timestamp>/final_experiment
```

Parallelism: `--workers` sets how many seeds run at once; for the record-confidence
runner `--threads` sets CP-SAT threads per solve (defaults use the cores the
process is allowed to use, which respects cluster allocations).
