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
The random prefix uses `seed + 20000` and, by default, samples a nested ordering
of up to 1000 indices without replacement from the full set. The manifest records
the selection policy, offset and pool size; package versions and run seed are
also recorded for reproducibility.

## Results and protocol stamping

Every `make run` creates one Stockholm-local timestamp directory, shared by the
experiment families in that run. Results are organized by timestamp, then type:

```
results/<timestamp>/
  final_experiment/
    manifest.json      protocol: every setting that changes what the numbers mean
    runs/*.json        one file per run: git commit, uncommitted changes, command,
                       host, package versions (safe for concurrent cluster array jobs)
    seed000.csv ...    one file per seed
    summary_*.txt      analysis output, starting with the protocol it came from
    *.pdf              figures
  record_confidence/
    manifest.json, runs/, m010/seed000.csv ...
    summary_*.txt, *.pdf
  calibration_experiment/
    manifest.json, runs/, seed000.csv ...
    summary_*.txt, *.pdf
```

- Timestamps use Stockholm local time in `YYYYMMDD_HHMMSS` format. Each experiment
  runner gets a fresh timestamp by default; Make shares one timestamp across all
  experiment families and analyses in a pipeline.
- A manifest records the protocol and each run's Git state, command, host and package versions.
- A runner refuses to write into a directory whose protocol differs from its own,
  or that holds unstamped files.
- Result directories are automatic and new by default, preserving earlier
  results. `--out-dir` and the Make `RUN_DIR` override are not supported.
- Analyses only read data through the manifest and refuse to compare result
  sets whose protocols differ beyond what is being compared.
- Analyses default to the latest timestamp for their experiment and save summaries
  and figures beside its CSVs. Their tables also print to the terminal.
- Before a new run exists, that latest folder can be an archived legacy run; check
  its manifest and protocol summary before interpreting the output.
- Result generation does not commit or publish artifacts.

Archived legacy results are kept in the same layout under their archive timestamp.
Their manifests preserve the original protocol, including sequential tree
selection and per-row labels where applicable.

The record-confidence result sets produced before timestamped directories
(2026-10-20; random and first-m tree selection, and the prior-guided variant
ablation) stay under `results/record_confidence/<selection>/` and
`results/record_confidence_variants/random/`. Analyze them with
`analyze_record_confidence.py --results-dirs <dir> [<dir> ...]`; several
directories that differ only in tree selection are compared per selection.

## Running

The Makefile keeps a short command set for bounded and complete runs:

```sh
make help
make setup
make test
make pilot                   # one seed of the main experiment
make run-final               # complete main experiment plus its analyses
make run                     # all experiment families, sequentially
make analyze                 # all analyses on the latest results
make report                  # local report PDF
```

The main defaults are n=1000, K=40 geometric picks from 1 to 500 trees
(34 distinct integer releases), A=marital-status, B=education, 350 seeds
starting at 0, and 180 seconds per solve. Workers default to the CPUs available
to the process. Full-set release is off by default.
`pilot` runs one configurable main-experiment seed with one worker. `run-final`
runs the main family; `run-record-confidence` and `run-calibration` run each
supplementary family. `run` runs all three families sequentially and stops on
failure. `analyze` runs all analyses on the latest results.

Main Make overrides: `N`, `K` or `K_STEPS`, `M_MAX`, `ATTR_A`, `ATTR_B`,
`ALL_TREES`, `TIMEOUT`, `N_SEEDS`, `SEED_START`, and `WORKERS`. The seed start
defaults to 0; `N_SEEDS` defaults to 350 for the main experiment. The sequence
is contiguous.
`pilot` also accepts `PILOT_SEED` and `PILOT_WORKERS`.

```sh
make pilot N=100 K=4 M_MAX=20
make pilot N=100 K_STEPS=5 M_MAX=20
make run-final K=40 M_MAX=500 WORKERS=1
```

Record-confidence settings: `RECORD_M` (default 10), `RECORD_SAMPLES` (20),
`RECORD_TIMEOUT` (30s),
`RECORD_WORKERS` (2), `RECORD_N_SEEDS` (20), `RECORD_SEED_START` (0), and
`RECORD_ONLY` (optional method subset). Its private sample remains n=100.
`THREADS` sets CP-SAT search workers per solve for every family (default 4).
Calibration settings: `CALIBRATION_N` (100), `CALIBRATION_K` (2) or
`CALIBRATION_K_STEPS` (linear increment; mutually exclusive),
`CALIBRATION_M_MAX` (500), `CALIBRATION_ATTR_A/B` (marital-status/education),
`CALIBRATION_ALL_TREES` (0), `CALIBRATION_TIMEOUT` (60s),
`CALIBRATION_WORKERS` (half the available CPUs, minimum one), and
`CALIBRATION_N_SEEDS` (35) and `CALIBRATION_SEED_START` (0). It measures
m=1/500 and retains 15 repeats per condition, rather than using the main grid.
Shared `N_SEEDS`, `SEED_START`, and `WORKERS` override the corresponding family
settings unless a family-specific override is supplied. Main family-specific overrides use `FINAL_N_SEEDS` and `FINAL_SEED_START`;
record-confidence and calibration use the `RECORD_*` and `CALIBRATION_*` names
above. `SEED_END` has been removed. None of the runners accepts a custom output
directory. Make rejects `RUN_DIR`, old `M_VALUES`/`M_STEP` settings (including
their `CALIBRATION_` forms), `M_PICKS`, and `CALIBRATION_M_PICKS`.

For code checks, use `.venv/bin/python -m pytest -q`, `ruff check .`,
`ruff format --check .`, and `.venv/bin/python -m compileall -q src experiments tests`.
`make report` builds the local LaTeX report when its sources and tools are available.

The same commands remain available directly for options not surfaced as Make
variables, such as record-confidence methods:

```sh
uv sync --locked --python 3.12
.venv/bin/python -m pytest -q

.venv/bin/python experiments/run_record_confidence.py --m 10 50 100 --n-seeds 20
.venv/bin/python experiments/analyze_record_confidence.py         # analyzes latest set

.venv/bin/python experiments/run_final_experiment.py --n-seeds 350
.venv/bin/python experiments/analyze_results.py                   # --results-dir picks another set
.venv/bin/python experiments/analyze_mechanism.py

# Re-analyse archived results
.venv/bin/python experiments/analyze_results.py --results-dir results/<archive-timestamp>/final_experiment
```

Parallelism: `--workers` sets how many seeds run at once and `--threads` how many
CP-SAT search workers each solve uses (default 4; `--workers` defaults to the
available CPUs divided by `--threads`). Both are saved in the run manifest.
Several search workers are needed, not just faster: they run different
strategies in parallel, and one worker alone stalls on larger releases. At
n=100 with 50 random trees, one worker found no table in 180 s (condition
None) while three found the optimum in 16 s; for prior-guided sampling one
worker found no table within 30 s at m=50 and never proved the optimum at
m=10. Runs that reach a wall-clock timeout, use several search workers, or use
different hardware/software versions are not guaranteed to be bitwise
identical; with several workers, which of several equally good tables is
returned can vary between runs.

Resuming: set `RTDS_RUN_TIMESTAMP` to an existing timestamp and rerun the same
command; finished seeds are skipped, and the runner refuses if the protocol
differs (including `--workers`, which is part of the protocol).

## Cluster runs

`hpc/seed_array.sh` runs one seed per array task, all tasks writing into one
shared timestamp directory (see the usage lines at its top; adapt the resource
and module lines to the cluster). Before submitting: create the virtualenv with
the cluster's Python module, run `pytest -q` in an interactive job, and download
the Adult data once on a login node, since compute nodes often have no internet
access:

```sh
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from rashomon.data import load_adult_raw; load_adult_raw('data')"
mkdir -p hpc/logs
```

## Background-knowledge configuration

Main and evaluator calibration accept `--n`, `--k` or `--k-steps`, `--m-max`,
`--attr-a`, `--attr-b`, `--all-trees` / `--no-all-trees`, seed count/start,
workers, and timeout. `--m-values`, `--m-step`, and `--m-picks` have been removed. Use
`--k-steps` only for a linear grid; otherwise `--k` selects a geometric grid.
For example, `--k-steps 10 --m-max 25` selects 1, 11, 21, and 25. The step
must be between 1 and `m_max`; the maximum is always included. The options are
mutually exclusive. Make equivalents are `K_STEPS` and `CALIBRATION_K_STEPS`.

Geometric release sizes are `unique(rint(geomspace(1, m_max, k)))`: equally
spaced on a logarithmic axis before rounding to the nearest integer (ties to
even) and removing duplicates. `k >= 2`, `m_max >= 1`; maximum 1 produces one
release. K is an upper bound on distinct measurements. The minimum is fixed at
1; both endpoints are retained. The explicit linear alternative generates
`1, 1 + k_steps, ...` and appends `m_max` when needed. Require
`1 <= k_steps <= m_max`. The manifest records the selected schedule, its
parameter, rounding/endpoint policy, effective count and release sizes.

```sh
.venv/bin/python experiments/run_final_experiment.py --n 200 --k 8 --m-max 50 --attr-b age --n-seeds 5 --workers 1
.venv/bin/python experiments/run_calibration_experiment.py --k 2 --m-max 100 --n-seeds 5 --workers 1
```

`--all-trees` explicitly appends each seed's entire Rashomon set, even beyond
`m_max`; it is off by default. The actual size is recorded as `n_trees_available`
and a full release has `m_is_all_trees=true`. Counts beyond the available set
are skipped; the effective number of seeds can therefore differ across m.
All conditions share the same release within each seed/m pair.
The tree pool is max(1000, m_max), or the full available set with all-trees enabled.
Changing the pool can change smaller prefixes; keep it fixed for paired contrasts.

The n=1000 default is conservative, not a claim of infeasibility at n=5000.
A bounded two-leaf synthetic construction check found five times as many solver
variables at 5000 rows as at 1000; matching's pairwise arrays grow quadratically.
No full Adult n=5000, m=500 feasibility benchmark has been run. `--n 5000`
remains available for explicitly budgeted experiments. The 180s timeout limits
solver search per condition; it excludes construction, fitting and matching.
Timeout without a candidate still stops the runner; it is not scored as an error.

A/B may be marital-status, education, age, hours-per-week or capital-gain, must be
distinct, and cannot be sex (T) or income (Y). Unknown attributes produce an error.
Numeric knowledge uses the existing threshold bits, not exact raw values. Adding
occupation or another absent attribute requires an explicit feature-schema change.
The current reconstruction protocol supplies true per-row Y; this setting remains
recorded as `labels=given_per_row`. The Swedish-inspired public-information scenario
and exact auxiliary knowledge are modeling assumptions, not population validation.
