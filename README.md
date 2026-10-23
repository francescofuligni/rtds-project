# RTDS Project

RTDS project for the Research Topics in Data Science course: reconstructing
training data from a released Rashomon set of decision trees (TreeFARMS), and
giving the attacker a per-record confidence it can compute without ground truth.

For a from-scratch explanation of what we did and why, with diagrams and all
results, open `docs/released_trees_explained.html` in a browser (one
self-contained file; works offline).

## Experiments

| Runner (`experiments/`) | Question | Analysis |
|---|---|---|
| `run_record_confidence.py` | Can an attacker attach a calibrated confidence to each reconstructed record? Compares per-tree counting, solver-seed sampling, prior-guided sampling and a no-trees public-statistics baseline. | `analyze_record_confidence.py` |
| `run_final_experiment.py` | Does background knowledge of A (marital-status) / B (education) lower reconstruction error on T (sex), as a function of the number of released trees m? | `analyze_results.py`, `analyze_mechanism.py`; `compare_background_pairs.py` compares result sets that differ only in A/B (e.g. B=education vs. B=age) |

Findings so far: `results/FINDINGS.md` (record confidence) and
`results/FINDINGS_background_knowledge.md` (main experiment).

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
- Result generation does not commit or publish artifacts.

The record-confidence result sets produced before timestamped directories
(2026-10-20; random and first-m tree selection, and the prior-guided variant
ablation) stay under `results/record_confidence/<selection>/` and
`results/record_confidence_variants/random/`. Analyze them with
`analyze_record_confidence.py --results-dirs <dir> [<dir> ...]`; several
directories that differ only in tree selection are compared per selection.
Sets that differ in tree selection, `public_n` or `public_prior` are compared
setting by setting, paired by seed; `--out-dir` puts such a comparison in its
own folder (committed comparisons live under `results/comparisons/`).

## Running

The Makefile keeps a short command set for bounded and complete runs:

```sh
make help
make setup
make test
make pilot                   # one seed of the main experiment
make run-final               # complete main experiment plus its analyses
make run                     # both experiment families, sequentially
make analyze                 # all analyses on the latest results
make report                  # local report PDF
```

The main defaults are n=100, K=15 geometric picks from 1 to 200 trees
(14 distinct integer releases), A=marital-status, B=education, 350 seeds
starting at 0, and 180 seconds per solve; this is the setting of the committed
main-experiment results (which used 100 seeds). Larger settings mostly time out
on a laptop: at n=1000 no table was found within 180 s even at m=10, and
m>=200 at n=1000 needs more than 5 GB per solve. Workers default to the CPUs
available to the process divided by the solver threads. Full-set release is off
by default.
`pilot` runs one configurable main-experiment seed with one worker. `run-final`
runs the main family and `run-record-confidence` the record-confidence family;
`run` runs both sequentially and stops on failure. `analyze` runs all analyses
on the latest results.

Main Make overrides: `N`, `K` or `K_STEPS`, `M_MAX`, `ATTR_A`, `ATTR_B`,
`ALL_TREES`, `TIMEOUT`, `N_SEEDS`, `SEED_START`, and `WORKERS`. The seed start
defaults to 0; `N_SEEDS` defaults to 350 for the main experiment. The sequence
is contiguous.
`pilot` also accepts `PILOT_SEED` and `PILOT_WORKERS`.

```sh
make pilot N=100 K=4 M_MAX=20
make pilot N=100 K_STEPS=5 M_MAX=20
make run-final ATTR_B=age N_SEEDS=100
```

Record-confidence settings: `RECORD_M` (default 10), `RECORD_SAMPLES` (20),
`RECORD_TIMEOUT` (30s),
`RECORD_WORKERS` (2), `RECORD_N_SEEDS` (20), `RECORD_SEED_START` (0), and
`RECORD_ONLY` (optional method subset). Its private sample remains n=100.
`THREADS` sets CP-SAT search workers per solve for every family (default 4;
record confidence 8).
Shared `N_SEEDS`, `SEED_START`, and `WORKERS` override the corresponding family
settings unless a family-specific override is supplied. Main family-specific overrides use `FINAL_N_SEEDS` and `FINAL_SEED_START`;
record confidence uses the `RECORD_*` names above. `SEED_END` has been removed.
None of the runners accepts a custom output directory. Make rejects `RUN_DIR`,
old `M_VALUES`/`M_STEP` settings and `M_PICKS`.

For code checks, use `.venv/bin/python -m pytest -q`, `ruff check .`,
`ruff format --check .`, and `.venv/bin/python -m compileall -q src experiments tests`.
`make report` builds the LaTeX report from a local `report/` folder, which is
not committed; it stops with a message when that folder is missing.

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

# Re-analyse an earlier set, or compare two sets that differ only in A/B
.venv/bin/python experiments/analyze_results.py --results-dir results/<timestamp>/final_experiment
.venv/bin/python experiments/compare_background_pairs.py --results-dirs results/<t1>/final_experiment results/<t2>/final_experiment --out-dir results/comparisons/<name>
```

Parallelism: `--workers` sets how many seeds run at once and `--threads` how many
CP-SAT search workers each solve uses (default 4, record confidence 8;
`--workers` defaults to the available CPUs divided by `--threads`). Both are saved in the run manifest.
Several search workers are needed, not just faster: they run different
strategies in parallel, and one worker alone stalls on larger releases. At
n=100 with 50 random trees, one worker found no table in 180 s (condition
None) while three found the optimum in 16 s; for prior-guided sampling one
worker found no table within 30 s at m=50 and never proved the optimum at
m=10; even 4 workers never proved it at m=10 (the portfolio then lacks the
strategies that bound the distance), while 6 and 8 did in 8-16 s. Runs that reach a wall-clock timeout, use several search workers, or use
different hardware/software versions are not guaranteed to be bitwise
identical; with several workers, which of several equally good tables is
returned can vary between runs.

Resuming: set `RTDS_RUN_TIMESTAMP` to an existing timestamp and rerun the same
command; finished seeds are skipped, and the runner refuses if the protocol
differs (including `--workers`, which is part of the protocol).

## Cluster runs

`hpc/seed_array.sh` runs one seed per array task, all tasks writing into one
shared timestamp directory (see the usage lines at its top; adapt the resource
and module lines to the cluster). Each task passes its allocated CPUs as
`--threads`. Before submitting: create the virtualenv with
the cluster's Python module, run `pytest -q` in an interactive job, and download
the Adult data once on a login node, since compute nodes often have no internet
access:

```sh
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from rashomon.data import load_adult_raw; load_adult_raw('data')"
mkdir -p hpc/logs
```

## Background-knowledge configuration

The main experiment accepts `--n`, `--k` or `--k-steps`, `--m-max`,
`--attr-a`, `--attr-b`, `--all-trees` / `--no-all-trees`, seed count/start,
workers, and timeout. `--m-values`, `--m-step`, and `--m-picks` have been removed. Use
`--k-steps` only for a linear grid; otherwise `--k` selects a geometric grid.
For example, `--k-steps 10 --m-max 25` selects 1, 11, 21, and 25. The step
must be between 1 and `m_max`; the maximum is always included. The options are
mutually exclusive. The Make equivalent is `K_STEPS`.

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
```

`--all-trees` explicitly appends each seed's entire Rashomon set, even beyond
`m_max`; it is off by default. The actual size is recorded as `n_trees_available`
and a full release has `m_is_all_trees=true`. Counts beyond the available set
are skipped; the effective number of seeds can therefore differ across m.
All conditions share the same release within each seed/m pair.
The tree pool is max(1000, m_max), or the full available set with all-trees enabled.
Changing the pool can change smaller prefixes; keep it fixed for paired contrasts.

Feasibility (Adult, condition None, 180 s per solve): with 4 workers n=100 solves
up to m=200 (69 s, 1.8 GB) but not m=500; n=200 solves m=50 (107 s); n=1000 finds
no table at m=10 even with 6 workers. Larger `--n` remains available for
explicitly budgeted experiments. The 180s timeout limits solver search per
condition; it excludes construction, fitting and matching. A solve that times
out without a table is recorded with status `UNKNOWN` and no error, and the
analyses leave it out; `compare_background_pairs.py` reports the solved share
per condition and m.

A/B may be marital-status, education, age, hours-per-week or capital-gain, must be
distinct, and cannot be sex (T) or income (Y). Unknown attributes produce an error.
Numeric knowledge uses the existing threshold bits, not exact raw values. Adding
occupation or another absent attribute requires an explicit feature-schema change.
The current reconstruction protocol supplies true per-row Y; this setting remains
recorded as `labels=given_per_row`. The Swedish-inspired public-information scenario
and exact auxiliary knowledge are modeling assumptions, not population validation.

### Tree selection

Both experiment runners accept `--tree-selection random|sequential`.
Omitting the option keeps random selection as the default. Sequential selection
releases trees at indices 0 through m-1 in TreeFARMS enumeration order; it is
not boosting. Both modes retain nested releases as m increases and record the
selection policy in the protocol manifest.

```sh
make run-final TREE_SELECTION=sequential
make pilot TREE_SELECTION=sequential
make run-record-confidence RECORD_TREE_SELECTION=sequential
```

`TREE_SELECTION` defaults to `random` and also supplies the record-confidence
default unless `RECORD_TREE_SELECTION` is set explicitly.

### Attack figures and future utility overlay

`experiments/plot_attack_figures.py` reads saved main results without solving:

```sh
MPLBACKEND=Agg .venv/bin/python experiments/plot_attack_figures.py \
  --education-dir results/20261020_195325/final_experiment \
  --age-dir results/20261020_195325_01/final_experiment \
  --output-dir results/figures/office_meeting_3
```

Optional `--sequential-dir` adds the A+B selection comparison over the measured release sizes shared by the selected runs.
Use a new sequential run with the same sample settings, solver budgets and
seed indices as the random run. Only shared valid seed/m observations enter
this figure. Historical runs with different budgets are rejected.
Both geometric and uniform release grids are read from actual measurements;
markers show measured reconstruction means joined by straight segments;
accuracy uses a thinner solid smooth curve without markers. Uncertainty bands are hidden; SEM remains in the aggregate CSVs.

Optional `--education-accuracy-dir` and `--age-accuracy-dir` overlay utility on
the respective feature figure. Each directory must contain `accuracy.csv`
(columns `seed,m,accuracy`, exactly covering the attack seed/m keys) and a
`manifest.json` with `experiment: predictive_accuracy`, `attack_protocol` equal
to the source attack protocol, `utility_metric: majority_vote`, and
`tie_policy: smallest_class`. For income classes 0/1, tied votes predict 0.
The future utility producer will evaluate majority voting on held-out Y;
no utility-computation module or new experiment is included yet.

### Code organisation

- `src/rashomon/build.py`: TreeFARMS construction.
- `src/rashomon/selection.py`: nested random/sequential selection and protocol metadata.
- `src/rashomon/publisher.py`: reproducible private/public sample indices.
- `src/attack/`: reconstruction constraints, knowledge, scoring and record confidence.
- `experiments/run_*.py`: experiment configuration, execution and output orchestration.
- `experiments/analysis/`: analysis implementations, validated poster inputs and aggregation.
- `experiments/plotting/`: poster rendering and shared styles.
- `experiments/analyze_*.py`, `compare_background_pairs.py` and `plot_attack_figures.py`:
  thin CLI entry points; existing commands remain available.

The utility-only accuracy producer remains future work. This regrouping preserves
sampling, preprocessing, solver budgets and scientific metrics. Legacy analysis
modules still contain their original rendering; separation is incremental.
