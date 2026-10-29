# RTDS: Reconstructing Data from Released Decision Trees

A Research Topics in Data Science course project at Stockholm University.
The project studies reconstruction of a sensitive attribute from selected
decision trees in a TreeFARMS Rashomon set and their exact per-leaf class counts.

The main experiment compares four background-knowledge conditions on the same
sample and released trees: None, A, B and A+B. The sensitive target is **T = sex**;
**Y = income** is the predictive label and is explicitly assumed known per row.
A and B are additional known attributes, by default marital status and education.
T is never supplied as background knowledge. A separate record-confidence
experiment evaluates scores attached to reconstructed record claims.

## Repository map

| Directory | Contents |
| --- | --- |
| [`src/rashomon/`](src/rashomon/) | Adult preprocessing, sampling, tree construction, selection and voting |
| [`src/attack/`](src/attack/) | Reconstruction constraints, knowledge conditions, baselines and scoring |
| [`experiments/`](experiments/) | Runners, analysis CLIs and shared configuration/output helpers |
| [`experiments/analysis/`](experiments/analysis/) | Numerical analysis and validated figure inputs |
| [`experiments/plotting/`](experiments/plotting/) | Figure rendering and styles |
| [`tests/`](tests/) | Small synthetic tests and bounded integration checks |
| [`results/`](results/) | Saved experiment data, protocol manifests and numerical summaries |
| [`figures/`](figures/) | Generated experiment PNGs |
| [`docs/`](docs/) | Background reading and the [selected source catalogue](docs/papers/papers.md) |
| [`report/`](report/) | LaTeX report skeleton, bibliography and build instructions |
| [`hpc/`](hpc/) | Slurm seed-array script; adapt resources to the cluster before use |

## Setup and first run

Install `uv` and use Python 3.12 for the locked environment (Python >=3.11 is
declared). Run commands from the repository root:

```sh
uv sync --locked --python 3.12
make help
make test
make pilot N=100 K=4 M_MAX=20
```

Tests use synthetic data and temporary files; they do not download Adult or run
full experiments. The pilot runs one main-experiment seed. Experiment commands
fetch the UCI Adult training data if the local cache is absent.

## Experiments, analysis and figures

```sh
# Main reconstruction experiment followed by its numerical analyses.
make run-final

# Both reconstruction and record-confidence families, with their analyses.
make run

# Example: five main seeds using age as B and sequential tree selection.
make run-final ATTR_B=age TREE_SELECTION=sequential N_SEEDS=5 WORKERS=1

# Numerical summaries for the latest saved run of each family.
make analyze

# Analyze one selected archive without rendering figures.
.venv/bin/python experiments/analyze_results.py --no-plots \
  --results-dir results/<timestamp>/final_experiment

# Render figures from saved education/age runs, without new experiments.
make figures \
  EDUCATION_DIR=results/<education-run>/final_experiment \
  AGE_DIR=results/<age-run>/final_experiment
```

Add `SEQUENTIAL_DIR=results/<sequential-run>/final_experiment` to the figure
command for the random/sequential A+B comparison. Inputs must have compatible
protocols and shared valid seed/release-size measurements. The command also
renders the two gap views and the background overview. Its None reference comes
from the education run; uniform and marginal controls are measured saved values.
Direct analysis CLIs render their own plots unless `--no-plots` is supplied.

### Main options

| Make variable | Default / meaning |
| --- | --- |
| `N` | `100` private training rows |
| `K`, `M_MAX` | `15` geometric picks up to `200` trees; rounding may reduce the count |
| `K_STEPS` | Optional linear step instead of `K`; includes the maximum |
| `ATTR_A`, `ATTR_B` | `marital-status`, `education`; supported alternatives include `age` |
| `TREE_SELECTION` | `random` nested prefixes; `sequential` uses TreeFARMS enumeration |
| `ALL_TREES` | `0`; set `1` to add the full available set |
| `N_SEEDS`, `SEED_START` | Main: `350`, `0`; confidence: `20`, `0` |
| `WORKERS` | Concurrent seeds; main defaults to available CPUs divided by solver threads, confidence to `2` |
| `THREADS` | CP-SAT workers per solve; main `4`, confidence `8` |
| `TIMEOUT` | Main solve limit: `180` seconds |
| `ACCURACY` | `0`; set `1` to calculate main predictive accuracy or plot embedded accuracy |

Full runs can be compute-intensive. `make run` executes one configured main
background pair and the confidence family. Run education and age separately
when producing the two figure inputs.

### Supplementary commands

| Command | Use |
| --- | --- |
| `make run-record-confidence` | Confidence only; `RECORD_M=10`, `RECORD_SAMPLES=20`, `RECORD_TIMEOUT=30` by default |
| `make run-record-confidence RECORD_ONLY=prior_guided` | Select a method; the runner's `--help` lists all methods |
| `make run-accuracy ATTACK_DIR=results/<run>/final_experiment` | Rebuild historical releases and score held-out income without reconstruction solves |
| `make run-accuracy ATTACK_DIR=results/<run>/final_experiment N_SEEDS=1` | Bound the historical accuracy replay to one seed |
| `.venv/bin/python experiments/compare_background_pairs.py --help` | Compare saved runs with different A/B choices |

Confidence also accepts `RECORD_SEED_START`, `RECORD_N_SEEDS`, `RECORD_WORKERS`
and `RECORD_TREE_SELECTION`. Shared `N_SEEDS`, `SEED_START`, `WORKERS` and
`THREADS` apply to both families; confidence-specific values override shared ones.
Runner CLIs expose additional settings through `--help`.

Accuracy is opt-in for main runs and plots. Use `make run-final ACCURACY=1` to
measure the exact in-memory releases, then `make figures ACCURACY=1` with saved
inputs to overlay those measurements. Only the two base education/age plots get
an accuracy axis; gap, overview and selection plots remain reconstruction-only.

Historical replay prints a fresh `results/<timestamp>/predictive_accuracy/`
directory. Pass complete replay outputs as `EDUCATION_ACCURACY_DIR` and
`AGE_ACCURACY_DIR` to `make figures` to request separate overlays explicitly.
A one-seed pilot cannot overlay a complete attack run.

## Outputs and interpretation

Runs use `results/<Stockholm timestamp>/<experiment>/`. Protocol manifests
record scientific settings; per-seed CSVs contain measurements, and `runs/`
records execution provenance and invocation outcomes. Confidence CSVs are grouped
by release size. New standalone invocations allocate fresh timestamps; `make run`
shares one timestamp across families. Existing raw archives are preserved.

Runners finish independent work after a seed/job fails, record missing candidates
with their solver status, and exit nonzero for incomplete invocations. Make still
attempts the associated analyses and remaining family. There are no automatic
retries. Analysis comparisons exclude missing pairs rather than treating them as
losses. Insufficient tree counts are reported as skipped release sizes.

`make figures` validates inputs and stages all renders before replacing existing
same-named PNGs and associated local plotting metadata. Raw experiment
inputs remain unchanged. Numerical analyses write summaries beside their inputs.

A feasible or optimal table need not be authentic or unique. Scoring matches rows
within Y groups using non-T truth; this is evaluator-assisted matching, not
attacker-side identification. Confidence scores are not calibrated probabilities
by construction. Uniform/marginal controls do not use A/B knowledge.

Predictive accuracy measures held-out majority-vote income predictions, with ties
resolved to class 0. Historical replay checks counts and structural diagnostics
but cannot certify identical historical trees without original hashes.
Preprocessing thresholds are fitted on the full cleaned table, so this is not a
fully training-only preprocessing evaluation.

## Report

`report/report.tex` includes the five course sections and an individual-contributions
section. The scientific text is intentionally empty; authors fill it themselves.
The title, author names and repository URL are retained.
[Selected sources](docs/papers/papers.md) explain the references maintained in
`report/references.bib`.

The supplied ICLR 2025 template is configured for one column and the course's
11-point, single-spaced text. `make report` produces `report/report.pdf`;
`make -C report pdf-direct` provides a pdfLaTeX/BibTeX fallback. The course limit
is six content pages, excluding references. Enable the bibliography after adding
the first citation. [Template notes](report/TEMPLATE.md) document provenance and
the small compatibility/course adaptations.

To rebuild automatically whenever a report source or bibliography is saved, run
`make -C report watch` and leave it running; stop it with Ctrl+C. This requires
latexmk and updates `report/report.pdf` after successful compilation, keeping
auxiliary files in `report/.build/`.

## Comments and text

Use English for maintained text, comments, docstrings, messages and plot labels.
Preserve original titles, names and scientific data. Write short, plain comments
that explain a reason, assumption or non-obvious constraint; omit narration of
obvious code and change history. Keep docstrings focused on behavior, inputs,
outputs and relevant limits. Distinguish implemented behavior, project assumptions
and scientific guarantees. Keep tests focused on meaningful properties and a few
representative workflow checks, without duplicating implementation details.
