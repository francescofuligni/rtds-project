# RTDS: Reconstructing Data from Released Decision Trees

This repository contains a research project for the Research Topics in Data
Science course at Stockholm University. It studies how an attacker can infer a
sensitive attribute from a released set of accurate decision trees, and how
public background knowledge affects that reconstruction.

The experiments build a Rashomon set with TreeFARMS and release nested subsets
of its trees. For each release size, the attacker reconstructs the sensitive
target `T` under four knowledge conditions: no extra knowledge, attribute `A`,
attribute `B`, or both `A+B`. The current main setup uses `T = sex`, known
per-row income labels `Y`, `A = marital-status`, and `B = education` (or another
supported attribute). Tree selection is random by default; sequential
TreeFARMS-order selection is also available for comparison. A separate
record-confidence experiment evaluates attacker-side confidence methods.

The project is a local, reproducible research pipeline. It is not a production
service. Experiment manifests record the protocol and run environment so that
results can be inspected and compared under explicit conditions.

## Set up

Requirements: Python 3.11 or later and `uv`. The lockfile is verified with
Python 3.12. From the repository root:

```sh
uv sync --locked --python 3.12
make help
make test
```

The test command may download the UCI Adult dataset if it is not already in the
local data cache. No full experiment is needed to run the test suite.

## Run experiments

Start with a small one-seed pilot:

```sh
make pilot N=100 K=4 M_MAX=20
```

Run the main experiment and its analyses, or both experiment families:

```sh
make run-final
make run
```

The main defaults are `n=100`, up to 15 geometric release sizes from 1 to 200
trees, marital status for `A`, education for `B`, 350 seeds, and a 180-second
solver timeout. Full runs can be compute-intensive. Use Make variables to adjust
the sample, release grid, attributes, tree selection, seed count, worker count,
and timeout; `make help` lists the command interface.

Random selection is the default. To run the main experiment with sequential
TreeFARMS enumeration instead:

```sh
make run-final TREE_SELECTION=sequential
```

Result folders are created under `results/<timestamp>/`, separated by
experiment family. Each contains a protocol `manifest.json`, per-run metadata,
per-seed CSVs, and analysis summaries. New runs do not replace earlier raw
results. Analyses validate protocol compatibility before comparing datasets.

## Analyze saved results and regenerate figures

`make analyze` writes numerical summaries of the latest saved results without
rendering historical plots. Direct analysis CLIs retain plotting by default;
pass `--no-plots` for numerical output only. To select a
specific result folder, use the corresponding analysis script with
`--results-dir` or `--results-dirs`; for example:

```sh
.venv/bin/python experiments/analyze_results.py \
  --results-dir results/<timestamp>/final_experiment
```

Regenerate the attack feature figures without rerunning any experiment by
pointing the plotting target at saved education and age results:

```sh
make figures \
  EDUCATION_DIR=results/<education-run>/final_experiment \
  AGE_DIR=results/<age-run>/final_experiment
```

Add `SEQUENTIAL_DIR=results/<sequential-run>/final_experiment` to include the
random/sequential comparison. Paired comparisons require compatible protocols
and shared seed/release-size observations. Each feature figure also gets a
`_gap` variant that shades the gap between no background knowledge and A+B.

Generated figures are PNGs in `figures/`; rerendering replaces files with the
same names. Derived plotting tables and `sources.json` are kept in
`figures/data/`. Raw seed data and manifests remain in their source folders
under timestamped folders in `results/`.

### Held-out predictive accuracy

New `make run`, `make run-final` and `make pilot` runs compute predictive income
accuracy by default on the same tree releases already in memory. No extra
TreeFARMS fit or reconstruction solve is required. The main seed CSVs include
`predictive_accuracy`, shared across the four knowledge conditions for each
seed/m; the manifest declares the vote, tie and holdout policies.

When `make figures` reads two new main runs, it automatically adds accuracy to
**only** `attack_marital_education.png` and `attack_marital_age.png`, with income
accuracy on the right axis and sex reconstruction error on the left. The `_gap`
variants and random/sequential comparison contain reconstruction error only.
The utility measurements used for each overlay are exported once per seed/m to
`figures/data/attack_marital_<education|age>_accuracy.csv`.

For example, run the main family separately for each background pair:

```sh
make run-final ATTR_B=education N_SEEDS=100 WORKERS=1
make run-final ATTR_B=age N_SEEDS=100 WORKERS=1
# Use the two printed final_experiment directories; no utility flags needed.
make figures EDUCATION_DIR=results/<education-run>/final_experiment \
  AGE_DIR=results/<age-run>/final_experiment
```

`make run` still runs the configured main pair plus record-confidence; it does
not implicitly launch both background pairs or render figures without their inputs.
Existing archives are unchanged. To add utility to historical runs, rebuild a saved main run's tree releases and evaluate majority-vote income
accuracy on its disjoint per-seed holdout, without rerunning reconstruction:

```sh
# Bounded pilot: one saved seed. Omit N_SEEDS to process every saved seed.
make run-accuracy ATTACK_DIR=results/20261020_195325/final_experiment N_SEEDS=1
make run-accuracy ATTACK_DIR=results/20261020_195325/final_experiment
make run-accuracy ATTACK_DIR=results/20261020_195325_01/final_experiment
```

Each invocation prints a new `results/<timestamp>/predictive_accuracy/` directory.
Seeds run serially; rebuilding TreeFARMS still costs computation. Parameters and
measured seed/m keys come from the source run, rather than current runner defaults.
Tied votes predict income class 0. Historical preprocessing is reproduced,
including thresholds fitted on the full cleaned table, so this is not a fully
training-only preprocessing evaluation. Tree counts and joint-split fractions
must match saved diagnostics. Historical tree hashes are absent: passing these
checks does not certify identical historical trees. Newly generated release
hashes, input hashes and execution metadata are retained for future comparisons.
Independent seeds continue after errors; the process exits nonzero for failures.
Partial/pilot accuracy files cannot overlay a complete attack run.

Pass the two complete output directories to the existing figure command:

```sh
make figures \
  EDUCATION_DIR=results/20261020_195325/final_experiment \
  AGE_DIR=results/20261020_195325_01/final_experiment \
  SEQUENTIAL_DIR=results/20261025_190500/final_experiment \
  EDUCATION_ACCURACY_DIR=results/<education-utility-run>/predictive_accuracy \
  AGE_ACCURACY_DIR=results/<age-utility-run>/predictive_accuracy
```

The right axis shows held-out income accuracy; the left retains sex reconstruction
error. Accuracy uses all measured releases, including those with missing attack
errors. The two utility runs are evaluated separately; equal curves are not assumed.

## Repository map

- `src/rashomon/` builds and selects trees from the Rashomon set.
- `src/attack/` implements reconstruction, scoring, and confidence methods.
- `experiments/` contains experiment runners, analysis commands, and plotting.
- `results/` stores manifests and raw experiment outputs.
- `figures/` stores generated PNGs and reproducibility data for attack plots.
- `tests/` contains the pytest suite.
- `docs/` contains project background material.

### Incomplete experiments

Runners finish independent work after individual failures. Main work units are
seeds; confidence work units are (seed, m) pairs. Solver outcomes without a
candidate retain their status and missing metric; unexpected exceptions discard
only the affected unit. No automatic retries are performed.

Each invocation records outcomes under `runs/outcomes/`, separately from protocol
manifests and scientific CSVs. Confidence coverage includes selection/scoring
attempts and successful solves for every method, including methods with no claims.
Candidate support is conditional on successful solves, not a calibrated posterior.
Historical archives without this coverage are reported as unavailable.

Runners and Make return a nonzero status for failures or missing candidates after
finishing independent jobs, analyses and experiment families. Insufficient tree
counts remain explicit skips. Analyses report incomplete coverage; no data means
no inferred statistics or figures. `make figures` validates all selected inputs
and completes temporary rendering before replacing existing output files.
