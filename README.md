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
odometer over shared subtrees, so the first m trees are near-copies and leak
far less than m distinct trees would. All runners therefore release a random
ordering's first m trees by default (`--selection random`, nested across m);
`--selection first` reproduces the legacy behaviour.

## Results and protocol stamping

Every result set is its own directory with a `manifest.json` (see
`experiments/provenance.py`):

```
results/<experiment>/<selection>/
  manifest.json      protocol: every setting that changes what the numbers mean
  runs/*.json        one file per run: git commit, uncommitted changes, command,
                     host, package versions (safe for concurrent cluster array jobs)
  seed000.csv ...    one file per seed (record_confidence: m010/seed000.csv ...)
  summary_*.txt      analysis output, starting with the protocol it came from
  *.pdf              figures
```

- A runner refuses to write into a directory whose protocol differs from its own,
  or that holds unstamped files. Changing a setting means a new directory
  (`--out-dir`).
- Runs are resumable: rerunning the same command skips finished seeds. Seed
  ranges (`--seed-start/--seed-end`) can be split across machines or cluster
  array jobs and land in the same directory.
- Analyses only read data through the manifest and refuse to compare result
  sets whose protocols differ beyond what is being compared.
- Stamped result sets are committed; scratch runs go in `results/dev/` (ignored).

`results/legacy/` holds the results produced before stamping (first-m tree
selection, per-row labels given). Their manifests describe how they were
produced; they are superseded once the experiments are rerun with random
selection.

## Running

```sh
uv sync                                                           # or: pip install -e . into .venv
.venv/bin/python -m pytest -q

.venv/bin/python experiments/run_record_confidence.py --m 10 50 100 --seed-end 20
.venv/bin/python experiments/analyze_record_confidence.py         # compares all result sets found

.venv/bin/python experiments/run_final_experiment.py --seed-end 350
.venv/bin/python experiments/analyze_results.py                   # --results-dir picks another set
.venv/bin/python experiments/analyze_mechanism.py

# Re-analyse the legacy results
.venv/bin/python experiments/analyze_results.py --results-dir results/legacy/final_experiment
```

Parallelism: `--workers` sets how many seeds run at once; for the record-confidence
runner `--threads` sets CP-SAT threads per solve (defaults use the cores the
process is allowed to use, which respects cluster allocations).

## Cluster runs

`hpc/seed_array.sh` runs one seed per array task (see the usage lines at its
top; adapt the resource and module lines to the cluster). Before submitting:
create the virtualenv with the cluster's Python module, run `pytest -q` in an
interactive job, and download the Adult data once on a login node, since
compute nodes often have no internet access:

```sh
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from rashomon.data import load_adult_raw; load_adult_raw('data')"
mkdir -p hpc/logs
```
