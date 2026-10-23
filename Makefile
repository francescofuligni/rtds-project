PYTHON ?= .venv/bin/python
UV ?= uv
# The output root is fixed. Only this internal timestamp is shared by sub-makes.
ifneq ($(origin RUN_DIR),undefined)
$(error RUN_DIR is no longer configurable; outputs use results/{timestamp}/{experiment})
endif
ifneq ($(strip $(M_VALUES)$(M_STEP)),)
$(error M_VALUES and M_STEP were removed; use K or K_STEPS with M_MAX)
endif
ifneq ($(origin M_PICKS),undefined)
$(error M_PICKS was removed; use K or K_STEPS)
endif
ifneq ($(strip $(K)),)
ifneq ($(strip $(K_STEPS)),)
$(error Set only one of K or K_STEPS)
endif
endif
ifneq ($(origin RECORD_THREADS),undefined)
$(error RECORD_THREADS was removed; use THREADS, which applies to every family)
endif
override RUN_TIMESTAMP := $(if $(RTDS_RUN_TIMESTAMP),$(RTDS_RUN_TIMESTAMP),$(shell python3 experiments/output_paths.py))
export RTDS_RUN_TIMESTAMP := $(RUN_TIMESTAMP)
override RESULT_DIR := results/$(RUN_TIMESTAMP)

# Shared overrides apply to every family in `make run` when supplied.
N_SEEDS ?=
SEED_START ?=
ifneq ($(origin SEED_END),undefined)
$(error SEED_END was removed; use N_SEEDS)
endif
WORKERS ?=
# CP-SAT search workers per solve (default 4, see experiments/configuration.py).
THREADS ?=
THREAD_ARGS = $(if $(strip $(THREADS)),--threads $(THREADS),)

# Main paired experiment defaults (see experiments/configuration.py).
N ?= 100
K ?=
K_STEPS ?=
M_MAX ?= 200
ATTR_A ?= marital-status
ATTR_B ?= education
TREE_SELECTION ?= random
RECORD_TREE_SELECTION ?= $(TREE_SELECTION)
ALL_TREES ?= 0
TIMEOUT ?= 180
PILOT_SEED ?= 0
PILOT_WORKERS ?= 1
FINAL_SEED_START ?= $(if $(strip $(SEED_START)),$(SEED_START),0)
FINAL_N_SEEDS ?= $(if $(strip $(N_SEEDS)),$(N_SEEDS),350)

FINAL_M_ARGS = $(if $(strip $(K_STEPS)),--k-steps $(K_STEPS),--k $(if $(strip $(K)),$(K),15)) --m-max $(M_MAX)
FINAL_TREE_ARGS = $(if $(filter 1 true yes,$(ALL_TREES)),--all-trees,--no-all-trees)
FINAL_WORKER_ARGS = $(if $(strip $(WORKERS)),$(WORKERS),)

# Record-confidence family defaults and supported overrides.
RECORD_M ?= 10
RECORD_SEED_START ?= $(if $(strip $(SEED_START)),$(SEED_START),0)
RECORD_N_SEEDS ?= $(if $(strip $(N_SEEDS)),$(N_SEEDS),20)
RECORD_SAMPLES ?= 20
RECORD_WORKERS ?= $(if $(strip $(WORKERS)),$(WORKERS),2)
RECORD_TIMEOUT ?= 30
RECORD_ONLY ?=

# Read-only attack-figure rendering from existing saved experiment outputs.
EDUCATION_DIR ?=
AGE_DIR ?=
SEQUENTIAL_DIR ?=

.DEFAULT_GOAL := help
.PHONY: help setup pilot run run-final run-record-confidence plot-attack-figures analyze test report

help:
	@printf '%s\n' \
	  'RTDS project commands:' \
	  '  make setup                  Sync the locked Python 3.12 environment' \
	  '  make pilot                  Run one configurable main-experiment seed' \
	  '  make run                    Run every experiment family and analysis' \
	  '  make run-final              Run only the main experiment family' \
	  '  make run-record-confidence  Run only the record-confidence family' \
	  '  make plot-attack-figures   Plot saved attack results without rerunning experiments' \
	  '  make analyze                Re-run all analyses on the latest results' \
	  '  make test                   Run pytest' \
	  '  make report                 Build the local report PDF' \
	  'Run overrides: N, K or K_STEPS, M_MAX, ATTR_A, ATTR_B, ALL_TREES, TREE_SELECTION,' \
	  '  N_SEEDS, SEED_START, WORKERS and TIMEOUT. Outputs are always results/{timestamp}/{experiment}.' \
	  'Supplementary overrides: RECORD_* (see README).'

setup:
	$(UV) sync --locked --python 3.12

# The pilot is one seed, but otherwise uses the same main protocol overrides.
pilot:
	$(PYTHON) experiments/run_final_experiment.py $(THREAD_ARGS) --seed-start $(PILOT_SEED) --n-seeds 1 --workers $(PILOT_WORKERS) --n $(N) $(FINAL_M_ARGS) --attr-a $(ATTR_A) --attr-b $(ATTR_B) $(FINAL_TREE_ARGS) --tree-selection $(TREE_SELECTION) --timeout $(TIMEOUT)

run-final:
	$(PYTHON) experiments/run_final_experiment.py $(THREAD_ARGS) --seed-start $(FINAL_SEED_START) --n-seeds $(FINAL_N_SEEDS) $(if $(strip $(FINAL_WORKER_ARGS)),--workers $(FINAL_WORKER_ARGS),) --n $(N) $(FINAL_M_ARGS) --attr-a $(ATTR_A) --attr-b $(ATTR_B) $(FINAL_TREE_ARGS) --tree-selection $(TREE_SELECTION) --timeout $(TIMEOUT)
	$(PYTHON) experiments/analyze_results.py --results-dir $(RESULT_DIR)/final_experiment
	$(PYTHON) experiments/analyze_mechanism.py --results-dir $(RESULT_DIR)/final_experiment

run-record-confidence:
	$(PYTHON) experiments/run_record_confidence.py $(THREAD_ARGS) --m $(RECORD_M) --tree-selection $(RECORD_TREE_SELECTION) --seed-start $(RECORD_SEED_START) --n-seeds $(RECORD_N_SEEDS) --samples $(RECORD_SAMPLES) --workers $(RECORD_WORKERS) --timeout $(RECORD_TIMEOUT) $(if $(strip $(RECORD_ONLY)),--only $(RECORD_ONLY),)
	$(PYTHON) experiments/analyze_record_confidence.py --results-dirs $(RESULT_DIR)/record_confidence

plot-attack-figures:
	@test -n "$(EDUCATION_DIR)" -a -n "$(AGE_DIR)" || { echo "Set EDUCATION_DIR and AGE_DIR to saved final_experiment directories"; exit 2; }
	MPLBACKEND=Agg $(PYTHON) experiments/plot_attack_figures.py --education-dir "$(EDUCATION_DIR)" --age-dir "$(AGE_DIR)" $(if $(strip $(SEQUENTIAL_DIR)),--sequential-dir "$(SEQUENTIAL_DIR)",)

# Keep the families sequential because each can be compute- and memory-intensive.
run:
	$(MAKE) --no-print-directory run-final
	$(MAKE) --no-print-directory run-record-confidence

analyze:
	$(PYTHON) experiments/analyze_results.py
	$(PYTHON) experiments/analyze_mechanism.py
	$(PYTHON) experiments/analyze_record_confidence.py

test:
	$(PYTHON) -m pytest -q

# The report sources are kept locally (report/ is not committed).
report:
	@test -f report/Makefile || { echo "report/ is not in the repository; add your local copy first"; exit 1; }
	$(MAKE) -C report pdf
