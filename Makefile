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
# CP-SAT workers per solve: main defaults to 4, confidence to 8.
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
ACCURACY ?= 0
ACCURACY_ARGS = $(if $(filter 1 true yes,$(ACCURACY)),--accuracy,)
FIGURE_ACCURACY_ARGS = $(if $(filter 1 true yes,$(ACCURACY)),--include-accuracy,)
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
ATTACK_DIR ?=
EDUCATION_ACCURACY_DIR ?=
AGE_ACCURACY_DIR ?=

.DEFAULT_GOAL := help
.PHONY: help setup pilot run run-final run-record-confidence run-accuracy figures analyze test report

help:
	@printf '%s\n' \
	  'RTDS project commands:' \
	  '  make setup                  Sync the locked Python 3.12 environment' \
	  '  make pilot                  Run one configurable main-experiment seed' \
	  '  make run                    Run every experiment family and analysis' \
	  '  make run-final              Run the main experiment (accuracy opt-in: ACCURACY=1)' \
	  '  make run-record-confidence  Run only the record-confidence family' \
	  '  make figures                Plot saved attack results without rerunning experiments' \
	  '  make run-accuracy           Rebuild ATTACK_DIR releases and score held-out income (no solves)' \
	  '  make analyze                Write numerical summaries of the latest results' \
	  '  make test                   Run pytest' \
	  '  make report                 Build the local report PDF' \
	  'Run overrides: N, K or K_STEPS, M_MAX, ATTR_A, ATTR_B, ALL_TREES, TREE_SELECTION,' \
	  '  N_SEEDS, SEED_START, WORKERS and TIMEOUT. Outputs are always results/{timestamp}/{experiment}.' \
	  'Supplementary overrides: RECORD_* (see README).'

setup:
	$(UV) sync --locked --python 3.12

# The pilot is one seed, but otherwise uses the same main protocol overrides.
pilot:
	$(PYTHON) experiments/run_final_experiment.py $(ACCURACY_ARGS) $(THREAD_ARGS) --seed-start $(PILOT_SEED) --n-seeds 1 --workers $(PILOT_WORKERS) --n $(N) $(FINAL_M_ARGS) --attr-a $(ATTR_A) --attr-b $(ATTR_B) $(FINAL_TREE_ARGS) --tree-selection $(TREE_SELECTION) --timeout $(TIMEOUT)

run-final:
	@status=0; \
	$(PYTHON) experiments/run_final_experiment.py $(ACCURACY_ARGS) $(THREAD_ARGS) --seed-start $(FINAL_SEED_START) --n-seeds $(FINAL_N_SEEDS) $(if $(strip $(FINAL_WORKER_ARGS)),--workers $(FINAL_WORKER_ARGS),) --n $(N) $(FINAL_M_ARGS) --attr-a $(ATTR_A) --attr-b $(ATTR_B) $(FINAL_TREE_ARGS) --tree-selection $(TREE_SELECTION) --timeout $(TIMEOUT) || status=1; \
	$(PYTHON) experiments/analyze_results.py --no-plots --results-dir $(RESULT_DIR)/final_experiment || status=1; \
	$(PYTHON) experiments/analyze_mechanism.py --no-plots --results-dir $(RESULT_DIR)/final_experiment || status=1; \
	exit $$status

run-record-confidence:
	@status=0; \
	$(PYTHON) experiments/run_record_confidence.py $(THREAD_ARGS) --m $(RECORD_M) --tree-selection $(RECORD_TREE_SELECTION) --seed-start $(RECORD_SEED_START) --n-seeds $(RECORD_N_SEEDS) --samples $(RECORD_SAMPLES) --workers $(RECORD_WORKERS) --timeout $(RECORD_TIMEOUT) $(if $(strip $(RECORD_ONLY)),--only $(RECORD_ONLY),) || status=1; \
	$(PYTHON) experiments/analyze_record_confidence.py --no-plots --results-dirs $(RESULT_DIR)/record_confidence || status=1; \
	exit $$status

run-accuracy:
	@test -n "$(ATTACK_DIR)" || { echo "Set ATTACK_DIR to a saved final_experiment directory"; exit 2; }
	$(PYTHON) experiments/run_predictive_accuracy.py --attack-dir "$(ATTACK_DIR)" $(if $(strip $(SEED_START)),--seed-start $(SEED_START),) $(if $(strip $(N_SEEDS)),--n-seeds $(N_SEEDS),)

figures:
	@test -n "$(EDUCATION_DIR)" -a -n "$(AGE_DIR)" || { echo "Set EDUCATION_DIR and AGE_DIR to saved final_experiment directories"; exit 2; }
	MPLBACKEND=Agg $(PYTHON) experiments/plot_attack_figures.py $(FIGURE_ACCURACY_ARGS) --education-dir "$(EDUCATION_DIR)" --age-dir "$(AGE_DIR)" $(if $(strip $(SEQUENTIAL_DIR)),--sequential-dir "$(SEQUENTIAL_DIR)",) $(if $(strip $(EDUCATION_ACCURACY_DIR)),--education-accuracy-dir "$(EDUCATION_ACCURACY_DIR)",) $(if $(strip $(AGE_ACCURACY_DIR)),--age-accuracy-dir "$(AGE_ACCURACY_DIR)",)

# Keep the families sequential because each can be compute- and memory-intensive.
run:
	@status=0; \
	$(MAKE) --no-print-directory run-final || status=1; \
	$(MAKE) --no-print-directory run-record-confidence || status=1; \
	exit $$status

analyze:
	@status=0; \
	$(PYTHON) experiments/analyze_results.py --no-plots || status=1; \
	$(PYTHON) experiments/analyze_mechanism.py --no-plots || status=1; \
	$(PYTHON) experiments/analyze_record_confidence.py --no-plots || status=1; \
	exit $$status

test:
	$(PYTHON) -m pytest -q

# The report sources are kept locally (report/ is not committed).
report:
	@test -f report/Makefile || { echo "report/ is not in the repository; add your local copy first"; exit 1; }
	$(MAKE) -C report pdf
