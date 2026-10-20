PYTHON ?= .venv/bin/python
UV ?= uv
# The output root is fixed. Only this internal timestamp is shared by sub-makes.
ifneq ($(origin RUN_DIR),undefined)
$(error RUN_DIR is no longer configurable; outputs use results/{timestamp}/{experiment})
endif
ifneq ($(strip $(M_VALUES)$(M_STEP)$(CALIBRATION_M_VALUES)$(CALIBRATION_M_STEP)),)
$(error M_VALUES and M_STEP were removed; use K or K_STEPS with M_MAX)
endif
ifneq ($(origin M_PICKS),undefined)
$(error M_PICKS was removed; use K or K_STEPS)
endif
ifneq ($(origin CALIBRATION_M_PICKS),undefined)
$(error CALIBRATION_M_PICKS was removed; use CALIBRATION_K or CALIBRATION_K_STEPS)
endif
ifneq ($(strip $(K)),)
ifneq ($(strip $(K_STEPS)),)
$(error Set only one of K or K_STEPS)
endif
endif
ifneq ($(strip $(CALIBRATION_K)),)
ifneq ($(strip $(CALIBRATION_K_STEPS)),)
$(error Set only one of CALIBRATION_K or CALIBRATION_K_STEPS)
endif
endif
ifneq ($(origin RECORD_THREADS),undefined)
$(error RECORD_THREADS was removed; CP-SAT solver threads are fixed at 1)
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

# Main paired experiment defaults.
N ?= 1000
K ?=
K_STEPS ?=
M_MAX ?= 500
ATTR_A ?= marital-status
ATTR_B ?= education
ALL_TREES ?= 0
TIMEOUT ?= 180
PILOT_SEED ?= 0
PILOT_WORKERS ?= 1
FINAL_SEED_START ?= $(if $(strip $(SEED_START)),$(SEED_START),0)
FINAL_N_SEEDS ?= $(if $(strip $(N_SEEDS)),$(N_SEEDS),350)

FINAL_M_ARGS = $(if $(strip $(K_STEPS)),--k-steps $(K_STEPS),--k $(if $(strip $(K)),$(K),40)) --m-max $(M_MAX)
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

# Evaluator-calibration family defaults and supported overrides.
CALIBRATION_N ?= 100
CALIBRATION_K ?=
CALIBRATION_K_STEPS ?=
CALIBRATION_M_MAX ?= 500
CALIBRATION_SEED_START ?= $(if $(strip $(SEED_START)),$(SEED_START),0)
CALIBRATION_N_SEEDS ?= $(if $(strip $(N_SEEDS)),$(N_SEEDS),35)
CALIBRATION_WORKERS ?= $(WORKERS)
CALIBRATION_ATTR_A ?= marital-status
CALIBRATION_ATTR_B ?= education
CALIBRATION_ALL_TREES ?= 0
CALIBRATION_TIMEOUT ?= 60

CALIBRATION_M_ARGS = $(if $(strip $(CALIBRATION_K_STEPS)),--k-steps $(CALIBRATION_K_STEPS),--k $(if $(strip $(CALIBRATION_K)),$(CALIBRATION_K),2)) --m-max $(CALIBRATION_M_MAX)
CALIBRATION_TREE_ARGS = $(if $(filter 1 true yes,$(CALIBRATION_ALL_TREES)),--all-trees,--no-all-trees)

.DEFAULT_GOAL := help
.PHONY: help setup pilot run run-final run-record-confidence run-calibration \
	analyze test report

help:
	@printf '%s\n' \
	  'RTDS project commands:' \
	  '  make setup                  Sync the locked Python 3.12 environment' \
	  '  make pilot                  Run one configurable main-experiment seed' \
	  '  make run                    Run every experiment family and analysis' \
	  '  make run-final              Run only the main experiment family' \
	  '  make run-record-confidence  Run only the record-confidence family' \
	  '  make run-calibration        Run only the evaluator-calibration family' \
	  '  make analyze                Re-run all analyses on the latest results' \
	  '  make test                   Run pytest' \
	  '  make report                 Build the local report PDF' \
	  'Run overrides: N, K or K_STEPS, M_MAX, ATTR_A, ATTR_B, ALL_TREES,' \
	  '  N_SEEDS, SEED_START, WORKERS and TIMEOUT. Outputs are always results/{timestamp}/{experiment}.' \
	  'Supplementary overrides: RECORD_* and CALIBRATION_* (see README).'

setup:
	$(UV) sync --locked --python 3.12

# The pilot is one seed, but otherwise uses the same main protocol overrides.
pilot:
	$(PYTHON) experiments/run_final_experiment.py --seed-start $(PILOT_SEED) --n-seeds 1 --workers $(PILOT_WORKERS) --n $(N) $(FINAL_M_ARGS) --attr-a $(ATTR_A) --attr-b $(ATTR_B) $(FINAL_TREE_ARGS) --timeout $(TIMEOUT)

run-final:
	$(PYTHON) experiments/run_final_experiment.py --seed-start $(FINAL_SEED_START) --n-seeds $(FINAL_N_SEEDS) $(if $(strip $(FINAL_WORKER_ARGS)),--workers $(FINAL_WORKER_ARGS),) --n $(N) $(FINAL_M_ARGS) --attr-a $(ATTR_A) --attr-b $(ATTR_B) $(FINAL_TREE_ARGS) --timeout $(TIMEOUT)
	$(PYTHON) experiments/analyze_results.py --results-dir $(RESULT_DIR)/final_experiment
	$(PYTHON) experiments/analyze_mechanism.py --results-dir $(RESULT_DIR)/final_experiment

run-record-confidence:
	$(PYTHON) experiments/run_record_confidence.py --m $(RECORD_M) --seed-start $(RECORD_SEED_START) --n-seeds $(RECORD_N_SEEDS) --samples $(RECORD_SAMPLES) --workers $(RECORD_WORKERS) --timeout $(RECORD_TIMEOUT) $(if $(strip $(RECORD_ONLY)),--only $(RECORD_ONLY),)
	$(PYTHON) experiments/analyze_record_confidence.py --results-dirs $(RESULT_DIR)/record_confidence

run-calibration:
	$(PYTHON) experiments/run_calibration_experiment.py --seed-start $(CALIBRATION_SEED_START) --n-seeds $(CALIBRATION_N_SEEDS) $(if $(strip $(CALIBRATION_WORKERS)),--workers $(CALIBRATION_WORKERS),) --n $(CALIBRATION_N) $(CALIBRATION_M_ARGS) --attr-a $(CALIBRATION_ATTR_A) --attr-b $(CALIBRATION_ATTR_B) $(CALIBRATION_TREE_ARGS) --timeout $(CALIBRATION_TIMEOUT)
	$(PYTHON) experiments/analyze_calibration.py --results-dir $(RESULT_DIR)/calibration_experiment
	$(PYTHON) experiments/calibrate_confidence.py --results-dir $(RESULT_DIR)/calibration_experiment

# Keep the families sequential because each can be compute- and memory-intensive.
run:
	$(MAKE) --no-print-directory run-final
	$(MAKE) --no-print-directory run-record-confidence
	$(MAKE) --no-print-directory run-calibration

analyze:
	$(PYTHON) experiments/analyze_results.py
	$(PYTHON) experiments/analyze_mechanism.py
	$(PYTHON) experiments/analyze_record_confidence.py
	$(PYTHON) experiments/analyze_calibration.py
	$(PYTHON) experiments/calibrate_confidence.py

test:
	$(PYTHON) -m pytest -q

report:
	$(MAKE) -C report pdf
