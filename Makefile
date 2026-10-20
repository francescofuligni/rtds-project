PYTHON ?= .venv/bin/python
UV ?= uv
RUN_TIMESTAMP := $(shell python3 experiments/output_paths.py)
RUN_RESULTS := results/$(RUN_TIMESTAMP)

.DEFAULT_GOAL := help
.PHONY: help setup run test

help:
	@printf '%s\n' \
	  'RTDS project commands:' \
	  '  make help   Show available commands' \
	  '  make setup  Sync the locked Python 3.12 environment' \
	  '  make run    Run all experiments, analyses, and artifact generation' \
	  '  make test   Run the pytest suite'

setup:
	$(UV) sync --locked --python 3.12

run:
	$(PYTHON) experiments/run_final_experiment.py --out-dir $(RUN_RESULTS)/final_experiment
	$(PYTHON) experiments/analyze_results.py --results-dir $(RUN_RESULTS)/final_experiment
	$(PYTHON) experiments/analyze_mechanism.py --results-dir $(RUN_RESULTS)/final_experiment
	$(PYTHON) experiments/run_record_confidence.py --out-dir $(RUN_RESULTS)/record_confidence
	$(PYTHON) experiments/analyze_record_confidence.py --results-dirs $(RUN_RESULTS)/record_confidence
	$(PYTHON) experiments/run_calibration_experiment.py --out-dir $(RUN_RESULTS)/calibration_experiment
	$(PYTHON) experiments/analyze_calibration.py --results-dir $(RUN_RESULTS)/calibration_experiment
	$(PYTHON) experiments/calibrate_confidence.py --results-dir $(RUN_RESULTS)/calibration_experiment

test:
	$(PYTHON) -m pytest -q
