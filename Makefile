PYTHON ?= .venv/bin/python
UV ?= uv

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
	$(PYTHON) experiments/run_final_experiment.py
	$(PYTHON) experiments/analyze_results.py
	$(PYTHON) experiments/analyze_mechanism.py
	$(PYTHON) experiments/run_record_confidence.py
	$(PYTHON) experiments/analyze_record_confidence.py
	$(PYTHON) experiments/run_calibration_experiment.py
	$(PYTHON) experiments/analyze_calibration.py
	$(PYTHON) experiments/calibrate_confidence.py

test:
	$(PYTHON) -m pytest -q
