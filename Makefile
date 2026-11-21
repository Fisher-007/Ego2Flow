PYTHON ?= python3

.PHONY: test check

test:
	$(PYTHON) -m pytest -q tests

check:
	$(PYTHON) -m compileall -q benchmark examples evaluate.py
	$(PYTHON) evaluate.py --help
