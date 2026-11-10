PYTHON ?= .venv/bin/python
TABLE_OUTPUT ?= artifacts/saved_tables

.PHONY: tables data test help

tables:
	$(PYTHON) scripts/reproduce_saved_tables.py --output $(TABLE_OUTPUT)

data:
	$(PYTHON) scripts/fetch_data.py

test:
	$(PYTHON) -m pytest -q

help:
	@echo "make tables [TABLE_OUTPUT=artifacts/new_tables]: reproduce saved numerical tables"
	@echo "make data: download, verify and prepare public application data"
	@echo "make test: run the numerical and workflow checks"
	@echo "Full experiment commands: docs/RUNNING.rst"
