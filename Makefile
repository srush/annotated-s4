SHELL := /bin/bash
.PHONY: help check autoformat
.DEFAULT: help

# Generates a useful overview/help message for various make features
help:
	@echo "make check"
	@echo "    Run code style and linting (black, flake, isort) *without* changing files!"
	@echo "make autoformat"
	@echo "    Run code styling (black, isort) and update in place - committing with pre-commit also does this."

check:
	isort --check .
	black --check .
	flake8 --show-source .

autoformat:
	isort --atomic .
	black .
	flake8 --show-source .
