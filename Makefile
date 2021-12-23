SHELL := /bin/bash
.PHONY: help check autoformat notebook html clean
.DEFAULT: help

# Generates a useful overview/help message for various make features
help:
	@echo "make check"
	@echo "    Run code style and linting (black, flake, isort) *without* changing files!"
	@echo "make autoformat"
	@echo "    Run code styling (black, isort) and update in place - committing with pre-commit also does this."
	@echo "make notebook"
	@echo "    Use jupytext-light to build a notebook (.ipynb) from the s4/s4.py script."
	@echo "make html"
	@echo "    Use jupyter & jupytext to do the two-step conversion from the python script, to the HTML blog post."
	@echo "make clean"
	@echo "    Delete the generated, top-level s4.ipynb notebook."

check:
	isort --check .
	black --check .
	flake8 --show-source .

autoformat:
	isort --atomic .
	black .
	flake8 --show-source .

notebook: s4/s4.py
	jupytext --to notebook s4/s4.py -o s4.ipynb

html: s4/s4.py
	jupytext --to notebook s4/s4.py -o s4.ipynb
	jupyter nbconvert --to html s4.ipynb

clean: s4.ipynb
	rm -f s4.ipynb
