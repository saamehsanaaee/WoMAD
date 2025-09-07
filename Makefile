.PHONY: install format lint test docs clean

install:
	pip install -e .

format:
	black .

lint:
	flake8 .

test:
	python -m unittest discover tests

docs:
	mkdocs build

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .venv venv build dist *.egg-info
	rm -rf site
