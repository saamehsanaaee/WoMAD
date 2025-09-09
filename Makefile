.PHONY: install format lint test docs clean deploy

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

deploy: docs
	git checkout gh-pages
	cp -r site/* .
	git add .
	git commit -m "Manual docs deployment"
	git push origin gh-pages
	git checkout main

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .venv venv build dist *.egg-info
	rm -rf site
