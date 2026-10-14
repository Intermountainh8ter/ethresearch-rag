PY ?= python
PIP ?= pip
INDEX ?= .index

.PHONY: install test lint cov eval eval-gate judge index fetch serve mcp docker clean

install:
	$(PIP) install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

cov:
	pytest -q --cov=ethrag --cov-report=term-missing

eval:
	$(PY) evals/run_evals.py --show-misses

eval-gate:
	$(PY) evals/run_evals.py --min-test-hybrid-recall5 0.80

judge:
	$(PY) evals/run_judge.py

index:
	$(PY) -m ethrag.cli build-index --data data --out $(INDEX)

fetch:
	$(PY) scripts/fetch_corpus.py

serve: index
	$(PY) -m ethrag.cli serve --index $(INDEX) --port 8080

mcp: index
	$(PY) -m ethrag.cli mcp --index $(INDEX)

docker:
	docker build -t ethrag .

clean:
	rm -rf $(INDEX) .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
