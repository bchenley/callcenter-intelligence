.PHONY: install test test-integration test-all lint format run clean

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/unit tests/security -v

test-integration:
	pytest tests/integration -v

test-all:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff check --fix . && ruff format .

run:
	python app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
