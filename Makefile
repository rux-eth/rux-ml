.DEFAULT_GOAL := help
.PHONY: help install lint format type test test-golden test-gpu clean

help:
	@echo "rux-ml — make targets"
	@echo "  install      uv sync (install + dev deps)"
	@echo "  lint         ruff check"
	@echo "  format       ruff format"
	@echo "  type         basedpyright src/ tests/"
	@echo "  test         pytest (default markers; excludes gpu/slow/golden)"
	@echo "  test-gpu     pytest -m gpu"
	@echo "  test-golden  pytest -m golden"
	@echo "  clean        remove caches and build artifacts"

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

type:
	uv run basedpyright src/ tests/

test:
	uv run pytest

test-gpu:
	uv run pytest -m gpu

test-golden:
	uv run pytest -m golden

regenerate-golden:
	uv run pytest -m golden --regenerate-golden

clean:
	rm -rf .pytest_cache .ruff_cache .basedpyright_cache .mypy_cache
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.py[codz]" -delete
