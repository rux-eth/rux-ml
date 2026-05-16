.DEFAULT_GOAL := help
.PHONY: help install lint format type test test-golden test-gpu regenerate-golden clean docker-build docker-run docker-shell

help:
	@echo "rux-ml — make targets"
	@echo "  install              uv sync (install + dev deps)"
	@echo "  lint                 ruff check"
	@echo "  format               ruff format"
	@echo "  type                 basedpyright src/ tests/"
	@echo "  test                 pytest (default markers; excludes gpu/slow/golden/docker)"
	@echo "  test-gpu             pytest -m gpu"
	@echo "  test-golden          pytest -m golden"
	@echo "  regenerate-golden    rewrite tests/golden/fixtures/golden_v1/ (manual; never CI)"
	@echo "  docker-build         build rux-ml:local image and capture digest to .docker-image-digest"
	@echo "  docker-run           run rux-ml in container (ARGS=\"<cli args>\")"
	@echo "  docker-shell         open a bash shell in the container"
	@echo "  clean                remove caches and build artifacts"

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

docker-build:
	DOCKER_BUILDKIT=1 docker build -t rux-ml:local .
	docker image inspect rux-ml:local --format='{{.Id}}' > .docker-image-digest
	@echo "Built rux-ml:local — digest:"
	@cat .docker-image-digest

docker-run:
	docker compose run --rm rux-ml $(ARGS)

docker-shell:
	docker compose run --rm --entrypoint /bin/bash rux-ml

clean:
	rm -rf .pytest_cache .ruff_cache .basedpyright_cache .mypy_cache
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.py[codz]" -delete
