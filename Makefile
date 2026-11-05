.PHONY: setup lint test test-mlx toy-run web egress-test

setup:
	uv sync
	cd web && npm ci

lint:
	uv run ruff check src tests
	uv run mypy src/zllf/core

test:
	uv run pytest -m "not mlx and not slow"

test-mlx:
	uv run pytest -m mlx

toy-run:
	uv run zllf run --corpus tests/fixtures/corpus --hardware toy --runs-dir runs --auto-approve

web:
	cd web && npm run build

egress-test:
	bash deploy/egress/run_egress_test.sh
