# PR-001: Project scaffold

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

---

## Scope

Bootstrap the repo so subsequent PRs have a foundation to build on. Concretely:

- `pyproject.toml` declaring the `rux_ml` package, Python ≥ 3.12, dependencies (initial: just `typer`, `pydantic`, `pydantic-settings`), build system (`hatchling` or per maturin's recommendation when Rust lands later)
- `uv.lock` committed
- `.python-version` pinning the interpreter
- `.gitignore` per `docs/CONVENTIONS.md`
- `.editorconfig`
- `Makefile` with `install`, `lint`, `format`, `test`, `clean` targets
- `src/rux_ml/__init__.py` with `__version__` and an empty `__all__`
- `src/rux_ml/py.typed` marker
- `tests/__init__.py` + `tests/conftest.py` (empty fixtures placeholder)
- `tests/test_smoke.py` with one test asserting `rux_ml.__version__` is a string
- `crates/.gitkeep`
- `configs/README.md` describing the three-tier composition (no actual config files yet)
- `ruff.toml` or `[tool.ruff]` in `pyproject.toml` with line length and rules
- `[tool.mypy]` (strict) or `pyrightconfig.json`
- `[tool.pytest.ini_options]` registering markers (`gpu`, `slow`, `golden`, `integration`) and `addopts = "-m 'not gpu and not slow and not golden'"`

NOT in scope: Dockerfile (PR-012), any layer module body, any CLI commands, any config models.

## Dependencies

None.

## Architecture section implemented

Repo skeleton per `docs/CONVENTIONS.md` (Directory Conventions table) and `docs/ARCHITECTURE.md` (System Overview).

## Verification criteria

- [ ] `uv sync` completes without error
- [ ] `uv run pytest` runs and passes (single smoke test)
- [ ] `uv run python -c "import rux_ml; print(rux_ml.__version__)"` prints a version
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy src/` (or pyright equivalent) passes
- [ ] `make test` succeeds end-to-end
- [ ] All gitignored runtime dirs (`studies/`, `registry/`, `data/`, `logs/`) are listed in `.gitignore` even though they don't exist yet

## Research backing

Tier 1 — directly research-backed at design time:

- D1 (uv + maturin + container pattern): [uv adoption](https://aleyan.com/blog/2026-why-arent-we-uv-yet/), [PyO3 docs](https://pyo3.rs/v0.28.0/building-and-distribution.html)
- D12 (pytest layout + markers): [pytest good practices](https://docs.pytest.org/en/latest/explanation/goodpractices.html)
- D14 (`src/` layout + commit lockfiles): [PyPA src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/), [Cargo FAQ](https://doc.rust-lang.org/cargo/faq.html)
- D15 (curated public API + `__all__`): [sklearn `__init__.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/__init__.py)

State assessment must verify nothing in the uv/PyPA ecosystem has shifted in ways that change the scaffold (e.g., `pyproject.toml` `[build-system]` recommendations).

## Notes

- The `__version__` is the package's source of truth; `[project]` in `pyproject.toml` reads from it via `dynamic = ["version"]` per modern hatchling/setuptools idioms.
- Line length and ruff rule selection are minor BEST-GUESS choices (88 vs 100, ALL vs E,F,I,W,...) — pick once, stay consistent. Document in `docs/CONVENTIONS.md` after this PR lands.
