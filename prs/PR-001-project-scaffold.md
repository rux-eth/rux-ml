# PR-001: Project scaffold

**Landed-in:** v0.0.1

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

### State Assessment (2026-05-15)

**Current state of the codebase**:

- Repo on `dev` after merge of "Initial architecture, PR plan, and design log (D1–D17)" (commit `ddfa7ba`)
- No `src/`, no `pyproject.toml`, no `uv.lock`, no `Makefile`, no `Dockerfile` — PR-001 will create the scaffold from scratch
- Existing files: `CLAUDE.md`, `README.md`, three `PROCEDURE-*.md`, six `docs/*.md`, fourteen `prs/PR-*.md`, the project's `.gitignore`
- No prior PR has touched PR-001's area (this is the first build PR)

**Assumptions at PR draft time** (drafted 2026-05-15 from D1, D12, D14, D15):

- `uv` is the canonical Python dep manager
- `[build-system]` uses `hatchling` (or per maturin's recommendation when Rust lands later)
- `src/<pkg>/` layout per PyPA
- `ruff` for lint+format
- `mypy --strict` (or pyright equivalent) for type checking — left as an open choice
- pytest with `[tool.pytest.ini_options]` markers + `addopts`
- `.python-version` for interpreter pinning
- `py.typed` marker per PEP 561

**Verification against current authoritative sources**:

| Item | Status | Source |
|---|---|---|
| uv `sync`/`run`/`uv.lock` workflow | **STILL CURRENT** | [uv docs](https://docs.astral.sh/uv/) |
| `[build-system]` default for new uv projects | **DRIFT FLAGGED** | [uv build backend](https://docs.astral.sh/uv/concepts/build-backend/), [uv init reference](https://docs.astral.sh/uv/reference/cli/#uv-init) |
| `src/<pkg>/` layout | **STILL CURRENT** | [PyPA src vs flat](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) |
| `ruff check` + `ruff format` defaults | **STILL CURRENT** | [Ruff tutorial](https://docs.astral.sh/ruff/tutorial/) |
| Type checker: mypy vs pyright | **DRIFT FLAGGED (soft)** | [ty vs mypy vs pyright 2026](https://www.danilchenko.dev/posts/ty-vs-mypy-vs-pyright/), [pydevtools](https://pydevtools.com/handbook/explanation/how-do-mypy-pyright-and-ty-compare/) |
| pytest 9.x marker + addopts syntax | **STILL CURRENT** | [pytest mark docs](https://docs.pytest.org/en/stable/how-to/mark.html) |
| `.python-version` for uv | **STILL CURRENT** | [uv python versions](https://docs.astral.sh/uv/concepts/python-versions/) |
| `py.typed` marker | **STILL CURRENT (note)** | [PEP 561](https://peps.python.org/pep-0561/) — historical; canonical guidance now at [typing spec site](https://typing.python.org/en/latest/spec/) |

**Stale assumptions** (where current state disagrees with PR assumptions):

1. **`[build-system]` backend**: PR-001 listed `hatchling` (or maturin's recommendation later) as the default. **`uv init` now defaults to Astral's own `uv_build` backend** (`uv_build>=0.11.14,<0.12`). `hatchling` is one option among `uv|hatch|flit|pdm|poetry|setuptools|maturin|scikit`. For a pure-Python package (no Rust yet at v0), `uv_build` is the canonical 2026 recommendation. When the first Rust crate lands per D13, switch to `maturin` per the maturin docs.

2. **Type checker**: PR-001 said "`mypy --strict` (or `pyrightconfig.json`)". **2026 community signal: pyright/basedpyright is the new-project recommendation; mypy is for legacy CI.** The "open choice" in PR-001 should now resolve to **basedpyright** (or pyright) for stricter defaults than mypy with better speed and spec conformance (~98% vs Astral's `ty` at ~53%, mypy slower than both).

**New constraints learned from prior PRs or codebase evolution**:

None — this is the first build PR. `dev` is empty of code; `docs/` has the full design.

**Synthesis Outcome: AMEND**

Two drifts surfaced. Both are scaffold-config changes (single-file `pyproject.toml` edits + dev-dependency swap), not architectural rework — they don't propagate to downstream PRs. Per `PROCEDURE-pr-research.md` Phase 4 Outcome Branch, **synthesis stopped pending explicit user decision** on the amendments below.

### Pending user decision (Phase 4 Amend gate)

- **Amendment A — `[build-system]`**: switch PR-001 spec from `hatchling` → **`uv_build`** for v0; document migration to `maturin` in the first Rust PR per D13.
- **Amendment B — Type checker**: resolve PR-001's open choice as **basedpyright** (or pyright if you prefer the stricter Microsoft default), removing mypy from the scaffold.

Both updates are append-only to PR-001's `## Scope` and `## Verification criteria` sections; no other PR file is affected.

_Phases 2–4 (Scope, Findings, Synthesis) and Phase 5 (Gate Check) will be appended after the user decides on amendments._

---

## Scope

Bootstrap the repo so subsequent PRs have a foundation to build on. Concretely:

- `pyproject.toml` declaring the `rux_ml` package, Python ≥ 3.12, dependencies (initial: just `typer`, `pydantic`, `pydantic-settings`), build system **`uv_build`** (per current `uv init` default; migrate to `maturin` when D13's first Rust crate lands — recorded in PR-001 Notes and the future Rust-crate PR)
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
- `[tool.basedpyright]` configured with strict defaults (basedpyright chosen per Phase 1 amendment B; replaces the original "mypy or pyright" open choice)
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
- [ ] `uv run basedpyright src/` passes
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

- The `__version__` is the package's source of truth. With `uv_build` as the backend, `[project]` reads it via `dynamic = ["version"]` and `[tool.uv_build]` (or equivalent) per Astral's docs.
- Line length and ruff rule selection are minor BEST-GUESS choices (88 vs 100, ALL vs E,F,I,W,...) — pick once, stay consistent. Document in `docs/CONVENTIONS.md` after this PR lands.
- **Doc updates ride with this PR's implementation commit** (per `PROCEDURE-pr-research.md` Phase 4 synthesis):
  - `docs/CONVENTIONS.md` "Code Style" — replace "`mypy --strict` (or `basedpyright`)" with "`basedpyright` (strict defaults)"
  - `CLAUDE.md` "Build & Test Commands" — replace `uv run mypy src/` with `uv run basedpyright src/`
- Future-Rust note: when D13's first Rust crate lands, that PR switches `[build-system]` from `uv_build` → `maturin`. The transition is a single-file change to `pyproject.toml`.

---

### Synthesis (2026-05-15)

**Outcome:** Amend (approved by user 2026-05-15).

**Changes to this PR from research:**
- `[build-system]` backend: `hatchling` → **`uv_build`** (Amendment A; PROVEN by [uv build backend docs](https://docs.astral.sh/uv/concepts/build-backend/), [uv init reference](https://docs.astral.sh/uv/reference/cli/#uv-init))
- Type checker: `mypy --strict` (or pyright) → **`basedpyright`** (strict) (Amendment B; CONVENTION 2026 per [ty vs mypy vs pyright 2026](https://www.danilchenko.dev/posts/ty-vs-mypy-vs-pyright/), [pydevtools type-checker comparison](https://pydevtools.com/handbook/explanation/how-do-mypy-pyright-and-ty-compare/))

**Changes to ARCHITECTURE.md:** None. Architecture is unchanged; the amendments are scaffold-config.

**Changes to CONSTRAINTS.md:** None.

**Changes to CONVENTIONS.md:** 1-line edit to "Code Style" — rides with PR-001 implementation commit (listed in Notes above).

**Changes to CLAUDE.md:** 1-line edit to "Build & Test Commands" table — rides with PR-001 implementation commit (listed in Notes above).

**New PRs that must come first:** None.

**Research-backed details now locked in this PR:**
- `[build-system]` = `uv_build` for v0 (pure-Python); migrate to `maturin` when first Rust crate lands per D13
- Type checker = `basedpyright` (strict defaults)

### Gate Check

- Premise still valid: ✓ (no architectural rework; both drifts are scaffold-config)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-05-15)
- Implementation cleared: ✓ (2026-05-15)
