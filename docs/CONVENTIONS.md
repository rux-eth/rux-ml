# Conventions

Agreed-upon patterns for `rux-ml`. Unlike constraints (hard rules — violating them is a bug), conventions are soft patterns: follow them unless you have a good reason not to, and document the exception.

Conventions emerge during design sessions and implementation. Add them here as they solidify.

---

## Directory Conventions

The repo skeleton was research-validated against `lightning-hydra-template`, the closest known precedent (see D14 in `docs/DESIGN-log.md`). Most top-level directories follow established conventions; a few are inventions. The mapping below pre-empts confusion for future readers.

| Directory | Status | Closest analog in established templates |
|---|---|---|
| `src/rux_ml/` | PROVEN convention | PyPA src layout; Kedro inner-src |
| `tests/` | PROVEN convention | pytest good practices; Lightning-Hydra-template |
| `configs/` (plural) | CONVENTION | Lightning-Hydra-template uses `configs/`; Kedro uses `conf/` |
| `docs/`, `prs/` | CONVENTION | Universal; `prs/` from the vibe-rails template starter |
| `crates/.gitkeep` | PROVEN | polars/ruff workspace pattern; deferred per D13 |
| `studies/` | INVENTION | No template covers Optuna-first workflows. Closest: Lightning-Hydra-template's `logs/multiruns/` |
| `registry/` | INVENTION | No filesystem-registry convention exists. Closest: CCDS `models/` with explicit promotion semantics layered on |
| `data/cas/` | INVENTION | DIY DVC replacement. CCDS uses `data/{raw,interim,processed,external}/` — different model |
| `data/manifests/` | INVENTION | JSON manifests indexing the CAS. No precedent |
| `logs/` | PROVEN convention | Lightning-Hydra-template uses exactly this name |

`studies/`, `registry/`, `data/`, `logs/` are runtime-generated and gitignored. They live at the repo root by convention (MLflow, DVC, Optuna all default to project-tree). The path root is overridable via `WORKBENCH_HOME` env var.

Source data lives **outside** the repo by default; the workbench's `data/cas/` + `data/manifests/` is the *versioned cache*, not source files. Paths to source data go in TOML config.

---

## Naming Conventions

### Files & directories

- **Module / file names:** `snake_case.py` (e.g., `objective.py`, `data_iter.py`)
- **Class names:** `PascalCase` (e.g., `RuxMLConfig`, `XGBoostTrainer`, `MemoryPressureError`)
- **Functions / variables:** `snake_case`
- **Constants:** `UPPER_SNAKE_CASE` at module scope
- **Private helpers:** leading underscore (`_resolve_toml_layers`)
- **TOML config files:** `kebab-case.toml` is acceptable; `snake_case.toml` matches Python module names — pick one per directory and stay consistent
- **Pydantic config classes:** `<Layer>Config` (e.g., `DataConfig`, `TrainingConfig`); root is `RuxMLConfig`

### CLI verbs

- **Verb groups:** `data`, `train`, `tune`, `runs`, `registry` (singular nouns or imperative verbs; chosen per D11)
- **Subcommands:** lowercase verb (`promote`, `rollback`, `list`, `show`, `compare`, `start`, `resume`, `status`, `retry-trial`)
- **Override syntax:** dot-path with `=` (`--training.learning_rate=0.05`); env var equivalent uses `__` for nesting (`RUXML_TRAINING__LEARNING_RATE`)

### Registry version strings

- **Format:** `v_<YYYY>_<MM>_<DD>_<short_hash>` (e.g., `v_2026_05_14_a8f3c2`)
- **Status:** BEST-GUESS (no canonical convention) — easy to change if it hurts
- **Rationale:** human-readable, sortable by date, hash provides uniqueness within a day

### Per-trial user_attr keys

Names recorded on every Optuna trial (constraints in `docs/CONSTRAINTS.md`):

- `data_hash` — dataset content hash (D9)
- `data_cfg_hash`, `features_cfg_hash`, `training_cfg_hash`, `tuning_cfg_hash` — per-layer config hashes (D17)
- `root_cfg_hash` — full config hash (alias also written as `config_hash` for backward-compat queries)
- `git_sha`, `entropy_hex`, `image_digest`, `xgboost_version`, `cuda_runtime_version`, `gpu_model`, `driver_version`, `omp_threads`, `peak_rss_mb`

---

## Module Dependency Rules

The package is layered (per D15). Dependencies flow downward only:

```
            cli/
              │
              ▼
   ┌──── tuning/ ───┐
   │       │        │
   │       ▼        │
   ▼   training/    ▼
runs/      │     registry/
   │       ▼
   │   features/
   │       │
   │       ▼
   └──── data/ ──── config/  (config is leaf upward)
                     │
                     ▼
                _internal/
```

Concretely:

- `cli/` may import from any layer. **No layer imports from `cli/`.**
- `config/` is imported by every layer. **`config/root.py` is the only file that imports ALL per-layer configs** — prevents cycles.
- `_internal/` is the only package importable from everywhere (utilities, hashing, env, memory, seeds).
- Cross-layer calls go through explicit factory functions (`make_data(cfg)`, `make_features(cfg)`, `make_trainer(cfg)`), never through deep-imports of internal modules.

---

## Public API Discipline

Per D15, `src/rux_ml/__init__.py` exposes a curated set with explicit `__all__`:

```python
__version__ = "..."
__all__ = ["RuxMLConfig", "Trainer", "load_run", "load_model", "list_runs", "list_registry"]
```

Everything else requires qualified imports (`from rux_ml.tuning import run_tuning`). Subpackage `__init__.py` files either stay empty or carry their own `__all__`. This is ruff-friendly and matches sklearn's discipline.

When adding a new public symbol: add it to `__all__` in the appropriate `__init__.py`, and document why a user would reach for it.

---

## Test Conventions

Per D12, registered pytest markers:

- `gpu` — requires CUDA; auto-skipped on CPU-only machines
- `slow` — runs longer than ~5s (excluded from default run)
- `golden` — golden-dataset regression test (excluded from default run)
- `integration` — multi-component integration test

`pyproject.toml` sets `addopts = "-m 'not gpu and not slow and not golden'"` so default `pytest` is fast.

Golden tests use `np.testing.assert_allclose(preds, golden, atol=1e-5, rtol=1e-4)` plus AUC/RMSE tolerance bands. **Never** exact hashes (forbidden by `docs/CONSTRAINTS.md`).

---

## Logging Conventions

- Use the project's structured logger from `rux_ml._internal.logging` — no `print` in library code
- **Trial-correlation field:** every log line emitted during a trial includes `trial_id` and `study_name`
- **Log levels:** `info` for high-level flow (start of run, end of trial); `debug` for per-iteration detail; `warning` for recoverable degradations (e.g., switched to ExtMemQuantileDMatrix because X-size exceeded threshold); `error` for raised exceptions
- **Sensitive data:** none expected (no auth, no PII), but if added later, never log raw config values containing secrets

---

## Configuration Conventions

Per D17, the three-tier composition layers in this order (lowest → highest priority):

1. `configs/base.toml`
2. `configs/problems/<name>.toml`
3. `configs/studies/<name>.toml`
4. `.env` file
5. Env vars (`RUXML_*` with `__` for nesting)
6. CLI dot-path overrides

`TomlConfigSettingsSource(deep_merge=True)` ensures nested tables overlay correctly. Lists replace; they do not concatenate.

`extra="forbid"` is set on all Pydantic models so typos surface as validation errors instead of silently ignored fields.

Hash elision: paths, timestamps, and runtime-only fields (`logs.path`, `studies.storage_url`) are excluded from `*_cfg_hash` computation. The elision list lives as a constant `_HASH_ELIDED_FIELDS` in `src/rux_ml/config/root.py`.

---

## Rust+PyO3 Conventions (when crates exist)

Per D13, no Rust at v0. When the first crate lands:

- Cargo workspace at root (`Cargo.toml` with `members = ["crates/*"]`)
- Crate naming: `rux_ml_<area>` (e.g., `rux_ml_kernels`)
- Build via `maturin` integrated with `uv` (`uv run maturin develop --uv`)
- Python import side: **try/except ImportError with pure-Python fallback** at the call site, with the pure-Python reference impl alongside the Rust function
- Cross-language test target: `make test` runs both `cargo test` (per crate) and `pytest`

---

## Code Style

- `ruff` for linting + formatting (config in `pyproject.toml`)
- `basedpyright` (strict defaults) for type checking — replaces the original mypy choice per PR-001 Phase 1 amendment ([2026 type-checker comparison](https://www.danilchenko.dev/posts/ty-vs-mypy-vs-pyright/))
- Line length: project default (88 / 100 — pick one in PR-001 and stay consistent)
- Imports: external → internal → relative, ruff-sorted

---

## Where new conventions go

When a convention emerges that isn't documented here, add it during the same PR that establishes it. Conventions added retroactively go stale fast.
