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
- `data_cfg_hash`, `features_cfg_hash`, `training_cfg_hash`, `tuning_cfg_hash`, `cv_cfg_hash` — per-layer config hashes (`cv_cfg_hash` added by PR-015)
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
- **CLI subpackage exception (`src/rux_ml/cli/**`):** ruff's `TC001/TC002/TC003` (move imports into `TYPE_CHECKING`) is disabled. Reason: Typer uses `inspect.signature(..., eval_str=True)` at command-registration time, and string annotations like `"typer.Context"` fail with `NameError` if the underlying module isn't importable at runtime. Configured in `pyproject.toml` `[tool.ruff.lint.per-file-ignores]`.

---

## CV strategy conventions (per PR-015)

The workbench distinguishes **one-shot** and **repeated** splitting at the API level:

- **One-shot** — `rux_ml.data.splits.train_val_test_split(df, *, ratios, seed) → dict[str, pl.DataFrame]`. Returns three materialised Polars frames in one call. Used by `rux-ml train` for the single-baseline path; pre-dates the Splitter Protocol and does not consume `cfg.cv`.
- **Repeated CV** — `rux_ml.data.cv.Splitter` Protocol (`split(X: pl.DataFrame, y, *, groups) → Iterator[(np.ndarray, np.ndarray)]`). Yields row-index pairs per sklearn convention. Used by PR-007's Optuna objective and any future HPO loop. The Splitter is constructed inside the trial subprocess via `make_splitter(cfg.cv, seed=…)`.

The shapes intentionally differ — one-shot returns DataFrames (cheap when K=1); repeated returns indices (avoids materialising K × DataFrames in memory-bound trials).

**Groups column-to-array convention** (`GroupKFoldCV`): the config carries `groups_column: str` (a column name on the input DataFrame). The **caller** resolves it to `np.ndarray` via `df[col].to_numpy()` before calling `splitter.split(..., groups=arr)`. The Splitter never holds DataFrame state. Two cited production precedents: sklearn user guide on Group K-Fold; mlxtend `GroupTimeSeriesSplit` user guide. This keeps the Splitter Protocol stateless and pickle-friendly even though PR-015's design builds the Splitter inside the trial child (so cross-process pickling is not exercised at v0).

**Per-strategy default selection by data shape:**

| Data shape | Default Splitter |
|---|---|
| IID tabular, balanced target | `KFoldCV(n_splits=5, shuffle=True)` |
| IID tabular, imbalanced classification target | `StratifiedKFoldCV` |
| Time-indexed (fixed horizon labels) | `TimeSeriesSplitCV(gap=<label_horizon>)` |
| Time-indexed (variable horizon labels, overlapping) | `CombinatorialPurgedCV` with `embargo_size ∈ [0.005·N, 0.02·N]` per AFML §7.4.2 |
| Grouped (entity ID, session ID, etc.) | `GroupKFoldCV(groups_column=…)` |

These are starting-point defaults; final choice is per-problem and lives in `configs/problems/<name>.toml`.

**ExtMem compatibility:** only `TimeSeriesSplitCV` is `extmem_compatible` at v0; pairing any other Splitter with `ExtMemQuantileDMatrix` raises `NotImplementedError` at training time (materialised fallback deferred to a follow-up PR).

---

## HPO objective shape (per PR-007)

The Optuna objective is **K-fold CV-mean** per the PR-007 Tier-2 research findings:

- Each trial calls `make_splitter(cfg.cv, seed=cfg.tuning.entropy)` (PR-015's Splitter Protocol) and runs `cfg.cv.n_splits` fits.
- Per-fold scores are reported via `trial.report(fold_score, step=fold_idx)` so `WilcoxonPruner` (the default — purpose-built for K-fold CV per Optuna 3.6+) can paired-test against running trials.
- The trial returns `statistics.fmean(fold_scores)` as the aggregate objective value (arithmetic mean; switch to median only after measured outlier evidence per Q1.b research).
- Features pipeline is re-fit per fold for leakage hygiene (cardinalities re-computed on each fold's train set).
- XGBoost-internal `early_stopping_rounds` runs against each fold's val partition (the test fold).
- **`XGBoostPruningCallback` is NOT wired inside the CV loop** — Optuna #3203 documents that the callback's per-fold `step=0,1,…` reports break iteration-level pruners. Within-fold pruning is owned by XGBoost; cross-fold pruning is owned by Optuna's fold-level pruner.

A future single-fit objective regime (no CV; one fit per trial) would re-enable `XGBoostPruningCallback` for iteration-level pruning. That regime is not exposed at v0; the literal preserves `hyperband` / `successive_halving` pruner choices for it.

## Trial config derivation (per PR-007)

`build_objective(base_cfg)` derives a fresh `trial_cfg` per trial by:
1. Walking `base_cfg.search_space` via `walk_search_space(...) → dict[str, Any]` (flat dot-path keys like `training.learning_rate`).
2. Unflattening dot-paths to a nested dict (`{"training": {"learning_rate": 0.05}}`).
3. Deep-merging with `base_cfg.model_dump()`.
4. Re-validating the merged dict via `RuxMLConfig.model_validate(merged)`.

This pattern avoids the pitfalls of `model_copy(update=…)` with nested fields (which replaces the whole sub-model with a raw dict). Trial-cfg-derivation re-validates the entire config so type/constraint errors surface clearly per trial.

---

## Subprocess-per-trial env pinning (per PR-008)

The subprocess-per-trial path (`cfg.tuning.trial_isolation = "subprocess"`, default) requires a strict module-loading order in `src/rux_ml/_internal/trial_runner.py`:

1. **Top-level imports stay stdlib-only** (`argparse`, `json`, `os`, `sys`, `pathlib`).
2. `main()` parses CLI args + loads the overrides JSON.
3. `from rux_ml.config import RuxMLConfig` (light — pydantic + tomllib only).
4. `cfg = RuxMLConfig.from_layers(...)`.
5. **Pin thread env vars from `cfg.memory` before any heavy import**:
   - `OMP_NUM_THREADS` = `cfg.memory.omp_threads`
   - `OPENBLAS_NUM_THREADS` = `cfg.memory.openblas_threads`
   - `MKL_NUM_THREADS` = `cfg.memory.mkl_threads`
   - `POLARS_MAX_THREADS` = `cfg.memory.polars_threads`
6. **Now** lazy-import `rux_ml.tuning` + the rest (which pull in numpy / polars / sklearn / xgboost).

**Why the order matters**: numpy / openblas / mkl read these env vars at import-time to size their thread pools. Setting them after the libraries are imported is a no-op against the existing pool — at best it affects subsequent operations and at worst it silently does nothing. The child sets them once before the first heavy import.

The parent-side dispatcher uses a manual `for _ in range(n_trials): subprocess.run([...])` loop (PR-008 sub-decision A1). It does **not** call `study.optimize(_spawn_trial_dispatcher, ...)` — that pattern would double-tell the trial score (parent's `optimize` calls `study.tell` after the dispatcher returns, while the child has already told the score itself). Each child runs its own `study.optimize(build_objective(cfg), n_trials=1)`; SQLite coordinates state.

---

## Container conventions (per PR-012)

The workbench ships a single GPU-enabled image, single Compose service. Choices that future container changes should respect:

- **Both base images are digest-pinned** (`docs/CONSTRAINTS.md` "Container Digest Pinning" — NON-NEGOTIABLE). This includes the NVIDIA CUDA base (`nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:...`) **and** the uv binary copy-in stage (`ghcr.io/astral-sh/uv:0.11.14@sha256:...`). Tags can be rebuilt or deleted; digests are immutable. New digests are resolved via the registry HTTP API (`Docker-Content-Digest` header on the manifest endpoint) or, on a host with Docker, `docker buildx imagetools inspect <tag>`.
- **uv-managed Python, not the deadsnakes PPA.** `uv python install 3.12` is Astral's current Docker recipe; it avoids a third-party apt repository and keeps Python a single uv-controlled artifact for both build and runtime.
- **`uv sync --locked`, not `--frozen`.** `--locked` is the current Astral canonical (slightly stricter "lockfile must exist and not need updating" check); `--frozen` still works but the docs example uses `--locked`. Two-step pattern in the Dockerfile: deps-only layer (`uv sync --locked --no-install-project --no-dev`) for cacheability, then full sync after `COPY src/`.
- **Compose GPU passthrough uses `deploy.resources.reservations.devices`**, not the legacy `gpus: all` service key (which Docker no longer documents as of 2026). The block specifies `driver: nvidia`, `count: 1`, `capabilities: [gpu]`. The CLI flag `--gpus all` still works for `docker run` / `docker compose run`.
- **Memory caps live in Compose**: `mem_limit: 32g` (hard cap via cgroup v2 `memory.max`) + `mem_reservation: 28g` (soft cap). The 28 GB soft cap matches `MemoryConfig.watchdog_threshold_gb` (per D10) so the in-process psutil watchdog and the kernel agree on what "memory pressure" means.
- **`tini` is PID 1** so SIGTERM/SIGINT forward to the `rux-ml` process and to subprocess-per-trial children spawned via `python -m rux_ml._internal.trial_runner` (PR-008 spawn-semantics requirement).
- **Non-root `rux` user (uid 1000)** so host-bind-mounted runtime dirs (`./studies`, `./registry`, `./data`, `./logs`) round-trip ownership to a typical host user.
- **Rust toolchain installed but unused at v0.** D1 mandated installing `rustup` + stable + maturin in the image even before the first crate lands (per D13's profile-driven trigger). The toolchain is installed to `/opt/cargo` + `/opt/rustup` (world-readable); maturin is `uv tool install`'d for symmetry with the `uv run maturin develop --uv` dev loop.
- **Build-time digest capture**: `make docker-build` writes the local image ID to `.docker-image-digest` via `docker image inspect rux-ml:local --format='{{.Id}}'`. This file is gitignored; PR-013 wires its contents into `TrialAttrs.image_digest`.
- **`.dockerignore` is the canonical build-context filter.** Workbench runtime dirs (`/studies`, `/registry`, `/data`, `/logs`), Python caches (`.venv`, `.pytest_cache`, `.ruff_cache`, `.basedpyright_cache`, `.hypothesis`), and `.git` are all excluded — none of them should ever be baked into a layer.
- **`xxhash` is NOT an apt dep.** The Python `xxhash>=3.7` wheel bundles its own C extension; system `xxhash` is unused and was dropped from the spec.

Container smoke tests are gated behind `RUXML_RUN_DOCKER_TESTS=1` + a working `docker` binary (`@pytest.mark.docker`); the static regression gates in `tests/container/test_container_static.py` run unconditionally and protect the Dockerfile / Compose / `.dockerignore` from drift without needing Docker.

---

## Seed management conventions (per PR-013)

Reproducibility-grade seed handling lives in `src/rux_ml/_internal/seeds.py`. Future PRs that introduce new randomized components should plumb through the `SeedBag` rather than adding a new top-level seed field.

- **Master entropy lives at one place: `cfg.tuning.entropy: int | None`.** `None` auto-pins via `os.urandom`; an integer pins for reproducibility across the whole study.
- **Per-trial derivation is `(master_entropy, trial.number)` deterministic.** `make_seed_bag` uses `SeedSequence(entropy=master, spawn_key=(trial.number,))` so two trials with the same pair produce the same bag, and distinct `trial.number`s produce distinct bags. NumPy's [`SeedSequence.spawn`](https://numpy.org/doc/stable/reference/random/parallel.html) is the documented mechanism for this.
- **Sampler seed is study-level**, not per-trial. The Optuna sampler is constructed once at study creation; both `_internal/trial_runner.py` and `cli/tune.py` derive a study-level bag with `trial_number=0` as the sentinel and feed `study_bag.sampler_seed` to `make_sampler`. The per-trial bags (inside the objective) carry their own `sampler_seed` slot for symmetry but it is unused.
- **`bag.entropy_hex` (32 lowercase hex chars) is the round-trip identity.** Storing it in `TrialAttrs.entropy_hex` is sufficient to reconstruct the full bag — no need for the master or trial number at promotion time. `registry/promote.py` reads the recorded `entropy_hex` and calls `make_seed_bag_from_hex` so the re-fit uses the EXACT same `split_seed` and `xgb_seed` the originating trial used. Without this contract, the promoted bundle would silently diverge from the trial's reported metrics.
- **Child seeds are int32-safe.** `_seed_from_child` masks to `[0, 2**31)` so JSON / SQLite TEXT round-trips and XGBoost's `random_state` handling are both stable.
- **`features/encoders.py` `NestedCVWrapper` keeps `random_state=0`.** Target-encoder internal CV is an implementation detail of the feature pipeline, not part of the outer `(split, cv, sampler, xgb)` bag. Future PRs that introduce additional randomized feature transforms may revisit if the determinism contract demands it.
- **CPU bit-exact contract**: `device="cpu"` + `tree_method="hist"` + `OMP_NUM_THREADS=1` + pinned `xgb_seed` → bit-exact predictions. The integration test in `tests/integration/test_determinism_cpu.py` asserts this with `np.testing.assert_array_equal`; subsampling (`subsample=0.8, colsample_bytree=0.8`) is enabled there so the seed actually drives randomness.
- **GPU near-determinism contract** (per D9): `device="cuda"` + pinned `xgb_seed` → predictions match within `atol=1e-5`. **Never** `assert_array_equal` on GPU output (forbidden by `docs/CONSTRAINTS.md` Tolerance-Based Golden Tests rule). If the GPU tolerance test fails intermittently, investigate against XGBoost release notes; do not loosen the tolerance silently.

## Environment version capture (per PR-013)

`_internal/env.py:get_versions(memory)` is the single source for the `TrialAttrs` environment block. Five fields, three required:

- **Required**: `xgboost_version` (`xgboost.__version__`), `cuda_runtime_version` (`xgboost.build_info()["CUDA_VERSION"]` formatted `"major.minor"`), `omp_threads` (from `cfg.memory.omp_threads`), `image_digest` (from `.docker-image-digest` file or the literal `"unknown"` outside the container).
- **Optional**: `gpu_model` + `driver_version` (from `nvidia-smi --query-gpu`, `None` on CPU-only hosts where nvidia-smi is absent).

`xgboost.build_info()` is the XGBoost-3.x canonical surface for build metadata; the legacy `xgboost.config_context()` is for *configuring* XGBoost, not querying the build, and does not expose `CUDA_VERSION`. The PR-012 container smoke output confirmed the API: `CUDA_VERSION: [12, 9]`.

---

## Regenerating golden fixtures (per PR-014)

The committed golden fixtures in `tests/golden/fixtures/golden_v1/` (`synthetic.parquet`, `preds.npy`, `metric.json`, `manifest.json`) are the workbench's full-stack regression gate. They MUST be treated as code, not as auto-refreshable artifacts.

**When `tests/golden/test_xgb_baseline.py::test_golden_xgb_baseline_in_process` fails**, follow this investigation procedure before regenerating:

1. **Diff `manifest.json` library versions** vs the current environment (`xgboost.__version__`, `cuda_runtime_version` from `xgboost.build_info()`, sklearn, numpy, polars, skops). If any version changed, that's the likely cause — check that library's release notes for prediction-affecting changes.
2. **Run the PR-013 CPU bit-exact contract test**: `uv run pytest tests/integration/test_determinism_cpu.py`. If THAT fails, the determinism contract itself regressed — fixing that is the higher priority. Do not regenerate the golden until the determinism contract is back.
3. **Only after both check out**, run `make regenerate-golden` and review the diff of `manifest.json` (especially `library_versions` and `entropy_hex`) before committing. The regen rewrites all four fixture files atomically.

`make regenerate-golden` is **never** run in CI. The regen path is explicitly manual + reviewable; CI-driven auto-regen would silently mask real regressions.

The `--regenerate-golden` pytest flag is wired via `tests/golden/conftest.py:pytest_addoption` per the [pytest docs](https://docs.pytest.org/en/stable/example/simple.html#pass-different-values-to-a-test-function-depending-on-command-line-options). The `golden` marker is already registered in `pyproject.toml`; default `pytest` excludes it via `addopts`.

**Tolerances** (lockdown values for the v0 fixture; revisit only with documented justification):

- Predictions: `np.testing.assert_allclose(actual, golden, atol=1e-5, rtol=1e-4)` per `docs/CONSTRAINTS.md` Tolerance-Based Golden Tests Only.
- AUC band: `abs_tol=0.005` — wide enough to survive XGBoost minor releases, tight enough to catch real regressions.

**Two surfaces are tested**:

- `test_golden_xgb_baseline_in_process` — in-process pipeline determinism (training-stack regressions).
- `test_golden_load_model_matches_in_process` — promote → `load_model` round-trip (PR-010 serialization regressions). Skipped under `--regenerate-golden` (it's a structural test, not a fixture comparison).

**Adding a new golden surface is deliberate.** Each additional golden fit increases the cost of any CUDA / XGBoost upgrade because every golden must be regen'd + diffed. The v0 milestone ships exactly one golden train+predict per PR-014's scope.

---

## Where new conventions go

When a convention emerges that isn't documented here, add it during the same PR that establishes it. Conventions added retroactively go stale fast.
