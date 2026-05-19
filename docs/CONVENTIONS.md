# Conventions

Agreed-upon patterns for `rux-ml`. Unlike constraints (hard rules — violating them is a bug), conventions are soft patterns: follow them unless you have a good reason not to, and document the exception.

Conventions emerge during design sessions and implementation. Add them here as they solidify.

---

## Directory Conventions

The repo skeleton was research-validated against `lightning-hydra-template`, the closest known precedent (see D14 in `docs/0.0/DESIGN-log.md`). Most top-level directories follow established conventions; a few are inventions. The mapping below pre-empts confusion for future readers. <!-- rewrite-doc-refs:skip-line -->

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
- **Override syntax:** repeatable `--set <dot-path>=<value>` (e.g. `--set training.learning_rate=0.05 --set tuning.n_trials=100`). Values are JSON-parsed when possible (numbers / bools / lists / objects) and fall back to raw strings otherwise. Typos in the dot-path surface as `ValidationError` because every Pydantic model has `extra="forbid"`. Env var equivalent uses `__` for nesting (`RUXML_TRAINING__LEARNING_RATE`).

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

**base.toml discriminator-table hygiene** (per PR-022): when a `base.toml` table corresponds to a Pydantic discriminated union (`[cv]`, `[training]`, `[solving]`, and `[tuning.search_space.*]` once those become base-settable), only fields present in EVERY variant of that union may live at the base-table position. Variant-specific knobs (`kfold.shuffle`, `xgboost.tree_method`, `time_series.gap`, `cpcv.embargo_size`, `cpcv.n_folds`, ...) go in `configs/problems/<n>.toml`. Reason: `deep_merge` is a plain-dict merge that carries base fields across the discriminator on `kind` switches; Pydantic's `extra="forbid"` then rejects fields the new variant doesn't declare. Both behaviors are intentional — the rule is about TOML *content*, not runtime behavior. The discriminator field itself is `kind` for `[cv]` / `[training]` / `[solving]` and `type` for `[tuning.search_space.*]` — the structural test handles both. Enforced by `tests/config/test_base_toml_discriminator_hygiene.py`; future contributors adding a variant-specific field to base fail CI, not the user at runtime.

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

- **One-shot** — `rux_ml.data.splits.train_val_test_split(df, *, ratios, seed) → dict[str, pl.DataFrame]` for the random / IID path, and `rux_ml.data.splits.temporal_train_val_test_split(df, *, time_column, ratios) → dict[str, pl.DataFrame]` for the time-ordered / time-series path (PR-024). Both return three materialised Polars frames. `rux-ml train` and `registry/promote.py` route through `make_splits(cfg, df, *, seed)` which dispatches on `cfg.data.split_kind ∈ {"random", "time_ordered"}` — two separate functions, no kind-knob (≥3-cited convention: sktime `temporal_train_test_split`, Darts `TimeSeries.split_before/split_after`, AutoGluon TimeSeriesPredictor, Nixtla `mlforecast.cross_validation`, mlfinlab).
- **Repeated CV** — `rux_ml.data.cv.Splitter` Protocol (`split(X: pl.DataFrame, y, *, groups) → Iterator[(np.ndarray, np.ndarray)]`). Yields row-index pairs per sklearn convention. Used by PR-007's Optuna objective and any future HPO loop. The Splitter is constructed inside the trial subprocess via `make_splitter(cfg.cv, seed=…)`.

The shapes intentionally differ — one-shot returns DataFrames (cheap when K=1); repeated returns indices (avoids materialising K × DataFrames in memory-bound trials).

**`data.split_kind` × `cv.kind` consistency** (PR-024). The `RuxMLConfig` model_validator enforces two cross-field rules at config-load time (fail-fast, ValueError):

- `data.split_kind == "time_ordered"` requires `data.time_column` to be set.
- `cv.kind ∈ {"time_series", "cpcv", "panel_cpcv"}` requires `data.split_kind == "time_ordered"` — otherwise the one-off `rux-ml train` baseline would random-shuffle while the HPO loop respects temporal ordering, producing meaningfully different train/val/test layouts between the two CLI paths and re-introducing the leakage profile PR-023 closed.

The temporal one-off split is **deterministic** (sort by `time_column` then slice by ratio; no seed). This preserves the reproducibility contract between `cli/train.py` and `registry/promote.py` automatically — promotion-time re-fit reproduces the same train/val/test layout the trial saw without any seed plumbing. Single-asset users can leave `data.split_kind = "random"` (default) for v0.1.1-identical behavior.

**Groups column-to-array convention** (`GroupKFoldCV`): the config carries `groups_column: str` (a column name on the input DataFrame). The **caller** resolves it to `np.ndarray` via `df[col].to_numpy()` before calling `splitter.split(..., groups=arr)`. The Splitter never holds DataFrame state. Two cited production precedents: sklearn user guide on Group K-Fold; mlxtend `GroupTimeSeriesSplit` user guide. This keeps the Splitter Protocol stateless and pickle-friendly even though PR-015's design builds the Splitter inside the trial child (so cross-process pickling is not exercised at v0).

**Per-strategy default selection by data shape:**

| Data shape | Default Splitter |
|---|---|
| IID tabular, balanced target | `KFoldCV(n_splits=5, shuffle=True)` |
| IID tabular, imbalanced classification target | `StratifiedKFoldCV` |
| Single-asset time series (fixed horizon labels) | `TimeSeriesSplitCV(gap=<label_horizon>)` |
| Single-asset time series (variable horizon labels, overlapping) | `CombinatorialPurgedCV` with `embargo_pct ∈ [0.005, 0.02]` per AFML §7.4.2 |
| **Stacked panel** (multiple rows per timestamp; per-asset forward labels) | `PanelCombinatorialPurgedCV(time_column=…, asset_column=…, target_horizon_bars=<horizon_in_timestamps>, embargo_pct=…)` per PR-023 D1 |
| **Stacked panel** (simple walk-forward, fixed time-unit embargo) | `TimeSeriesSplitCV(time_column=…, embargo_time="24h")` per PR-023 D2 |
| Grouped (entity ID, session ID, etc.) | `GroupKFoldCV(groups_column=…)` |

These are starting-point defaults; final choice is per-problem and lives in `configs/problems/<name>.toml`.

**ExtMem compatibility:** only `TimeSeriesSplitCV` is `extmem_compatible` at v0.1+; pairing any other Splitter with `ExtMemQuantileDMatrix` raises `NotImplementedError` at training time (materialised fallback deferred to a follow-up PR). `PanelCombinatorialPurgedCV` is not ExtMem-compatible (requires the full `time_column` materialised at split time).

**`TimeSeriesSplitCV.gap` is row-count, not time-units** (per PR-022). The `gap` field excludes N **rows** between train-end and test-start — directly from sklearn's `TimeSeriesSplit` semantics. On a stacked panel with K rows per timestamp (e.g., K assets × hourly bars), `gap=N` rows ≈ `N/K` timestamps of separation per asset. Setting `gap=24` on a 1084-asset panel produces <1 hour of per-asset embargo, not 24 hours. For single-asset time series this isn't an issue (1 row = 1 bar); for panels use the panel-aware paths landed in PR-023 (`TimeSeriesSplitCV.embargo_time="24h"` for simple walk-forward; `PanelCombinatorialPurgedCV` for CPCV). Live finding from the 2026-05-18 first-real-dataset run.

**`TimeSeriesSplitCV.embargo_time` polymorphic semantics** (per PR-023 D2). When set, `embargo_time` wins over `gap`:

- `int` → row-count gap, same semantics as `gap` (kept for the field-uniformity convention).
- `str` (e.g., `"24h"`, `"3d"`) → requires `time_column`; parsed via `pandas.Timedelta`. The splitter computes the median delta between unique sorted timestamps and the mean rows-per-unique-timestamp, then translates the requested duration into a row-count gap. On regular bars this is exact; on irregular bars it's a **conservative best-guess** (out-of-scope precision improvement tracked in `docs/0.2/RESEARCH-BACKLOG.md` A2). Convention precedent: sktime `SlidingWindowSplitter`, Nixtla `mlforecast.cross_validation`, Darts `historical_forecasts`.

**`TimeSeriesSplitCV.time_unit` for Int64 timestamp columns** (per PR-027). When `time_column` is `pl.Int64` (e.g., Unix-seconds from a database column — the crypto-h3 shape), `time_unit` MUST be set to one of `"ns"`, `"us"`, `"ms"`, `"s"`. The splitter promotes the column via `pl.from_epoch(col, time_unit=time_unit)` before computing the median delta — polars' `.cast(pl.Datetime("ns"))` on Int64 reinterprets the integers AS nanoseconds-since-epoch with no unit conversion, which silently mis-counts otherwise. `pl.Datetime` columns carry their own unit intrinsically; setting `time_unit` on a `pl.Datetime` column raises `ValueError` (signals user confusion, fail-fast per PR-024 cross-field validator convention). Convention precedent: polars `pl.from_epoch(time_unit=...)`, pandas `pd.to_datetime(unit=...)`, Nixtla `validate_freq` (4-of-6 surveyed CV libraries enforce explicit unit declaration; see PR-027 Phase 3).

**Panel CPCV** (`PanelCombinatorialPurgedCV`, per PR-023 D1). Folds over **unique sorted timestamps** read from `time_column`; skfolio CPCV runs on the timestamp axis (so `target_horizon_bars` and `embargo_pct` are interpreted in **timestamp units**, not rows). Each timestamp-level fold is mapped back to row indices by selecting every row at that timestamp — per-asset purge is timestamp-atomic (drop a timestamp from train → drop every asset's row at that timestamp). `asset_column` is required and validated at split time so the wrapper fails loudly when given mismatched data. Convention precedent: mlfinlab `StackedCombinatorialPurgedKFold`, Numerai era-wise CV.

**skfolio purge precision gap** (PR-023 D4, MANDATORY note). skfolio's `CombinatorialPurgedCV` uses a **scalar two-sided `purged_size` in indexes** — it drops `purged_size` rows on each side of every test span. This is a row-count simplification of AFML §7.4.2's **interval-overlap purge**, which drops only the train observations whose actual label window overlaps the test indexes. The skfolio model is **conservative-correct** (drops more train rows than strictly needed; never leaks), but coarser than AFML's interval-overlap. Workbench accepts this tradeoff per D4 — it keeps skfolio as the CPCV backend without reinventing the AFML interval algorithm. Users who need interval-overlap precision should reach for `mlfinlab.cross_validation.PurgedKFold` or `timeseriescv` directly; both are paid/licensed alternatives and not bundled. Skfolio docs: `skfolio.model_selection._combinatorial.py:81–220`. AFML reference: López de Prado, *Advances in Financial Machine Learning*, Snippet 7.1 (purge), Snippet 7.3 (`mbrg = int(X.shape[0] * pctEmbargo)`).

---

## HPO objective shape (per PR-007)

The Optuna objective is **K-fold CV-mean** per the PR-007 Tier-2 research findings:

- Each trial calls `make_splitter(cfg.cv, seed=cfg.tuning.entropy)` (PR-015's Splitter Protocol) and runs `cfg.cv.n_splits` fits.
- Per-fold scores are reported via `trial.report(fold_score, step=fold_idx)` so `WilcoxonPruner` (the default — purpose-built for K-fold CV per Optuna 3.6+) can paired-test against running trials.
- The trial returns `statistics.fmean(fold_scores)` as the aggregate objective value (arithmetic mean; switch to median only after measured outlier evidence per Q1.b research).
- Features pipeline is re-fit per fold for leakage hygiene (cardinalities re-computed on each fold's train set).
- XGBoost-internal `early_stopping_rounds` runs against each fold's test partition (the held-out fold is passed as `eval_set`). **This is a pragmatic deviation from textbook CV, not a clean convention** — research-backed in PR-022 Phase 3 (2026-05-18). Specifically:
  - It matches the library-blessed default of `xgboost.cv()` / `lightgbm.cv()` / `catboost.cv()` (the built-ins use the held-out fold as their early-stopping watch-list) and the XGBoost sklearn-API doc example.
  - It produces an **optimism bias** in the per-fold metric, because `best_iteration_` is HP-selected on the same fold the score is computed on. XGBoost's own docs call this out: *"using early stopping during cross validation may not be a perfect approach because it changes the model's number of trees for each validation fold."* — https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#early-stopping
  - The bias **compounds across HPO trials**: Optuna selects HPs whose `best_iteration` on the test fold maximizes test-fold score. Across hundreds of trials, the selection itself adapts to the test folds. Workbench CV metrics should be treated as point estimates for HP *ranking* — not as unbiased generalization estimates.
  - It is **not** inherited from Optuna's WilcoxonPruner recipe; that tutorial uses independent problem instances, not CV folds with early stopping (https://optuna.readthedocs.io/en/latest/tutorial/20_recipes/013_wilcoxon_pruner.html). The workbench's "clean responsibility separation" framing (XGBoost owns within-fold, Optuna owns across-fold) is a deliberate workbench *choice* given this tradeoff.
  - **Alternatives explicitly available**: Position B (carve an inner val from the train fold; sklearn's `HistGradientBoosting*` default) and Position C (no early stopping in CV; XGBoost's *own* recommendation for CV-with-HPO — *"A better approach is to retrain the model after cross validation using the best hyperparameters along with early stopping."*).
  - **PR-025 calibrated the A/B/C tradeoff empirically on the workbench's first real dataset** (crypto-h3, 1084-asset stacked panel, 20.7M rows, `PanelCombinatorialPurgedCV(n_folds=5, n_test_folds=2)` per PR-023). On the SAME splits + seeds + search space, 10 Optuna trials per position (TPE + WilcoxonPruner — 6–8 completed per position after pruning):
    - **Position A** (current default): median RMSE **0.027101**, within-position spread 0.000006, best 0.027096.
    - **Position B** (inner val from train): median RMSE **0.027101** — identical to A to 6 decimals; spread 0.000013, best 0.027096.
    - **Position C** (no early stopping): median RMSE **0.027172** — **+0.26% vs A**, ~60% slower wallclock.
    A's "optimism bias" did NOT materialize empirically here — A == B within Optuna's own search noise. Implementation cost of Shape 2 (switch the workbench to Position B) — `_fold_scores` rewrite + per-family eval_set adapters across XGBoost / LightGBM Pattern-A shim / CatBoost — would buy a 0% RMSE improvement on this problem. **Decision: stay on Position A; this rule remains the workbench default**. The optimism-bias warning above still stands — workbench CV scores are point estimates for HP *ranking*, not unbiased generalization estimates — but on the workbench's first real dataset the bias is operationally zero. Per memory `project_cv_strategy_tier2`, per-problem CV revisits remain a known pattern; future problems may surface a B advantage and warrant a separate calibration. PR-025 calibration receipt: `prs/PR-025-calibration-results.json`; harness: `scripts/calibrate_pr025.py`.
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

## Per-family optional extras naming (per PR-018 A3)

Optional dependencies are PEP 631 extras named **after the family**, not after the backend package. Each `[project.optional-dependencies]` entry installs one family's runtime deps:

```toml
[project.optional-dependencies]
lightgbm = ["lightgbm>=4.6.0"]
# catboost = ["catboost>=1.2"]    # PR-019
# osqp     = ["osqp>=0.6"]        # PR-020
```

Rationale: the family name (`lightgbm`) matches `cfg.training.kind` and the `TRAINER_FAMILIES["lightgbm"]` registry key. Users who want a family run `uv sync --extra lightgbm`; the workbench's hard `[project.dependencies]` covers the default-experience family (xgboost) per Q-Dep (PR-017).

Note: PR-018 introduced the `[project.optional-dependencies]` table for the first time. PR-019 and PR-020 add new entries without restructuring.

## Solver layer (per PR-020)

The Solver layer is a parallel top-level surface alongside the Trainer layer. Per `docs/0.1/DESIGN-log.md` Q1 + PR-020 Q-Shape (partial mirror), <!-- rewrite-doc-refs:skip-line --> the solving layer reuses the workbench's config/registry/factory/study/provenance substrate but bypasses `cfg.data` / `cfg.cv` / fittable-model registry.

**Subpackage layout** mirrors the Trainer layer:

```
src/rux_ml/solving/
├── __init__.py        # public API + SOLVER_FAMILIES registry
├── base.py            # SolvingBase (family-agnostic config: problem_module)
├── factory.py         # top-level make_solver dispatcher
├── protocol.py        # Solver typing.Protocol
├── result.py          # SolverResult dataclass
└── <family>/
    ├── __init__.py    # public API for this family
    ├── config.py      # <Family>Solving Pydantic variant
    └── factory.py     # make_<family>_solver
```

**`rux-ml solve` verb convention.** Single command (analog of `rux-ml train`), NOT a typer-group. With solver-internal HPO deferred to a follow-up PR, there's no `start/resume/status` to need. User invokes `rux-ml solve --problem <problem-name>` (or `--config <path>`) for a one-shot solve. The CLI:

1. Loads `cfg.solving` from the TOML stack.
2. Imports `cfg.solving.problem_module` (a Python module path) and calls its `build_problem() → <family-specific problem>`.
3. Dispatches via `make_solver(cfg.solving)`.
4. Calls `solver.solve(problem)` and records a 1-trial Optuna trial with `TrialAttrs` (provenance + solver-runtime fields).

**Problem-module contract**: the user defines `def build_problem() -> cvxpy.Problem` (for the CVXPY family) in any importable Python module. The CLI imports fresh on each invocation. Keep problem modules inside the workbench's git so `git_sha` captures their content (the `solving_cfg_hash` field covers only the module's import path, not its source — known limitation).

**Solver trials vs Trainer trials**: same `TrialAttrs` schema with Optional fields. Solver trials leave `data_hash` / `data_bytes_hash` / `data_logical_hash` as `"none:solver-trial"` placeholders (no Parquet input) and fill `solving_cfg_hash` + `solver_status` + `objective_value` + `solver_iter_count` + `solve_time_s`. Trainer trials work unchanged — they leave the solver fields None.

**Solver-internal HPO is deferred** to a follow-up Tier-2 PR. v0.1's `rux-ml solve` is one-shot only; users sweep over solver hyperparameters by running multiple invocations with different overrides.

## Per-family threading model (per PR-019 Q-Parallel)

The workbench's `pin_threads()` (PR-011) exports `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `POLARS_MAX_THREADS` env vars to the trial subprocess. **Not every family honors these.**

| Family | Threading library | Honors `OMP_NUM_THREADS`? | Workbench transport |
|---|---|---|---|
| XGBoost | OpenMP | Yes | env var (no factory action) |
| LightGBM | OpenMP | Yes | env var (no factory action) |
| **CatBoost** | **Intel TBB** | **No** | factory reads env, passes `thread_count=` explicitly |

When adding a family whose threading library is NOT OpenMP, the factory must read `OMP_NUM_THREADS` from the env and translate explicitly. For CatBoost, the helper is `_resolve_thread_count()` in `src/rux_ml/training/catboost/factory.py`.

**CPU bit-exact determinism (PR-013 contract)** also varies by family:

- XGBoost: `random_state` + `OMP_NUM_THREADS=1` + `tree_method="hist"` suffices.
- LightGBM: `random_state` + `deterministic=True` + `OMP_NUM_THREADS=1`.
- CatBoost: `random_seed` + `thread_count=1` + `bootstrap_type='No'` + `rsm=1` + `random_strength=0` + `has_time=True` + `boosting_type='Plain'`. **A dedicated determinism test for CatBoost is deferred to a follow-up Tier-2 PR** (PR-019 scope reduction — recipe is exotic, deserves its own focused review).

## Per-family fit-time callbacks / kwarg translation (per PR-018 Q-Wrap)

Each family's factory translates `TrainingBase` and per-variant config fields to the family's upstream kwarg names. The Trainer Protocol surface (`fit(X, y, **kwargs) -> Trainer`; `predict(X) -> ArrayLike`) is family-agnostic; callers (cli/train.py, tuning/objective.py, registry/promote.py) call `make_trainer(cfg.training, seed=...).fit(x, y, ...)` without knowing the family.

Family-specific translation patterns observed so far:

- **Constructor-kwarg families** (XGBoost): `early_stopping_rounds` is a constructor kwarg; the factory bakes it into the estimator at construction time. The caller passes `eval_set=[...]` to `fit()` and the library uses the stored `early_stopping_rounds`.
- **Fit-time-callback families** (LightGBM): `early_stopping_rounds` is NOT a constructor kwarg in v4.5+; it lives in `callbacks=[lightgbm.early_stopping(N)]` passed to `fit()`. The factory returns a thin shim whose `fit()` injects the callback when `eval_set` is present (Pattern A per Phase-4 PR-018 sub-decision). Callers remain unchanged.

When a new family is added (PR-019, PR-020, ...), the family's factory takes responsibility for adapting upstream kwargs to the workbench's surface. If the family's `__init__` rejects unknown kwargs (CatBoost — Q-MK in PR-017 design session), the family's variant config OMITS `model_kwargs` and only exposes typed fields.

Metric translation: if the family doesn't accept the workbench's metric-registry name (`auc`, `logloss`, `rmse`, `mae`), the factory maintains a `_METRIC_TRANSLATE` map (LightGBM: `logloss` → `binary_logloss`).

## Multi-family Trainer extensibility (per PR-017)

Per `docs/0.1/DESIGN-log.md` Q1–Q5 + Q6: <!-- rewrite-doc-refs:skip-line -->

**Subpackage per family.** Every Trainer family lives at `src/rux_ml/training/<family>/`:

```
src/rux_ml/training/
├── __init__.py        # public API + TRAINER_FAMILIES registry
├── base.py            # TrainingBase (family-agnostic fields)
├── factory.py         # top-level make_trainer dispatcher
├── metrics.py         # family-agnostic metric registry
├── protocol.py        # Trainer typing.Protocol
└── <family>/
    ├── __init__.py    # public API for this family's subpackage
    ├── config.py      # <Family>Training Pydantic variant of TrainingConfig
    ├── factory.py     # make_<family>_trainer
    └── (family-specific extras: ingest.py, etc.)
```

**B-explicit registry.** Families register themselves in `src/rux_ml/training/__init__.py`'s `TRAINER_FAMILIES: dict[str, Callable]` — one entry per family, hand-maintained, no decorator-driven registration. Pattern anchored on HF transformers' `MODEL_MAPPING_NAMES`. The conformance test (`tests/training/test_registry_conformance.py`) parametrizes over this dict; any registered family that fails the end-to-end fit-predict smoke fails CI.

**Discriminated-union config.** `TrainingConfig` (`src/rux_ml/config/training.py`) is `Annotated[XGBoostTraining | <Future>Training, Field(discriminator="kind")]`. Each variant inherits from `TrainingBase` and declares `kind: Literal["<family>"]` as the discriminator. The `kind` field is NOT on `TrainingBase` (it would force every variant to break `reportIncompatibleVariableOverride`). The dispatcher takes `cfg: TrainingConfig` so basedpyright narrows on each variant.

**TOML configs MUST declare `kind` explicitly** under `[training]`. Pydantic's discriminator dispatch runs BEFORE field defaults are applied, so even though variants have `kind = "xgboost"` defaults, the TOML loader requires the key to be present (same pattern as `[cv]`). Example:

```toml
[training]
kind = "xgboost"
device = "cpu"
metric = "auc"
```

**Per-family extras naming.** Optional dependencies follow the per-family pattern — `[xgboost]`, `[lightgbm]`, `[catboost]` — named after the family, not the backend package. (XGBoost stays hard-required per Q-Dep research; PR-018+ add the optional siblings.) Codified by PR-018 in the same commit that adds the first sibling.

**Family removal**: deprecate with `FutureWarning` from the family's factory in release `0.y`, remove in `0.(y+1)`. MINOR bump per `docs/VERSIONING.md §1`; bundle-loadability break is documented in `CHANGELOG.md`, not elevated to MAJOR.

**Conformance test invariant.** `tests/training/test_registry_conformance.py` is the stale-path tripwire. Adding a family without a working subpackage fails the test; removing a subpackage without removing the registry entry fails the test. Each new family adds (a) its registry entry, (b) its minimal-cfg entry in the test's `_MINIMAL_CFG` map (with a fail-loudly assertion that every registered family is in the map). No skipping, no XFAILing.

**Native escape hatches** (per D5): each family may expose a `use_native: bool` flag on its variant to bypass the sklearn-wrapper API for the ~5 % of cases where the wrapper is insufficient (XGBoost: `xgb.train()`; LightGBM: `lgb.train()`; CatBoost: `cb.train()`). Not used by default; flagged on the per-family variant when it lands.

## Where new conventions go

When a convention emerges that isn't documented here, add it during the same PR that establishes it. Conventions added retroactively go stale fast.
