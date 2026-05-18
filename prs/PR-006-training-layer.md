# PR-006: Training layer

**Landed-in:** v0.0.1

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16)

**Current state of the codebase**:

- `dev` at `57af8ef` (PR-005 merged, plus the PR-015 planning commit `3b5c1a1` inserting Tier-2 CV strategy between PR-006 and PR-007). Working tree clean.
- All v0 PRs through PR-005 are merged. No `src/rux_ml/training/` subpackage yet.
- Installed runtime deps (per `pyproject.toml`): `typer>=0.12`, `pydantic>=2.6`, `pydantic-settings>=2.6`, `xxhash>=3.7`, `polars>=1.40`, `xgboost>=3.1` (installed: **3.2.0**), `numpy>=2.0`, `scikit-learn>=1.4`, `category-encoders>=2.6`, `pyarrow>=17`. **`optuna` is NOT installed.**
- `TrainingConfig` (PR-002, `src/rux_ml/config/training.py`) already pins every field PR-006 consumes: `kind`, `device`, `metric`, `enable_categorical`, `tree_method`, `use_native`, `learning_rate`, `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`, `early_stopping_rounds`, `model_kwargs`.
- `DataConfig.gpu_in_memory_x_gb_max = 18.0` and `DataConfig.split_ratios` already in place. `data.splits.train_val_test_split(df, *, ratios, seed)` and `data.versioning.compute_data_hash(path)` and `data.data_iter.ParquetDataIter(files, target_column, *, cache_prefix)` exist and are tested.
- `features.make_features(cfg, cardinalities=...)` requires cardinalities pre-computed via `cardinalities_from(df, columns)`; pipeline shape is Polars → pandas waist → Polars (PR-005 pattern PR-006 mirrors at the trainer boundary).
- `cli/train.py` is a single no-op leaf via `not_implemented(...)`; the root callback already constructs `GlobalOptions(config, problem, study, overrides, ...)` and the `data` CLI is the live "real-body" pattern (loads `RuxMLConfig.from_layers(...)`, calls layer functions, prints results).
- `_internal/hashing.py` exposes `canonical_json`, `sha256_canonical`, `xxh3_64_bytes`, `xxh3_64_file`. `config.root` exposes `cfg_hash(cfg)` (root) and `layer_cfg_hash(cfg, layer)` for per-layer hashes. **No `git_sha` helper exists yet.**
- `configs/base.toml` has a `[training]` section header with only comments — PR-006 doesn't need to edit it. No `configs/problems/` or `configs/studies/` exist (`from_layers` only loads them when `--problem` / `--study` is passed).
- Tests follow the PR-005/PR-004 split: `tests/<layer>/test_*.py` for unit + integration, `tests/cli/test_<verb>_subcommands.py` for `typer.testing.CliRunner` end-to-end tests with `tmp_path` configs.

**Assumptions at PR draft time** (drafted 2026-05-15 from D5 + D3 + D7):

- XGBoost sklearn wrapper exposes `device`, `enable_categorical`, `tree_method`, `eval_set`, `callbacks`, `best_iteration_` (per the PR file's `State assessment must verify…` clause).
- `XGBClassifier(device=cfg.device, enable_categorical=True, tree_method="hist", **model_kwargs)` is the default path; `xgb.train()` is the ~5 % escape hatch when `cfg.use_native=True`.
- Ingest path selector picks `QuantileDMatrix` below `cfg.gpu_in_memory_x_gb_max`, `ExtMemQuantileDMatrix` above — with `cache_host_ratio` "from `MemoryConfig`" (PR-006 scope text).
- `rux-ml train` runs through `study.ask()` / `study.tell()` so it lands as a 1-trial entry in `studies/studies.db` (per D7).
- Per-trial `user_attrs` records ≥ {per-layer + root `*_cfg_hash`, `data_hash`, `git_sha`}; `entropy_hex` and the full provenance triple land in PR-013.

**Verification against current installed state** (local-only, no web fetch):

| Item | Status | Where verified |
|---|---|---|
| `XGBClassifier.get_params()` exposes `device`, `enable_categorical`, `tree_method`, `early_stopping_rounds`, `callbacks` | **STILL CURRENT** (XGBoost 3.2.0) | `uv run python -c "from xgboost import XGBClassifier; XGBClassifier().get_params()"` |
| `XGBClassifier.fit(X, y, *, eval_set, verbose, ...)` accepts `eval_set` | **STILL CURRENT** | `inspect.signature(XGBClassifier().fit).parameters` |
| `best_iteration` attribute populated after fit with `early_stopping_rounds` | **STILL CURRENT** | live fit returned `best_iteration=4` |
| `xgb.QuantileDMatrix` and `xgb.ExtMemQuantileDMatrix` importable | **STILL CURRENT** | `hasattr(xgb, "QuantileDMatrix")` and `…ExtMemQuantileDMatrix` both `True` |
| `ParquetDataIter` already plugs into `ExtMemQuantileDMatrix` | **STILL CURRENT** | `tests/data/test_data_iter.py::test_dataiter_constructs_extmem_quantile_dmatrix` passes |
| `optuna` installed | **NOT INSTALLED** | `ModuleNotFoundError: No module named 'optuna'` |

**Stale assumptions**:

1. **XGBoost `callbacks` location** — PR-006's data-flow example implies a `fit(..., callbacks=[...])` kwarg. In XGBoost 3.x the `callbacks` parameter is a **constructor / `set_params` argument**, not a `.fit()` kwarg. **Scope-impact for PR-006: NONE** — the only callback the project uses is `XGBoostPruningCallback`, which PR-006 explicitly defers to PR-007. PR-006 needs no callbacks. Implementation note for PR-007: wire via constructor / `set_params`, not `.fit()`.
2. **`best_iteration_` vs `best_iteration`** — `Trainer` Protocol (per ARCHITECTURE.md line 207) declares `best_iteration_: int | None`. XGBoost 3.2 exposes both `best_iteration` (canonical) and `best_iteration_` (sklearn-style trailing underscore) on the fitted estimator. **Scope-impact: NONE** — Protocol is structural; either name type-checks. PR-006 picks `best_iteration_` (sklearn convention).

**New constraints learned from prior PRs / codebase evolution**:

1. **`MemoryConfig` has no `cache_host_ratio` field today.** ARCHITECTURE.md "Decision Rules: XGBoost ingest path" + this PR's scope text ("`cache_host_ratio` from `MemoryConfig`") assume it exists. Current `MemoryConfig` has only `watchdog_threshold_gb`, `watchdog_sample_hz`, `omp_threads`, `openblas_threads`, `mkl_threads`, `polars_threads`. **PR-006 must add `cache_host_ratio: float | None = None` to `MemoryConfig`** (XGBoost auto-estimates when `None`). Mechanical, additive, zero-default — per `CONSTRAINTS.md` "Zero Hardcoded Parameters" this knob must live in TOML, not in `ingest.py`.
2. **No `git_sha` helper exists.** PR-006 records `git_sha` in `user_attrs`. Lands as a tiny `rux_ml._internal.git.git_sha()` (subprocess `git rev-parse HEAD` with a graceful fallback to `"unknown"`). Additive only; matches the existing `_internal/` pattern.
3. **Optuna not in runtime deps.** PR-006 must add `optuna>=4.0` to `pyproject.toml` `dependencies`.
4. **PR-005's pandas-waist pattern is in place.** `features.make_features` returns Polars from its final stage; XGBoost's sklearn wrapper accepts numpy/pandas. PR-006's trainer boundary mirrors the pattern: Polars → pandas (preserving `Categorical` dtype when `enable_categorical=True`) → `XGBClassifier.fit`.
5. **PR-015 (Tier-2 CV strategy) is now inserted between PR-006 and PR-007.** PR-006 must use the existing naive `data.splits.train_val_test_split` and explicitly document that PR-015 will replace it. **PR-006 must NOT introduce any CV abstraction.**
6. **`configs/problems/` and `configs/studies/` directories do not exist yet.** `from_layers` already handles their absence gracefully. PR-006's integration test builds a minimal in-`tmp_path` layered-TOML stack (mirror `tests/cli/test_data_subcommands.py`'s `workdir` fixture).
7. **`basedpyright` execution-environment relaxation for `tests/`** (added by PR-005) covers sklearn / category_encoders / hypothesis. XGBoost ships type stubs; PR-006 likely doesn't need a further relaxation, but watch for noise during implementation.

**Synthesis Outcome: CONFIRM** — zero substantive drift in the PR-006 core path (XGBoost sklearn API + `QuantileDMatrix`/`ExtMemQuantileDMatrix` + Optuna `ask`/`tell` + provenance-triple subset). The "new constraints" above are all small, additive, and implied by the spec; none change the architecture.

### Synthesis (2026-05-16)

**Outcome:** Confirm.

**Changes to this PR from research (small, mechanical):**

- Add `cache_host_ratio: float | None = None` to `MemoryConfig` (single field; XGBoost auto-estimates when `None`).
- Add `rux_ml._internal.git.git_sha()` (subprocess `git rev-parse HEAD`, falls back to `"unknown"` on failure).
- Add `optuna>=4.0` to runtime `dependencies` in `pyproject.toml`.
- Implementation note: pass XGBoost callbacks (when PR-007 adds them) via constructor / `set_params`, not `fit(..., callbacks=...)`. **No callbacks needed in PR-006 itself.**
- Implementation note: trainer boundary converts Polars → pandas (preserving `Categorical` dtype) before `XGBClassifier.fit`; mirrors PR-005's waist.
- Implementation note: PR-006 uses the existing `data.splits.train_val_test_split` unchanged; PR-015 (Tier-2) will replace it before PR-007 lands.

**Changes to ARCHITECTURE.md:** One-line addition under "Memory & Parallelism Architecture" naming `cache_host_ratio` as a `MemoryConfig` knob (already implied by the existing "Decision Rules: XGBoost ingest path" section; making the config location explicit).

**Changes to CONSTRAINTS.md / CONVENTIONS.md / CLAUDE.md:** None.

**New PRs that must come first:** None. PR-015 (Tier-2) is sequenced *after* PR-006 by design.

**Phase 3 (web research) intentionally skipped:** Per `PROCEDURE-pr-research.md` and the `feedback_phase1_scope` memory, Tier-1 PRs with zero substantive drift run Phases 2-4 light and go straight to Gate Check. Local verification of XGBoost 3.2.0 confirmed every API the spec leans on; the spec's own "State assessment must verify…" checklist passes. Re-verifying via WebSearch would duplicate the design-time D5/D3/D7 research without new questions.

**Research-backed details now locked in this PR:**

- XGBoost 3.2.0 sklearn wrapper (`XGBClassifier` / `XGBRegressor`) with `device=cfg.device`, `enable_categorical=True`, `tree_method="hist"`, `early_stopping_rounds=cfg.early_stopping_rounds`, `**cfg.model_kwargs`
- Native `xgb.train()` escape hatch behind `cfg.use_native=True` for the `QuantileDMatrix(ref=...)` / `ExtMemQuantileDMatrix` paths (per D5)
- `QuantileDMatrix(device="cuda", tree_method="hist")` below `cfg.gpu_in_memory_x_gb_max`; `ExtMemQuantileDMatrix` via existing `data.data_iter.ParquetDataIter` above (per D3)
- 1-trial Optuna study via `study.ask()` + `study.tell()`, SQLite at `runs.storage_url` (per D7)
- Per-trial `user_attrs` ⊇ `{data_cfg_hash, features_cfg_hash, training_cfg_hash, tuning_cfg_hash, root_cfg_hash, data_hash, git_sha}` (full provenance triple completes in PR-013)
- Polars → pandas at the trainer boundary (mirrors PR-005)
- CV strategy untouched; `data.splits.train_val_test_split` consumed as-is (PR-015 deferral)

### Gate Check

- Premise still valid: ✓ (zero substantive drift; mechanical additions only)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

### Implementation notes (2026-05-16)

Two findings surfaced during implementation that warrant documenting:

1. **`Trainer` Protocol narrowed to the universal subset (`fit` + `predict`).** The Phase 1 spec assumed the 4-method shape from ARCHITECTURE.md (`fit / predict / predict_proba / best_iteration_`). Implementation surfaced two structural conflicts:
   - `XGBRegressor` has no `predict_proba` — it's a classifier-only sklearn surface, so a 4-method Protocol would have excluded regressors from `make_trainer`'s return type.
   - XGBoost's type stubs don't declare `best_iteration_` (it's set at runtime, only after a fit with `early_stopping_rounds` configured), so `XGBClassifier` failed the static Protocol check too.

   Resolution: keep the Protocol minimal (the universally-present sklearn methods) and narrow at call sites — `predict_proba` is reached via a typed cast inside `_predict_proba` in `training/metrics.py` (only invoked from AUC / logloss paths), and `best_iteration` is read via `getattr(trainer, "best_iteration", None)` in `cli/train.py`. ARCHITECTURE.md updated in the same commit.

2. **File-level `pyright` pragma on `training/metrics.py`** for `reportUnknownVariableType` + `reportUnknownArgumentType`. `sklearn.metrics` scorers ship parameter stubs typed as `Unknown` across the board (return types are correctly typed). Localizing the relaxation to the one module that imports from `sklearn.metrics` keeps the rest of `src/` strict, matching PR-005's "library code in src/ stays strict; surgical relaxation where the third-party stubs force it" pattern.

3. **`tests/cli/test_subcommands.py` train-stub assertion removed.** `train` is now a real CLI body (covered by `tests/cli/test_train_subcommand.py`), so the "not yet implemented" parametrize row for `train` was deleted — same pattern as the `data` verbs after PR-004.

4. **`MemoryConfig.cache_host_ratio` is `None` by default** so XGBoost auto-estimates; the field is read by `select_ingest`'s caller (when the ExtMem branch executes) rather than by `select_ingest` itself. PR-006 exercises only the in-memory branch end-to-end; the ExtMem execution path lands when a real problem exceeds the threshold (the iterator from PR-004 + this config knob already cover the prerequisites).

5. **TOML has no `null` literal.** The CLI integration test config omits `early_stopping_rounds` rather than setting it to `null`; users who want to disable early stopping rely on the Pydantic field default (`50`) being inert when `n_estimators` is smaller, or override via `--set training.early_stopping_rounds=0` (or env var).

---

## Scope

Implement the training layer per D5 + D3 ingest decision rule. After this PR, `rux-ml train` runs a single baseline training end-to-end through data → features → trainer → score, with every reproducibility hash recorded.

- `src/rux_ml/training/protocol.py` — `Trainer(Protocol)` with `fit / predict / predict_proba / best_iteration_` per D5
- `src/rux_ml/training/factory.py` — `make_trainer(cfg: TrainingConfig) -> Trainer`:
  - `cfg.kind == "xgboost"` → `XGBClassifier` or `XGBRegressor` with `device=cfg.device` (`"cuda"` default), `enable_categorical=True`, `tree_method="hist"`, `**cfg.model_kwargs`
  - native-`xgb.train()` path (`cfg.use_native = True`) for QuantileDMatrix(ref=) and ExtMemQuantileDMatrix cases
- `src/rux_ml/training/metrics.py` — metric registry (`auc`, `logloss`, `rmse`, `mae`); metric chosen via `cfg.metric`
- `src/rux_ml/training/ingest.py` — implements the D3 decision rule:
  - estimate `X_bytes` from the materialized features
  - if ≲ `cfg.gpu_in_memory_x_gb_max` → `QuantileDMatrix(device="cuda", tree_method="hist")`
  - else → `ExtMemQuantileDMatrix` via `data.data_iter.build_iter(cfg)` with `cache_host_ratio` from `MemoryConfig`
- CLI body for `rux-ml train` (`src/rux_ml/cli/train.py`):
  - load `RuxMLConfig` (with `--problem` + `--study` resolved to TOML files)
  - run via `study.ask()` + `study.tell()` so it appears in `studies/studies.db` as a 1-trial study (per D7)
  - record per-trial `user_attrs` (config hashes, `data_hash`, `git_sha` — `entropy_hex` lands in PR-013)
  - print final score + storage location
- Tests:
  - Unit: `make_trainer(cfg)` returns the right concrete estimator with the right kwargs
  - Integration: `rux-ml train` on a tiny synthetic dataset succeeds, score within sane range
  - GPU-gated (`@pytest.mark.gpu`): same integration test against `device="cuda"`
  - Decision-rule unit test: `select_ingest(X_bytes, cfg)` returns `QuantileDMatrix` below threshold, `ExtMemQuantileDMatrix` above

NOT in scope: HPO sweeps (PR-007), subprocess isolation (PR-008), full provenance triple — entropy_hex lands in PR-013.

## Dependencies

PR-005.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Training" component row, "Decision Rules: XGBoost ingest path", "Key Abstractions: `Trainer` Protocol", "Data Flow" (one-off run path).

## Verification criteria

- [ ] `Trainer` Protocol type-checks against `XGBClassifier` (no errors)
- [ ] `make_trainer(cfg)` honors `device`, `enable_categorical`, `tree_method`, `**model_kwargs`
- [ ] Ingest decision rule selects correctly given a fake size estimator
- [ ] `rux-ml train --problem <p> --study <s>` runs end-to-end on a synthetic Parquet, records a trial, prints the score
- [ ] Recorded `user_attrs` includes (at minimum) the `*_cfg_hash` set + `data_hash` + `git_sha`
- [ ] GPU-gated integration test passes when CUDA is present and is correctly skipped on CPU-only

## Research backing

Tier 1:

- D5: [XGBoost sklearn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html), [XGBoost callbacks](https://xgboost.readthedocs.io/en/stable/python/callbacks.html)
- D3: [XGBoost ExtMem tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html), [QuantileDMatrix API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- D7: [Optuna `ask`/`tell`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html)

State assessment must verify the XGBoost sklearn wrapper still exposes `device`, `enable_categorical`, `eval_set`, `callbacks`, `best_iteration_`.

## Notes

- The "5 % escape hatch" to `xgb.train()` is configurable per D5 — keep it as a clean code path, not a hidden branch deep in the factory.
- Do not start writing the registry yet (PR-010); just record the trial. Promotion is explicit and lives in its own PR.
