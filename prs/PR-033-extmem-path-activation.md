# PR-033: ExtMem out-of-core ingest path activation

**Landed-in:** (Unreleased — will be bundled into v0.3.0 by PR-036)

## Before Implementation (NON-NEGOTIABLE)

This PR was implemented after `PROCEDURE-pr-research.md` completed all 5 phases on 2026-05-21. Findings appended to `## Research findings` below.

**Tier-2 PR** — research-pending at scaffold time; the architectural sub-decision D2 (ExtMem activation pattern) was resolved via this PR's own Phase 1 state assessment (per the option-1 plan, 2026-05-20).

This PR triggers a MINOR bump (v0.3.0) per `docs/VERSIONING.md §1` — new behavior + new public Trainer-Protocol-compatible adapter.

## Research findings

### Phase 1 — State assessment + prior-art audit (2026-05-21)

**Drift watch**: ExtMem APIs in `xgboost` lock file confirmed unchanged. `ExtMemQuantileDMatrix(cache_host_ratio=...)` still present; `ParquetDataIter` (rux-ml) unchanged since PR-006.

**Prior-art audit**: `git log -p src/rux_ml/training/xgboost/factory.py src/rux_ml/data/data_iter.py src/rux_ml/training/xgboost/ingest.py` surfaced:
- PR-006: `ParquetDataIter` + `select_ingest` shape; `_xgb_kwargs` parameter assembly convention (sklearn-wrapper).
- PR-013: `seed` plumbing → `random_state` on sklearn wrapper.
- PR-017: factory moved into the family subpackage; back-compat re-exports from `rux_ml.training`.
- PR-032: `_BoosterTrainerShim` precedent in `registry/scorer.py` — pure-score adapter over raw `Booster` exposing `predict`/`predict_proba` via `xgb.DMatrix(x, enable_categorical=True) + booster.predict(...)`. Pattern is the strongest in-repo anchor for PR-033's adapter shape.

**Scope reduction surfaced at Phase 1**:
- `registry/promote.py` — re-fit operates on the train+val substrate (~85% of source data); fits in VRAM at workbench scale. **Native path here adds risk for no benefit at v0.3**. Deferred.
- `tuning/objective.py` — per-fold ExtMem rebuild cost is prohibitive (multi-minute per fold × `cv.n_splits`). Existing `_check_extmem_compat` guard preserves the "HPO + ExtMem = `NotImplementedError`" semantic for free. **Leave it untouched**.
- Auto-on-ExtMem-trigger — XGBoost 3.2 tutorial warns ExtMem is slower than `QuantileDMatrix` when data fits in RAM; opt-in via `use_native=True` only. Auto-trigger deferred to follow-up once a real workload exceeds 18 GB.

**Tier inheritance check**: PR-033 was already Tier-2; scope-reduction *removes* sub-decisions rather than adding them, so no further reclassification needed.

### Phase 2 — Reuse-over-reinvent (2026-05-21)

No mature library covers the 4-element combination directly. Each ingredient is reusable:
- `xgb.train` (XGBoost 3.2 native API)
- `xgb.ExtMemQuantileDMatrix` (XGBoost 3.2 out-of-core path)
- `xgb.QuantileDMatrix` (XGBoost 3.2 in-VRAM path)
- `ParquetDataIter` (PR-006 in-repo; thin wrapper on `xgb.DataIter`)
- `_BoosterTrainerShim` (PR-032 in-repo precedent for sklearn-Trainer-Protocol over raw Booster)
- `single_source_iter` (PR-033 new helper; Polars `iter_slices` is stock — no third-party library covers single-source chunking for XGBoost out-of-core)

### Phase 3 — Web research + Group D MCP-Verification (2026-05-21)

**Q1 — Native `xgb.train` adapter design.** XGBoost 3.2 [`python_api.html`](https://xgboost.readthedocs.io/en/release_3.2.0/python/python_api.html) parameter mapping table located. Sklearn → native renames:
- `random_state` → `params["seed"]`
- `n_estimators` → `xgb.train(num_boost_round=...)` kwarg
- `early_stopping_rounds` → `xgb.train(early_stopping_rounds=...)` kwarg (only honored when `evals` non-empty)
- `enable_categorical`, `max_bin`, `cache_host_ratio` → DMatrix constructor kwargs (not booster params)
- `objective` → `params["objective"]` (native does not auto-detect from data)

**Q2 — `ParquetDataIter` chunking strategy.** XGBoost 3.2 [`external_memory.py` demo](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/external_memory.py) cited — verbatim split-on-write pattern (Polars `iter_slices` → N temp parquets → wrap in iterator). NVIDIA recommendation: 5-10 GB per batch on a 36 GB host. PR-033 uses `_DEFAULT_EXTMEM_BATCH_COUNT = 4` (small enough that even tiny test frames produce >1 batch; large enough for workbench-scale frames at the 18 GB threshold).

**Q3 — Default `cache_host_ratio`.** XGBoost 3.2 [`external_memory.html`](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html) explicitly recommends `null` (auto-estimate) for first runs. `MemoryConfig.cache_host_ratio: float | None = None` already defaults to `None`; no change needed.

**Q4 — `use_native` semantics.** Opt-in only (`cfg.training.use_native=True` required). Auto-trigger DEFERRED per XGBoost 3.2 tutorial warning that ExtMem is slower than `QuantileDMatrix` when data fits in RAM.

**Q5 — Test-mode flag.** Reduced `gpu_in_memory_x_gb_max=1e-9` in test `DataConfig` forces the ExtMem branch. No new env-var surface.

**Q6 — `compute_score` compatibility.** Adapter exposes `predict_proba` via the same `xgb.DMatrix(x, enable_categorical=True) + booster.predict(...)` pattern as PR-032's `_BoosterTrainerShim` — metric registry stays homogeneous.

**Q7 — Per-fold ExtMem rebuild cost in HPO.** Prohibitive (multi-minute per fold). `tuning/objective.py` stays on sklearn-wrapper path; `_check_extmem_compat` semantic preserved.

**Q8 — Backward compatibility.** No breakage — `use_native=False` is the default; existing studies route through the unchanged sklearn-wrapper path.

**Q-Group-D MCP-Verification Round**:
- **Probe 1 (Schema-Integrity)** — every named identifier exists at the pinned `xgboost` version (`xgb.train`, `xgb.ExtMemQuantileDMatrix`, `xgb.QuantileDMatrix`, `cache_host_ratio` kwarg, `enable_categorical` kwarg, `early_stopping_rounds` kwarg, `num_boost_round` kwarg, `seed` param) AND in `dev` HEAD (`ParquetDataIter`, `select_ingest`, `cfg.training.use_native`, `cfg.memory.cache_host_ratio`). VERIFIED.
- **Probe 2 (Synthesis-Verification)** — searched for a single cited working example combining all 4 elements (`xgb.train` + `ExtMemQuantileDMatrix` + `cache_host_ratio` + sklearn-Trainer-Protocol wrapper). **NO cite found.** Each ingredient is independently sourced; the synthesis is the workbench's own composition. **Status downgraded from `convention` to `best-guess-given-constraints`**.
- **Probe 3 (Binding-at-creation)** — N/A. `XGBoostNativeAdapter` is constructor-instantiated, not registered with any external system.

### Phase 4 — Synthesis (Outcome: Amend → Apply, 2026-05-21)

4 amendments locked at user approval:
1. **Status label**: `convention` → `best-guess-given-constraints` for the 4-element synthesis combination (per Probe 2 result).
2. **Mandatory numeric-agreement test**: `test_native_adapter_predict_proba_matches_sklearn_wrapper_within_tol` (`atol=1e-5`) bounds the BGGC risk by asserting bit-exact agreement with the sklearn-wrapper path on shared params + seed + data. Without this test the BGGC label would be unbounded.
3. **Parallel adapter classes (NOT inheritance refactor)**: `_BoosterTrainerShim` stays in `registry/scorer.py` (score-only after bundle load); `XGBoostNativeAdapter` is new in `training/xgboost/native_adapter.py` (real fit + predict + predict_proba). Different lifecycle, different ownership.
4. **CONVENTIONS.md honest activation warning** citing XGBoost 3.2 tutorial: ExtMem is experimental; `QuantileDMatrix` is faster when data fits in RAM; `use_native=True` opt-in only when X > VRAM+RAM + NVMe-on-PCIe-4 storage.

### Phase 5 — Gate check (2026-05-21)

All verification criteria from the scope section below cleared at Phase 5; user approved implementation.

## Scope (final, post Phase 1 reduction)

Activate the ExtMem out-of-core ingest path so the workbench can train on datasets larger than VRAM. Completes phantom #2 + #2b from the 2026-05-20 audit.

### Item 1 — `XGBoostNativeAdapter` (NEW `src/rux_ml/training/xgboost/native_adapter.py`)

Trainer-Protocol-compatible wrapper around `xgb.train`:

- `fit(X, y, *, eval_set=None, verbose=False)` — branches on `select_ingest(estimate_x_bytes(X), cfg.data)`:
  - `QuantileDMatrix`: build directly from `X, y`.
  - `ExtMemQuantileDMatrix`: chunk via `single_source_iter` (Item 3) → wrap in `ParquetDataIter` → build with `cache_host_ratio` threaded through (GPU-only; CPU build rejects the kwarg).
  - `xgb.train(params, dtrain, num_boost_round, evals=..., early_stopping_rounds=...)`.
- `predict(x)` — `xgb.DMatrix(x, enable_categorical=True) → booster.predict(...)`. Accepts pandas/polars/ndarray uniformly (mirrors PR-032 shim).
- `predict_proba(x)` — 2D `[1-p, p]` shape for binary classification.
- `best_iteration` property — read from booster after early-stopping fit.

### Item 2 — `cli/train.py` branches on `cfg.training.use_native`

`_fit_and_score`: `cfg.training.use_native=True` routes through `XGBoostNativeAdapter`; `False` (default) stays on `make_trainer`. Honest stderr log: `ingest path: <DMatrixCls> (native API|sklearn wrapper)`.

### Item 3 — `single_source_iter` helper (`src/rux_ml/data/data_iter.py`)

```python
def single_source_iter(
    df: pl.DataFrame, target_column: str, *,
    batch_count: int, tmp_dir: Path, cache_prefix: str,
) -> ParquetDataIter: ...
```

Polars `iter_slices(n_rows=ceil(N/batch_count))` → N temp parquets → `ParquetDataIter`. Caller owns `tmp_dir` cleanup (adapter creates an owned `tempfile.TemporaryDirectory` per fit call).

### Item 4 — `cache_host_ratio` threaded through (GPU-only)

Adapter passes `cfg.memory.cache_host_ratio` to `ExtMemQuantileDMatrix(...)` when `cfg.training.device == "cuda"`. CPU build rejects the kwarg with `Check failed: detail::HostRatioIsAuto(config.cache_host_ratio)`.

### Item 5 — `cfg.training.use_native` actually read

For the first time in production — by `cli/train.py::_fit_and_score`.

### Item 6 — Tests

8 in `tests/training/test_native_adapter.py`:
1. `test_native_adapter_fit_predict_smoke`
2. **`test_native_adapter_predict_proba_matches_sklearn_wrapper_within_tol`** (MANDATORY BGGC mitigation)
3. `test_native_adapter_predict_proba_returns_2d_for_binary`
4. `test_native_adapter_uses_quantile_dmatrix_for_small_x` (monkey-patched `select_ingest`)
5. `test_native_adapter_threads_cache_host_ratio_when_device_cuda` (monkey-patched DMatrix; verifies GPU-gated threading)
6. `test_native_adapter_uses_extmem_for_large_x_with_force_flag` (real ExtMem on CPU via `gpu_in_memory_x_gb_max=1e-9`)
7. `test_single_source_iter_chunks_into_n_files`
8. `test_single_source_iter_rejects_invalid_inputs`

1 in `tests/cli/test_train_subcommand.py`:
- `test_train_use_native_smoke` — CLI `--set training.use_native=true` end-to-end + asserts "native API" in stderr.

### Out of scope (deferred per Phase 1)

- `registry/promote.py` native path (sklearn-wrapper sufficient on train+val substrate that fits in VRAM)
- HPO objective native path (per-fold ExtMem rebuild prohibitive; `_check_extmem_compat` guard preserves existing "HPO + ExtMem = `NotImplementedError`" semantic)
- Auto-on-ExtMem-trigger (opt-in via `use_native=True` only)

### Docs updates (same commit)

- `docs/ARCHITECTURE.md` — D3 entry updated from "documented aspiration" → "opt-in active via `use_native=True`"; Memory & Parallelism `cache_host_ratio` row gains honest GPU-only semantic.
- `docs/CONVENTIONS.md` — new "XGBoost native API + ExtMem path (per PR-033)" subsection: honest activation warning citing XGBoost 3.2 tutorial; parameter-mapping table; BGGC label + mitigation test pointer.
- `docs/0.3/ROADMAP.md` PR-033 row `[ ] → [x]`.
- `docs/0.3/RESEARCH-BACKLOG.md` PR-033 row → `state-assessed 2026-05-21` + `fully-researched 2026-05-21` + `implementation-cleared 2026-05-21`.
- `docs/0.3/DESIGN-log.md` — new "Session: 2026-05-21 — PR-033 ExtMem path activation" entry.
- `CHANGELOG.md [Unreleased] ### Added` — loud entry.

## Dependencies

- **PR-030** (sprint scaffolding) — merged.
- **PR-032** (`_BoosterTrainerShim` precedent in `registry/scorer.py`) — merged.

## Architecture section implemented

`docs/ARCHITECTURE.md` "Decision Rules" → "XGBoost ingest path" (D3); Memory & Parallelism table `cache_host_ratio` row.

## Verification criteria

- [x] `XGBoostNativeAdapter` exists in `src/rux_ml/training/xgboost/native_adapter.py` and conforms to the `Trainer` Protocol surface (`fit` + `predict` + `predict_proba` + `best_iteration`).
- [x] `cli/train.py` branches on `cfg.training.use_native` and routes through `XGBoostNativeAdapter` when `True`.
- [x] `cache_host_ratio` is read by `XGBoostNativeAdapter` and threaded into `ExtMemQuantileDMatrix(...)` when on GPU.
- [x] `use_native` is read and respected.
- [x] Test-mode forcing the path on small data passes end-to-end via reduced `gpu_in_memory_x_gb_max`.
- [x] `compute_score` returns a sensible metric for an `XGBoostNativeAdapter` instance.
- [x] **MANDATORY BGGC mitigation test passes**: adapter `predict_proba` matches sklearn-wrapper within `atol=1e-5` on shared params + seed + data.
- [x] `registry/promote.py` UNCHANGED.
- [x] `tuning/objective.py` UNCHANGED, including `_check_extmem_compat`.
- [x] `docs/ARCHITECTURE.md` D3 entry updated from aspiration → opt-in active.
- [x] `docs/0.3/ROADMAP.md` PR-033 row flipped `[ ]` → `[x]`.
- [x] `CHANGELOG.md [Unreleased] ### Added` loud entry.

## Research backing

Tier-2 — D2 resolved via this PR's own Phase 1 state assessment (per the option-1 plan, 2026-05-20). Phase 3 web research anchored on:

- XGBoost 3.2 [`external_memory.html`](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html) — chunking pattern (Option B, split-on-write); honest "ExtMem slower than QuantileDMatrix when data fits in RAM" warning.
- XGBoost 3.2 [`external_memory.py` demo](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/external_memory.py) — verbatim `ParquetDataIter` shape.
- XGBoost 3.2 [`python_api.html`](https://xgboost.readthedocs.io/en/release_3.2.0/python/python_api.html) — parameter mapping table.
- PR-032 `_BoosterTrainerShim` (in-repo precedent for sklearn-Trainer-Protocol over raw Booster).

**Status**: `best-guess-given-constraints` for the 4-element combination (`xgb.train` + `ExtMemQuantileDMatrix` + `cache_host_ratio` + sklearn-Trainer-Protocol wrapper) — Group D Probe 2 returned NO cite. Mitigation: numeric-agreement test.

## Notes

- **Largest scope in v0.3 sprint** — Phase 1 scope-reduction kept it tractable.
- **Triggers MINOR bump** (v0.3.0) per `docs/VERSIONING.md §1`.
- **No real-world workload triggers this today.** Crypto-h3 is 1.4 GB, well under the 18 GB threshold. The PR's primary value is making the documented behavior real before a workload needs it. Test exercises happen via forced-flag.
- Per memory `feedback_roadmap_flip_in_pr`: PR-033's implementation commit flips `docs/0.3/ROADMAP.md` PR-033 row `[ ]` → `[x]` in the same commit.
- This PR also resolves the "borderline phantom 2b" (`select_ingest` return drove telemetry only, not behavior) by making the return value actually drive construction in the opt-in path.
