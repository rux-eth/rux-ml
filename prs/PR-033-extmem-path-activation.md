# PR-033: ExtMem out-of-core ingest path activation

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** — the architectural sub-decision D2 (ExtMem activation pattern) is **locked by the v0.3 design session** before this PR's Phase 1 begins. Multiple non-preference sub-decisions: native API adapter design; ParquetDataIter chunking; default `cache_host_ratio`; `compute_score` compatibility on raw Booster; per-fold ExtMem cost in HPO.

This PR triggers a MINOR bump (v0.3.0) per `docs/VERSIONING.md §1` — new behavior + new public Trainer-Protocol-compatible adapter.

**This PR has the largest scope of the v0.3 sprint.** Phase 1 may surface drift severe enough to split the PR (e.g., the adapter + native xgb.train wire-up as one PR; the ParquetDataIter chunking + cli/train integration as a second).

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Open research questions** (must be resolved before implementation):

1. **Native `xgb.train` adapter design** — wrap `xgb.Booster` in a `Trainer`-Protocol-compatible shim with `fit/predict` methods. Survey: how do other workbenches that use both sklearn and native XGBoost APIs unify them? `sklearn.compose.TransformedTargetRegressor` pattern? Custom shim?
2. **`ParquetDataIter` chunking strategy** — current `ParquetDataIter` takes a list of batch file paths. For a single-file source (crypto-h3 is one parquet), how do we split into batches? Options: (a) Polars `scan_parquet().collect(streaming=True)` row-group-by-row-group; (b) split-on-write into N intermediate parquets; (c) something else. Phase 3 web research on XGBoost out-of-core practitioners.
3. **Default `cache_host_ratio`** — `null` (XGBoost auto-estimate) vs explicit fraction. XGBoost docs recommend auto for first runs. Lean: keep `null` default; document the override pattern.
4. **`use_native` semantics** — `cfg.training.use_native` as opt-in OR auto-flip when `select_ingest` returns `ExtMemQuantileDMatrix`? Lean: auto-flip on ExtMem dispatch; `use_native=True` as an override forcing native even when sklearn-wrapper would work (for advanced users).
5. **Test-mode flag** — how do we exercise the path on small data? Options: (a) `RUXML_FORCE_EXTMEM=1` env var; (b) drastically reduce `gpu_in_memory_x_gb_max` in test config; (c) explicit `--ingest=extmem` CLI flag. Phase 4 to lock.
6. **`compute_score` compatibility** — metric registry assumes sklearn-wrapper (`trainer.predict_proba` etc.). For raw Booster, need either: (a) add Booster-aware paths to `_rmse` / `_auc` / `_logloss` / `_mae`; (b) the new adapter exposes `predict_proba` etc. via `Booster.predict` wrapping. Lean: (b) — adapter unifies the surface.
7. **Per-fold ExtMem rebuild cost** — in HPO, each fold reconstructs the DMatrix. ExtMem rebuild cost may be prohibitive (multi-minute per fold). Lean: ExtMem path active ONLY for `cli/train.py` (one-fit baseline) + `registry/promote.py` (one-fit re-fit). HPO objective stays on in-memory regardless (compatibility guard at `objective.py:153` already enforces this — but the guard would need to actually be reachable for the first time).
8. **Backward compatibility** — does this break any existing study or test? The existing `select_ingest` decision exists but is unused; wiring it up could expose latent bugs in the `_check_extmem_compat` guard.

## Scope

Activate the ExtMem out-of-core ingest path so the workbench can train on datasets larger than VRAM. Completes phantom #2 from the 2026-05-20 audit.

### Item 1 — Trainer-Protocol-compatible native-API adapter

New `src/rux_ml/training/xgboost/native_adapter.py` exposing:

```python
class XGBoostNativeAdapter:
    """Wraps a native xgb.Booster with a sklearn-compatible fit/predict surface."""
    def fit(self, dmatrix: xgb.DMatrix, ..., eval_set: ...) -> "XGBoostNativeAdapter": ...
    def predict(self, x: Any) -> NDArray: ...
    def predict_proba(self, x: Any) -> NDArray: ...
    @property
    def best_iteration(self) -> int: ...
```

Implements the `Trainer` Protocol via duck-typing. Wraps `xgb.train(params, dmatrix, ...)` internally.

### Item 2 — `select_ingest` decision honored in production

Modify `cli/train.py:_fit_and_score` (and `registry/promote.py:_refit`) to:

1. Call `select_ingest(estimate_x_bytes(x_train_t), cfg.data)` to choose the class.
2. If `QuantileDMatrix`: existing sklearn-wrapper path (no change).
3. If `ExtMemQuantileDMatrix` OR `cfg.training.use_native == True`: native path via `XGBoostNativeAdapter` + `ParquetDataIter`.

Branch point lives in a new `make_trainer_for_ingest(cfg, ingest_cls, ...)` factory or in the existing `make_trainer` extended with an `ingest_cls` argument.

### Item 3 — `ParquetDataIter` chunking from single-file source

Extend `src/rux_ml/data/data_iter.py` or add a sibling module to support construction from a single `pl.DataFrame` (after `make_splits` carving). Chunking strategy locked in Phase 4 from research Q2.

### Item 4 — `cache_host_ratio` threaded through

`MemoryConfig.cache_host_ratio` (currently dead) becomes a real consumer in `XGBoostNativeAdapter.__init__` → passed to `ExtMemQuantileDMatrix(..., cache_host_ratio=cfg.memory.cache_host_ratio)`.

### Item 5 — `use_native` flag actually read

`XGBoostTraining.use_native` (currently dead) read by `make_trainer_for_ingest` to force the native path even when sklearn-wrapper would work.

### Item 6 — `_check_extmem_compat` becomes reachable

`tuning/objective.py:153`'s ExtMem compatibility guard fires for the first time when a real workload + non-extmem-compatible splitter combine. Either:
- Fold-rebuild cost makes ExtMem-in-HPO impractical → keep the guard but document the limitation
- OR: HPO objective stays on in-memory regardless (per Q7 lean); guard tightens to "ExtMem path active only for baseline + promote"

### Item 7 — Test strategy for the path

- Synthetic-large dataset test (forced via test-mode flag from Q5).
- Property test: `select_ingest(...) == ExtMemQuantileDMatrix` triggers `XGBoostNativeAdapter` construction, not `XGBClassifier` / `XGBRegressor`.
- End-to-end test: synthetic dataset > forced threshold trains via native path, returns predictions matching the in-memory path within tolerance.
- Marker: `@pytest.mark.slow` and/or `@pytest.mark.gpu` (4090 only).

### Item 8 — Docs update

- `docs/ARCHITECTURE.md` "Decision Rules" → "XGBoost ingest path (per D3)": update from documented-aspiration to documented-active.
- `docs/ARCHITECTURE.md` Memory & Parallelism table: `cache_host_ratio` field now actually consumed.
- `docs/CONVENTIONS.md`: native-API escape hatch subsection added (when `use_native=True` is appropriate).

### Out of scope

- LightGBM / CatBoost out-of-core paths (deferred per D3; LightGBM uses binned datasets that fit in RAM; CatBoost has its own ExtMem mechanism).
- ExtMem cross-fold sharing in HPO (per-fold rebuild cost; deferred indefinitely).

## Dependencies

- **PR-030** (sprint scaffolding) lands first.
- **v0.3 design session** runs and locks D2 before this PR's Phase 1.

## Architecture section implemented

`docs/ARCHITECTURE.md` "Decision Rules" → "XGBoost ingest path" (D3); Memory & Parallelism table `cache_host_ratio` row.

## Verification criteria

Populated after Phase 1 state assessment. Initial sketch:

- [ ] `XGBoostNativeAdapter` exists in `src/rux_ml/training/xgboost/native_adapter.py` and passes the `Trainer` Protocol conformance test in `tests/training/test_registry_conformance.py`.
- [ ] `cli/train.py` and `registry/promote.py` route through `select_ingest` decision honestly.
- [ ] `cache_host_ratio` is read by `XGBoostNativeAdapter.__init__` and passed to `ExtMemQuantileDMatrix`.
- [ ] `use_native` is read and respected.
- [ ] Test-mode forcing the path on small data passes end-to-end.
- [ ] `compute_score` returns sensible RMSE for an `XGBoostNativeAdapter` instance.
- [ ] `docs/ARCHITECTURE.md` D3 entry updated from "documented-aspiration" framing to current-active.
- [ ] `docs/0.3/ROADMAP.md` PR-033 row flipped `[ ]` → `[x]`.
- [ ] `CHANGELOG.md [Unreleased] ### Added` (new adapter, new behavior) + `### Fixed` (ExtMem path now actually executes when threshold exceeded).

## Research backing

Tier-2 — D3 originally researched the dispatch decision; the v0.3 wire-up shape is open. Locked by v0.3 design session D2 + Phase 3 web research per the 8 open questions.

Anchored on:
- D3 (PR-006) original research on XGBoost ingest tiers
- `ParquetDataIter` (PR-006)
- `select_ingest` (PR-006/PR-017)
- v0.3 design session D2 (to be written)

## Notes

- **Largest scope in v0.3 sprint.** May split if Phase 1 surfaces drift.
- **Triggers MINOR bump** (v0.3.0) per `docs/VERSIONING.md §1`.
- **No real-world workload triggers this today.** Crypto-h3 is 1.4 GB, well under the 18 GB threshold. The PR's primary value is making the documented behavior real before a workload needs it. Test exercises happen via forced-flag.
- Per memory `feedback_roadmap_flip_in_pr`: PR-033's implementation commit must flip `docs/0.3/ROADMAP.md` PR-033 row `[ ]` → `[x]` in the same commit.
- Phase 1 should verify the `ExtMemQuantileDMatrix` + `cache_host_ratio` kwarg has not changed in the currently pinned XGBoost minor version.
- This PR also resolves the "borderline phantom 2b" (`select_ingest` return drives telemetry only, not behavior) by making the return value actually drive construction.
