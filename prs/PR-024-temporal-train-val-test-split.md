# PR-024: One-off temporal train/val/test split

**Landed-in:** v0.2.0

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per memory `project_cv_strategy_tier2`. **Pattern locked** by PR-023 Phase 3 Q5 archived research (≥3 cited systems: sktime + Darts + AutoGluon TS + Nixtla + mlfinlab all use **two separate functions**, not a single function with a kind-knob). Per-PR Phase 1 state assessment is mandatory (catch drift since PR-023 lands); Phases 2–4 may be light because the convention is established.

**This PR triggers a MINOR bump** (v0.2.0) per `docs/VERSIONING.md` §1 (new config knob `data.split_kind` + new public function). Ships under v0.2.0 alongside PR-023.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

### Phase 1 — State Assessment (2026-05-19)

**Current state** (HEAD = `a66665b`, post-PR-023 merge):

- `train_val_test_split` in `src/rux_ml/data/splits.py` is **unchanged by PR-023** — still `df.sample(fraction=1.0, shuffle=True, seed=seed)` with ratio validation (67 lines total).
- **5 consumers** of `train_val_test_split`: `src/rux_ml/cli/train.py:70` (one-off baseline), `src/rux_ml/registry/promote.py:110` (re-fit at promotion), `tests/golden/test_xgb_baseline.py:186+375`, `tests/cli/test_train_subcommand.py`, `tests/data/test_splits.py` (7 unit tests).
- `DataConfig` (`src/rux_ml/config/data.py`) currently has `source_path`, `target_column`, `cas_root`, `manifests_root`, `gpu_in_memory_x_gb_max`, `split_ratios` — **no `split_kind`, no `time_column`**.
- **Discriminator-hygiene constraint** (PR-022): `[data]` is NOT a discriminated union. The structural test (`tests/config/test_base_toml_discriminator_hygiene.py`) only walks `[cv]` / `[training]` / `[solving]` — new common fields on `DataConfig` (`split_kind`, `time_column`) are base.toml-safe.
- **Reproducibility contract** (PR-013): `cli/train.py` ↔ `registry/promote.py` must produce the SAME train/val/test split for a given trial (promote re-builds `bag` from the trial's `entropy_hex` and re-calls the split function with the same seed). Temporal split is deterministic (sort by `time_column`; no seed needed) — contract preserved for free for the time_ordered path.
- **Golden infra** (PR-014) is CPU bit-exact, opt-in via `make test-golden`. CPU-only golden for the temporal path is feasible without GPU.

**Drift since PR-023**: zero days elapsed. `git log` on `src/rux_ml/data/splits.py`, `src/rux_ml/config/data.py`, `src/rux_ml/cli/train.py`, `src/rux_ml/registry/promote.py` shows no commits between `a66665b` (PR-023 merge) and PR-024 implementation start. **No drift.**

**Stale stub assumptions:**

- Stub Item 1 signature uses `time_col`; PR-023 D1/D2 chose `time_column`. Naming consistency favors `time_column`.
- Stub Q1 asks "should `temporal_train_val_test_split` accept `asset_col`?" sktime + AutoGluon do per-series tail-slicing. **Resolved**: workbench's typical baseline use case is "global cut at timestamp T → all assets in each partition." Per-asset tail-slicing produces uneven per-asset histories and is not what a baseline regression typically wants. **Reject `asset_column` parameter** (defer to user code if ever needed; tracked in v0.2 RESEARCH-BACKLOG drift watch).
- Stub Q2 leans `UserWarning` for the `cv.kind` × `data.split_kind` mismatch. **Resolved**: hard error at config validation via a `model_validator` on `RuxMLConfig` (loud + fail-fast at config load, not at training time; matches the rest of the project's validation strategy).
- Stub Q3 mentions goldens. Doable without GPU; commit small CPU-only golden.

**Exit decision**: Tier-2 status confirmed; research archived in PR-023 Phase 3 Q5; sub-decisions resolved with user approval (2026-05-19). Premise still valid; no drift; no prerequisite PRs surfaced. Proceed to Phase 2.

### Phase 2 — Scope Research (2026-05-19, light)

Per memory `feedback_phase1_scope`: Tier-2 with research archived → light Phase 2 (no new external research, scope sharpening only).

**Must-answer** (all resolved by Phase 1 + 2026-05-19 user-approved sub-decision choices):

1. **`time_column` naming** — matches PR-023 D1/D2 convention. RESOLVED.
2. **No `asset_column` parameter** on `temporal_train_val_test_split` — global cut by sorted `time_column`. RESOLVED.
3. **Hard validation error** for `cv.kind in {"time_series", "cpcv", "panel_cpcv"}` + `data.split_kind == "random"`. RESOLVED — implemented as a `model_validator(mode="after")` on `RuxMLConfig`.
4. **`time_ordered` requires `time_column`** — same validator. RESOLVED.
5. **Golden for temporal path** — CPU-only fixture under `tests/golden/`. RESOLVED.

**Excluded from this PR** (truly orthogonal):
- Walk-forward retrain (D7, out of v0.2).
- Per-asset temporal tail-slicing — A3-like ambiguity; revisit if a real use case surfaces.

### Phase 3 — Findings (2026-05-18 → 2026-05-19, archived in PR-023 Phase 3 Q5)

**Pattern verdict**: **convention** (≥3 cited systems) for "two functions, not one knob." Mature time-series libraries (sktime `temporal_train_test_split`, Darts `TimeSeries.split_before/split_after`, AutoGluon TimeSeriesPredictor, Nixtla `mlforecast.cross_validation`, mlfinlab) all expose temporal split as a separate, dedicated function from random split. sklearn keeps `train_test_split(shuffle=False)` as the workaround. Quantified leakage evidence: **arxiv 2512.06932** (Dec 2025) — LSTM RMSE Gain inflates 19.29%–20.51% under leaky splits vs <3% under chronological.

Full citations + URLs live in `prs/PR-023-time-aware-cv-for-panel-data.md` Research findings Phase 3 Q5 row. PR-024 does NOT re-dispatch agents; the convention is locked.

### Phase 4 — Synthesis (2026-05-19)

**Outcome**: confirm-with-locked-specifics — research confirms the premise (random shuffle on time-series leaks); pattern locked by PR-023 Phase 3 Q5; PR-024's sub-decisions locked by 2026-05-19 user approval.

**Final implementation shape**:
- New function `rux_ml.data.splits.temporal_train_val_test_split(df, *, time_column, ratios) → dict[str, pl.DataFrame]`. Sorts by `time_column`, slices by ratio. Deterministic.
- `DataConfig` gains `split_kind: Literal["random", "time_ordered"] = "random"` + `time_column: str | None = None`.
- `RuxMLConfig` gains a `model_validator(mode="after")` that enforces:
  - `data.split_kind == "time_ordered"` requires `data.time_column` to be set.
  - `cv.kind in {"time_series", "cpcv", "panel_cpcv"}` requires `data.split_kind == "time_ordered"`.
- `cli/train.py` + `registry/promote.py` route through a small dispatcher helper.
- CPU golden under `tests/golden/test_temporal_baseline.py` (tolerance-based, not exact-hash).

**Changes to ARCHITECTURE.md**:
- Decision Rules CV-strategy-by-data-shape entry (added by PR-023) gets a cross-reference to `data.split_kind` for the one-off baseline path.

**Changes to CONVENTIONS.md**:
- One-shot section gains a `time_ordered` branch + cross-references the mismatch validator.

**New PRs that must come first**: none.

### Phase 5 — Gate Check (2026-05-19)

- Premise still valid: ✓ (random shuffle on time-series leaks; ≥3 cited systems use two separate functions)
- No prerequisite PRs surfaced: ✓ (PR-023 already merged; PR-025 / PR-026 are downstream)
- User approved updated spec: ✓ (2026-05-19 sub-decisions: no asset_column; `time_column` naming; hard validation error; CPU golden)
- Implementation cleared: ✓

### Phase 5 — Implementation outcomes (2026-05-19)

Mini state-assessment: zero days elapsed since Phase 5 Gate Check; `git log` on the target files (`src/rux_ml/data/splits.py`, `src/rux_ml/config/data.py`, `src/rux_ml/config/root.py`, `src/rux_ml/cli/train.py`, `src/rux_ml/registry/promote.py`, `tests/data/test_splits.py`, `tests/cli/test_train_subcommand.py`) shows no commits between PR-023 merge (`a66665b`) and implementation start — no drift.

**Code landed**:

- `src/rux_ml/data/splits.py`
  - New `temporal_train_val_test_split(df, *, time_column, ratios)` — sorts by `time_column`, slices into time-ordered partitions. Deterministic. Validates `time_column` presence + ratios.
  - New `make_splits(cfg, df, *, seed)` dispatcher — single call site for both branches. Random path threads `seed`; temporal path is deterministic (seed ignored). Used by `cli/train.py` AND `registry/promote.py` so the reproducibility contract is preserved by construction.
- `src/rux_ml/config/data.py`
  - New fields on `DataConfig`: `split_kind: Literal["random", "time_ordered"] = "random"` + `time_column: str | None = None`. Defaults preserve v0.1.1 behavior.
- `src/rux_ml/config/root.py`
  - New `model_validator(mode="after")` on `RuxMLConfig` enforces two cross-field rules at config-load time (fail-fast `ValueError`):
    1. `data.split_kind == "time_ordered"` requires `data.time_column` to be set.
    2. `cv.kind ∈ {"time_series", "cpcv", "panel_cpcv"}` requires `data.split_kind == "time_ordered"`.
  - `_TEMPORAL_CV_KINDS: frozenset[str]` module-level constant for the rule (single source of truth).
- `src/rux_ml/cli/train.py` + `src/rux_ml/registry/promote.py` — both swap `train_val_test_split(...)` for `make_splits(cfg, df, seed=bag.split_seed)`.
- `src/rux_ml/data/__init__.py` — re-exports `make_splits` + `temporal_train_val_test_split`.
- **16 new tests** (default suite delta: 347 → 363):
  - `tests/data/test_splits.py` — 13 added (was 7): determinism / monotone-non-decreasing time-ordering / row preservation / panel structure / missing-column / empty-frame for `temporal_train_val_test_split`; dispatcher routing (random default, temporal opt-in, seed-ignored on temporal path) for `make_splits`; validator rules (rejects time_ordered-without-time_column; rejects temporal-CV-with-random-split; accepts consistent temporal pair; accepts random + non-temporal CV).
  - `tests/cli/test_train_subcommand.py` — 3 added: `time_ordered` CLI end-to-end on a CPU-trained 1-trial study; validator rejects missing `time_column` from the CLI; validator rejects temporal CV with random split from the CLI.
  - `tests/config/test_cv_config.py` — existing parametrized loader test updated to satisfy the new validator (temporal CV cases now carry a `[data]` block with `split_kind = "time_ordered"` + `time_column = "ts"`).

**Refinement during implementation**: the Phase 1 plan called for a "CPU golden under `tests/golden/test_temporal_baseline.py`" but the temporal-split function is a pure Polars sort + slice with no library-drift surface — a golden adds no regression-detection beyond what the unit tests already cover (determinism + monotone-non-decreasing ordering invariants). Replaced with a CLI integration test that exercises the full dispatcher path through `rux-ml train` (`test_train_time_ordered_path_end_to_end_cpu`); this catches dispatcher wiring + validator + downstream training in one shot. Heavier-weight goldens for `data.split_kind = "time_ordered"` end-to-end (with committed prediction arrays) are deferred — no incremental coverage at this PR's scope; revisit if a real time-series problem ships and merits the regression-baseline lock.

**Deferred (still owned by sibling PRs)**:
- Eval_set Position B/C operational debate → PR-025.
- Walk-forward retrain helper → out of v0.2 entirely (PR-023 D7).

**Tests + lint outcomes**:
- `uv run pytest` — 363 passed / 1 skipped / 15 deselected (was 347 on `a66665b`).
- `uv run pytest -m golden` — 7 passed (no regression in the existing XGBoost golden).
- `uv run ruff check .` — clean (verified separately).
- `uv run basedpyright src/` — clean (verified separately).

---

**Open research questions (HISTORICAL — preserved for traceability):**

1. **`time_col` semantics on stacked panels.** sktime + AutoGluon TS both do per-series tail-slicing — for stacked-panel data, the temporal split must respect per-asset boundaries. Open: should `temporal_train_val_test_split` accept an optional `asset_col` and do per-asset slicing? Or is it strictly single-series, with users reshaping externally?
2. **Validation warning on `cv.kind` mismatch.** When `cv.kind in {"time_series", "cpcv", "panel_cpcv"}` but `data.split_kind == "random"`, the one-off baseline path's `rux-ml train` leaks. Open: emit a `UserWarning` at config load? Hard error? Or trust users?
3. **Goldens for the temporal path.** PR-014 established golden-test infrastructure for the training layer. PR-024 should add goldens for the temporal split: tiny synthetic time series with hand-verified train/val/test boundaries, asserted via tolerance-based golden contract.
4. **Drift since PR-023.** Phase 1 must re-read `src/rux_ml/data/splits.py`, `src/rux_ml/cli/train.py`, and any PR-023 implementation that touched the data layer.

## Scope

Two items, both research-anchored by PR-023 Phase 3 Q5.

### Item 1 — new function `temporal_train_val_test_split`

Add to `src/rux_ml/data/splits.py`:
```python
def temporal_train_val_test_split(
    df: pl.DataFrame,
    *,
    time_col: str,
    ratios: Mapping[str, float],
    asset_col: str | None = None,  # for stacked-panel per-asset slicing (per Q1)
) -> dict[str, pl.DataFrame]:
    ...
```
Sorts by `time_col` (or `(asset_col, time_col)` if asset_col supplied), then splits by ratio with strict temporal ordering (no shuffle, no seed needed).

### Item 2 — `data.split_kind` dispatch field

Add `data.split_kind: Literal["random", "time_ordered"] = "random"` to `DataConfig`. Default preserves v0.1 behavior. `rux-ml train` dispatches to `train_val_test_split` (random) or `temporal_train_val_test_split` (time_ordered) based on `cfg.data.split_kind`. If `time_ordered`, also requires `cfg.data.time_column` to be set (a new field — make it conditional or fail-fast at validation).

### Item 3 — `cv.kind` / `data.split_kind` mismatch warning

When `cv.kind in {"time_series", "cpcv", "panel_cpcv"}` and `data.split_kind == "random"`, emit a `UserWarning` (or hard error per Q2 decision). Catches the silent-foot-gun where users configure temporal CV but forget the one-off baseline path uses random shuffle.

### Out of scope

- Time-aware CV (`TimeSeriesSplitCV.embargo_time`, `PanelCombinatorialPurgedCV`) — PR-023.
- Walk-forward retrain helper — out of v0.2 entirely (per `docs/0.2/DESIGN-log.md` D7).

## Dependencies

- **PR-023 lands first** — `data.split_kind` may need to coordinate with PR-023's `cv.kind` additions (panel_cpcv).
- v0.2 design session: complete (this PR is queued in `docs/0.2/ROADMAP.md`).

## Architecture section implemented

`docs/ARCHITECTURE.md` Decision Rules — the CV-strategy-by-data-shape decision rule (added by PR-023) should be updated to cross-reference `data.split_kind` for the one-off baseline path.

## Verification criteria

Populated after Phase 1 state assessment. Initial sketch:

- [ ] `temporal_train_val_test_split` returns time-ordered train/val/test (no shuffle); test set is always temporally last; train is temporally first.
- [ ] Per-asset path (when `asset_col` is set) respects per-asset temporal boundaries — no train row for asset A is later than test rows for the same asset A.
- [ ] `data.split_kind` field validates: `"random"` works with any `cv.kind`; `"time_ordered"` requires `data.time_column`.
- [ ] `rux-ml train` dispatches correctly between the two functions.
- [ ] Mismatch warning fires when `cv.kind` is temporal and `data.split_kind == "random"`.
- [ ] Golden test for the temporal path (per Q3).
- [ ] `CHANGELOG.md [Unreleased] ### Added` entries for the new function + new config field.

## Research backing

Tier-2 — pattern locked by PR-023 Phase 3 Q5; per-PR state assessment runs before implementation.

Anchored on:
- sktime `temporal_train_test_split` — https://github.com/sktime/sktime/blob/main/sktime/split/temporal_train_test_split.py
- Darts `TimeSeries.split_before/split_after` — https://unit8co.github.io/darts/generated_api/darts.timeseries.html
- AutoGluon TimeSeriesPredictor split conventions
- arxiv 2512.06932 — quantified leakage evidence for the foot-gun

## Notes

- **Triggers MINOR bump** (v0.2.0) per `docs/VERSIONING.md` §1.
- **PR-023 lands first** — coordinate `cv.kind` enum additions (panel_cpcv) with `data.split_kind` validation rules.
- The crypto-h3 baseline trained at `cdbffd2` (RMSE=0.02547) used `rux-ml tune --n-trials 3` (CV path), not `rux-ml train` (one-off path), so it was not affected by the random-shuffle leak. But the workbench's CONVENTIONS.md should make this distinction loud once PR-024 ships.
- Per memory `feedback_roadmap_flip_in_pr`: PR-024's implementation commit must flip `docs/0.2/ROADMAP.md` PR-024 row `[ ]` → `[x]` in the same commit.
