# PR-024: One-off temporal train/val/test split

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per memory `project_cv_strategy_tier2`. **Pattern locked** by PR-023 Phase 3 Q5 archived research (≥3 cited systems: sktime + Darts + AutoGluon TS + Nixtla + mlfinlab all use **two separate functions**, not a single function with a kind-knob). Per-PR Phase 1 state assessment is mandatory (catch drift since PR-023 lands); Phases 2–4 may be light because the convention is established.

**This PR triggers a MINOR bump** (v0.2.0) per `docs/VERSIONING.md` §1 (new config knob `data.split_kind` + new public function). Ships under v0.2.0 alongside PR-023.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Pattern surveyed in PR-023 Phase 3 (2026-05-18→2026-05-19)** — research findings archived in `prs/PR-023-time-aware-cv-for-panel-data.md` Phase 3 Q5. Verdict: **convention** (≥3 cited systems) for "two functions, not one knob." Mature time-series libraries (sktime `temporal_train_test_split`, Darts `TimeSeries.split_before/split_after`, AutoGluon TS, Nixtla `mlforecast.cross_validation`, mlfinlab) all expose temporal split as a separate, dedicated function from random split. sklearn keeps `train_test_split(shuffle=False)` as the workaround. Quantified leakage evidence: **arxiv 2512.06932** (Dec 2025) — LSTM RMSE Gain inflates 19.29%–20.51% under leaky splits vs <3% under chronological.

**Open research questions** (must be resolved at Phase 1 state assessment + Phase 2 scoping):

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
