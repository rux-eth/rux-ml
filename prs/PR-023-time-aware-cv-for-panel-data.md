# PR-023: Time-aware CV for panel data — time-unit embargo + label-overlap purge

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per memory `project_cv_strategy_tier2` (CV strategy work is universally Tier-2 in this project — proper AFML purge/embargo sizing, panel-data semantics, and skfolio CPCV behavior on stacked panels all need fresh research).

**Additionally, this PR triggers a MINOR bump** (v0.2.0) per `docs/VERSIONING.md` §1 (new built-in Splitter behavior / new config knobs). Per `docs/VERSIONING.md` §2, **a v0.2 design session via `PROCEDURE-design-planning.md` is required before this PR is finalized.**

Skipping the PR research procedure or the design-session prerequisite is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md` (after the v0.2 design session)._

**Surfaced during the first real-dataset run** on 2026-05-18 (chat session `f085a0c0-afc8-4386-8e48-be8acb810c12`). The investigation confirmed the v0.1.0 CV machinery silently under-embargoes any panel data with forward-looking targets; the resulting metric is meaningfully better-than-real on any stacked-panel problem (not just the crypto OHLCV case that happened to surface it).

**Open research questions** (must be resolved before implementation):

1. **Time-unit vs row-count embargo API** — `TimeSeriesSplitCV.gap` is row-count. For stacked panels with N rows per timestamp, `gap=K` ≈ `K/N` timestamps of separation, collapsing to <1 unit of per-asset embargo on the live crypto dataset (1084 assets × hourly bars). Same gap-vs-time mismatch affects any panel with a multiple-rows-per-timestamp shape (e.g. multi-instrument trading datasets, multi-store retail data, multi-patient medical time series). Options:
   - **A**: Add `time_column: str | None` and `embargo_time: float | None` fields to `TimeSeriesSplitCV`. Splitter converts time-unit embargo to row-counts using the actual timestamps. Reuses existing variant.
   - **B**: Add a new `PanelTimeSeriesSplitCV` discriminator variant with explicit `time_column` + `asset_column` + `embargo_bars` fields. Separate from `TimeSeriesSplitCV` (which stays single-asset).
   - **C**: Document `gap` as row-count + add a runtime warning when the data appears stacked. Don't change the schema. (Already partly in PR-022 Q3.)
   
   Phase 3 research: AFML Ch. 7, skfolio `CombinatorialPurgedCV` source, mlfinlab `PurgedKFold` API, sklearn-evaluation patterns. Which mature projects expose embargo in time-units vs row-counts?

2. **Label-overlap purging for forward-looking targets** — with h-bar forward targets, train rows whose target window overlaps test indices leak future information. AFML's purge condition drops such train rows; the workbench's `CombinatorialPurgedCV` (PR-015) wraps skfolio's CPCV which takes `purged_size` but currently has no surfaced way to express "drop rows whose forward-h target window overlaps a test index." Options:
   - **A**: Surface `target_horizon_bars` field on `CombinatorialPurgedCV`; convert to `purged_size` internally given `n_splits` and dataset length. Hides the AFML formula behind a workbench-level knob.
   - **B**: Pass through to skfolio's low-level args directly; document the calibration in `docs/CONVENTIONS.md`. Easier to ship, leaves the conversion math on the user.
   - **C**: Implement purging in the workbench's Splitter wrapper, independent of skfolio's `purged_size` semantics. Most control, most code.
   
   Phase 3 research: skfolio CPCV semantics (`purged_size` vs the AFML formula), whether purge is one-sided or two-sided in skfolio's implementation, whether the AFML embargo formula `embargo ∈ [0.005, 0.02] · N` from chapter 7 generalizes to panel data with multiple rows per timestamp.

3. **`CombinatorialPurgedCV` on stacked panels** — skfolio's CPCV was designed for single-asset time series. On a stacked panel sorted by `(timestamp, asset)`, does CPCV give meaningful per-asset embargo, or does it need a `GroupKFold`-shaped overlay (group by asset, then time-purge within each asset group)? Probably requires either:
   - A new `PanelCombinatorialPurgedCV` variant that purges per-asset
   - A pre-CV reshape: pivot data to `(asset_id_lane, timestamp)` before CPCV, then unpivot
   - Document CPCV as single-asset-only and require users to stack-split externally
   
   Phase 3 research: how mlfinlab / fastquant / quantitative-finance shops handle panel CPCV; whether anyone has shipped a panel-aware CPCV implementation.

4. **Family-agnostic calibration test** — the leakage is in the Splitter, not the trainer. Calibration must work across all 3 trainer families (XGBoost, LightGBM, CatBoost) — same Splitter on the same data should produce the same train/test row sets regardless of which family scores them. Open: synthetic dataset shape that exhibits the leakage clearly enough to be a regression test? Options:
   - **A**: Synthetic stacked panel where target = `close[t+h]` from a deterministic price series; without embargo, CV metric is artificially good; with embargo, metric drops to no-skill. Family-blind.
   - **B**: Randomized-target null model (shuffle `y`, retrain). Properly-embargoed CV should produce RMSE ≈ target_std; if it doesn't, leakage remains.

5. **Cohabitation with `rux-ml train` baseline path** — `train_val_test_split` (used by `rux-ml train`) does random shuffle, which is even worse for time-series than the CV path. Options:
   - **A**: Add `split_kind: Literal["random", "time_ordered"]` to `DataConfig`; default `"random"` preserves v0.1 behavior; time-series problems opt into `"time_ordered"`. New config knob → MINOR.
   - **B**: Document that `rux-ml train` is unsafe for time-series problems; require users to use `rux-ml tune --n-trials 1` instead. Cheaper.
   - **C**: Out of scope — handle `rux-ml train` baseline path in a separate follow-up PR.
   
   Lean (intuition, flagged): **A** if it fits cleanly in the design session; otherwise C.

---

## Scope

Time-aware, panel-aware CV. Concrete shape decided by v0.2 design session + Phase 4 synthesis. Listed at the option-level below, not pre-committed.

**Anticipated changes** (refine post-research):
- Schema update for `[cv]` discriminated union (per Q1 + Q3): either extend existing variants OR add new panel-aware variants.
- `src/rux_ml/data/cv.py` Splitter implementations.
- `src/rux_ml/config/cv.py` schema.
- `configs/base.toml` and inline comments for new fields.
- `docs/ARCHITECTURE.md` Decision Rules section — add panel-CV selection rule.
- `docs/CONVENTIONS.md` — soft-pattern guidance on CV-for-panel-data.
- Synthetic-panel calibration tests (per Q4); family-blind by construction.
- CHANGELOG `[Unreleased]` entries (likely `### Added` for new variants, `### Changed` for any modified `TimeSeriesSplitCV` semantics, `### Fixed` for the leakage).

### Out of scope

- Walk-forward retraining for live deployment (CV vs deployment separation).
- Per-fold artifact / per-asset metric decomposition (orthogonal observability concern).
- New trainer families (none planned in v0.2 yet).
- `rux-ml train` random-shuffle leakage if Q5 is resolved as C.

## Dependencies

- **v0.2 design session via `PROCEDURE-design-planning.md`** must complete before this PR is finalized. Other v0.2 work (TBD) may bundle here.
- **PR-022 lands first as v0.1.1 PATCH** so the eval_set / test-fold leakage isn't entangled with the panel-CV redesign.

## Architecture section implemented

`docs/ARCHITECTURE.md` Decision Rules — adds a new rule for CV-strategy-selection on panel data (the existing kfold / time_series / group_kfold / cpcv decision tree doesn't account for stacked panels with per-asset time semantics).

## Verification criteria

Populated after research + design session. Initial sketch:

- [ ] Time-unit embargo respected: synthetic panel with known label horizon → test asserts train/test boundary is `>= embargo_time` per asset, not just per row index.
- [ ] Label-overlap purge: synthetic forward-target data → test asserts no train row has a target window overlapping any test row's input window.
- [ ] Panel-CPCV: synthetic stacked panel → test asserts per-asset embargo is honored within every combinatorial split.
- [ ] **Family-agnostic** calibration: same synthetic data scored by XGBoost, LightGBM, AND CatBoost under the new Splitter — all 3 produce expected metric range (no-skill on randomized-target null model, sensible signal-detection on planted-signal panel). Parametrized over `TRAINER_FAMILIES`.
- [ ] Crypto regression baseline (PR-022-cleaned version of today's run) re-runs with time-aware CV; RMSE delta is measured. Expected: rise from 0.02547 → calibration baseline.
- [ ] `docs/CONVENTIONS.md` documents when to use which Splitter for panel data.
- [ ] No regression in existing `tests/data/test_cv.py` — old Splitter variants behave identically (or are explicitly migrated with deprecation per `docs/VERSIONING.md` §1 MINOR rules).

## Research backing

Tier-2 — research happens during `PROCEDURE-pr-research.md` Phases 2-3 (and during the v0.2 design session Phase 2 / Phase 3 ahead of that).

Anchored on:
- **López de Prado, AFML** — Ch. 7, Purged K-Fold + Combinatorial Purged Cross-Validation.
- **skfolio** — `model_selection.CombinatorialPurgedCV` (current implementation; PR-015 wrapper).
- **mlfinlab** — alternative AFML implementations.
- **scikit-learn** — `TimeSeriesSplit` semantics being departed from.
- **Industry surveys** — panel-CPCV practice across quantitative-finance / multi-instrument trading shops, multi-store retail forecasting, multi-patient medical time series.

## Notes

- **Triggers MINOR bump** (v0.2.0) per `docs/VERSIONING.md` §1.
- **v0.2 design session required** per `docs/VERSIONING.md` §2 before this PR is finalized. Design session may surface additional v0.2 work that should land alongside.
- **Family-agnostic by construction**: the Splitter operates on row indices and an optional `time_column` / `asset_column` — none of the trainer families enter the CV-correctness picture. The verification criteria parametrize over `TRAINER_FAMILIES` to enforce family-blindness, but the underlying fix has no per-family code path.
- **The crypto-h3 baseline trained on 2026-05-18 (RMSE=0.02547 at cdbffd2) is leakage-contaminated.** Once PR-022 ships the eval_set fix and PR-023 ships time-aware embargo, that model's metric must be re-computed. The original number stays in the runs DB as a historical artifact (per memory `feedback_pr_spec_historicity`).
- Per memory `project_cv_strategy_tier2`, **K-fold / walk-forward / CPCV / GroupKFold all need dedicated Tier-2 PRs with full research**. This PR covers the panel-data + time-aware case; other CV strategy revisits (per-problem CPCV calibration, walk-forward live-deployment helper, etc.) remain separate work.
- PR-022 is the sibling PATCH that addresses the eval_set / test-fold leakage and discriminator-hygiene prevention; that lands first.
- **`feedback_roadmap_flip_in_pr` rule applies once the v0.2 ROADMAP exists** (created during the v0.2 design session). Until then, no row to flip — call this out in implementation commits.
