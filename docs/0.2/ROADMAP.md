# Roadmap — v0.2

Ordered list of PRs for the v0.2 cut. Each PR is a single, reviewable change. Dependencies flow downward — no PR should be started before its predecessors are merged.

Full PR descriptions live in `prs/`. This file is the index.

The v0.1 roadmap (PR-017 through PR-021) is frozen at [`docs/0.1/ROADMAP.md`](../0.1/ROADMAP.md) — every row is `[x]`. The v0.0 roadmap is at [`docs/0.0/ROADMAP.md`](../0.0/ROADMAP.md).

**Status legend:** `[ ]` pending | `[x]` merged | `[~]` in progress

---

## Phase I: Time-aware CV correctness

| PR | Description | Status | Depends on |
|----|-------------|--------|------------|
| [PR-022](../../prs/PR-022-workbench-correctness-hygiene.md) | **Tier-1** workbench correctness hygiene — `[cv] shuffle = true` carryover fix + structural test enforcing base.toml discriminator hygiene across all 3 unions + CV-eval_set docs reframe (research-backed: Position A is library-blessed default but with documented HPO bias; reframed as pragmatic deviation, not Optuna-inherited pattern). Ships under v0.1.1 PATCH. | `[x]` | — |
| [PR-023](../../prs/PR-023-time-aware-cv-for-panel-data.md) | **Tier-2** time-aware CV for panel data — new `PanelCombinatorialPurgedCV` discriminator variant (folds over unique timestamps; per-asset purge); `TimeSeriesSplitCV` extended with polymorphic `embargo_time: int \| str` + `time_column`; workbench-level `target_horizon_bars` + `embargo_pct` on `CombinatorialPurgedCV` converting internally to skfolio's `purged_size` + `embargo_size = int(N * embargo_pct)`; layered tests (per-asset hand-verified fixtures + property + shuffle-null tripwire); `docs/ARCHITECTURE.md` Decision Rules entry for CV-strategy-by-data-shape; mandatory `docs/CONVENTIONS.md` note on skfolio's coarser purge model vs AFML interval-overlap | `[x]` | — |
| [PR-024](../../prs/PR-024-temporal-train-val-test-split.md) | **Tier-2** one-off temporal split — new `temporal_train_val_test_split(df, time_col, ratios, ...)` function in `src/rux_ml/data/splits.py` + new `data.split_kind: Literal["random", "time_ordered"]` field on `DataConfig` (dispatches between the two functions). Default preserves v0.1 behavior; users opt into time-ordered for time-series. Research mostly archived in PR-023 Phase 3 Q5; Tier-2 PR runs its own light state assessment | `[x]` | — |
| [PR-025](../../prs/PR-025-cv-eval-set-design-study.md) | **Tier-2** eval_set Position B/C operational design study — research-anchored decision on whether to switch the CV objective from Position A (test fold = eval_set, current/library-blessed-but-biased) to Position B (inner val from train fold; sklearn HistGradientBoosting default) or Position C (XGBoost's own recommendation: no early stopping in CV + post-CV retrain). Research mostly archived in PR-022 Phase 3; this PR re-opens with focus on calibration impact on the workbench's specific configuration | `[x]` | — |
| [PR-026](../../prs/PR-026-v0-2-0-cut.md) | **Tier-2** v0.2.0 version cut — script versioned→versioned migration (`docs/0.1/* → docs/0.2/*` in CLAUDE.md / README.md / `docs/SSOT/*`); bump `pyproject.toml` version `0.1.0 → 0.2.0`; rewrite `CHANGELOG.md` `[Unreleased]` → `[0.2.0]`; tag `v0.2.0`. Analog of PR-021 for the v0.1 cut | `[ ]` | PR-023, PR-024, PR-025 |

---

## Notes

- **PR-023 is the focal work** for v0.2 — surfaced by the 2026-05-18 first-real-dataset run (crypto OHLCV breakout-return regression). PR-024 and PR-025 are siblings with research already archived in PR-022/PR-023; they get their own state assessments + light Phase 2 before implementation, not full design-planning sessions.
- v0.2 is the **first time** the workbench faces panel-data correctness issues. The crypto-h3 baseline trained on 2026-05-18 (RMSE=0.02547 at `cdbffd2`) is leakage-contaminated by ≥2 mechanisms: per-asset embargo collapse (PR-023 fix) + HPO compounding bias (PR-025 deferred study). Re-running it under proper time-aware CV is a calibration target after PR-023 lands.
- Each v0.2 PR is **Tier-2** per memory `feedback_tier_inheritance` and the per-instance research surface — the panel-CV architectural pattern was research-backed by the 2026-05-19 design session (compressed flow; see [`DESIGN-log.md`](DESIGN-log.md) D1–D7), but per-PR design details + fixture patterns are not. Each PR runs the full 5-phase `PROCEDURE-pr-research.md` before implementation; Phase 1 (State Assessment) is mandatory regardless of design-time research.
- **PR-023, PR-024 are parallelizable** — independent of each other. PR-025 is loosely coupled (its decision may revisit `_fold_scores` which PR-023 might also touch); draft sequentially after PR-023 to avoid merge friction.
- PR-026 is the **version-cut ritual** per `docs/VERSIONING.md §2` workflow step 6. It lands last, after all v0.2 implementation PRs are merged on `dev`. The `v0.2.0` git tag lands at PR-026's merge commit.
- **PR-022 carries v0.1.1** — it's a PATCH PR that landed in the v0.2 sprint window but ships in the next PATCH cut (likely PR-026 rolls v0.1.1 and v0.2.0 together; verify at cut time).
- Five A-deferred / acknowledged ambiguities (A1–A5 in [`DESIGN-log.md`](DESIGN-log.md) "Deferred / acknowledged ambiguities" section) — each tagged to its resolution path.
- Phase I is intentionally small — v0.1 added breadth (multi-family Trainer + Solver); v0.2 adds correctness depth for the first real-data use case. CV strategy revisits per problem ([`feedback_cv_strategy_tier2`](../../MEMORY.md)) and the first Rust crate per D13's profile-driven trigger remain post-v0.2 work.
- Per `docs/CONSTRAINTS.md` (Per-Phase Approval Gate, Research-Backed Decisions, No Phantom Implementations): each PR includes a real test exercising actual end-to-end behavior on a synthetic panel that demonstrates the leakage absent the fix.
