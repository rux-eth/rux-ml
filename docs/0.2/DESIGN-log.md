# Design Log — v0.2

Design decisions and rationale from planning sessions for the v0.2 cut. Append new sessions below.

The canonical record of v0.1 design (Q1–Q6 + A1–A6 deferred ambiguities) lives in [`docs/0.1/DESIGN-log.md`](../0.1/DESIGN-log.md). Decisions and constraints from v0.1 remain in force unless an entry in this log explicitly supersedes them. The frozen v0.0 record is at [`docs/0.0/DESIGN-log.md`](../0.0/DESIGN-log.md).

---

## Session: 2026-05-19 — Time-aware CV for panel data (D1–D7)

### Context

Surfaced by the **first real-dataset run** on 2026-05-18 (crypto OHLCV breakout-return regression, 1084-asset stacked panel, 20.7M rows, h=3 forward target). The v0.1 CV machinery silently under-embargoes stacked-panel data with forward-looking labels: `TimeSeriesSplitCV.gap` is row-count, so `gap=24` on a 1084-asset hourly panel ≈ 0.022 timestamps of per-asset separation, not 24 hours. The reported metric was meaningfully better-than-real.

Investigation transcript: chat session `f085a0c0-afc8-4386-8e48-be8acb810c12`. Adjacent findings:

- **PR-022 (landed v0.1.1-bound)** — discriminator-carryover prevention + CV-eval_set docs reframe (the eval_set-as-test-fold design is now honestly characterized as a pragmatic deviation, not a clean Optuna-WilcoxonPruner-inherited pattern). The eval_set Position B/C operational debate was deferred to a separate Tier-2 study.
- **PR-023 (this session's main work)** — time-aware CV for panel data; full 5-phase `PROCEDURE-pr-research.md` ran with 6 research questions (Q1, Q2, Q3, Q4, Q5, Q7 — Q6 absorbed into Q3) and 6 cited findings reports.

The decision to plan this as a v0.2 cut followed from `docs/VERSIONING.md §1`: PR-023 introduces ≥4 new config knobs + 1 new `Splitter` family, both MINOR triggers; per §2, `PROCEDURE-design-planning.md` from Phase 1 is required before any v0.2 implementation PR lands. This session compresses Phases 1–3 because PR-023's Phase 3 research already produced the decisions; Phase 4 (this file) records them.

### Decisions

Each decision below is anchored on cited research findings in `prs/PR-023-time-aware-cv-for-panel-data.md` Research findings section (Phase 3 round 1: 2026-05-18 → 2026-05-19). Status labels per `PROCEDURE-pr-research.md` Phase 3 vocabulary.

#### D1 — `PanelCombinatorialPurgedCV` new discriminator variant

- **Decision**: add a new variant to the `CVConfig` discriminated union. Folds are computed over **unique sorted timestamps** (not row indices); each timestamp-fold maps to all K asset rows; purge / embargo applied in **timestamp units** translated to rows internally. Preserves skfolio's combinatorial-path machinery via post-hoc `groupby(asset)` on the test-row index (row indices implicitly carry `(timestamp, asset)`).
- **Rejected alternatives**:
  - Pivot panel → wide `(timestamps × assets)` then run skfolio CPCV — loses per-asset y; degenerate on unequal histories; no production OSS precedent.
  - Switch to mlfinlab `StackedCombinatorialPurgedKFold` — mlfinlab is paid/licensed since v0.16; license verification required and rejected for the workbench's "free deps only" stance.
  - Docs-only single-asset-only — workbench's tabular scope is *typically* panel; punting is a usability cliff.
- **Status**: **convention** (≥2 cited systems: mlfinlab `StackedCombinatorialPurgedKFold` + Numerai era-wise CV both implement timestamp-atomicity + per-asset purge).
- **Research citation**: Q3 findings in `prs/PR-023-*.md` — read skfolio `_combinatorial.py:424-466` to prove CPCV is row-contiguous and non-panel-aware.

#### D2 — `TimeSeriesSplitCV` extended with `time_column` + polymorphic `embargo_time`

- **Decision**: extend the existing `TimeSeriesSplitCV` schema with optional `time_column: str | None` + `embargo_time: int | str | None` fields. `embargo_time` is polymorphic — accepts row-count `int` or a human-readable string like `"24h"` parsed via `pandas.Timedelta`. Existing `gap: int` stays for backward compatibility (and for single-asset row-aligned cases).
- **Rejected alternatives**:
  - New `PanelTimeSeriesSplitCV` variant — no cited production library separates panel CV into a discriminator variant. Inflates the union; redundant with the polymorphic-field approach.
  - Keep row-count only, add runtime warning — Position C from Q1. Rejected because it's the *exact* foot-gun that motivated this PR; "warn don't fix" is unsatisfying.
- **Status**: **convention** (sktime + Nixtla + Darts all expose polymorphic `int | timedelta | DateOffset` for splitter time fields).
- **Research citation**: Q1 findings in `prs/PR-023-*.md` — sktime `SlidingWindowSplitter` source + Nixtla `mlforecast.cross_validation` freq-aware integers + Darts `historical_forecasts(stride=...)`.

#### D3 — Ergonomic `target_horizon_bars` + `embargo_pct` on `CombinatorialPurgedCV`

- **Decision**: surface workbench-level `target_horizon_bars: int` + `embargo_pct: float` on `CombinatorialPurgedCV` (the existing skfolio-wrapping variant). Convert internally: `purged_size = target_horizon_bars` (symmetric, conservative-correct on regular bars) and `embargo_size = int(N * embargo_pct)` (matches AFML Snippet 7.3 verbatim). Defaults stay at `target_horizon_bars=0` / `embargo_pct=0.0` — "no embargo" is a meaningful operational state matching skfolio and AFML.
- **Rejected alternatives**:
  - Pass-through `purged_size` / `embargo_size` int with docs — user repeats horizon-to-rows conversion; risk of silent miscount; violates the workbench's soft convention of ergonomic knobs.
  - Workbench reimplements AFML interval-overlap purge — violates "reuse over reinvent" (`docs/CONSTRAINTS.md`); CPCV combinatorics are non-obvious to re-derive.
- **Status**: **convention** for the conversion formulas (mlfinlab + timeseriescv + AFML reference code all use AFML formulas); **proven** for regular-bar correctness; **best-guess-given-constraints** for irregular bars (out of scope — flagged in `docs/0.2/RESEARCH-BACKLOG.md` drift watch).
- **Research citation**: Q2 findings — skfolio `_combinatorial.py:81-220` (two-sided row-count semantics); AFML Snippet 7.3 `mbrg = int(X.shape[0] * pctEmbargo)`.

#### D4 — Accept skfolio's symmetric / row-count purge as conservative-correct

- **Decision**: keep skfolio as the CPCV backend. Its scalar two-sided `purged_size` is a row-count simplification of AFML's interval-overlap purge — drops MORE train rows than strictly needed but **never leaks**. Mandatory `docs/CONVENTIONS.md` note documents the precision gap and points users to mlfinlab/timeseriescv for interval-overlap-precise alternatives.
- **Rejected alternatives**:
  - Switch CPCV backend to mlfinlab — license issue (D1).
  - Reimplement interval-overlap purge in the workbench — scope/maintenance creep (D3).
- **Status**: **proven** (skfolio docstring lines 82-84 confirm conservative-correct behavior; tested for non-leakage in skfolio's own suite).
- **Research citation**: Q2 findings — `skfolio/model_selection/_combinatorial.py` docstring + lines 338-350 (purge before/after + embargo after).

#### D5 — Layered tests; panel fixtures designed in-house

- **Decision**: PR-023's tests use a 3-layer approach:
  1. **Primary**: per-asset hand-verified index fixtures — tiny synthetic panels (e.g., 3 assets × 6 timestamps, label_horizon=1, embargo_time=1) with expected train/test indices computed by hand and asserted via `np.array_equal`.
  2. **Property tests**: pairwise `np.intersect1d(train, test).size == 0` + per-asset `max(train_time) + embargo < min(test_time)` loop + label-overlap assertion.
  3. **Secondary smoke**: one shuffle-null end-to-end run asserting fold-mean RMSE within tolerance of `y.std()` (sklearn `permutation_test_score`-style tripwire — catches feature-side leakage that index assertions miss).
- **Rejected alternatives**:
  - Planted-signal panel — used in academic papers for backtest-overfitting research, NOT for splitter correctness; conflates model-capacity with splitter-bug; stochastic with magic thresholds.
  - Layer-3-only (shuffle-null primary) — sklearn's `permutation_test_score` is for model evaluation, not splitter unit testing; no surveyed library uses it as a correctness oracle.
- **Status**: **convention** for layers 1–2 (sklearn `test_time_series_gap` + skfolio `assert_split_equal` + timeseriescv `TestPurge` / `TestEmbargo` as pure-function tests); **best-guess-given-constraints** for layer 3 + panel-fixture design (no production-grade precedent for testing stacked-panel CV — confirmed by surveying skfolio / sktime / mlfinlab / sklearn test suites).
- **Research citation**: Q4 findings — direct URLs in `prs/PR-023-*.md`.

#### D6 — One-off temporal split is PR-024 (separate MINOR)

- **Decision**: the random-shuffle leak in `train_val_test_split` (`src/rux_ml/data/splits.py`) is real but orthogonal to PR-023's CV-path fix. Split to **PR-024**, a separate Tier-2 MINOR PR. PR-023 archives the Q5 research findings; PR-024 runs its own state assessment + light Phase 2 (research is mostly archived).
- **Rejected alternatives**:
  - Bundle in PR-023 — adds 4 sub-decisions (time_col, multi-series panel handling, validation warning, goldens); risks scope creep. Per memory `feedback_tier_inheritance`, multiple non-preference sub-decisions warrant a dedicated PR.
  - Out of v0.2 entirely — the leak is the same class of bug PR-023 fixes; ignoring it would be inconsistent.
- **Status**: **convention** for the eventual API shape (≥3 cited: sktime `temporal_train_test_split` + Darts `split_before/split_after` + AutoGluon TS + Nixtla + mlfinlab all use two separate functions, not a single function with a kind-knob).
- **Research citation**: Q5 findings in `prs/PR-023-*.md` Phase 3.

#### D7 — Walk-forward retrain helper is out of v0.2

- **Decision**: do NOT build a `walk_forward_retrain` / `rux-ml retrain --schedule` helper at v0.2. Periodic retrain stays user-script territory. Optional short `docs/CONVENTIONS.md` paragraph citing sktime `UpdateRefitsEvery` / Darts `retrain=int` as the conventional shape (post-`registry promote` workflow). Reconsider at v0.3+ if/when a real deployment use case lands.
- **Rejected alternatives**:
  - New CLI verb (`rux-ml retrain`) — violates `docs/CONSTRAINTS.md` "no server" constraint (scheduler concern); no production-grade workbench ships this.
  - Python helper (`walk_forward_retrain` function) — only sktime ships this shape; 3 of 4 surveyed (Darts, Nixtla, AutoGluon) delegate to user code. Risks phantom-implementation territory (no consumer in the workbench yet).
- **Status**: **convention** (≥3 cited systems: Darts + Nixtla + AutoGluon all delegate; only sktime wraps).
- **Research citation**: Q7 findings in `prs/PR-023-*.md` Phase 3.

### Deferred / acknowledged ambiguities

- **A1 — Embargo-magnitude defaults**: AFML doesn't pin a single value; skfolio defaults to 0; workbench defaults to 0 too. Users opt into AFML-recommended `embargo_pct ∈ [0.005, 0.02]` per problem. Same precedent as PR-015. Resolved in PR-023 docs; no further research needed.
- **A2 — Event-time / dollar bars**: the `target_horizon_bars → purged_size` identity assumes regular bars. Event-time / dollar bars need re-derivation. Tracked in `docs/0.2/RESEARCH-BACKLOG.md` drift watch; out of v0.2.
- **A3 — Panel-fixture pattern (D5 layer 3)** is in-house design with `best-guess-given-constraints` status. Revisit if community CV-test conventions evolve (e.g., if sktime ships hierarchical splitter tests).
- **A4 — Eval_set Position B/C debate** (deferred from PR-022 Phase 3) — research-archived in PR-022, separate Tier-2 PR when queued (target: PR-025).
- **A5 — One-off temporal split** (D6) — research-archived in PR-023, PR-024 owns it.

### v0.2 implementation plan

| PR | Status | Notes |
|----|--------|-------|
| PR-023 | open | Time-aware CV for panel data — main v0.2 work; full 5-phase research run 2026-05-18→2026-05-19 |
| PR-024 | queued | One-off temporal split (`train_val_test_split` + `data.split_kind`); research archived in PR-023 Q5 |
| PR-025 | queued | Eval_set Position B/C operational design study; research archived in PR-022 Phase 3 |
| PR-026 | queued | v0.2.0 version cut — analogous to PR-021; runs after PR-023..PR-025 merge on `dev` |

### Process notes

- This session **compressed Phases 1–3** of `PROCEDURE-design-planning.md` because PR-023's Phase 3 research had already produced the decisions. User approved the compressed flow explicitly on 2026-05-19.
- Per-Phase Approval Gate held: halts at Phase 1 close (Idea restatement), Phase 2/3 close (Decisions + Convergence summary), Phase 4 close (this doc commit) — 3 explicit user approvals.
- The eval_set research that surfaced during PR-022 + PR-023 work is the strongest argument yet for the research-discipline rule. Two design assumptions that turned out to be unsourced intuition (eval_set "antipattern"; skfolio CPCV "single-asset built") were caught only because PR-022 / PR-023 Phase 3 forced cited-source verification.
