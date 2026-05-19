# PR-023: Time-aware CV for panel data — time-unit embargo + label-overlap purge

**Landed-in:** v0.2.0

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per memory `project_cv_strategy_tier2` (CV strategy work is universally Tier-2 in this project — proper AFML purge/embargo sizing, panel-data semantics, and skfolio CPCV behavior on stacked panels all need fresh research).

**Bump classification (PATCH vs MINOR) is deferred to Phase 4 Synthesis.** The stub previously pre-committed to MINOR, but that was premature. Per `docs/VERSIONING.md` §1, MINOR is triggered ONLY if the fix adds a new config knob, a new Splitter family, or new user-visible behavior. If research-backed synthesis finds a pure-docs / pure-internal solution, this is PATCH (no design session required). If a schema-surface change wins, this is MINOR and `PROCEDURE-design-planning.md` v0.2 session must run before implementation per §2. Either way, Phases 1-3 of `PROCEDURE-pr-research.md` run first and the call is made at Phase 4 from the actual cited findings.

Skipping the PR research procedure (regardless of bump classification) is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_Populated by `PROCEDURE-pr-research.md`. Do not begin implementation until all required phases are complete._

**Surfaced during the first real-dataset run** on 2026-05-18 (chat session `f085a0c0-afc8-4386-8e48-be8acb810c12`). The investigation confirmed the v0.1.0 CV machinery silently under-embargoes any panel data with forward-looking targets; the resulting metric is meaningfully better-than-real on any stacked-panel problem (not just the crypto OHLCV case that happened to surface it).

### State Assessment (2026-05-18)

**Current state** (HEAD = `027c3b0`, post-PR-022 merge):

- **5 Splitter implementations** in `src/rux_ml/data/cv.py` — all stateless wrappers around sklearn (KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold) + skfolio (CombinatorialPurgedCV). `make_splitter` dispatcher uses `match/case` on `cfg` type. **No panel-aware Splitter exists.**
- **`CombinatorialPurgedSplitter`** wraps `skfolio.model_selection.CombinatorialPurgedCV` and exposes `purged_size: int` + `embargo_size: int` (passed through opaquely). Flattens skfolio's `(train, list[test_path])` to single `(train, test_idx)` for sklearn-shape conformance; per-path decomposition is "out of scope for v0" per the wrapper's docstring.
- **`train_val_test_split`** in `src/rux_ml/data/splits.py` is `df.sample(fraction=1.0, shuffle=True, seed=...)` — random shuffle. Used by `rux-ml train`. Confirmed unchanged. PR-023 Q5 framing is correct.
- **Existing CV tests** in `tests/data/`: `test_cv.py` (203 lines, mostly contract tests), `test_cv_leakage.py` (119 lines, time-leakage + group-leakage). **No panel-data tests; no calibration tests.**
- **`docs/ARCHITECTURE.md` "Decision Rules" section** (line 113) covers CatBoost ingest + categorical handling + LightGBM ingest + categorical handling. **No CV-strategy decision rule exists.** Adding one is genuine new content for PR-023.
- **`docs/ARCHITECTURE.md:382-383`** has a CV-comparison table mentioning `TimeSeriesSplit` (gap + non-shuffled) and `CombinatorialPurgedCV` (purge + embargo, AFML §7.4.2). No panel awareness.
- **`docs/CONVENTIONS.md:199`** has a CV-default-by-data-shape table that already mentions `CombinatorialPurgedCV` for "variable horizon labels, overlapping" with `embargo_size ∈ [0.005·N, 0.02·N]` per AFML §7.4.2.
- **`docs/CONVENTIONS.md:206`** (added by PR-022 just now): explicit `TimeSeriesSplitCV.gap` row-vs-time subsection with forward-reference to PR-023.
- **`docs/0.1/DESIGN-log.md`** has zero panel-CV discussion (Q1–Q6 + A1–A6 don't touch it). v0.1 design didn't anticipate this.
- **`src/`-wide grep** for "panel" / "stacked" returns nothing in `data/` or `config/` — the workbench has no panel-data concept at code level. The crypto-h3 dataset was the first panel exercise.

**Assumptions at PR draft time** (PR-023 stub authored before any code reading):

- The stub assumed PR-023 would need to add a `CombinatorialPurgedCV` subsection to `docs/CONVENTIONS.md`. **Stale**: that surface already exists at line 199. PR-023's docs scope is narrower than the stub anticipated.
- The stub assumed PR-023 would need to add a `TimeSeriesSplitCV.gap` row-vs-time docs subsection. **Stale**: PR-022 just landed this (CONVENTIONS.md:206) with explicit forward-reference to PR-023. The forward-reference now needs PR-023 to actually ship the structural fix.
- The stub asserted "skfolio's CPCV was built for single-asset time series" as fact in Q3. **Unverified intuition** — flagged per memory `feedback_research_discipline`. Phase 3 must confirm or refute via skfolio source.
- The stub assumed the eval_set / test-fold concern was bundled here. **Stale**: separated into its own PR-022-Phase-3 research round (already complete, docs reframe shipped in PR-022). PR-023 does not own that question anymore.
- The stub pre-committed to MINOR + v0.2 design session. **Premature** (per the procedural correction just landed in this stub's "Before Implementation" section). Bump classification deferred to Phase 4 Synthesis.

**Stale assumptions** (where current state disagrees with PR-023's draft):

- PR-023 Item "Document `TimeSeriesSplit.gap` row-vs-time semantics" is **DONE upstream by PR-022**. Remove from this PR's scope.
- PR-023's anticipated `docs/CONVENTIONS.md` additions need re-scoping — most of what was planned is now either landed (gap docs) or already-present (CPCV in the default-by-shape table). The genuinely-new docs work is panel-data-CV decision rule + the actual fix's behavior documentation.
- None of the stale assumptions change PR-023's premise — they narrow scope, not invalidate it. **No design-planning loop needed.** Proceed.

**New constraints** (learned from prior PRs + codebase evolution):

- **base.toml discriminator-hygiene structural test** (PR-022): any new field on `[cv]` schema (e.g., `time_column`, `embargo_time` on `TimeSeriesSplitCV` per Q1-A) must NOT appear in `base.toml`. The test catches it. Either: (a) keep new fields out of base, OR (b) add field to every variant if truly common.
- **Discriminator-union tripwire test** (PR-022): if PR-023 adds a NEW Splitter variant (`PanelTimeSeriesSplitCV` per Q1-B / `PanelCombinatorialPurgedCV` per Q3-A), it auto-flows through `make_splitter`'s `match/case` AND through `DISCRIMINATED_TABLES` introspection. No new test maintenance for the structural test itself.
- **`CombinatorialPurgedSplitter` flatten-to-(train, test) shape** is a v0 simplification. If PR-023's panel-CPCV needs per-path decomposition (e.g., to score per-asset paths separately for panel calibration), the wrapper shape changes — scope-expansion warning for Phase 4.
- **`skfolio.CombinatorialPurgedCV` is loaded lazily** (line 257-260 in cv.py — `from skfolio.model_selection import ... # noqa: PLC0415`). If PR-023 introduces panel-CPCV as a new variant, decide whether it shares this lazy-load pattern.
- **Memory `feedback_research_discipline`** (re-affirmed by the PR-022 eval_set incident): each of PR-023's 5 open questions MUST be answered with cited sources in Phase 3. No "intuition labeled as convention."

**Exit decision**: stale assumptions narrow scope (good), don't invalidate premise (good). New constraints from PR-022 are additive guardrails, not blockers. **Proceed to Phase 2.**

### Phase 2 — Scope Research (2026-05-18)

Sharpening the PR-023 draft's Q1–Q5 against current state + classifying MUST-ANSWER vs NICE-TO-HAVE per `PROCEDURE-pr-research.md` Phase 2.

**Must-answer** (blocks implementation; expanded per 2026-05-19 user-feedback that multi-round research is preferred over deferral):

1. **Q1 — Time-unit embargo API on `TimeSeriesSplitCV`.** Per the live finding: `gap` is row-count; on panel data this collapses to <1 hour of per-asset embargo. Need an API that takes embargo in time-units (bars / hours / days / whatever the user's index dimension is).
   - **Success criteria**: a concrete API decision (extend `TimeSeriesSplitCV` with `time_column` + `embargo_time` fields per Q1-A; add `PanelTimeSeriesSplitCV` variant per Q1-B; OR keep gap-row-count and add a runtime warning per Q1-C). Backed by **≥2 cited production-grade CV systems** that expose time-unit embargo (candidates: skfolio, mlfinlab, sklearn, sktime, tsfresh).

2. **Q2 — Label-overlap purge for forward-looking targets.** Per AFML Ch. 7: train rows whose target windows overlap test indices must be PURGED from train, otherwise leak. The workbench's `CombinatorialPurgedSplitter` passes `purged_size: int` to skfolio opaquely; `TimeSeriesSplitCV` has no purge field at all. Open: skfolio `purged_size` semantics (bars / one-sided / two-sided)? AFML §7.4.2 formula? Workbench-level knob (`target_horizon_bars`) vs pass-through?
   - **Success criteria**: skfolio source confirmation of `purged_size` semantics. Cited AFML chapter for the purge formula. Decision: workbench-level knob + conversion formula OR pass-through + docs.

3. **Q3 — Panel CPCV semantics.** Confirm or refute the stub's "skfolio CPCV is single-asset" claim via skfolio source. If single-asset, decide between: new `PanelCombinatorialPurgedCV` variant, pre-CV pivot helper, or single-asset-only documentation.
   - **Success criteria**: skfolio source confirmation. **≥2 cited mature systems** (mlfinlab, QuantConnect, quant-finance OSS) shipping panel-CPCV — what's their API?

4. **Q4 — Calibration test methodology for stacked-panel CV.** Two competing approaches in literature: (a) **synthetic planted-signal panel** — generate data with known signal, verify CV captures it; (b) **randomized-target null model** — shuffle `y`, verify properly-embargoed CV produces RMSE ≈ target_std. Open: which is the convention in mature systems? Is there a third approach we missed?
   - **Success criteria**: cite ≥2 production-grade or peer-reviewed sources demonstrating panel-CV correctness tests. Pick (a) / (b) / both / something else, with rationale.

5. **Q5 — `rux-ml train` random-shuffle on time-series.** `train_val_test_split` (used by `rux-ml train` one-off path) is `df.sample(shuffle=True, seed=...)`. For time-series problems this leaks. Open: what do mature workbenches (sktime, mlfinlab, AutoGluon time-series, Darts, Nixtla) do for their one-off baseline path on temporal data?
   - **Success criteria**: cite ≥2 production-grade systems' one-off-baseline split policy for time-series. Decide: bundle a fix into PR-023 (new `DataConfig.split_kind` knob) or split to a separate PR-024 (research is archived in PR-023 either way).

6. **Q7 — Walk-forward retrain vs single-CV-fit for deployment.** How does the CV-correctness story map to live deployment? Practitioners typically pick HPs via CV, then retrain on full data once — but for time-series this may need walk-forward retrain. Open: what's the convention? Is there a workbench-level helper worth building?
   - **Success criteria**: ≥2 cited systems' CV→deployment patterns. Decision: in scope for v0.2 / separate concern / docs-only.

**Round-2 follow-ups** (dependent on Round-1 findings):

7. **Q3 follow-up** runs after Q2 returns — purge mechanism informs panel-CPCV design.
8. **Q6 — Per-fold CPCV path decomposition** runs only if Q3 surfaces a panel-CPCV design that requires per-asset path scoring. Conditional dispatch.

**Excluded from this round** (truly orthogonal):

- **Eval_set / Position-B-C debate** — owned by a separate future Tier-2 PR per PR-022's research outcome.
- **Bump classification (PATCH vs MINOR)** — deferred to Phase 4 Synthesis (versioning-policy mapping, not research).

**Dependencies:**

- **Q3 depends on Q2** (purge mechanism informs panel-CPCV design).
- **Q1, Q4, Q5, Q7 independent** of each other and of Q2/Q3.

**Dispatch plan for Phase 3:**

- **Round 1 (parallel)**: 5 agents for Q1, Q2, Q4, Q5, Q7.
- **Round 2 (sequential after Q2 returns)**: 1 agent for Q3; conditional 1 agent for Q6 if Q3 forces it.
- Estimated 15-25 min total wall time.

Each agent gets the same hard requirements as the PR-022 eval_set agent: cited URLs/files only, ≥2 alternatives with pros/cons per question, disconfirming-evidence search, status labels (proven / convention / best-guess-given-constraints), explicit "no clean convention found" allowed.

### Phase 3 — Findings (2026-05-18 → 2026-05-19)

6 research questions dispatched (Q1, Q2, Q4, Q5, Q7 in parallel Round 1; Q3 sequential Round 2 after Q2). Full per-question agent reports captured in the chat session transcript at `f085a0c0-afc8-4386-8e48-be8acb810c12`; the canonical synthesis lives in [`docs/0.2/DESIGN-log.md`](../docs/0.2/DESIGN-log.md) D1–D7.

| Q | Verdict | Status | Cites | Locked decision |
|---|---|---|---|---|
| Q1 | Two conventions coexist (sktime/Nixtla/Darts polymorphic units vs sklearn/skfolio/mlfinlab row-count). Workbench picks polymorphic per Option A. | convention | sktime `SlidingWindowSplitter` source; Nixtla `mlforecast.cross_validation`; Darts `historical_forecasts` | D2 — `TimeSeriesSplitCV` extended with `time_column` + polymorphic `embargo_time: int \| str` |
| Q2 | skfolio `purged_size` = two-sided int rows; `embargo_size = int(N * pct)` per AFML Snippet 7.3. Interval-overlap is canonical AFML but skfolio simplifies. | convention | skfolio `_combinatorial.py:81-220`; AFML Snippet 7.3 (BlackArbsCEO + WongYatChun mirrors); mlfinlab + timeseriescv interval-overlap | D3 — surface workbench `target_horizon_bars` + `embargo_pct` knobs |
| Q3 | skfolio CPCV is NOT panel-aware (`_combinatorial.py:424` — contiguous row ranges; no sort, no duplicate-timestamp handling). Panel-CPCV convention is dict-of-assets + per-asset purge (mlfinlab StackedCPCV + Numerai era-CV). | proven (non-panel) / convention (recommended pattern) | skfolio `_combinatorial.py:424-466`; mlfinlab `StackedCombinatorialPurgedKFold` line 108-151; Numerai era-wise CV forum | D1 — new `PanelCombinatorialPurgedCV` discriminator variant |
| Q4 | Hand-verified index assertions + property tests + purge/embargo unit tests is the convention. Planted-signal is academic. Panel-aware fixtures NOT FOUND in any of 4 surveyed test suites. | convention (layers 1-2) / best-guess (layer 3 + panel fixture) | sklearn `test_split.py` line 1557-1607; skfolio `test_combinatorial.py:20-349`; timeseriescv `test_cross_validation.py:171-226` | D5 — layered tests; panel-fixture pattern designed in-house |
| Q5 | Two functions, not one knob (convention across ≥3 mature systems). Quantified leakage: 19-21% RMSE inflation under leaky splits (arxiv 2512.06932). | convention | sktime `temporal_train_test_split`; Darts `split_before/split_after`; AutoGluon TS; Nixtla; mlfinlab; arxiv 2512.06932 | D6 — research archived for PR-024; **out of PR-023 scope** |
| Q7 | Periodic retrain stays outside the library (3 of 4 surveyed delegate to user code; only sktime ships `UpdateRefitsEvery`). | convention | sktime + Darts + Nixtla + AutoGluon TS | D7 — **out of v0.2 entirely**; optional CONVENTIONS.md note |

### Phase 4 — Synthesis (2026-05-19)

**Outcome**: **Confirm with locked specifics** — research confirms PR-023's premise (panel data needs time-aware CV); locks the specific shape via D1–D7 in [`docs/0.2/DESIGN-log.md`](../docs/0.2/DESIGN-log.md). No premise invalidation.

**Changes to this PR** from research:
- Item 1: TimeSeriesSplit `gap` row-vs-time docs subsection **removed** from PR-023 scope — already landed in PR-022 (CONVENTIONS.md:206 with explicit forward-reference to PR-023). PR-023's job is to satisfy that forward-reference with the actual schema fix (D2).
- Item 5: walk-forward retrain helper **removed** from PR-023 scope (D7 — out of v0.2).
- Item 6: `train_val_test_split` random-shuffle fix **split** to PR-024 (D6).
- Items 1-4 (per `## Scope` below) **kept**, refined per D1–D5.

**Changes to ARCHITECTURE.md**:
- New entry in `## Decision Rules` for **CV-strategy-by-data-shape** (single-asset vs stacked panel; row-count gap vs time-unit embargo). Lands with PR-023 implementation.

**Changes to CONVENTIONS.md**:
- New mandatory subsection on **skfolio's symmetric/coarser purge vs AFML interval-overlap** — documents the precision gap accepted in D4. Points users to mlfinlab/timeseriescv for interval-overlap-precise alternatives.
- Updates to the existing CV-default-by-data-shape table (CONVENTIONS.md:199) — add panel row.

**Changes to CONSTRAINTS.md**:
- None.

**New PRs that must come first**:
- None. PR-024 / PR-025 / PR-026 are queued **after** PR-023 in `docs/0.2/ROADMAP.md`.

**Research-backed details now locked in this PR**:
- D1 (`PanelCombinatorialPurgedCV` new variant) + D2 (polymorphic `embargo_time` extension) + D3 (`target_horizon_bars` + `embargo_pct` workbench knobs) + D4 (accept skfolio's coarser model) + D5 (layered tests + panel-fixture in-house) — all in `docs/0.2/DESIGN-log.md`.

### Phase 5 — Gate Check (2026-05-19)

- Premise still valid: ✓ (research confirms panel data needs time-aware CV)
- No prerequisite PRs surfaced: ✓ (PR-024 / PR-025 / PR-026 are post-PR-023, not blockers)
- User approved updated spec: ✓ (2026-05-19 Phase 3 convergence)
- Implementation cleared: ✓

### Phase 5 — Implementation outcomes (2026-05-19)

Mini state-assessment: zero days elapsed since Phase 5 Gate Check; `git log` on the target files (`src/rux_ml/config/cv.py`, `src/rux_ml/data/cv.py`, `src/rux_ml/tuning/objective.py`, `configs/base.toml`, `tests/data/`, `tests/config/test_base_toml_discriminator_hygiene.py`) shows no commits between Gate Check (`c8f4fc6`) and implementation start — no drift to handle.

**Code landed** (D1–D7 specifics from the locked design):

- `src/rux_ml/config/cv.py`
  - **D1**: `PanelCombinatorialPurgedCV` new discriminator variant added to the `CVConfig` tagged union — fields `n_folds`, `n_test_folds`, **`time_column: str`** (required), **`asset_column: str`** (required), `target_horizon_bars: int = 0`, `embargo_pct: float = 0.0`. `kind = "panel_cpcv"`.
  - **D2**: `TimeSeriesSplitCV` extended with `time_column: str | None = None` + `embargo_time: int | str | None = None`. Defaults preserve v0.1.1 row-count behavior.
  - **D3**: `CombinatorialPurgedCV` extended with `target_horizon_bars: int = 0` + `embargo_pct: float = 0.0`. Existing `purged_size` / `embargo_size` fields kept; row-count siblings win when both are non-zero (strict opt-in).
- `src/rux_ml/data/cv.py`
  - `PanelCombinatorialPurgedSplitter` (D1) — extracts unique sorted timestamps from `time_column`; runs `skfolio.CombinatorialPurgedCV` on the timestamp axis (`purged_size = target_horizon_bars`, `embargo_size = int(n_unique_ts * embargo_pct)`); maps each timestamp-level fold back to row indices via `np.searchsorted(unique_ts, times)`. Per-asset purge is automatic (timestamp-atomic). Validates `time_column` + `asset_column` presence at split time; raises if `n_unique_ts < n_folds`.
  - `TimeSeriesSplitter` (D2) — inner `TimeSeriesSplit` is now built at `.split()` time (no longer in `__init__`) so the effective `gap` can be computed from the input DataFrame's timestamps. `_effective_gap(X)` computes `ceil(pd.Timedelta(embargo_time) / median_delta) * (X.height / n_unique_ts)` for string durations; row-count `int` and `None` paths preserve v0.1.1 behavior exactly (validated by `test_time_series_embargo_time_none_falls_back_to_gap`).
  - `CombinatorialPurgedSplitter` (D3) — inner skfolio splitter built at `.split()` time so `embargo_pct` resolves against actual `N`. Conversion: `effective_purged_size = purged_size or target_horizon_bars`; `effective_embargo_size = embargo_size or int(N * embargo_pct)`. Backward-compat asserted by `test_cpcv_ergonomic_knob_defaults_preserve_v0_1_behavior` + `test_cpcv_row_count_wins_when_both_set`.
  - `make_splitter` dispatcher extended with `PanelCombinatorialPurgedCV` case; threads all new fields through unchanged.
- `src/rux_ml/data/__init__.py` + `src/rux_ml/config/__init__.py` — re-exports updated.
- `tests/data/test_panel_cv.py` (NEW; 20 tests, D5 three-layer design)
  - Layer 1 — hand-verified fixtures: 3 assets × 9 timestamps with C(3,2)=3 splits; per-asset purge timestamp-atomicity assertion on 4 assets × 16 timestamps.
  - Layer 2 — property tests: pairwise-disjoint train/test; no timestamp appears in both partitions; `C(n_folds, n_test_folds)` splits; missing-column / too-few-timestamps validation; ergonomic-knob equivalence to row-count siblings; `embargo_time` int / str / None paths.
  - Layer 3 — shuffle-null tripwire: 4 assets × 40 timestamps random target; constant-mean predictor; fold-mean MSE within `[0.6, 1.6]·var(y)` (no-skill baseline) — catches feature-side leakage that pure index assertions miss.
- `docs/ARCHITECTURE.md` — new `### CV strategy by data shape (per PR-023 D1/D2)` entry in the `## Decision Rules` section + per-strategy leakage-guarantees table updated with `TimeSeriesSplit.embargo_time` row and new `PanelCombinatorialPurgedCV` row.
- `docs/CONVENTIONS.md` — CV-default-by-shape table gains stacked-panel rows; `TimeSeriesSplitCV.embargo_time` polymorphic semantics subsection; `PanelCombinatorialPurgedCV` subsection; **mandatory skfolio purge precision gap note (D4)** documenting conservative-correctness vs AFML interval-overlap and pointing users to mlfinlab/timeseriescv for interval-precise alternatives.
- `CHANGELOG.md` `[Unreleased]` — new `### Fixed` entry for stacked-panel under-embargo + `### Added` entries (new variant, ergonomic knobs, docs additions, tests) + `### Changed` entry for `TimeSeriesSplitCV` polymorphic embargo.
- `docs/0.2/ROADMAP.md` — PR-023 row flipped `[ ]` → `[x]` per `feedback_roadmap_flip_in_pr`.

**Test outcomes**: full default suite `347 passed, 1 skipped, 15 deselected` (was 327 passed on the merge-base `c8f4fc6`; +20 new panel-CV tests). Ruff + basedpyright clean (verified separately).

**Deferred (still owned by sibling PRs)**:
- `train_val_test_split` random-shuffle leak → PR-024 (untouched here; D6).
- `eval_set` Position B/C operational debate → PR-025 (untouched here; A4).
- Walk-forward retrain helper → out of v0.2 (D7). Optional CONVENTIONS.md citing-paragraph not added at this PR (defer until a real deployment use case lands; A3 ambiguity preserved).

**ExtMem compatibility decision** (in-PR): `PanelCombinatorialPurgedSplitter.extmem_compatible = False`. The splitter must read the full `time_column` at split time to compute the timestamp-axis fold; the file-level ExtMem path is incompatible. Documented in CONVENTIONS.md.

---

**Open research questions** (HISTORICAL — resolved by Phase 3 above; preserved for traceability):

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

- **Bump classification deferred to Phase 4 Synthesis.** MINOR/PATCH depends on which fix option wins (see Open Question 1 + 2 + 3 + 5). If a schema-surface change lands, MINOR + v0.2 design session per `docs/VERSIONING.md` §2. If pure-docs / pure-internal, PATCH.
- **v0.2 design session required ONLY if Phase 4 classifies this as MINOR.** Design session may surface additional v0.2 work that should land alongside if triggered.
- **Family-agnostic by construction**: the Splitter operates on row indices and an optional `time_column` / `asset_column` — none of the trainer families enter the CV-correctness picture. The verification criteria parametrize over `TRAINER_FAMILIES` to enforce family-blindness, but the underlying fix has no per-family code path.
- **The crypto-h3 baseline trained on 2026-05-18 (RMSE=0.02547 at cdbffd2) is leakage-contaminated.** Once PR-022 ships the eval_set fix and PR-023 ships time-aware embargo, that model's metric must be re-computed. The original number stays in the runs DB as a historical artifact (per memory `feedback_pr_spec_historicity`).
- Per memory `project_cv_strategy_tier2`, **K-fold / walk-forward / CPCV / GroupKFold all need dedicated Tier-2 PRs with full research**. This PR covers the panel-data + time-aware case; other CV strategy revisits (per-problem CPCV calibration, walk-forward live-deployment helper, etc.) remain separate work.
- PR-022 is the sibling PATCH that addresses the eval_set / test-fold leakage and discriminator-hygiene prevention; that lands first.
- **`feedback_roadmap_flip_in_pr` rule applies once the v0.2 ROADMAP exists** (created during the v0.2 design session). Until then, no row to flip — call this out in implementation commits.
