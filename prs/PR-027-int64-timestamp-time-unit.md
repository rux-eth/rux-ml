# PR-027: Int64-timestamp time-unit handling in `TimeSeriesSplitter._effective_gap`

**Landed-in:** v0.2.0

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per memory `project_cv_strategy_tier2` — CV strategy machinery is universally Tier-2. The bug is concrete and reproduced; the design space (schema-field vs refusal vs auto-detect) needs Phase 2 scope-sharpening before implementation.

**Bump classification: PATCH** per `docs/VERSIONING.md` §1. Either fix shape (schema field added with `None` default OR a clearer error message) preserves all v0.2.0-pre-PR-026 behavior — the affected code path (`TimeSeriesSplitCV(embargo_time="<duration>", time_column="<Int64 col>")`) currently raises a sklearn ValueError, so no committed config exercises it. Ships under v0.2.0 alongside PR-023/24/25 (rolled in PR-026).

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_Populated by `PROCEDURE-pr-research.md`. Do not begin implementation until all required phases are complete._

**Surfaced during PR-025 Phase 2** (2026-05-19) when reading the PR-023 `TimeSeriesSplitter._effective_gap` implementation against the crypto-h3 dataset. PR-025 sidestepped via `PanelCombinatorialPurgedCV` (which operates on unique-timestamp indices and doesn't go through `_effective_gap`); the bug remained un-fixed at v0.2.0 cut-time. **Empirically reproduced 2026-05-19** on a synthetic crypto-h3-shaped panel — see "Empirical evidence" below.

### Empirical evidence (2026-05-19, pre-Phase-1)

Synthetic stacked panel: 100 assets × 200 daily timestamps; `embargo_time="24h"` (= 1 daily timestamp). Intended `_effective_gap`: 100 rows.

| `ts` dtype | suspect `cast(Datetime("ns"))` median Δ | computed `_effective_gap` | `.split()` outcome |
|---|---|---|---|
| `Int64` (Unix-seconds; crypto-h3 shape) | **86,400** (taken as ns ⇒ 86.4 μs) | **100,000,000,000** | sklearn raises `ValueError: Too many splits=5 for number of samples=20000 with test_size=3333 and gap=100000000000.` |
| `Datetime(time_unit='us')` | 86,400,000,000,000 | 100 (correct) | works; 48h per-asset gap |
| `Datetime(time_unit='ns')` | 86,400,000,000,000 | 100 (correct) | works; 48h per-asset gap |

**Bug characterization**: loud crash on `.split()` (not a silent leakage). The polars `.cast(pl.Datetime("ns"))` on an `Int64` column reinterprets the integers AS nanoseconds-since-epoch — no unit conversion. Reproduction script at `/tmp/v0.2-empirical/repro_int64_bug.py` (research artifact; not for commit).

### Phase 1 — State Assessment (2026-05-19)

**Current state** (HEAD = `47c4891`, PR-025 merge; PR-027 branch `pr-027/int64-timestamp-time-unit` cut from there):

- **Bug surface (1 file, 1 method)**: `src/rux_ml/data/cv.py:219-250` (`TimeSeriesSplitter._effective_gap`). Empirically reproduced above. No other splitter goes through `_effective_gap`:
  - `KFoldSplitter` / `StratifiedKFoldSplitter` / `GroupKFoldSplitter` — no time semantics.
  - `CombinatorialPurgedSplitter` — uses row-count `purged_size` / `embargo_size` (or the workbench's `target_horizon_bars` + `embargo_pct`); no duration-string parsing.
  - `PanelCombinatorialPurgedSplitter` — operates on unique sorted timestamps via `np.unique(X[time_column].to_numpy())`; the dtype just has to be `np.unique`-sortable. Polars Int64 + polars Datetime both sort correctly without a `cast(Datetime("ns"))` round-trip. **No bug** on this path. Confirmed by inspection of `src/rux_ml/data/cv.py:436-487`.
- **Schema surface (1 schema)**: `TimeSeriesSplitCV` in `src/rux_ml/config/cv.py:56-81` carries the polymorphic `embargo_time: int | str | None` + `time_column: str | None` fields added by PR-023 D2. Adding `time_unit` (Option A) goes here.
- **Factory dispatch (1 site)**: `make_splitter` at `src/rux_ml/data/cv.py:509-516` threads `cfg.time_column` + `cfg.embargo_time` through; adding `time_unit` is a one-line dispatch change (Option A) or zero (Option B).
- **Test coverage of `_effective_gap`** lives at `tests/data/test_panel_cv.py:366-432` (4 D2-polymorphic-`embargo_time` tests, plus the `int` / `None` fallback tests). **All four tests use `pl.Datetime` columns** — no Int64 case is exercised, which is why the bug shipped in PR-023.
- **No drift since PR-023 merge (`a66665b`, 2026-05-19)**: `git log src/rux_ml/data/cv.py src/rux_ml/config/cv.py tests/data/test_panel_cv.py` shows only PR-023's implementation commit; nothing else has touched `_effective_gap` or its tests. PR-024 (`3e1f2c0`) added `DataConfig.time_column` for the temporal one-off split — a different concern (the temporal split uses `pl.DataFrame.sort()` which is dtype-agnostic and unaffected).
- **Naming convention check**: polars (`pl.Datetime(time_unit=...)`, `pl.from_epoch(time_unit=...)`), pandas (`pd.Timestamp(unit=...)`), and pyarrow (`pa.timestamp("us")`) all use the term **`time_unit`** at the API boundary. If Option A wins, the field is `time_unit: Literal["ns", "us", "ms", "s"] | None = None` — matches the ecosystem.
- **Discriminator hygiene check (PR-022 backstop)**: `configs/base.toml [cv]` only carries `kind = "kfold"`. Adding `time_unit` to `TimeSeriesSplitCV` is base.toml-safe (variant-specific field, stays out of base). The PR-022 structural test continues to pass.

**Stale assumptions in PR draft**:

- None of consequence. The Empirical evidence section above and the two-option scope are accurate against current state.

**New constraints** (from PR-022/23/24/25):

- **Test pattern**: PR-023 D5 codified a three-layer test design (hand-verified fixtures + property + shuffle-null). PR-027's Int64 repro test should match the existing `tests/data/test_panel_cv.py` style (line 396-415's pattern for the `"2h"` case is the closest analog) — add an Int64 sibling test.
- **Tier inheritance** (per `feedback_tier_inheritance`): Phase 1 surfaces exactly **one** non-preference sub-decision (A vs B). Per the rule, this could be downgraded to Tier-1 light — but the (A) vs (B) call is a schema-surface design decision, not just a preference, so keeping **Tier-2 with light Phase 2** is the right call. Phase 3 web research is likely unnecessary (the bug is project-specific polars cast behavior, empirically reproduced; no external API drift to survey).
- **No prerequisite PRs**: PR-027 is independent of PR-026's version-cut work. The two can land in either order, but per the user's choice on 2026-05-19, PR-027 lands first so the `### Fixed` entry rolls into `[0.2.0]` cleanly.

**Exit decision**: zero drift; bug verified pre-Phase-1; decision space narrows to (A) vs (B). **Proceed to Phase 2 — light scope sharpening, no design-planning loop needed.** Tier-2 classification confirmed; full Phase 1-5 procedure runs but Phases 2-3 may be light.

### Phase 2 — Scope Research (2026-05-19, light)

Per memory `feedback_phase1_scope`: Tier-2 with no design-time research lock → Phase 3 must run, **targeted** at the open A-vs-B question. Phase 2 sharpens the question and lists conditional sub-decisions to resolve in Phase 4 from Phase 3 findings.

**Must-answer** (blocks implementation):

1. **Q1 — Option A (add `time_unit` field) vs Option B (refuse Int64 with cast guidance).**
   - **Success criteria**: ≥2 cited mature CV / time-series libraries (sktime, mlfinlab, skfolio, Darts, Nixtla, AutoGluon TS) and how they surface time-unit ambiguity when their input can be `Int64` vs `Datetime`. If a convention exists (≥2 systems agree on one shape), inherit it. If the field is genuinely split (B vs A), pick by workbench fit and label as **best-guess-given-constraints**.

**Conditional sub-decisions** (resolved in Phase 4 from Phase 1 + Phase 3 findings):

- **Q1a (if A wins)** — Default value of `time_unit`. Phase 1 lean (intuition, flagged per `feedback_research_discipline`): `None` (auto-detect from Datetime dtype; require explicit set on Int64).
- **Q1b (if A wins)** — Behavior when user sets `time_unit` on a `pl.Datetime` column. Three options: (i) silently ignored (Datetime carries its own unit, redundant); (ii) `UserWarning` (PR-022-style soft signal); (iii) hard `ValueError` (PR-024-style fail-fast). Lean (intuition, flagged): (iii) hard error — workbench convention is fail-fast validation per the PR-024 cross-field model_validator pattern; silent ignoring conflicts with the PR-022/24 explicit-error preference.
- **Q1c (if A wins)** — Accepted literal values. Phase 1 lean (intuition, flagged): `Literal["ns", "us", "ms", "s"]` only, matching polars `pl.Datetime` precision range. The longer durations (`"min"`, `"h"`, `"D"`) belong on `embargo_time` (already polymorphic), not `time_unit`.
- **Q1d (if B wins)** — Exact error message + cast guidance. Phase 1 lean (intuition, flagged): point users at `pl.from_epoch(pl.col(time_column), time_unit="s")` (polars-native cast), not a manual `* 1_000_000_000` int multiply.

**Round-2 follow-ups**: none expected. The convention question is independent of the sub-decisions — Phase 3 returns once, Phase 4 picks A or B, then locks Q1a-d from there.

**Excluded from this PR** (truly orthogonal):
- Auto-detect from Int64 magnitudes (Option C from the Scope section above). **Rejected pre-Phase-1** per `feedback_research_discipline` — heuristic without a citation.
- Extending the fix to `CombinatorialPurgedSplitter` — unaffected (uses row-count `purged_size`, not duration parsing).
- Extending the fix to `PanelCombinatorialPurgedSplitter` — Phase 1 confirmed it operates via `np.unique(...).to_numpy()` and is dtype-agnostic. No bug.
- Goldens for the Int64 path — the fix is testable via a single unit test on a synthetic panel; no library-drift surface a golden would capture beyond what the unit test already covers.

**Dispatch plan for Phase 3**:

- **1 light-touch agent**, single targeted question (Q1). Survey ≥4 of {sktime, mlfinlab, skfolio, Darts, Nixtla, AutoGluon TS}. Hard requirements: cited URLs/files only; ≥2 alternatives noted with pros/cons; explicit "no clean convention found" allowed if libraries truly split. Estimated 5-10 min wall time.

**Exit decision**: 1 must-answer question, targeted Phase 3 dispatch, light. Halt for approval before dispatching the agent.

### Phase 3 — Findings (2026-05-19)

1 light-touch agent dispatched (transcript captured). Survey verdict: **Option A inherits library convention (CONVENTION label, ≥2 production-grade systems agree)**. Option C has zero adopters across the surveyed libraries.

| Library | API surface | Integer-column handling | Evidence label | Citation |
|---|---|---|---|---|
| sktime | `SlidingWindowSplitter`, `ForecastingHorizon` (`fh: Union[int, list, ndarray, pd.Index]`); explicit `freq` only when caller passes datetime values | A — positional/relative steps default; integer indices first-class with row-count semantics | CONVENTION | [SlidingWindowSplitter v0.20.0](https://www.sktime.net/en/v0.20.0/api_reference/auto_generated/sktime.forecasting.model_selection.SlidingWindowSplitter.html) |
| mlfinlab | `PurgedKFold`, `CombinatorialPurgedKFold` — `pct_embargo: float`; embargo = `int(times.shape[0] * pctEmbargo)` row offset; `samples_info_sets` requires `DatetimeIndex` | A (de facto) — embargo is a row fraction; no time_unit knob | CONVENTION (AFML §7) | [Adv_Fin_ML_Exercises/snippets.py](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/src/features/snippets.py) |
| skfolio | `CombinatorialPurgedCV` — `purged_size: int`, `embargo_size: int` (both row-counts) | A — row-count only, no duration string | PROVEN (API signature) | [skfolio CombinatorialPurgedCV](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html) |
| Darts | `TimeSeries.split_before/after`, `historical_forecasts` — accepts datetime, integer, or float (proportion); RangeIndex `freq` = constant step size | A — integer IS the time axis, no translation; explicit dispatch on dtype | CONVENTION | [Darts TimeSeries API](https://unit8co.github.io/darts/generated_api/darts.timeseries.html) |
| Nixtla | `mlforecast.cross_validation`; `utilsforecast.validate_freq` raises `ValueError` on dtype/freq mismatch — `int dtype + non-int freq` or `dt dtype + int freq` both rejected | A enforced as a fail-fast typed error | PROVEN | [utilsforecast/validation.py](https://github.com/Nixtla/utilsforecast/blob/main/utilsforecast/validation.py) |
| AutoGluon TS | `TimeSeriesDataFrame` validation refuses non-`datetime64` columns; auto-attempts `pd.to_datetime` (silently coerces Int64 as ns — same footgun this PR fixes) | B — refuse integer; implementation has the exact bug-class our PR patches | PROVEN | [autogluon ts_dataframe.py](https://github.com/autogluon/autogluon/blob/master/timeseries/src/autogluon/timeseries/dataset/ts_dataframe.py) |

**Cross-library synthesis**: 4 of 6 libraries on Option A (sktime, mlfinlab, skfolio, Nixtla); 1 on Option B (AutoGluon, with a documented footgun in its implementation); 1 hybrid leaning A (Darts). **Option C — zero adopters**; both pandas `pd.to_datetime(..., unit=...)` and polars `pl.from_epoch(..., time_unit=...)` require explicit unit at the ecosystem-canonical primitive boundary.

**Disconfirming evidence**:

- AutoGluon's `pd.to_datetime(df[TIMESTAMP])` silent coercion is exactly the bug class this PR is patching — disconfirming case against silent coercion / Option C, NOT against Option B's intent.
- No library found auto-detecting time unit from integer magnitudes (Option C).
- No deprecation cycles found between A/B/C; libraries that picked A or B have stayed on that pick.

**Workbench-fit note**: Nixtla's `validate_freq` pattern (typed `ValueError` on dtype/freq mismatch) is structurally identical to PR-024's `RuxMLConfig._validate_split_kind_consistency` cross-field validator — the workbench already has this pattern in production. Option A inherits both library convention AND workbench convention.

### Phase 4 — Synthesis (2026-05-19)

**Outcome**: **Confirm with locked specifics** — Phase 3 convention is unambiguous. PR-027 implements Option A.

| Q | Lean (Phase 2) | Final decision (Phase 4) | Status |
|---|---|---|---|
| Q1 — A vs B | (no lean per `feedback_research_discipline`) | **Option A** — add `time_unit: Literal["ns","us","ms","s"] \| None = None` to `TimeSeriesSplitCV` | **CONVENTION** (4/6 libraries; Nixtla `validate_freq` + Darts RangeIndex `freq` are the strongest) |
| Q1a — Default | `None`, auto-detect from Datetime dtype | **`None`** — explicit set only required when column is `pl.Int64`; `pl.Datetime` columns ignore the field | CONVENTION (Nixtla's pattern: freq required only for int dtype) |
| Q1b — Datetime + `time_unit` set | hard `ValueError` (PR-024 fail-fast pattern) | **Hard `ValueError`** — explicit set on a `pl.Datetime` column is meaningless and signals user confusion; fail-loud matches PR-024 cross-field validator pattern | **BEST-GUESS** (Nixtla allows the consistent case `dt + duration freq` silently; we're stricter for workbench fail-fast hygiene) |
| Q1c — Accepted values | `Literal["ns","us","ms","s"]` matching polars precision range | **`Literal["ns","us","ms","s"]`** | CONVENTION (matches polars `pl.Datetime(time_unit=...)` and `pl.from_epoch(time_unit=...)` accepted values) |
| Q1d — B's cast guidance | (skipped — Option B not selected) | N/A | — |

**Changes to PR scope from Phase 3**:

- Lock Option A. Drop Option B and Option C from the scope.
- Field name `time_unit` (matches polars). Default `None`. Accepted: `Literal["ns","us","ms","s"]`.
- Behavior matrix:
  - `pl.Datetime` column + `time_unit=None`: works as today (preserves v0.2.0-pre-PR-027 behavior).
  - `pl.Datetime` column + `time_unit` set: raise `ValueError` ("`time_unit` is meaningless on a Datetime column; the column carries its own unit. Remove `time_unit` from the config.").
  - `pl.Int64` column + `time_unit=None`: raise `ValueError` ("`time_column` is `pl.Int64` and requires `time_unit` to be set on `TimeSeriesSplitCV`. Set one of `'ns'`, `'us'`, `'ms'`, `'s'` per your integer's interpretation.").
  - `pl.Int64` column + `time_unit` set: convert via `pl.from_epoch(col, time_unit=time_unit).cast(pl.Datetime("ns"))` then proceed with existing logic.

**Changes to ARCHITECTURE.md**: none — `_effective_gap` is below the Decision Rules level.

**Changes to CONVENTIONS.md**: one new bullet under the existing `TimeSeriesSplitCV.embargo_time` subsection — "When `time_column` is `pl.Int64` (e.g., Unix-seconds from a database column), set `time_unit` to the integer's interpretation. `pl.Datetime` columns carry their unit intrinsically; setting `time_unit` on them raises a `ValueError`."

**Changes to CONSTRAINTS.md**: none.

**New PRs that must come first**: none.

**Research-backed details now locked**:

- Field name + accepted values inherit polars convention.
- Behavior matrix inherits Nixtla's validate-on-mismatch pattern + workbench's PR-024 fail-fast convention.
- Q1b (Datetime + `time_unit` set raises) is the only sub-decision labeled **BEST-GUESS** — stricter than Nixtla; user-acknowledgement required at Phase 5 Gate.

### Phase 5 — Gate Check (2026-05-19)

- Premise still valid: ✓ (bug empirically reproduced pre-Phase 1 with concrete numerics)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-05-19 "leans approved" — Option A locked with Q1a-c CONVENTION + Q1b BEST-GUESS workbench-fit pick)
- Implementation cleared: ✓

### Phase 5 — Implementation outcomes (2026-05-19)

Mini state-assessment: zero days elapsed since Phase 4; `git log` on the target files (`src/rux_ml/data/cv.py`, `src/rux_ml/config/cv.py`, `tests/data/test_panel_cv.py`) shows only PR-023's implementation commit between Gate Check and implementation start — no drift.

**Code landed**:

- `src/rux_ml/config/cv.py`
  - **PR-027**: `TimeSeriesSplitCV` gains `time_unit: Literal["ns", "us", "ms", "s"] | None = None` (default preserves v0.2-pre-PR-027 behavior on `pl.Datetime` columns; required when `time_column` is `pl.Int64`). Docstring updated to describe the behavior matrix.
- `src/rux_ml/data/cv.py`
  - `TimeSeriesSplitter.__init__` accepts `time_unit: Literal["ns", "us", "ms", "s"] | None`.
  - `_effective_gap` gains the PR-027 behavior matrix: (a) `pl.Int64` + `time_unit=None` raises `ValueError` with cast guidance; (b) `pl.Datetime` + `time_unit` set raises `ValueError` (signals user confusion, fail-fast per PR-024 convention); (c) `pl.Int64` + `time_unit` set promotes via `pl.from_epoch(col, time_unit=time_unit)` before computing the median delta; (d) `pl.Datetime` + `time_unit=None` falls through to the existing path unchanged.
  - `make_splitter` factory threads `cfg.time_unit` through to `TimeSeriesSplitter`.
- `tests/data/test_panel_cv.py` — **4 new tests** for the PR-027 paths:
  - `test_time_series_int64_without_time_unit_raises` — pl.Int64 + time_unit=None → ValueError.
  - `test_time_series_int64_with_time_unit_translates_correctly` — pl.Int64 Unix-seconds + time_unit="s" produces bit-identical folds to the pl.Datetime equivalent (the structural correctness invariant).
  - `test_time_series_datetime_with_time_unit_set_raises` — pl.Datetime + time_unit set → ValueError.
  - `test_time_series_datetime_without_time_unit_unchanged` — pl.Datetime + time_unit=None bit-identical to pre-PR-027 behavior (regression check).
- `docs/CONVENTIONS.md` — new `TimeSeriesSplitCV.time_unit` subsection under the existing `TimeSeriesSplitCV.embargo_time` block.
- `CHANGELOG.md [Unreleased] ### Fixed` — PR-027 entry (rolls into `[0.2.0]` at PR-026).
- `docs/0.2/ROADMAP.md` PR-027 row flipped `[ ]` → `[x]` per `feedback_roadmap_flip_in_pr`.
- `docs/0.2/RESEARCH-BACKLOG.md` PR-027 row marked `fully-researched 2026-05-19` + `implementation-cleared 2026-05-19`.

**Test outcomes**:
- `uv run pytest` — **367 passed, 1 skipped, 15 deselected** (was 363 on `47c4891`; +4 new PR-027 tests).
- `uv run ruff check .` — clean (verified).
- `uv run basedpyright src/` — clean (0 errors).

**Deferred / out of scope**:
- AutoGluon-style integer column refusal (Option B) — Phase 3 confirmed AutoGluon's implementation has the exact silent-coercion footgun this PR patches; Option A is strictly more expressive.
- Auto-detect from integer magnitudes (Option C) — zero library adopters; rejected pre-Phase-1.
- Extending the fix to longer Polars time durations (e.g., `"d"`) — `pl.from_epoch` accepts `"d"` but `pl.Datetime` doesn't; out of scope.

**Followup tracker for future contributors**: this PR shipped under v0.2.0 alongside PR-022 / PR-023 / PR-024 / PR-025 via PR-026's version-cut commit. `Landed-in:` flips to `v0.2.0` at PR-026.

---

## Scope

Two-option decision space (pick one in Phase 4 Synthesis):

### Option A — explicit `time_unit` declaration on `TimeSeriesSplitCV`

Add a `time_unit: Literal["ns", "us", "ms", "s"] | None = None` field. Semantics:

- `pl.Datetime` columns: ignore (the column already carries its unit); `time_unit` is silently accepted for forward compat or rejected with a clear error.
- `pl.Int64` columns: `time_unit` MUST be set; otherwise raise a clear `ValueError` at split time pointing at this field.

Schema surface: 1 new field on `TimeSeriesSplitCV` schema; no change to `PanelCombinatorialPurgedCV` (which doesn't go through `_effective_gap`).

Pros: explicit, no silent magic, matches how pandas / polars / arrow surface this distinction at the API boundary.

Cons: extra config knob; user must remember to set it when using Int64.

### Option B — refuse `Int64` with a clear error message

Don't change the schema. At split time, if `X[time_column].dtype` is `pl.Int64`, raise:

```
TimeSeriesSplitCV.embargo_time as duration string requires time_column to
have a Polars Datetime dtype with an explicit time unit; got pl.Int64. Cast
the column upstream via df.with_columns(pl.from_epoch(pl.col("<col>"),
time_unit="s")).
```

Schema surface: 0 changes. One added validation branch in `_effective_gap`.

Pros: minimal surface; nudges users toward the better data hygiene of carrying explicit-unit Datetime columns; consistent with the "cast at the boundary" pattern.

Cons: shifts the unit declaration onto the user's data-loader code rather than the CV config; users with Int64 timestamps must cast every time.

### Option C — auto-detect from magnitude (rejected upfront)

Heuristic: Int64 ~1e9 ⇒ seconds, ~1e12 ⇒ ms, ~1e18 ⇒ ns. **Rejected** per memory `feedback_research_discipline` — the heuristic would silently miscount for any out-of-range dataset, and "the explicit `time_unit` field is one keystroke." Not worth the magic.

## Dependencies

None. Built on v0.2-dev `47c4891` (PR-025 merge).

## Architecture section implemented

None — bug fix + test. `docs/CONVENTIONS.md` CV-strategy subsection gains a one-line note on Int64-Unix-seconds time columns (Phase 4 will decide which option's text lands).

## Verification criteria

To be sharpened in Phase 2. Initial sketch:

- [ ] Repro test in `tests/data/test_cv.py`: a stacked panel with `pl.Int64` Unix-seconds `time_column` + `embargo_time="24h"` produces the **correct** `_effective_gap` (no `1e11`) — or under Option B, raises a typed error.
- [ ] Existing `pl.Datetime` paths unchanged (no regression in current panel-CV tests).
- [ ] `CHANGELOG.md [Unreleased] ### Fixed` entry (rolls into `[0.2.0]` at PR-026).
- [ ] `docs/CONVENTIONS.md` gains a one-line note (Phase 4 picks the wording).
- [ ] `docs/0.2/ROADMAP.md` PR-027 row flipped `[ ]` → `[x]` in the same commit.

## Research backing

Tier-2 — but research surface is narrow:
- The bug is project-specific polars cast behavior (verified empirically). No external API drift to survey.
- Phase 3 web research may be light or unneeded; the design decision is (A) vs (B) on a single schema knob.

Anchored on:
- Empirical repro above (synthetic crypto-h3-shaped panel).
- `src/rux_ml/data/cv.py:219-250` (`TimeSeriesSplitter._effective_gap`).
- `configs/problems/crypto_breakout_h3.toml:25` (documents that crypto-h3 `timestamp` is Int64 Unix-seconds).
- Polars docs for `pl.from_epoch` and `pl.Datetime` casts.

## Notes

- **PATCH bump (v0.2.0)** per `docs/VERSIONING.md` §1: bug fix with no design-cycle work. Ships under v0.2.0 alongside PR-023/24/25 (CHANGELOG `[0.2.0] ### Fixed` entry; rolled by PR-026's version cut).
- **`feedback_roadmap_flip_in_pr` rule**: PR-027's implementation commit must flip `docs/0.2/ROADMAP.md` PR-027 row `[ ]` → `[x]` in the same commit. The row is added to the v0.2 ROADMAP as part of this PR's scaffolding commit (or merged into the implementation commit).
- **Tier reclassification possible at Phase 1 exit**: if Phase 1 surfaces only the (A)-vs-(B) sub-decision and no other drift, this could be downgraded to a Tier-1 light procedure per `feedback_tier_inheritance`. Default stays Tier-2 until Phase 1 reports.
