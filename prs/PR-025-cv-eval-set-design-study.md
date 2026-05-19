# PR-025: CV eval_set Position B/C operational design study

**Landed-in:** v0.2.0 (rolled in PR-026)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per memory `project_cv_strategy_tier2`. **Pattern surveyed** by PR-022 Phase 3 archived research; this PR re-opens the operational design decision (switch the workbench's CV objective from Position A to B / C, or document the tradeoff more loudly while staying on A). Phase 3 may dispatch additional targeted research per family (the existing PR-022 research surveyed the question at the convention level; PR-025 needs to quantify the workbench's specific HPO-compounding bias).

**Bump classification deferred to Phase 4 Synthesis.** If the decision is "switch to B or C" → MINOR (new config knob or changed CV objective behavior). If "stay on A + document better" → PATCH (docs-only). Default expectation: MINOR.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Pattern surveyed in PR-022 Phase 3 (2026-05-18)** — research findings archived in `prs/PR-022-workbench-correctness-hygiene.md` Phase 3. Verdict: **real practitioner split, no clean convention**. Status: **best-guess-given-constraints**, leaning slightly toward Position A as the *de-facto* library convention.

Four positions identified:

- **Position A — test fold = eval_set** (workbench current; library-blessed default of `xgb.cv` / `lgb.cv` / `cb.cv`). Documented optimism bias.
- **Position B — inner val from train fold** (sklearn `HistGradientBoosting*` default; xgboosting.com / APXML tutorials).
- **Position C — no early stopping in CV + retrain after** (XGBoost's *own* doc recommendation for CV-with-HPO).
- **Position D — full nested CV** (academic only; prohibitive cost).

### Phase 1 — State Assessment (2026-05-19)

**Current state** (HEAD = `3e1f2c0`, post-PR-024 merge):

- **`_fold_scores`** (`src/rux_ml/tuning/objective.py:174-225`) is pass-through Position A. `eval_set=[(x_te_pd, y_te.to_numpy())]` on line 215 hands the test fold to XGBoost's `early_stopping_rounds`. Untouched since PR-013 (last commit on the file = `519803c`). Zero drift from PR-023 / PR-024.
- **Eval_set surface per family**: XGBoost (constructor `early_stopping_rounds` + `fit(eval_set=[(Xv, yv)])` list), LightGBM (Pattern-A shim injects `lightgbm.early_stopping(N)` callback at fit-time per PR-018), CatBoost (constructor `early_stopping_rounds` + `fit(eval_set=(Xv, yv))` tuple).
- **`TrainingBase.early_stopping_rounds: int | None = 50`** — already supports Position C semantics (set to `None`).
- **`data.split_ratios.val = 0.15`** already exists; usable for Position B inner-val carve.
- **Calibration inputs**: local-only `configs/problems/crypto_breakout_h3.toml` + `configs/studies/crypto_breakout_h3_baseline.toml` — were drafted pre-PR-024 and are unusable on `dev` HEAD (the new cross-field validator rejects them). Must be promoted + updated to PR-023 (`panel_cpcv` + `target_horizon_bars=3`) + PR-024 (`split_kind="time_ordered"` + `time_column="timestamp"`) machinery BEFORE Phase 3 can run.

**Drift since PR-024**: zero days. No commits on `_fold_scores` or trainer factories.

**Stale stub assumptions**: stub Q3's "per-problem knob (Shape 4)" rejected at Phase 1 — single-user workbench, deferring forever unless ≥2 problems with conflicting needs surface. Stub Q1's "quantify the HPO compounding bias" survives unchanged.

**Tier-2 classification confirmed**: research archived in PR-022 Phase 3 covers the convention; the **empirical quantification** is what's missing.

**Exit decision**: Phase 1 cleared; promotion of local configs is a prerequisite for Phase 3; user approved (2026-05-19) the four sub-decisions (recommended options 1.i / 2.i / 3.i / 4.i).

### Phase 2 — Light Scoping (2026-05-19)

User-approved choices:
- `time_column` naming consistent with PR-023 D1/D2 (not `time_col`).
- No `asset_column` parameter on `temporal_train_val_test_split` (PR-024 already shipped without it; not in PR-025 scope).
- Hard error at config validation via `model_validator` on `RuxMLConfig` — already landed in PR-024.
- Calibration CV: **`PanelCombinatorialPurgedCV(target_horizon_bars=3, embargo_pct=0.0, n_folds=5, n_test_folds=2)`** — avoids the Int64-time-unit edge case in PR-023's `TimeSeriesSplitter._effective_gap` (the dataset's `timestamp` is Unix-seconds, but the splitter assumes nanoseconds-since-epoch; tracked as a separate follow-up). PanelCPCV operates on unique-timestamp INDICES → no time-unit ambiguity.
- 10 trials × 3 positions on the SAME splits + seeds + search space. Statistical threshold for switching: `|ΔMedian| / Median_A > 1%`.

**Dispatch plan**: workbench-bound empirical run (not agent-dispatch research). SSH to `rux@100.90.42.41`, sync via git, run `scripts/calibrate_pr025.py`, commit results back via git.

### Phase 3 — Empirical Calibration (2026-05-19, workbench)

**Setup**:
- Branch `pr-025/cv-eval-set-design-study` at commits `725fd7b` (config promotion + initial harness) → `6e11b05` (timing instrumentation) → `0ec3f63` (numpy-native fit path + direct `xgb.train`) → `2ec4bb5` (memory peak tracking).
- Configs promoted to repo and updated:
  - `configs/problems/crypto_breakout_h3.toml`: `[data] split_kind="time_ordered"`, `time_column="timestamp"`; `[cv] kind="panel_cpcv"` with `target_horizon_bars=3`, `embargo_pct=0.0`, `asset_column="symbol"`, `n_folds=5`, `n_test_folds=2`.
  - `configs/studies/crypto_breakout_h3_baseline.toml`: `n_trials = 10` (bumped from sanity-check 3).
- Calibration harness `scripts/calibrate_pr025.py` runs Positions A/B/C with hardcoded variants of the inner fit-and-score function; identical splits, seeds, search space across positions.

**Perf optimization (this PR introduced)**: initial run showed bursty GPU utilization (98% peaks, 1.3 GB / 24 GB VRAM, ~200 W / 450 W TDP). Per-step profile pointed at:

| Step | Original | Optimized | Delta |
|---|---|---|---|
| `df_row_select` | 14.46s | 2.06s | -86% |
| `dmatrix_build` | (in trainer_fit) | 14.74s | separated |
| `trainer_fit` | 39.41s | 21.80s | -45% |
| `predict_score` | 3.56s | 3.37s | -5% |
| **Trial wallclock** | **63.10s** | **31.97-44.57s** | **1.41-1.97x** |

Fix: pre-convert `x_full`/`y_full` to numpy ONCE; skip the sklearn Pipeline for zero-categorical problems; build `xgb.QuantileDMatrix` directly + call `xgb.train` low-level (no pandas roundtrip, no CPU↔GPU bounce on predict). User-approved bar: ≥1.5x. Median across 2-trial probe: 1.6x. Memory peaks captured: host RSS 11.4 GB on a 31 GB system (23 GB headroom; watchdog threshold 28 GB untouched), GPU 1.1 GiB on 24 GiB (22.9 GiB headroom).

**Calibration results** (per-position, completed-trials-only median per Phase 4 honest filter):

| Position | Completed | Median RMSE | Best | Within-pos spread | Wall/trial (median) | Host RSS peak | GPU peak |
|---|---|---|---|---|---|---|---|
| **A** (test fold = eval_set; current default) | 7/10 | **0.027101** | 0.027096 | 0.000006 | 37.0s | 7.8 GB | 1.1 GiB |
| **B** (inner val carved from train fold) | 6/10 | **0.027101** | 0.027096 | 0.000013 | 31.3s | 11.3 GB | 1.1 GiB |
| **C** (no early stopping in CV) | 8/10 | **0.027172** | 0.027111 | 0.000263 | 58.7s | 11.4 GB | 1.1 GiB |

Raw calibration JSON at `prs/PR-025-calibration-results.json` (committed).

**Note on the run-end "best=0.02353" summary line for Position A**: that's a print artifact — the script's `min(trial_means)` initially included pruned trials whose `trial.value` was set to a partial-fold mean. After filtering by `state == TrialState.COMPLETE`, A's true best is 0.027096 — matching the median, confirming A is well-converged on this dataset. The completed-only filter is the load-bearing comparison; the original print line was misleading and is documented here for honesty.

### Phase 4 — Synthesis (2026-05-19)

**Decision: Shape 1 — stay on Position A + document loudly.**

Reasoning:
1. **A vs B medians are identical to 6 decimals** (0.027101). Within-position spreads (0.000006 for A, 0.000013 for B) overlap completely. A's "optimism bias compounding" — the entire motivating concern from PR-022 Phase 3 — is **operationally zero** on the workbench's first real dataset.
2. **C is +0.26% worse than A** (0.027172 vs 0.027101) AND 60% slower wallclock. Empirically worse; no reason to switch.
3. **Implementation cost of Shape 2** is high: `_fold_scores` rewrite + per-family inner-val-carve adapters across XGBoost / LightGBM Pattern-A shim / CatBoost + likely a new config knob. Buying a 0% RMSE improvement is poor leverage.
4. The optimism-bias *warning* in `docs/CONVENTIONS.md` HPO objective section still stands — it's a per-problem caveat, and this calibration shows the bias is dataset-dependent. Future problems may surface a B advantage; per memory `project_cv_strategy_tier2`, per-problem CV revisits remain a known pattern.

**Bump classification: PATCH** (docs-only change; no schema, no runtime, no test fixture additions).

**Out-of-scope follow-ups surfaced during PR-025**:
- **Int64 timestamp time-unit ambiguity** in `TimeSeriesSplitter._effective_gap` (PR-023 D2 surface). Current code casts `Int64 → pl.Datetime("ns")` assuming nanoseconds-since-epoch; crypto-h3's `timestamp` is Unix-seconds. PR-025 sidestepped via `PanelCombinatorialPurgedCV` (operates on unique-timestamp indices, no duration parsing). File a separate fix PR.
- **HPO budget on calibration**: 10 trials × 3 positions left only 6–8 completed trials per position after WilcoxonPruner ran. Larger budget could be useful; not blocking the decision since within-position spread is tiny.

### Phase 5 — Implementation outcomes (2026-05-19)

**Code landed** (this PR's contribution):
- `configs/problems/crypto_breakout_h3.toml` + `configs/studies/crypto_breakout_h3_baseline.toml` — promoted from local-only to repo, updated to PR-023 (`panel_cpcv`) + PR-024 (`split_kind="time_ordered"`) machinery.
- `scripts/calibrate_pr025.py` — Phase 3 calibration harness; remains in the repo as the reproducibility receipt (research artifact, not a runtime path).
- `prs/PR-025-calibration-results.json` — raw measurement output (committed 2026-05-19 from workbench).
- `docs/CONVENTIONS.md` HPO objective subsection: appended quantified bias number (A == B to 6 decimals; +0.26% A→C); confirms Position A as workbench default.
- `docs/ARCHITECTURE.md` HPO components subsection: same — adds the empirical numbers, points to calibration receipt.
- `CHANGELOG.md` `[Unreleased]` `### Changed`: new entry describing the calibration outcome.
- `docs/0.2/ROADMAP.md` PR-025 row flipped `[ ]` → `[x]`.

**No code changes** to `src/`:
- `_fold_scores` unchanged.
- No new config knob (no `cv.eval_set_strategy`).
- No Trainer Protocol changes.

**Tests + lint**: no new tests (no behavior change); default suite remains at 363 passed (no regression). Ruff + basedpyright clean.

---

## Scope

To be locked at Phase 4 Synthesis. Three candidate shapes:

### Shape 1 — Stay on A, document loudly (PATCH)

If Phase 3 calibration shows A's bias is small on the workbench's setup: do nothing operationally. Strengthen `docs/CONVENTIONS.md` `HPO objective shape` section with the quantified bias number from Phase 3 + explicit acknowledgment that the workbench is on Position A by choice.

### Shape 2 — Switch to B (MINOR)

If Phase 3 shows A's bias is meaningful AND carving inner val is clean across all 3 families: change `_fold_scores` (`src/rux_ml/tuning/objective.py`) to carve inner val from train fold using `cfg.data.split_ratios.val`. Update `docs/CONVENTIONS.md` + `docs/ARCHITECTURE.md` to reflect the new design. Parametrized conformance test across `TRAINER_FAMILIES` asserting eval_set indices are disjoint from scoring-set indices.

### Shape 3 — Switch to C (MINOR)

If Phase 3 shows A's bias is meaningful AND inner-val carve is messy: disable `early_stopping_rounds` in the CV path; require `n_estimators` to be an explicit HPO knob (with reasonable defaults). Post-CV retrain uses early stopping on a held-out split (orthogonal — separate `rux-ml train` path).

### Shape 4 — Per-problem knob (MINOR)

Surface `cfg.cv.eval_set_strategy` as a new config field. Lets advanced users pick A / B / C per problem. More schema surface; deferred unless community asks.

### Out of scope

- Anything not directly about the eval_set placement in CV.
- Walk-forward retrain helper — covered by `docs/0.2/DESIGN-log.md` D7 (out of v0.2).

## Dependencies

- **PR-023 lands first** — PR-025's Phase 3 calibration uses the time-aware-CV machinery PR-023 ships. Re-running crypto-h3 baseline requires panel-aware CV to be in place.
- **PR-022 (already merged)** — eval_set Phase 3 archived research lives there.

## Architecture section implemented

If Shape 2 or 3 ships: `docs/ARCHITECTURE.md` HPO objective subsection (the part PR-022 reframed) updates again. If Shape 1: same section gains the quantified bias number.

## Verification criteria

Populated after Phase 4 Synthesis. Initial sketch:

- [ ] Phase 3 calibration: crypto-h3 baseline RMSE under A / B / C reported with cited methodology.
- [ ] Decision documented: which shape, why.
- [ ] If Shape 2 or 3: parametrized conformance test across `TRAINER_FAMILIES`.
- [ ] If Shape 4: new config field with TOML override semantics tested.
- [ ] `docs/CONVENTIONS.md` HPO objective subsection updated with the quantified bias OR new design.
- [ ] `docs/ARCHITECTURE.md` HPO components subsection updated likewise.
- [ ] `CHANGELOG.md [Unreleased]` entry appropriate to the shape chosen.

## Research backing

Tier-2 — pattern surveyed by PR-022 Phase 3; PR-025 re-opens with workbench-specific calibration.

Anchored on:
- PR-022 Phase 3 archived findings — 4 positions with cited sources
- XGBoost sklearn-API doc: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#early-stopping
- sklearn `HistGradientBoostingClassifier` Position B default: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

## Notes

- **Bump classification deferred to Phase 4** per the PR-023 / PR-022 precedent.
- The crypto-h3 baseline at `cdbffd2` is leakage-contaminated by ≥2 mechanisms: per-asset embargo collapse (PR-023 fix) + HPO compounding bias (this PR's question). PR-023 must ship first; PR-025's calibration runs on the cleaner machinery.
- Per memory `feedback_roadmap_flip_in_pr`: PR-025's implementation commit must flip `docs/0.2/ROADMAP.md` PR-025 row `[ ]` → `[x]` in the same commit.
- If Phase 3 calibration shows A's bias is < 1% RMSE delta on the workbench's actual datasets, **the right answer is probably Shape 1** (stay on A + document loudly). Don't over-engineer.
