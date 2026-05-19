# PR-025: CV eval_set Position B/C operational design study

**Landed-in:** (not yet landed)

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

**Open research questions** (must be resolved during PR-025 Phase 3):

1. **Quantify the workbench-specific HPO compounding bias.** Re-run the crypto-h3 baseline (RMSE=0.02547 at `cdbffd2`) under three configurations on the **same time-aware splits** (post-PR-023): (a) Position A as-is, (b) Position B with inner val carved from train fold via `cfg.data.split_ratios.val`, (c) Position C with `n_estimators` fixed, no early stopping. Report the RMSE delta. If A vs B differ by < tolerance, Position A's optimism bias is small enough to keep on operational grounds. If > tolerance, switching is justified.
2. **Implementation cost of carving inner val across all 3 trainer families.** XGBoost takes `eval_set=[(X_val, y_val)]` (sklearn-style). LightGBM uses Pattern-A shim per PR-018 (`callbacks=[lightgbm.early_stopping(N)]` injected at fit time). CatBoost uses constructor `early_stopping_rounds` + `eval_set=(X_val, y_val)` (tuple, not list). Verify a single inner-val carve works across all three; design the Trainer-Protocol or factory-level surface change.
3. **Decision**: stay on A + document more loudly / switch to B / switch to C / make it a per-problem knob (`cfg.cv.eval_set_strategy: Literal["test_fold", "inner_val", "no_early_stop"] = "test_fold"`).

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
