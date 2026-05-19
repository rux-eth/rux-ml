# PR-022: Workbench correctness hygiene — discriminator-carryover prevention + CV eval_set fix

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-1.** Each item is bounded and anchored on the source as it stood at v0.1.0 (cdbffd2). Phase 1 (State Assessment) is required to catch any drift since then; Phases 2–4 may be light if no drift is found. Phase 3 web research is unlikely (the CV eval_set antipattern is textbook; carryover semantics are Pydantic discriminator + `deep_merge` behavior).

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Surfaced during the first real-dataset run** on 2026-05-18 (chat session `f085a0c0-afc8-4386-8e48-be8acb810c12`). The investigation surfaced two related correctness gaps; this PR addresses both with **prevention-oriented fixes** (enforced by tests / structural changes) rather than per-issue patches that hope future contributors notice.

**Open research questions** (must be resolved before implementation):

1. **Discriminator-hygiene enforcement** — the root cause of the `[cv] shuffle = true` bug (hit live during the 2026-05-18 run) was that fields in a `base.toml` discriminator-table leak across variants on `deep_merge`. The same risk exists today for every discriminated union in the schema. Open: what's the right enforcement mechanism?
   - **A**: Programmatic test that walks `base.toml`, introspects each Pydantic discriminator's variant set, and asserts every base field is in EVERY variant. Test failure = caught before merge, not at user time.
   - **B**: Schema-level marker — declare in code which fields are "base-safe" per discriminator. Heavier; requires schema annotation.
   - **C**: Move all discriminator tables out of `base.toml` entirely (require every problem TOML to set them). Strictest; loses the "shared default" affordance.
   
   Phase 3 may not need web research (Pydantic-specific); decide between A / B / C in Phase 4 synthesis. Lean (intuition, flagged): **A** — single test in `tests/config/`, no schema annotation churn, fails CI early.

2. **CV eval_set / test-fold leakage** — `src/rux_ml/tuning/objective.py` (`_fold_scores`) passes the test fold as XGBoost's `eval_set` then scores against the same fold. `best_iteration_` is HP-selected on the same fold the score is computed on → metric is better-than-real. Affects every CV run regardless of trainer family. Open: how to remove the leak across all 3 families (XGBoost, LightGBM, CatBoost — each has its own early-stopping mechanism per PR-017/018/019)?
   - **A**: Carve an inner val slice from each train fold using the existing `cfg.data.split_ratios.val` knob. eval_set = inner val; score on the untouched test fold. No new config.
   - **B**: New `cv.inner_val_fraction` knob (problem-TOML-overridable). More explicit; new MINOR-worthy knob.
   - **C**: Add an explicit `Trainer.fit_with_early_stopping(X_tr, y_tr, X_val, y_val) → best_iteration` Protocol method; each family implements it natively. Cleanest API; biggest refactor.
   
   Phase 3 may want a light-touch survey of how sklearn-evaluation / mlflow / Optuna's own examples handle this. Lean (intuition, flagged): **A** for v0.1.1 — uses an existing knob, no new MINOR surface, family-agnostic via the existing sklearn-style `eval_set` kwarg every Trainer family already accepts. Test-side: parametrized conformance across `TRAINER_FAMILIES` asserting eval_set indices are disjoint from the scoring set's indices.

3. **TimeSeriesSplit row-vs-time documentation** — `gap` on `TimeSeriesSplitCV` is row-count. On stacked panels with N assets per timestamp, `gap=K` rows ≈ `K/N` timestamps — `gap=24` on a 1084-asset panel means <1 hour of per-asset embargo (live finding from the 2026-05-18 run). Open: docs-only (this PR) or runtime warning (this PR or PR-023)?
   - **A**: Docs in `docs/CONVENTIONS.md` only; defer schema change to PR-023.
   - **B**: Docs + runtime `UserWarning` triggered when `gap > 0` AND the data appears stacked (heuristic: timestamp column has duplicates within the first K rows).
   - **C**: Add a `gap_unit: Literal["rows", "time"] = "rows"` knob — explicit, but conflates with PR-023's time-aware CV scope.
   
   Lean (intuition, flagged): **A** for v0.1.1; structural fix belongs in PR-023 which already covers time-aware embargo for panel data.

---

## Scope

Three items, all model-/family-agnostic and prevention-oriented. Concrete implementation shape decided post-research in Phase 4 synthesis — listed at the option-level below, not pre-committed.

### Item 1 — base.toml `[cv] shuffle = true` + structural prevention

- Remove `shuffle = true` from `base.toml [cv]` (concrete fix for the live bug).
- Add an enforcement test (mechanism per Q1) that walks every discriminated union in the schema and asserts no base-table field is variant-specific. Future regressions of the same shape — e.g., a contributor adding `learning_rate = 0.1` to `base.toml [training]` (would break problem-level `kind = "catboost"` since CatBoost has no `learning_rate`-equivalent in our schema — wait, it does, but `subsample` doesn't always exist; the point stands) — fail the test, not the user.

Applies to every discriminated union in the schema: `[cv]` (5 variants), `[training]` (3 variants), `[solving]` (1 variant today), `[tuning.search_space.*]` (3 variants — internal but worth verifying). All 4 are covered by a single parametrized test, not per-union code.

### Item 2 — `objective.py` eval_set / test-fold leakage

- Fix the leakage per Q2; the fix must work across all 3 trainer families (XGBoost, LightGBM, CatBoost). Validated by parametrized test over `TRAINER_FAMILIES` that asserts eval_set indices are disjoint from scoring-set indices on a synthetic regression task.
- Affects `tests/tuning/test_objective.py` (existing tests pinning specific metric values must update to reflect the corrected, slightly-higher metric).
- Calibration check: re-run the crypto baseline (PR-021 reference: RMSE=0.02547 at c29886c → cdbffd2) and report the delta. Expected: small rise (early stopping was overfitting to test fold). Delta is documented in the PR description, not pinned in a test.

### Item 3 — `docs/CONVENTIONS.md` discriminator-carryover + `TimeSeriesSplit.gap` semantics

- Document the discriminator-carryover gotcha (forward-references Item 1's test as the structural enforcement).
- Document `TimeSeriesSplit.gap` as row-count, with the panel-data implication and forward-reference to PR-023 for time-aware embargo.

### Out of scope

- Time-aware CV / per-asset embargo / label-overlap purging — owned by PR-023.
- Random-shuffle leakage in `rux-ml train`'s baseline path (`train_val_test_split`) — separate concern; tag in PR-023 open questions.
- New `gap_unit` config knob — option C in Q3, deferred to PR-023.

## Dependencies

None. Built on v0.1.0 (cdbffd2).

## Architecture section implemented

None — bug fix + structural test + docs. No new architecture.

`docs/CONVENTIONS.md` gains two short subsections per Item 3. No SSOT rewrites.

## Verification criteria

Populated after Phase 4 synthesis. Initial sketch (refine post-research):

- [ ] `base.toml [cv]` no longer carries `shuffle = true`
- [ ] New test in `tests/config/` walks every discriminated union; asserts every base-table field is in every variant; fails clearly if violated. Parametrized over the 4 unions, not 4 separate tests.
- [ ] `objective.py` eval_set is disjoint from the scoring fold; parametrized conformance test across `TRAINER_FAMILIES` confirms for XGBoost, LightGBM, CatBoost.
- [ ] Existing `tests/tuning/test_objective.py` value-pinning tests updated to reflect corrected metric.
- [ ] Calibration: crypto-h3 baseline re-run produces a measurable delta from RMSE=0.02547. Delta documented in PR description.
- [ ] `docs/CONVENTIONS.md` has discriminator-carryover subsection.
- [ ] `docs/CONVENTIONS.md` has `TimeSeriesSplit.gap` row-vs-time subsection with PR-023 forward-reference.
- [ ] `CHANGELOG.md [Unreleased] ### Fixed` entries for both Item 1 and Item 2.
- [ ] `make test` green; lint + type-check green.

## Research backing

Tier-1 — anchored on:
- v0.1.0 source code as of `cdbffd2` (Phase 1 State Assessment will verify no drift).
- Pydantic-settings v2 `deep_merge=True` behavior on discriminated unions (verified live during the 2026-05-18 run).
- CV `eval_set = test-fold` antipattern: standard ML textbook material (Hastie/Tibshirani Ch. 7, Bishop Ch. 1, sklearn's `cross_validate` docs).

Phase 3 web research likely 0 agents unless Q2 option-C surfaces a meaningfully better pattern that other workbenches have shipped.

## Notes

- **PATCH bump (v0.1.1)** per `docs/VERSIONING.md` §1: bug fixes + docs + internal test additions. No design session needed per §2.
- **Out-of-procedure hot-patch was rejected by the user** on 2026-05-18. An earlier draft of this work landed direct on `dev` as `c29886c` (since reverted) without running `PROCEDURE-pr-research.md`. This PR is the proper procedural redo.
- **Model-agnostic by construction**: both the discriminator-hygiene test (walks every discriminated union, not just `[cv]`) and the eval_set fix (parametrized over `TRAINER_FAMILIES`) are family-blind. New trainer families added in future PRs inherit the same protection automatically.
- **PR-023 is the sibling** addressing time-aware CV embargo for panel data. Tier-2, MINOR, requires v0.2 design session. This PR-022 deliberately stays PATCH-shaped.
- **`feedback_roadmap_flip_in_pr` rule**: post-v0.1.0 cut there's no active `docs/<x.y>/ROADMAP.md` for in-flight work. For PATCH bumps the established convention is for the version-cut PR (v0.1.1) to roll up Unreleased entries — PR-022 itself has no roadmap row to flip. Call this out explicitly in the implementation commit so future-Claude doesn't flag a missed flip.
