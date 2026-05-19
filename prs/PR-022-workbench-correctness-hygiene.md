# PR-022: Workbench correctness hygiene — discriminator-carryover prevention + CV-eval_set doc accuracy

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-1.** Each item is bounded and anchored on the source as it stood at v0.1.0 (cdbffd2). Phase 1 (State Assessment) is required to catch any drift since then; Phases 2–4 may be light if no drift is found. Phase 3 web research is unlikely (carryover semantics are Pydantic discriminator + `deep_merge` behavior, project-specific).

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_Populated by `PROCEDURE-pr-research.md`. Do not begin implementation until all required phases are complete._

**Surfaced during the first real-dataset run** on 2026-05-18 (chat session `f085a0c0-afc8-4386-8e48-be8acb810c12`). The investigation surfaced multiple related correctness concerns; this PR addresses base.toml hygiene + docs with **prevention-oriented fixes** (enforced by a test / docs that reference the test as the structural backstop).

### State Assessment (2026-05-18)

**Current state** (HEAD = `cadda31`, 0 days post-v0.1.0 cut at `cdbffd2`):

- `configs/base.toml [cv]` still has `shuffle = true` at line 47 — the live bug surfaced during the 2026-05-18 run.
- `configs/base.toml [training]` has only `kind = "xgboost"` set; the variant-specific defaults (`learning_rate`, `max_depth`, etc.) are LISTED in the inline comment but not SET. No carryover risk today; latent if a contributor "promotes" a comment to a TOML assignment.
- 4 discriminated unions in the schema (confirmed by `grep`):
  - `[cv]` — 5 variants (`KFoldCV`, `StratifiedKFoldCV`, `TimeSeriesSplitCV`, `GroupKFoldCV`, `CombinatorialPurgedCV`), discriminator `kind`.
  - `[training]` — 3 variants (`XGBoostTraining`, `LightGBMTraining`, `CatBoostTraining`), discriminator `kind`.
  - `[solving]` — 1 variant (`CVXPYSolving`) today, discriminator `kind`.
  - `[tuning.search_space.*]` — 3 variants (`FloatSpec`, `IntSpec`, `CatSpec`), **discriminator `type`, not `kind`** (correction vs PR-022 draft).
- All variant schemas inherit `StrictModel` → `extra="forbid"`. Pydantic strict validation gives a clear error when an unknown field leaks across.
- `docs/CONVENTIONS.md:424-426` already documents the discriminated-union pattern for `[training]` and references `[cv]`. The **base.toml-table carryover gotcha** specifically is NOT documented.
- `tests/config/` has 5 test files (`test_loader.py`, `test_overrides.py`, `test_cv_config.py`, `test_search_space.py`, `test_hashing.py`). Natural home for the new discriminator-hygiene test.
- `tests/tuning/test_objective.py` is 147 lines; verifies `trial.report` is called per fold, completion count. **No pinned-value metric tests** (correcting PR-022's draft assumption).
- No code touched in `objective.py / cv.py / base.toml` between v0.1.0 (`cdbffd2`) and HEAD (`cadda31`). **Zero drift.**

**Assumptions at PR draft time**:

- The eval_set / test-fold pattern in `objective.py` was an unintentional CV antipattern ("standard textbook material").
- `tests/tuning/test_objective.py` had pinned-value metric tests that would need updating.
- The 4 discriminator-union tables would all key on `kind`.

**Stale assumptions**:

- **Eval_set as test-fold is documented INTENTIONAL design**, not a bug. `docs/CONVENTIONS.md:214` ("XGBoost-internal `early_stopping_rounds` runs against each fold's val partition (the test fold)") and `docs/ARCHITECTURE.md:223` ("Per-fold XGBoost-internal `early_stopping_rounds` handles within-fold pruning; Optuna pruning operates only at fold granularity via `trial.report(fold_score, fold_idx)`") both describe the design as a deliberate split: XGBoost handles within-fold convergence, Optuna handles across-fold pruning. The textbook bias concern (using HP-selected iteration to score) is real but the workbench accepted it as a known tradeoff. **My draft's "antipattern" framing was unsourced intuition.** — `feedback_research_discipline` violation flagged.
- `tests/tuning/test_objective.py` has NO pinned-value metric tests. Concern was misplaced.
- The `[tuning.search_space.*]` discriminator uses `type`, not `kind`. The structural test must handle both discriminator field names (or be told them explicitly).

**New constraints**:

- The discriminator-hygiene test must NOT assume `kind` is the universal discriminator field — it has to introspect each Pydantic union's actual discriminator (`type` for `SearchSpec`, `kind` for the other three).

**Exit decision**: The eval_set stale assumption is severe enough to change Item 2's premise (the original Item 2 was "fix the leakage" — but it's not necessarily a leakage). Per Phase 1 exit criteria, that item is **deferred** out of PR-022 to a separate research round (dispatched 2026-05-18; agent ID redacted but transcript captured). The deferred item may end up as a Tier-2 PR if research finds the workbench design is wrong; if research confirms the design, no PR follows and the docs gain a clearer tradeoff note.

PR-022's scope is now Items 1 + 2 (was Items 1 + 3 in the original draft, with the old Item 2 dropped). Both surviving items are unaffected by the eval_set research outcome and proceed to Phase 2 independently.

---

**Open research questions** (must be resolved before implementation):

1. **Discriminator-hygiene enforcement** — the root cause of the `[cv] shuffle = true` bug (hit live during the 2026-05-18 run) is **base.toml content**, not runtime behavior. Pydantic's strict validation (`extra="forbid"` on `StrictModel`) and `deep_merge`'s plain-dict merge are both working as designed — and the resulting `ValidationError` is actually a *good* diagnostic that points straight at the leaked field. Silent discriminator-aware field-dropping would be worse for debugging. The fix lives in `configs/base.toml`: only fields common to every variant of a discriminator belong at the base-table position. The same risk exists today for every discriminated union. Open: what's the right enforcement mechanism so this kind of base-TOML hygiene violation can't recur silently?
   - **A**: Programmatic test that introspects each Pydantic discriminator's variant set and asserts every field at `base.toml`'s discriminator table is in EVERY variant. Test failure = caught at PR review, not at user time. **No `src/` code change** — just a new test file under `tests/config/`.
   - **B**: Schema-level marker — declare in code which fields are "base-safe" per discriminator. Heavier; requires per-field schema annotation.
   - **C**: Move all discriminator tables out of `base.toml` entirely (require every problem TOML to set them). Strictest; loses the "shared default" affordance.
   
   Phase 3 may not need web research (the problem is Pydantic-discriminator + multi-layer TOML, project-specific); decide between A / B / C in Phase 4 synthesis. Lean (intuition, flagged): **A** — single test, no schema annotation churn, fails CI early, leaves runtime behavior untouched.

2. **TimeSeriesSplit row-vs-time documentation** — `gap` on `TimeSeriesSplitCV` is row-count. On stacked panels with N assets per timestamp, `gap=K` rows ≈ `K/N` timestamps — `gap=24` on a 1084-asset panel means <1 hour of per-asset embargo (live finding from the 2026-05-18 run). Open: docs-only (this PR) or runtime warning (this PR or PR-023)?
   - **A**: Docs in `docs/CONVENTIONS.md` only; defer schema change to PR-023.
   - **B**: Docs + runtime `UserWarning` triggered when `gap > 0` AND the data appears stacked (heuristic: timestamp column has duplicates within the first K rows).
   - **C**: Add a `gap_unit: Literal["rows", "time"] = "rows"` knob — explicit, but conflates with PR-023's time-aware CV scope.
   
   Lean (intuition, flagged): **A** for v0.1.1; structural fix belongs in PR-023 which already covers time-aware embargo for panel data.

**Resolved by 2026-05-18 research round** (see Phase 3 Findings below): the original "antipattern" framing was wrong; Position A (test fold = eval_set) is library-blessed and used by serious systems. But the workbench docs' framing is *also* misleading — implies the design is inherited from Optuna / a clean pattern, when it's actually a pragmatic deviation with documented HPO bias compounding risk that the docs don't acknowledge. Phase 4 outcome: **Amend — add Item 3 to scope (docs reframe). Operational design stays. Bigger A-vs-B-vs-C debate deferred to a future Tier-2 study.**

---

### Phase 3 — Findings (2026-05-18)

**Q (resolved):** *When using gradient-boosted models with `early_stopping_rounds` inside K-fold CV, where should `eval_set` point?*

Web research dispatched 2026-05-18 (agent transcript captured). Verdict: **real practitioner split — no clean convention.** Status: **best-guess-given-constraints**, leaning slightly toward Position A as the library-blessed default but with documented bias. Four positions found:

**Position A — test fold = eval_set** (workbench's current design)

*Sources*:
- XGBoost sklearn-API doc example fits `eval_set=[(X_test, y_test)]` then scores on same `X_test, y_test` (https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#early-stopping)
- `lightgbm.cv()` internal: `add_valid(valid_set, "valid")` with `test_idx` (https://github.com/microsoft/LightGBM/blob/master/python-package/lightgbm/engine.py)
- `xgboost.cv()` / `catboost.cv()` built-ins use the held-out fold for early stopping
- Optuna's `xgboost_cv.py` example uses `xgb.cv(..., early_stopping_rounds=100, ...)` and reports `test-auc-mean` (https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_cv.py)
- Kaggle Home-Credit LightGBM production code (https://randlow.github.io/posts/machine-learning/kaggle-home-loan-credit-risk-model-gbm/)

*Pros*: no extra data carved off train; library-blessed default in all 3 GBM `cv()` functions; operationally simple.

*Cons*: `best_iteration` HP-selected on the same data being scored → CV score optimistically biased. XGBoost docs explicitly call this out: *"using early stopping during cross validation may not be a perfect approach because it changes the model's number of trees for each validation fold."*

**Position B — inner val carved from train fold**

*Sources*:
- xgboosting.com tutorial (https://xgboosting.com/xgboost-early-stopping-with-cross-validation/) — explicit: *"the test set is not used for early stopping as it will make the accuracy estimate on the test set invalid"*
- APXML course (https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-8-boosting-hyperparameter-optimization/cross-validation-tuning)
- Macaluso practitioner blog (https://macalusojeff.github.io/post/HyperparameterTuningXGB/)
- **sklearn's `HistGradientBoostingClassifier` default** when `early_stopping=True` — carves `validation_fraction=0.1` from the training fold internally (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

*Pros*: test fold genuinely independent → unbiased CV score (textbook invariant).

*Cons*: steals 10–20% from train fold; more variance in `best_iteration` (inner val smaller); on panel data with GroupKFold/CPCV, inner splitter must also respect grouping/embargo — non-trivial.

**Position C — no early stopping in CV; retrain after**

*Sources*:
- **XGBoost's own docs explicit recommendation**: *"A better approach is to retrain the model after cross validation using the best hyperparameters along with early stopping."* (https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#early-stopping)
- XGBoost issue #2261 community pattern (https://github.com/dmlc/xgboost/issues/2261)
- AutoGluon tabular XGBoost model uses single train/val + no internal CV (https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/xgboost/xgboost_model.py)

*Pros*: cleanest unbiased CV; no leakage debate.

*Cons*: `n_estimators` becomes an explicit HPO param OR fixed high cap → wasted compute; doesn't match workbench's "let XGBoost own within-fold convergence" design.

**Position D — full nested CV**: academic only (arxiv-grade); prohibitive cost; no production-grade tabular system found using true nested-CV with inner early-stopping CV inside outer model-selection CV.

*Disconfirming evidence sought*:
- Searched for production claims that Position A is the *recommended* convention → none found. Library defaults implement A but the surrounding docs warn against it for CV-with-HPO.
- arxiv 2405.03389 (early-stopping bias in CV) is about across-fold pruning (Optuna-style), not eval_set placement.
- sklearn `cross_validate` user guide is silent on the issue (https://scikit-learn.org/stable/modules/cross_validation.html).
- AFML (Lopez de Prado) Ch. 7 focuses on label-leakage from overlapping observations, NOT early-stopping bias — no AFML rule on eval_set placement.
- Optuna's WilcoxonPruner tutorial (https://optuna.readthedocs.io/en/latest/tutorial/20_recipes/013_wilcoxon_pruner.html) uses independent TSP instances, not CV folds with early stopping. **Critically: WilcoxonPruner does NOT prescribe Position A** — it only requires per-fold scores reported via `trial.report`. The workbench's "clean responsibility separation" framing is a design *choice*, not a pattern inherited from Optuna's recipe.

*Workbench-specific risks the general analysis missed*:
- **HPO bias compounding**: Optuna selects HPs whose `best_iteration` on the test fold maximizes test-fold score. Across hundreds of trials, the selection itself adapts to the test folds — bigger problem than single-trial bias. Not acknowledged in workbench docs.
- **Panel data + B**: switching to Position B requires the inner splitter to respect the same grouping/embargo as the outer Splitter. Concrete implementation cost.
- **Docs misframing**: `CONVENTIONS.md:214` / `ARCHITECTURE.md:223` imply the eval_set design is inherited from Optuna's WilcoxonPruner pattern. It isn't. The framing should be "explicit pragmatic deviation from textbook CV with known bias", not "acknowledged standard."

*Recommendation*: keep Position A operationally (defensible; library-blessed default; switching to B/C has real costs on panel data); **fix the docs** to honestly characterize it as a pragmatic deviation with documented bias.

*Status label*: **best-guess-given-constraints** (real practitioner split; no clean ≥2-source convention for any single position as the universal answer).

*Risks accepted (Position A operational stays)*:
- CV scores remain optimistic generalization estimates (consistent-bias, useful for HP ranking but not for absolute reporting).
- HPO compounding bias is unmitigated — partially manageable by treating final-model selection as a separate validation step on held-out data.

### Phase 4 — Synthesis (2026-05-18)

**Outcome**: **Amend** — research surfaces a docs-accuracy problem (workbench docs misframe the eval_set design); operational design is defensible and stays as-is.

**Changes to this PR** from research:
- Add **Item 3** (CV-eval_set docs reframe) to scope. Docs-only — no `src/` change.

**Changes to ARCHITECTURE.md**:
- §223 paragraph: reframe from "clean responsibility separation" → "pragmatic deviation from textbook CV with known optimism bias; switching to B/C deferred to future Tier-2 study". Cite Phase 3 findings.

**Changes to CONVENTIONS.md**:
- §214 paragraph: reframe similarly + acknowledge HPO compounding bias risk.

**Changes to CONSTRAINTS.md**:
- None.

**New PRs that must come first**:
- None.

**Research-backed details now locked in this PR**:
- Position A stays operational (no `src/rux_ml/tuning/objective.py` changes).
- Docs reframe is anchored in cited findings (Phase 3 sources above).
- A-vs-B-vs-C operational change deferred — would need a Tier-2 PR with its own design session per `project_cv_strategy_tier2`.

### Phase 5 — Gate Check + Implementation (2026-05-18)

- Premise still valid: ✓ (Items 1 + 2 unchanged by research; Item 3 added directly from research findings)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-05-18)
- Implementation cleared: ✓

Implementation outcomes (verified):

- `configs/base.toml [cv]` now only carries `kind = "kfold"`. Removed `shuffle = true` (the live bug) AND `n_splits = 5` (bonus latent bug surfaced by the new structural test: `CombinatorialPurgedCV` uses `n_folds`, not `n_splits`).
- New `tests/config/test_base_toml_discriminator_hygiene.py`: 4 test cases (3 parametrized over discriminator unions + 1 tripwire). Verified to FAIL on the pre-fix base.toml (catches both `shuffle` and `n_splits`); PASSES post-fix. Tripwire passes (all 3 top-level discriminator unions covered).
- `docs/CONVENTIONS.md` Configuration Conventions: new base.toml-discriminator-hygiene subsection.
- `docs/CONVENTIONS.md` CV strategy conventions: new `TimeSeriesSplitCV.gap` row-vs-time subsection (forward-references PR-023).
- `docs/CONVENTIONS.md` HPO objective shape (line 214): reframe of eval_set design per Phase 3 findings.
- `docs/ARCHITECTURE.md` HPO components (line 223): parallel reframe.
- `CHANGELOG.md [Unreleased]`: `### Fixed` (base.toml carryover + docs reframe) + `### Added` (new CONVENTIONS.md subsections).
- Verified: 327 tests pass / 1 skip / 15 deselected; `ruff check .` clean; `basedpyright src/` clean.

---

## Scope

Three items. All model-/family-agnostic; prevention-oriented where applicable. Implementation shape locked by Phase 4 synthesis above.

(Original Item 2 — operational CV eval_set / test-fold "leakage" fix — **dropped after research**. Position A stays operational; Item 3 below covers the docs-accuracy gap instead.)

### Item 1 — base.toml `[cv] shuffle = true` + structural prevention

This is a **config (TOML content) fix**, not a runtime behavior change. Pydantic's `extra="forbid"` and `deep_merge` both stay as-is.

- Remove `shuffle = true` from `base.toml [cv]` (concrete fix for the live bug).
- Add an enforcement test (mechanism per Q1) that walks every discriminated union in the schema and asserts no base-table field is variant-specific. Future regressions of the same shape — e.g., a contributor adding `subsample = 0.8` to `base.toml [training]` would break problem-level `kind = "lightgbm"` (LightGBM's schema has `bagging_fraction`, not `subsample`) — fail the test at CI time, not the user at runtime.

Applies to every discriminated union in the schema: `[cv]` (5 variants), `[training]` (3 variants), `[solving]` (1 variant today), `[tuning.search_space.*]` (3 variants — internal but worth verifying). All 4 are covered by a single parametrized test, not per-union code. **No changes under `src/`** — TOML edit + new test file under `tests/config/`.

### Item 2 — `docs/CONVENTIONS.md` base.toml-hygiene + `TimeSeriesSplit.gap` semantics

- Document the **base.toml discriminator-table rule**: only fields common to every variant of a discriminator (`[cv]`, `[training]`, `[solving]`, `[tuning.search_space.*]`) belong at the base-table position. Variant-specific knobs go in `configs/problems/<n>.toml`. Reference Item 1's enforcement test as the structural backstop. Make clear that Pydantic's strict validation + `deep_merge`'s plain-dict semantics are intentional — the rule is about TOML *content*, not runtime behavior. Note: the `[tuning.search_space.*]` discriminator field is `type`, not `kind`, while the other three use `kind` — call this out.
- Document `TimeSeriesSplit.gap` as row-count, with the panel-data implication and forward-reference to PR-023 for time-aware embargo.

### Item 3 — CV eval_set docs accuracy reframe

Research-backed correction (Phase 3 above). Operational design stays; **docs change only**.

- **`docs/CONVENTIONS.md:214`** — rewrite the "XGBoost-internal `early_stopping_rounds` runs against each fold's val partition (the test fold)" line to honestly characterize:
  - it's Position A out of four positions (A/B/C/D) practitioners use
  - it's the library-blessed default of `xgb.cv` / `lgb.cv` / `cb.cv` (cite XGBoost sklearn-API doc + LightGBM `engine.py`)
  - it produces an optimism bias in the CV-reported metric that the workbench accepts as a tradeoff
  - the HPO bias compounds across trials (concrete risk, currently undocumented)
  - it is **not** inherited from Optuna's WilcoxonPruner tutorial (the docs implication that it is, is incorrect)
  - Position B (sklearn HistGradientBoosting's default) and Position C (XGBoost's *own* recommended-for-CV-HPO alternative) are explicitly available alternatives, deferred to future Tier-2 study per memory `project_cv_strategy_tier2`

- **`docs/ARCHITECTURE.md:223`** — parallel rewrite of the "Per-fold XGBoost-internal `early_stopping_rounds` handles within-fold pruning; Optuna pruning operates only at fold granularity" paragraph. Same reframe direction.

Both edits cite the Phase 3 sources inline. **No `src/` change. No tests change.**

### Out of scope

- Time-aware CV / per-asset embargo / label-overlap purging — owned by PR-023.
- Random-shuffle leakage in `rux-ml train`'s baseline path (`train_val_test_split`) — separate concern; tag in PR-023 open questions.
- New `gap_unit` config knob — option C in Q3, deferred to PR-023.

## Dependencies

None. Built on v0.1.0 (cdbffd2).

## Architecture section implemented

None — bug fix + structural test + docs. No new architecture.

`docs/CONVENTIONS.md` gains subsections per Item 2 (new) and Item 3 (paragraph reframe of §214). `docs/ARCHITECTURE.md` §223 paragraph is reframed per Item 3 — paragraph-level clarification, not structural rewrite. Both Phase-3-cited.

## Verification criteria

Populated after Phase 4 synthesis. Initial sketch (refine post-research):

- [ ] `base.toml [cv]` no longer carries `shuffle = true`
- [ ] New test in `tests/config/` walks every discriminated union; asserts every base-table field is in every variant; fails clearly if violated. Parametrized over the 4 unions, not 4 separate tests.
- [ ] `docs/CONVENTIONS.md` has base.toml-hygiene subsection (variant-specific fields stay out of discriminator tables; references the Item 1 test as enforcement).
- [ ] `docs/CONVENTIONS.md` has `TimeSeriesSplit.gap` row-vs-time subsection with PR-023 forward-reference.
- [ ] `docs/CONVENTIONS.md:214` reframed: position A characterization + cited library defaults + HPO compounding bias acknowledged + Optuna-inheritance implication removed + B/C alternatives noted as deferred Tier-2.
- [ ] `docs/ARCHITECTURE.md:223` reframed: parallel direction to CONVENTIONS.md:214.
- [ ] `CHANGELOG.md [Unreleased] ### Fixed` entry for the `[cv] shuffle = true` carryover bug.
- [ ] `CHANGELOG.md [Unreleased] ### Changed` entry for the docs reframe (or `### Fixed` if framed as "docs were inaccurate"; pick per the doc-edit conventions in `docs/VERSIONING.md` §6).
- [ ] `make test` green; lint + type-check green.

## Research backing

Tier-1 — anchored on:
- v0.1.0 source code as of `cdbffd2` (Phase 1 State Assessment will verify no drift).
- Pydantic-settings v2 `deep_merge=True` behavior on discriminated unions (verified live during the 2026-05-18 run).
Phase 3 web research completed 2026-05-18 for the (originally-Item-2, now-Item-3) eval_set question — see Phase 3 — Findings section. Findings cited inline in Item 3's doc edits.

Phase 3 not needed for Items 1 + 2 — the discriminator-hygiene problem is project-specific (Pydantic discriminated union + multi-layer TOML), and the docs subsections are straightforward.

## Notes

- **PATCH bump (v0.1.1)** per `docs/VERSIONING.md` §1: bug fixes + docs + internal test additions. No design session needed per §2.
- **Out-of-procedure hot-patch was rejected by the user** on 2026-05-18. An earlier draft of this work landed direct on `dev` as `c29886c` (since reverted) without running `PROCEDURE-pr-research.md`. This PR is the proper procedural redo.
- **Model-agnostic by construction**: the discriminator-hygiene test walks every discriminated union (not just `[cv]`), so new trainer families / solver families / search-space variants added in future PRs inherit the same protection automatically.
- **PR-023 is the sibling** addressing time-aware CV embargo for panel data. Tier-2, MINOR, requires v0.2 design session. This PR-022 deliberately stays PATCH-shaped.
- **`feedback_roadmap_flip_in_pr` rule**: post-v0.1.0 cut there's no active `docs/<x.y>/ROADMAP.md` for in-flight work. For PATCH bumps the established convention is for the version-cut PR (v0.1.1) to roll up Unreleased entries — PR-022 itself has no roadmap row to flip. Call this out explicitly in the implementation commit so future-Claude doesn't flag a missed flip.
