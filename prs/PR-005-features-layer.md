# PR-005: Features layer

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

### State Assessment (2026-05-16)

**Current state of the codebase**:

- `dev` at `32aad23` (PR-004 merged: data layer)
- `src/rux_ml/` has `config/`, `cli/`, `_internal/`, `data/`, `__main__.py` — **no `features/` subpackage yet**
- `FeaturesConfig` (PR-002, `src/rux_ml/config/features.py`) already exists with:
  - `categorical_low_card_threshold: int | None = None` (BEST-GUESS per D4 — no default; tune on first dataset)
  - `spec: dict[str, Any] = Field(default_factory=dict)` (placeholder; PR-005 may narrow into a typed schema)
- `configs/base.toml` has the `[features]` table header with comment-only content — PR-005 doesn't need to edit it
- Installed runtime deps: `polars 1.40.1`, `xgboost 3.2.0`, `numpy 2.4.5`, `xxhash 3.7.0`, `pydantic`, `pydantic-settings`, `typer`
- **Not yet installed**: `scikit-learn`, `category-encoders`, `hypothesis` — PR-005 will add them

**Assumptions at PR draft time** (drafted 2026-05-15 from D4 + D12):

- sklearn `Pipeline` + `ColumnTransformer` + `FunctionTransformer` orchestrator
- `set_output(transform="polars")` for end-to-end Polars I/O (sklearn 1.4+)
- `category_encoders.NestedCVWrapper(TargetEncoder)` for high-cardinality categoricals
- `hypothesis` for property-based transformer-invariant tests
- Categorical decision rule per D4: low-card → pass-through to XGBoost `enable_categorical=True`; high-card → wrapped target encoder
- Consume `FeaturesConfig` from PR-002

**Stale assumptions**: None. PR-002's `FeaturesConfig` schema matches what PR-005 consumes; PR-004's data layer produces Polars DataFrames that flow cleanly into the planned features pipeline; no design decision since 2026-05-15 contradicts the PR-005 spec.

**New constraints learned from prior PRs**:

- `FeaturesConfig.spec` is intentionally a placeholder; PR-005 may narrow it into a typed schema (this PR's scope explicitly anticipates this).
- PR-003's CLI has no `features` verb group — correct; feature transforms run inside the training/tuning flow, not as a user-invoked CLI step.
- Test markers (`gpu`, `slow`, `golden`, `integration`) are registered project-wide; PR-005's tests don't need a new marker.
- `pyproject.toml` uses PEP 735 `[dependency-groups]` for dev deps — `hypothesis` goes there; `scikit-learn` and `category-encoders` go in runtime `dependencies`.

**Synthesis Outcome: CONFIRM** — zero drift detected via local state assessment. PR-005 spec stands.

### Synthesis (2026-05-16)

**Outcome:** Confirm.

**Changes to this PR from research:** None (the deps additions are already implied by the PR's scope; spelling them out for clarity).

**Changes to ARCHITECTURE.md / CONSTRAINTS.md / CONVENTIONS.md / CLAUDE.md:** None.

**New PRs that must come first:** None.

**Phase 3 (web research) intentionally skipped:** Per `PROCEDURE-pr-research.md`, Tier-1 PRs run Phases 2-4 "light if no drift found." Web verification of sklearn / category_encoders / hypothesis APIs would duplicate the design-time research from D4 + D12 without justification.

**Research-backed details now locked in this PR:**
- sklearn `Pipeline` + `ColumnTransformer` + `FunctionTransformer` (per D4)
- `set_output(transform="polars")` end-to-end (per D4 — PROVEN at design time)
- `category_encoders.NestedCVWrapper(TargetEncoder)` for very-high-cardinality columns (per D4 — PROVEN at design time)
- Low/med-card categoricals pass through to XGBoost `enable_categorical=True` (per D4)
- `hypothesis` selective use for transformer invariants (per D12)

**Runtime dependency additions** (mechanical):
- `scikit-learn>=1.4` (for `set_output("polars")`)
- `category-encoders>=2.6`

**Dev dependency additions** (mechanical):
- `hypothesis>=6`

### Gate Check

- Premise still valid: ✓ (zero drift)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-05-16)
- Implementation cleared: ✓ (2026-05-16)

### CV scope note (2026-05-16)

The only CV-touching code in this PR is `category_encoders.NestedCVWrapper`, which internally uses sklearn's `StratifiedKFold(n_splits=5)` for the narrow purpose of preventing within-train leakage when target-encoding **high-cardinality categoricals**. This is **library-provided**, not implemented in this repo — and it's a single decision (5 random folds, stratified by target) limited to the target-encoder hot path.

Per user direction (recorded in project memory `project-cv-strategy-tier2`), **any project-wide CV strategy** (K-fold variants, walk-forward, CPCV, GroupKFold, embargo windows for time-series, etc.) **needs its own Tier-2 PR** before PR-007 (Optuna `objective` design). That PR must research the library/tool choice, address data-leakage prevention, and address optimization/parallelism interactions with the sequential `n_jobs=1` constraint from D6 + subprocess-per-trial from D10.

PR-005 does **not** preempt that decision; the current `NestedCVWrapper(StratifiedKFold)` use can be revisited and overridden by the future CV-strategy PR if its choice conflicts.

### Implementation notes (2026-05-16)

1. **`ColumnTransformer` replaced with a small custom `_ColumnRouter`.** ColumnTransformer's `set_output("polars")` propagation breaks on `NestedCVWrapper` (legacy sklearn-API, no `set_output`), and routing via a custom transformer lets us call `fit_transform` correctly so NestedCV's K-fold un-leaked encoding fires.
2. **Pipeline shape:** `Polars in → polars_select_then_pandas → _ColumnRouter (pandas) → to_polars → Polars out`. Pandas is the "waist" because legacy sklearn transformers need it; conversions happen once on entry and once on exit.
3. **`FeaturesConfig.spec`** narrowed from `dict[str, Any]` to a typed `FeaturesSpec(numeric_columns, categorical_columns)`. Derived columns (ratios, datetime parts, etc.) deferred to a later PR with a concrete use case.
4. **New runtime deps**: `scikit-learn>=1.4`, `category-encoders>=2.6`, `pyarrow>=17` (needed for `pl.DataFrame.to_pandas`). New dev dep: `hypothesis>=6`.
5. **Test config relaxation:** `[tool.basedpyright]` execution environment for `tests/` disables `reportUnknown*` + `reportPrivateUsage` — sklearn / category_encoders / hypothesis ship limited type info and the test-side noise didn't reflect real bugs. Library code in `src/` stays under the strict default.

---

## Scope

Implement the feature engineering layer per D4 — sklearn `Pipeline` + `ColumnTransformer` orchestrator, with Polars-expression stateless transforms wrapped in `FunctionTransformer`.

- `src/rux_ml/features/pipeline.py` — `make_features(cfg: FeaturesConfig) -> sklearn.Pipeline`:
  - Stateless Polars block via `FunctionTransformer` (composes whatever derived columns the config spec requests: ratios, datetime parts, aggregates)
  - sklearn `ColumnTransformer` for stateful steps (impute, scale where needed; default behavior leaves XGBoost native handling untouched)
  - `set_output(transform="polars")` so downstream stays in Polars
- `src/rux_ml/features/polars_steps.py` — small helpers building Polars expressions from the TOML feature spec
- `src/rux_ml/features/encoders.py` — `make_categorical_encoder(col, cardinality, cfg) -> Transformer` implementing the D4 decision rule:
  - `cardinality ≤ cfg.categorical_low_card_threshold` → pass-through (rely on XGBoost `enable_categorical=True` downstream)
  - else → `category_encoders.TargetEncoder` wrapped in `NestedCVWrapper`
  - threshold has **no default** in `configs/base.toml` per D4 BEST-GUESS — surfaces as an explicit choice on first dataset
- `configs/base.toml` extended with `[features]` section (skeleton spec format documented in `configs/README.md`)
- Tests:
  - Unit: each Polars step is idempotent; round-trip through `Pipeline` returns Polars
  - Common contract: every custom transformer passes a fit-on-train / transform-on-test discipline check
  - Property-based (`hypothesis`): shape and dtype preservation on synthetic frames
  - GPU-irrelevant — no `@pytest.mark.gpu`

NOT in scope: training the model (PR-006); the actual choice of which features for which problem (TOML config per problem).

## Dependencies

PR-004.

## Architecture section implemented

`docs/ARCHITECTURE.md` → "Features" component row, "Decision Rules: Categorical encoding".

## Verification criteria

- [ ] `make_features(cfg)` returns a fittable sklearn `Pipeline`
- [ ] `pipeline.fit(X_train).transform(X_test)` returns a Polars DataFrame (not pandas, not NumPy)
- [ ] `set_output("polars")` is in effect end-to-end
- [ ] Categorical decision rule: low-card columns pass through; high-card columns receive `NestedCVWrapper(TargetEncoder)`
- [ ] No leakage in target encoding (verified by a unit test fitting on train, transforming test, checking that test encodings depend only on train statistics)
- [ ] Common-contract tests pass for every transformer
- [ ] Property-based tests cover shape/dtype invariants

## Research backing

Tier 1:

- D4: [sklearn `set_output("polars")` 1.4 release notes](https://scikit-learn.org/stable/whats_new/v1.4.html), [XGBoost categorical tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html), [Hopsworks Pandas2+Polars](https://www.hopsworks.ai/post/pandas2-and-polars-for-feature-engineering), [Made With ML feature-store overkill](https://madewithml.com/courses/mlops/feature-store/)
- D12: hypothesis selectively for transformer invariants

State assessment must verify sklearn `set_output` API + `category_encoders.NestedCVWrapper` are unchanged.

## Notes

- We deliberately do NOT add `feature-engine` per D4 — sklearn + Polars + `category_encoders` covers the scope without an extra dep.
- Per D4, the cardinality threshold is `BEST-GUESS` — leave the field optional in `FeaturesConfig` and require explicit configuration before any training run that touches categoricals.
