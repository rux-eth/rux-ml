# PR-005: Features layer

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PR** (research-backed at design time).

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._

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
