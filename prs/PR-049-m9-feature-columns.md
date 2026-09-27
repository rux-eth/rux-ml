# PR-049: The M9 problems train on the `feat__` columns of their own subtree

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1.** Delegated by the rux-capital program lead; covered by the operator's standing approval (the features accepted 2026-09-27; program D38: the model's features are the `feat__` columns). Config + tests + docs; no `src/` change.

## Research findings

### State Assessment (2026-09-27)

Baselines: rux-ml `dev` @ `70c9b4a` (PR-048 merged, #54). Harness schema `spec/m9_training_schema.json` v2 (rumpy-harness `18bda8b`, sha256 `fc900836…e5e3`): the live file in the PR-024 worktree is byte-identical to the vendored `tests/fixtures/m9_training_schema.json`.

- **No prefix-selection mechanism exists**: `FeaturesSpec` is two explicit lists (`numeric_columns`, `categorical_columns`) and `select_columns` keeps exactly those. So each problem names its `feat__` columns explicitly (auditable), and a test pins the list to the schema.
- **The schema**: 28 `feat__` columns in `fill/`, 27 in `walk/` (`feat__queue_ahead_usd` fill-only). `targets.<t>.columns` filtered to `feat__` equals the `feat` entries in order. Every one is `float64`. (PR-048's note said 29 / 28, which was a miscount.)
- **Possibly-empty features, per the schema's definitions** ("null when …"): `near_walk_bp_q1000/q3000` (thin book), `ofi_60s/300s` (no transition), `rv_bp_300s/3600s` (no pair), and in `fill/` `queue_ahead_usd` (beyond the deepest level). Pinned by test.
- **The null path to the model**: `train` hands XGBoost `pipeline.transform(x).to_pandas()`. A Polars Float64 null becomes pandas NaN, and `XGBRegressor(missing=nan)` (the default; the configs don't override it) sends it down the learned default branch. The native path (`use_native`, off) uses `to_numpy()`, which also gives NaN.
- **The registry, probed locally** (xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10; 400 rows, 20 % NaN in a numeric column): all three families fit and predict with NaN in a **numeric** feature (XGBoost `missing`, LightGBM `use_missing`, CatBoost `nan_mode="Min"`). **CatBoost refuses NaN in a categorical feature** (`CatBoostError: Invalid type for cat_feature … =NaN`). The only M9 categorical is `kind`, a C6 key that is never null, and every M9 problem is XGBoost. No registered family fails on the M9 features.
- **PR-043's refusal**: no `feat__` column is in the `y__` namespace or listed as a diagnostic. A label appended to the same list is still refused (tested); the CLI refusal tests pass unchanged.

**Decisions (routine):** the order attributes stay first (`p_bp, q_usd, h_ms, side`, then `kind` categorical), then the `feat__` columns in the schema's order. No imputation: NaN is XGBoost's native missing value.

---

## Scope

- `configs/problems/m9_{fill_frac,markout_bp,walk_bp}.toml` `[features.spec]`: `numeric_columns` = order attributes + every `feat__` column of the problem's own subtree.
- Tests: `tests/config/test_m9_problem_configs.py` (the list equals the schema's, per subtree; queue-ahead is fill-only; numeric features have numeric schema dtypes; the label quarantine passes and still bites; the nullable set is pinned and fed to a NaN-native family). `tests/cli/test_m9_train_cli.py::test_the_fit_sees_the_feat_columns_with_nulls_as_missing` (per problem, through the real CLI: the fitted and eval frames hold exactly the configured columns, numeric as numbers, each schema null as NaN, all held-out rows scored). `tests/cli/conftest.py` (`c6_set` writes nulls in the schema's possibly-empty features; draws nothing from the rng). `tests/conftest.py` (`m9_feature_columns`, `m9_nullable_features`).
- `docs/ARCHITECTURE.md` "M9 problem layer" (Features bullet); `CHANGELOG.md`; PR-041 Q3 closed.

## Dependencies

`dev` @ `70c9b4a`.

## Verification criteria

- [x] Per problem, `features.spec` = order attributes + the subtree's `feat__` columns in schema order (fill 28, walk 27). The config tests and the CLI test failed first against `dev`'s configs.
- [x] No numeric feature has a non-numeric schema dtype; no `y__` / diagnostic among the features (PR-043 check passes; a label added to the list is refused).
- [x] Schema-empty features reach the XGBoost fit as NaN; `train` exits 0 and scores every held-out row.
- [x] Default suite, ruff, basedpyright `src/`: see the commit message.

## Notes — for the program lead

- Feature count per problem: `m9_fill_frac` 33 and `m9_markout_bp` 33 (32 numeric + `kind`), `m9_walk_bp` 32 (31 + `kind`).
- The M9 problems' `root_cfg_hash` changes because the feature spec is part of it. They have not trained on the real set yet, so no recorded trial is affected.
- The schema says nothing about nulls for the other `feat__` columns (for example `spread_bp` with no l2 row in the lookback). Any null there takes the same NaN path.
