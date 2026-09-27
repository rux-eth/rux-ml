# PR-041: The M9 problem configs — one label per problem, the others as diagnostics

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (the tier is the operator's call at review).** The contract is ruled outside this repo — rux-capital program v0.3 D37, D41, D45 #3 (operator-accepted 2026-09-24) and ACCEPTANCE C9 — and the rux-ml side is mechanical: three TOML files, one optional config layer, one metric. Built by the R-14 **track session** (delegated by the program's lead session, 2026-09-26; operator not present), so the Phase 5 gate below is recorded as track-autonomous and **the merge to `dev` is left to the operator**.

## Research findings

### State Assessment (2026-09-26)

Baselines: rux-ml `dev` @ `fcdbf3f` (PR-040 merge), program @ `79eed8f` (read-only). Local Mac, CPython 3.12, polars 1.40.1, xgboost / sklearn 1.8.0 per `uv.lock`.

**Current state**:
- A previous track agent (out of API credit mid-PR) left this PR uncommitted on `pr-041/m9-problem-configs`: the three problem TOMLs, `config/m9.py` (`M9Config.diagnostic_columns`), the `RuxMLConfig.m9` field with a target/feature validator, an optional-layer hash rule, the `brier` metric, XGBoost's `brier → rmse` eval-metric translation, per-fold diagnostic summaries in `train`, and tests. Written test-first; all its tests passed when resumed.
- `DataConfig` already carries `target_column`, `split_kind = "time_ordered"`, `time_column` (PR-024, PR-027 Int64 epoch-ms); the metric registry had `auc / logloss / rmse / mae` (`training/metrics.py`).
- `RuxMLConfig.solving: SolvingConfig | None` (PR-020) is the precedent for an optional top-level layer — and the precedent for its hazard: PR-020 moved every `root_cfg_hash` silently (PR-040 §Prior-art).
- `categorical_low_card_threshold` has no default (D4 BEST-GUESS); any config with categoricals must set it (`features/encoders.py:61-67`).

**Assumptions at draft time** (program side): the table carries `y__fill_frac`, `y__markout_bp` (at the decision interval), `y__walk_bp` (D45 #3) with keys `stamp_ms, coin, side, kind, p_bp, q_usd, h_ms` (ACCEPTANCE C6); scoring = Brier for the fill fraction, MAE in bp for markout and walk (D45 #3); default-config GBT per target (C9).

**Stale assumptions / defects found on resume** (fixed test-first here):
1. **`brier` broke two families.** Adding it to the shared `METRIC_REGISTRY` without LightGBM / CatBoost translations meant `kind="lightgbm", metric="brier"` handed LightGBM an unknown metric and `kind="catboost"` raised `KeyError` in `_METRIC_TRANSLATE[cfg.metric]`. Both now translate to RMSE (tests `test_lightgbm_brier_metric_translated_to_rmse`, `test_catboost_brier_metric_translated_to_rmse` failed first).
2. **Diagnostics reported nulls as values.** `n` counted nulls, and an all-null column reported `mean = 0.0` (`float(None or 0.0)`). A markout is undefined on an unfilled rung (D37 #2 "conditional on fill by construction"), so nulls are expected. Now `n` counts values, `null_count` counts nulls + NaNs, and an empty statistic is `null` (test `test_label_diagnostics_count_nulls_apart_and_never_report_a_missing_mean_as_zero`, hand-computed values, failed first).
3. The program's D38 #2 key list omits `kind`; ACCEPTANCE C6 (later) includes it. The configs follow C6.

**New constraints**:
- An optional layer must be **hash-neutral while unset** (pinned: `test_unset_layer_leaves_every_pinned_root_hash_unchanged`, the PR-040 pins `06f13e9c…` / `83c3090f…`).
- The training root, the schema file and the label column names are program PR-024's, **not landed**: the source path is a placeholder and the feature list is the order's own attributes only (questions below).

**Prior-art carried forward**: PR-020 (optional layer moved hashes → avoided); PR-024 (config-only refusals live in a `RuxMLConfig` model validator); PR-040 (refusals exit 2 via `typer.BadParameter`; pinned-hash regression tests); LightGBM `_METRIC_TRANSLATE` (PR-018) as the translation precedent.

### Research Questions, Findings, Group D (light — Tier-1)

1. *Is sklearn's `brier_score_loss` usable for a fractional outcome?* **No — proven locally**: sklearn 1.8.0 raises `ValueError: The type of the target inferred from y_true is continuous but should be binary`. The general form mean((p − y)²) is the Brier score for probabilistic forecasts of an outcome in [0, 1]; computed as `mean_squared_error` and cross-checked by hand (0.140625 on y = {0, .25, 1, .5}, p = .5). Alternative considered: log loss on the fraction (cross-entropy) — rejected, the program ruled Brier (D45 #3).
2. *Early stopping on Brier.* Brier = MSE = RMSE² on the same rows, a monotone transform, so every family early-stops on its RMSE metric and selects the same iteration. **Proven** (arithmetic).
3. *Fitting a fraction in [0, 1].* XGBoost `objective="reg:logistic"` accepts labels in [0, 1] and predicts in (0, 1), so the Brier scorer's range check never refuses a trained forecast. Alternative: `reg:squarederror` with clipping — rejected (clipping hides out-of-range forecasts). **Convention** (XGBoost documents `reg:logistic` for probability regression). The markout / walk problems take no objective override (default `reg:squarederror`) — C9 says default config; MAE is the reported score, not the objective (flagged as a question).
4. Web search was not run (track session, budget-capped, no agents). Every identifier above was verified against the installed libraries (Group D, Probe 1: `brier_score_loss` refusal reproduced; `XGBRegressor.get_params()["objective"] == "reg:logistic"` asserted in tests; LightGBM `metric="rmse"` and CatBoost `eval_metric="RMSE"` asserted on constructed trainers).

### Synthesis

**Outcome: Confirm** (the resumed work stands; two defects fixed test-first).
- ARCHITECTURE gains "M9 problem layer"; `RuxMLConfig` listing gains `m9`. CONSTRAINTS unchanged. No new prerequisite PR.

### Gate Check

- Premise still valid: ✓ (the program's C9 contract).
- No prerequisite PRs surfaced: ✓ (program PR-024's schema file is an input, not a rux-ml prerequisite; the placeholders are named).
- User approved updated spec: **pending** — track-autonomous build under the program's parallel-session ruling (2026-09-21); the branch is local and unmerged for the operator's review.

---

## Scope

- `configs/problems/m9_fill_frac.toml` (target `y__fill_frac`, metric `brier`, `objective = "reg:logistic"`), `m9_markout_bp.toml` (target `y__markout_bp`, `mae`), `m9_walk_bp.toml` (target `y__walk_bp`, `mae`); each time-ordered on `stamp_ms`, default XGBoost hyperparameters, features = the order attributes `p_bp, q_usd, h_ms` (numeric) + `side, kind` (categorical), the other two labels as `[m9] diagnostic_columns`.
- `M9Config` (`[m9]`, optional, hash-neutral while unset); `RuxMLConfig` validator: target ∉ diagnostics, diagnostics ∩ features = ∅.
- `brier` metric (regression, minimize), with RMSE early-stopping translations for XGBoost / LightGBM / CatBoost.
- `rux-ml train`: per-fold diagnostic summaries in `fold_meta.json` (`diagnostics`); a missing diagnostic exits 2.

Out of scope (their own R-14 PRs): the symbol-holdout split (PR-042), the `y__`-as-feature refusal on real files (PR-043), the honesty metric (PR-044), the leakage tests (PR-045), the learning curve (PR-046).

## Dependencies

`dev` @ `fcdbf3f`. Consumes program PR-024's training table when it lands (placeholders named below).

## Architecture section implemented

`docs/ARCHITECTURE.md` "M9 problem layer" (new); `RuxMLConfig` (the `m9` field).

## Verification criteria

- [x] Each problem loads through the real layered loader with its target, metric and the other two labels as diagnostics (`tests/config/test_m9_problem_configs.py`, 15 tests).
- [x] No `y__` / `oracle__` column and no key beyond the order attributes among the features; target never a diagnostic; diagnostics never features (config refusals, `tests/config/test_m9_config.py`).
- [x] Default-config GBT: no hyperparameter differs from `XGBoostTraining` defaults.
- [x] Unset `[m9]` leaves both pinned `root_cfg_hash` values unchanged; a set layer moves it.
- [x] `brier` = mean((p − y)²), refuses either side outside [0, 1]; all three families early-stop on RMSE.
- [x] `train` records diagnostics per fold with nulls counted apart; a missing diagnostic exits 2.
- [x] Default suite: `dev` 467 passed / 1 skipped → branch 499 passed / 1 skipped (all extras installed), 0 failures; ruff clean; basedpyright adds no error in touched files.
- [ ] Run on the real materialized set — blocked on program PR-024.

## Research backing

Program `docs/0.3/DESIGN-log.md` D37, D41, D45 #3; `docs/0.3/ACCEPTANCE.md` C6, C9. Local library probes above.

## Notes — questions for the program lead (placeholders in the configs)

- Q1: the training root's path and set-id convention (`data.source_path` is `data/m9/training_set`, a placeholder).
  - **ANSWERED 2026-09-26 by program PR-024 amendment A7 (operator-approved):** the set has two per-target subtrees — `fill/` (post-only rows: `y__fill_frac`, `y__markout_bp`) and `walk/` (taker rows: `y__walk_bp`). Each problem's `data.source_path` now names its subtree (`data/m9/training_set/{fill,walk}`); pinned by `tests/config/test_m9_problem_configs.py::test_problem_reads_its_own_training_subtree`. The root's location stays a placeholder until PR-024's materializer lands.
- Q2: `categorical_low_card_threshold = 8` is BEST-GUESS; for `side` / `kind` (2–3 values) any value ≥ 3 behaves identically, so it is not load-bearing until a higher-cardinality categorical appears.
- Q3: the `feat__` feature list — to be read from PR-024's schema file (C10); today the features are the order attributes only.
  - **ANSWERED 2026-09-27 by PR-049** (program D38: the model's features are the `feat__` columns; C10: "the feature list from the schema file"; the features accepted by the operator 2026-09-27): the list is read from rumpy-harness `spec/m9_training_schema.json` v2 (harness `18bda8b`, sha256 `fc900836…e5e3`; vendored verbatim at `tests/fixtures/m9_training_schema.json`), `targets.<subtree>.columns` filtered to the `feat__` prefix — 28 columns in `fill/`, 27 in `walk/` (`feat__queue_ahead_usd` is fill-only). Each problem names them explicitly after the order attributes in `features.spec.numeric_columns`; pinned by `tests/config/test_m9_problem_configs.py::test_features_are_the_subtrees_feat_columns_plus_the_order_attributes`, and driven end to end by `tests/cli/test_m9_train_cli.py::test_the_fit_sees_the_feat_columns_with_nulls_as_missing`.
- Q4: the exact label column names (is the decision-interval markout literally `y__markout_bp`, or a horizon-suffixed column?), the horizon-curve and time-to-fill columns to list as diagnostics, and **the null semantics of `y__markout_bp` on unfilled rows** — whether the markout problem trains on filled rows only (a row filter the program has not defined; today a null target reaches XGBoost and fails unnamed).
  - **Null semantics ANSWERED 2026-09-26 by program PR-024 amendment A9 (operator-approved):** `y__markout_bp` is null on an unfilled row, so the markout problem trains and scores on filled rows only — `[m9] row_filter_non_null = ["y__markout_bp"]`, applied in `make_splits` before the split (train / tune / promote / score); a null (or NaN) target no longer reaches XGBoost (`tests/cli/test_train_subcommand.py::test_row_filter_keeps_null_targets_out_of_the_xgboost_fit`, `tests/data/test_splits.py`). `y__fill_frac` / `y__walk_bp` unchanged. The column-name and horizon-curve / time-to-fill diagnostics parts stay open for PR-024's schema file.
- Q5: the markout / walk problems fit `reg:squarederror` (default config) but are scored by MAE; D45 #3 does not say whether the objective should be `reg:absoluteerror` to match.
