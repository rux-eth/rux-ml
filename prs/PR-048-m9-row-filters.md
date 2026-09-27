# PR-048: The M9 problems fit the materialized set — row filters, own-subtree diagnostics, `side` numeric

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1.** Delegated by the rux-capital program lead after program PR-024 B-4 landed the materializer; **operator-approved for merge 2026-09-27 ("go")**. Config + tests only; no `src/` change.

## Research findings

### State Assessment (2026-09-27)

Baselines: rux-ml `dev` @ `f9acc68` (PR-041 … PR-047 merged, #47–#53), default suite 595 passed / 2 skipped. Harness schema: `rumpy-harness` `spec/m9_training_schema.json` v2 (commit `18bda8b`, sha256 `fc900836…e5e3`), read from the PR-024 worktree.

What the materializer writes (schema v2): `<set>/{fill,walk}/<day>.parquet`, flat; C6's keys, then `y__`, then `feat__`; the EV targets unsuffixed (`y__fill_frac`, `y__markout_bp`, `y__walk_bp`); `no_book` / `below_one_lot` / `alo_expired` rows WRITTEN with `y__outcome` and NaN targets (the label builder fills `np.nan`); `y__walk_bp` null where the IOC filled nothing (≈ 2 %).

**Defects found (each shown failing first on a schema-faithful fixture):**
1. `m9_fill_frac` and `m9_walk_bp` had no row filter: a NaN target reached XGBoost — `XGBoostError: Label contains NaN` (an unnamed crash, not exit 2). Markout already filtered (PR-041 / A9).
2. PR-041's diagnostics crossed subtrees (`y__walk_bp` for the two `fill/` problems; `y__fill_frac`, `y__markout_bp` for the `walk/` problem): on the real layout `train` exits 2 ("m9.diagnostic_columns … not in the training set columns").
3. **Found on the way (not in the order):** `side` is `int8` (±1) in the set but was a categorical feature; Polars refuses `cast(pl.Categorical)` from `i8` (`InvalidOperationError`, probed on polars 1.40.1), so every problem failed at the feature step. `side` now enters as numeric (a binary ±1 splits identically).
4. Target names: all three already equal the schema's unsuffixed EV targets — no change; now pinned by test.

**Decisions (routine, flagged):**
- Diagnostics = every other `y__` label of the problem's own subtree, except `y__outcome` (constant `ok` on the rows the target filter keeps) and `y__ttf_first_ms` / `y__ttf_full_ms` ("-1 when none": a sentinel the summary would average in as a value, the defect class PR-041 fixed for nulls). This also resolves PR-041 Q4's horizon-curve part from the schema file.
- The schema is vendored verbatim at `tests/fixtures/m9_training_schema.json`; `RUXML_M9_SCHEMA_PATH` runs a drift test against the live harness file (passed against `18bda8b`).
- The CLI fixture `c6_set` now writes the schema's layout and exact columns / dtypes (NaN on no-row outcomes, null on unfilled markouts and ≈ 2 % of walks), so every M9 CLI test runs on what the materializer writes.

---

## Scope

- `configs/problems/m9_fill_frac.toml`, `m9_walk_bp.toml`: `[m9] row_filter_non_null = [<target>]`.
- All three problems: `diagnostic_columns` from the own subtree; `side` numeric, `kind` categorical.
- Tests: `tests/config/test_m9_problem_configs.py` (every configured column in the own subtree's schema; diagnostics rule; unsuffixed EV target; categoricals are strings; drift), `tests/cli/test_m9_train_cli.py::test_no_missing_target_reaches_the_xgboost_fit` (per problem: an XGBoost spy sees no NaN, `row_filter_dropped` = the independently counted missing targets, `n_scored == n_rows`), `tests/cli/conftest.py` (schema-faithful `c6_set`), `tests/conftest.py` (schema helpers).

Out of scope: appending the schema's `feat__` columns to the features (PR-041 Q3 / C10); the real-set runs.

## Dependencies

`dev` @ `f9acc68`.

## Architecture section implemented

`docs/ARCHITECTURE.md` "M9 problem layer" (own-subtree diagnostics; every problem filters its target; `side` numeric).

## Verification criteria

- [x] Per problem, a fixture with NaN / null target rows trains with exit 0 and no missing value in the fitted or eval target (failed first for fill and walk with "Label contains NaN").
- [x] Every configured column (target, time, group, diagnostics, row filter, features) is in the problem's own subtree per the schema (failed first on all three).
- [x] The three targets equal the schema's unsuffixed EV targets, one problem each.
- [x] Default suite 595 → 610 passed, 2 → 3 skipped (the env-gated drift test), 0 failed; ruff clean; basedpyright `src/` clean.

## Notes — for the program lead

- The `feat__` columns (29 in `fill/`, 28 in `walk/`) are not yet features: the features remain the order attributes (PR-041 Q3 / C10 still open on the rux-ml side).
- `kind` and `h_ms` (and `p_bp` in `walk/`) are constant per subtree: harmless to XGBoost, uninformative.
