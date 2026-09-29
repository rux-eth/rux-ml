# PR-054: the M9 fits stream per-day batches through XGBoost's native API

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** R-14 track, program PR-027 build step **R2** on the rux-capital program lead's instruction of 2026-09-28. The research is program PR-027 §Findings Phase 3 Rounds 1–3 (Q2's copy inventory, R2-b / R2-b′ / R2-c / R2-d) and its Phase 4 amendments **A1, A3, A4, A11**, operator-approved 2026-09-28 ("go on all", the Phase 4 → 5 gate), A4 as a ruling that changes partition membership. The merge is operator-gated.

## Research findings

### State Assessment (2026-09-28)

Baselines: rux-ml `dev` @ `aec5464` (PR-053 merged), xgboost 3.2.0, Polars 1.40.1, pandas 3.0.3, NumPy 2.4.5; the default suite 680 passed / 3 skipped.

**Current state (read):**
- `cli/train.py::_fit_and_score` materialises the whole source (`materialize(load_parquet(...))`), splits it in memory (`make_splits`), summarises the ten `[m9] diagnostic_columns` from the partitions, runs the feature pipeline (Polars → pandas → Polars) and converts to pandas again for the sklearn wrapper. Program PR-027 R2-b measured the result: 127–137 GiB predicted at 156 days as wired, 52 GiB for the best per-partition pandas variant; the walk as wired is OOM-killed at 20 GiB.
- The row-random split is `df.sample(fraction=1.0, shuffle=True, seed)`: a permutation of the collected frame, not a rule a batch can apply, and not reproducible outside rux-ml's Polars version.
- The time-block split sorts by stamp (an unstable sort) and slices by row count, so on a stacked panel the cut lands inside a stamp and the tied rows are split arbitrarily (the earlier side is then purged by the embargo).
- The native path that exists (PR-033 `XGBoostNativeAdapter`) takes whole frames; its ExtMem branch writes the whole frame to temp Parquets through `ParquetDataIter`, which feeds every non-target column (labels included) and applies no filter or partition.
- No trial records how many threads the booster used; `omp_threads` (24) does not bound a fit whose `n_jobs` is set (program PR-027 R2-c: `nthread` 32).

**Drift found while building (each decided in the Synthesis):**
- **Time-block ties.** A partition rule that a single batch can apply is a stamp predicate, and a stamp predicate cannot split one stamp's rows. The rows tied on a boundary stamp therefore move: they go to the later partition, where the row-count slice had split them by an unstable sort and purged the earlier side. The spike that measured this route (program PR-027 `r2_spike.py` `lever_p`) made the same choice. Every other row stays where the slice put it; on unique stamps the partitions are identical.
- **`[m9]` on a classification problem.** Three existing tests put an `[m9]` layer on the synthetic AUC problem. No M9 problem is a classifier (Brier / MAE), and the sklearn classifier sets its objective and label encoding inside `fit`. So the batch path covers regression targets, and an `[m9]` classification fit keeps the in-memory path under the same partition rule.
- **The native adapter predicts with every tree (pre-existing, not changed).** `Booster.predict` defaults to `iteration_range=(0, 0)`, which means all trees (xgboost 3.2 `core.py`). The sklearn wrapper predicts up to `best_iteration + 1`, but PR-033's `XGBoostNativeAdapter.predict` passes no range. With early stopping it therefore scores with the `early_stopping_rounds` trees after the best as well. The batch path here follows the wrapper. The adapter is flagged below, not fixed.

**New constraints (prior art carried forward):**
- PR-053: a per-file scan must use the first file's schema, or it silently accepts what one scan refuses. `SourceBatches` does this.
- PR-027 R2-b′: a float32 cast before XGBoost is value-neutral.
- PR-027 R2-b: freed Polars / Arrow memory stays resident within the run, so the path must never collect a large frame in the first place.

### Synthesis / Gate

**Outcome: Confirm (A1 / A3 / A4 / A11 as approved), with four decisions the build forced:**
1. Time-block boundary ties go to the later partition (above).
2. The batch path is regression-only; an `[m9]` classification fit stays in memory.
3. X above `data.gpu_in_memory_x_gb_max` is refused, not silently moved to `ExtMemQuantileDMatrix`. At 156 days it is about 8.6 GB of float32, so it is not needed, and it is not built.
4. `[m9] row_key_columns` is refused at the split, not at config load, so configs that never split row-random stay valid.

**Gate:** track-autonomous; the operator's approval is pending at merge.

---

## Scope

- **`rux_ml.data.partitions`** (new): the `[m9]` split as a rule of the row.
  - `plan_partitions` computes the rule's inputs from the key columns, batch by batch: the time-block stamp cuts, the symbol-holdout membership, the row-random key spec, and the train-prefix cut.
  - It also builds a per-batch tally `(fold, stamp, group) → rows`.
  - `PartitionPlan.assign` gives any row its fold.
  - The keyed hash is `splitmix64` / `row_random_unit`.
  - `SourceBatches` reads a day file per batch (the first file's schema, the row filter in the scan); `FrameBatches` does the same for in-memory frames.
  - `split_record` builds the split record, and `diagnostics_by_batches` the label summary (A3, moments merged per batch).
- **`rux_ml.training.xgboost.batches`** (new): the DataIter and the fit.
  - `FoldIter` hands XGBoost one file's partition rows at a time, through `fold_batch` (float32 numeric columns, one fixed category list).
  - `fit_batches` builds the train `QuantileDMatrix` and the val one (`ref=` train), then `xgb.train` with the sklearn wrapper's own parameters.
  - `predict_fold` predicts batch by batch as `XGBModel.predict` does. `Scored` scores stored predictions.
  - `category_levels` gives the fixed category lists (it refuses a categorical that would need a fitted encoder). `booster_threads_device` implements A11.
- **`cli/train.py`**: `_fit_and_score_batches` for `[m9]` XGBoost regression fits. The in-memory path is unchanged except for `split_definition(..., seed=)` and the booster record. `_leakage` is split into the window, the audit and the verdict, so the counts audit shares the verdict.
- **`data/splits.py`**: `make_splits` / `split_definition` apply the plan to an `[m9]` frame (non-`[m9]` splits are unchanged). The ratio, unit-interval and row-filter helpers moved to `partitions` (aliases kept).
- **`data/leakage.py`**: `leakage_audit_counts`. `_within` is the one core shared with `leakage_audit`.
- **Config and records:**
  - `[m9] row_key_columns` is new; the three M9 problem configs list the harness schema's keys.
  - `TrialAttrs.booster_nthread` / `booster_device`, recorded in `train` (every path) and in `tune` (the last fold).
  - `XGBoostNativeAdapter.booster`.
- **Tests (new):**
  - `tests/data/test_m9_membership_rules.py` (7): the public surface with a pure-Python reference of the hash.
  - `tests/data/test_m9_partitions.py` (19).
  - `tests/cli/test_m9_batches.py` (6): bit-identity and the batch hook.
  - `tests/cli/test_m9_batch_train_cli.py` (5): the CLI end to end.
- **Tests (updated inputs):**
  - `test_m9_train_cli.py`'s two spies move from `XGBRegressor.fit` to the batch hook and check the same properties over the train and val batches.
  - `test_train_subcommand.py` (three `[m9]` fits on the synthetic set) and `test_splits.py` (one) name `row_key_columns`. Their partition sizes are read from `split_definition` rather than `int(n · ratio)`.
- Docs: `docs/ARCHITECTURE.md` (§ the M9 per-day batch fit; the `random` split kind), `CHANGELOG.md`, `docs/0.3/RESEARCH-BACKLOG.md`.

## Dependencies

PR-053 (merged, `aec5464`).

## Architecture section implemented

`docs/ARCHITECTURE.md` § "The M9 per-day batch fit and the partition rules (per PR-054)"; § One-off split kinds (`random`).

## Verification criteria

- [x] **Failing first**, on `dev` @ `aec5464` with the new tests only:
  - `test_m9_membership_rules.py` + `test_m9_batch_train_cli.py`: **11 failed, 1 passed** (the lock that a non-`[m9]` random split keeps the seeded shuffle).
  - Behavioural failures:
    - the time-block partitions differed at the tied rows (`AssertionError`);
    - `the [m9] fit collected the whole source` (the `materialize` refusal);
    - `KeyError: 'booster_nthread'`;
    - `DID NOT RAISE` for a row-random split without keys.
  - Missing-surface failures: the `row_key_columns` knob (`ValidationError`) and `split_definition(seed=)` (`TypeError`).
  - `test_m9_partitions.py` / `test_m9_batches.py`: collection errors on the missing modules.
- [x] **A1, bit-identity.** On the C6-layout fixture (three day files, CPU, `subsample` = `colsample_bytree` = 0.8 so that the seed and the row order matter), the batch fit and the pre-PR in-memory fit (the sklearn wrapper on the whole pandas partitions) were compared for all three regimes:
  - equal: the val and test **predictions (sha256)**, the val score, `best_iteration` and the train row count;
  - the realized test targets are equal too.
  - **Sensitivity checked:** reversing the file order made all three fail.
- [x] **A1, never a whole frame.** XGBoost receives only one day file's rows of one partition per batch:
  - the train matrix comes from several batches;
  - float32 numeric columns plus the category, and no key, target or diagnostic column;
  - `rux-ml train` on an M9 problem completes with `materialize` patched to raise.
- [x] **A3.** The diagnostics, split definition and leakage audit a fit records equal those computed from the whole frame split in memory, for each regime. The diagnostics hold to 1e-12 relative (moments merged per batch); the other two are exactly equal. A missing diagnostic column still exits 2.
- [x] **A4.** Row-random membership equals a pure-Python splitmix64 / sha256 reference row by row. It is:
  - independent of the row order;
  - keyed by the seed;
  - refused without keys, or with a null / NaN / unsupported key.
  - The published splitmix64 vector (seed 0 → `0xE220A8397B1DCDAF`) holds.
- [x] **A11.** With `n_jobs = 3` the trial records `booster_nthread = 3` and `booster_device = "cpu"`, which is also in `fold_meta.json` `booster`.
- [x] **The plan is layout-invariant.** Day batches vs one frame give the same assignment, tally and record for each regime, with and without the prefix. The symbol plan equals `symbol_holdout_split`, and the time prefix equals `train_prefix`.
- [x] Default suite **717 passed, 3 skipped** (680 + 37 new); `ruff check`; `ruff format --check` on the touched files (the one pre-existing `scripts/calibrate_pr025.py` untouched); `basedpyright src/` 0 errors.
- [ ] **Desktop measurement** (program PR-027 R2's run, the fill fit at 39 and 156 days, GPU, in a 20 GiB scope): recorded below when run.

## Research backing

Program PR-027 §Findings Phase 3 Round 1 Q2 (the copy inventory; the XGBoost float32 storage proof; the partition-as-predicate candidate), Rounds 2–3 (R2-b memory, R2-b′ float32 bit-equality, R2-c/d wall), Phase 4 A1 / A3 / A4 / A11. xgboost 3.2.0 `sklearn.py` (`get_xgb_params`, `_create_dmatrix`, `_get_iteration_range`, `predict`) and `core.py` (`Booster.predict`'s `(0, 0)` default) read for the parameter and prediction identity.

## Notes — for the program lead

- **Membership changes (A4 as ruled; also the time-block ties).** Row-random partitions differ from the shuffle's, and their sizes are the ratios in expectation (a 156-day fill set ≈ 93M rows: the deviation is ≈ 0.01 %). Time-block val / test gain the rows tied on the boundary stamps: fewer than one stamp's rows at each cut (≈ 3.9k fill rows at 163 coins × 24).
- **Promote / tune** use `make_splits` on the whole frame: the same membership, but still the in-memory path. A 156-day `[m9]` promote re-fit would need the batch path too (not in R2's row).
- **The PR-033 adapter's predictions** use every tree when early stopping ran (see the State Assessment). Nothing on the M9 path uses it; a one-line fix for a later PR.
- **Wall.** The batch path re-reads the day files: the plan pass, the diagnostics pass, about two to three passes per `QuantileDMatrix`, and the predictions. The desktop measurement reports the per-stage seconds (`fold_meta.json` `ingest.seconds`).
