# PR-055: an M9 fit exports its val / test rows; the M9 promote re-fit runs on the batch path

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** R-14 track, program PR-027 build step **R3** on the rux-capital program lead's instruction of 2026-09-28. The research is program PR-027 §Findings Phase 3 **Q7** (the floor as comparator: the proposed interface — rux-ml exports each fit's val / test row keys with the prediction and the realized label, the harness fits the floor on the complement) and Phase 4 amendment **A8**, operator-approved 2026-09-28 ("go on all"). The promote half is **the operator's ruling of 2026-09-28** (program PR-027 "Build progress 2026-09-28"): the promote re-fit, still in memory at 156 days, moves to the batch path in R3. The merge is operator-gated.

## Research findings

### State Assessment (2026-09-28)

Baselines: rux-ml `dev` @ `b1a3186` (PR-054 merged), xgboost 3.2.0, Polars 1.40.1, pyarrow ≥ 17; the default suite 717 passed / 3 skipped.

**Current state (read):**
- `cli/train.py::_fit_and_score_batches` (PR-054) predicts val and test batch by batch (`predict_fold`) and keeps only the scores: `fold_meta.json` carries `split_definition` (the membership rule since PR-054: stamp cuts, keyed hash, symbol lists, prefix stamp) but no row of any partition. The harness therefore cannot score a comparator on the model's test rows, nor fit it on the model's train rows.
- `registry/promote.py::_refit` collects the whole source (`materialize(load_parquet(...))`), splits it in memory, fits the pipeline and the sklearn wrapper on whole pandas partitions. PR-054's notes: "A 156-day `[m9]` promote re-fit would need the batch path too" — program PR-027 R2-b predicted 52–137 GiB for in-memory routes at 156 days against the 20 GiB scope.
- The trial record is the Optuna trial (`TrialAttrs` user attributes, `extra="ignore"` on read) plus the per-study `FileSystemArtifactStore` holding `metrics.json` / `fold_meta.json` (PR-034).

**Prior art carried forward:**
- PR-054: the batch path's predictions are bit-identical to the in-memory fit's; its membership is a rule of the row (the keyed hash, the stamp cuts, the tied-boundary rule); `[m9] row_key_columns` is refused at the split, not at config load.
- PR-034: artifact uploads are in-objective, post-fit, unguarded; Optuna persists their metadata in the trial's system attributes.
- PR-027 R2-b: freed Polars / Arrow memory stays resident — never collect a large frame (the export is streamed, not concatenated).

**Drift:** none against Q7 / A8. Two decisions the build forced:
1. **The purged rows are not exported.** Q7 names them as the harness's to subtract ("minus the purged rows that `split_definition` records"). The rule is stated precisely (below) and proven row for row by a test; exporting them would add a prefix fit's cut rows (up to ≈ 49M at a 25 % prefix).
2. **The promote pipeline is fitted on one day's train rows.** On the batch path it is stateless (every categorical passes through; `category_levels` refuses one above the threshold), so fitting records only the input's column names.

### Synthesis / Gate

**Outcome: Confirm (Q7's interface as proposed, A8 as approved; the operator's promote ruling).** **Gate:** track-autonomous under program PR-027's approved build gate; the operator's approval is pending at merge.

---

## Scope

- **`rux_ml.runs.oos_export`** (new): `OosExportWriter` streams the export through a `pyarrow` `ParquetWriter` (one row group per day file and partition), `close()` returns its record; `file_sha256`; the schema constants.
- **`training/xgboost/batches.py`**: `uses_batch_fit(cfg)` (the one question `train` and `promote` ask); `fold_rows` / `_xy` factored out of `fold_batch` (unchanged behaviour); `predict_fold(..., keys=, sink=)` hands each batch's keys, predictions and realized targets to a sink as they are made.
- **`cli/train.py`**: `_fit_and_score(..., export_dir=)`; the batch fit writes the export into a temporary directory; `run_command` uploads it as an artifact of the trial and records `fold_meta.json` `oos_export` + the user attribute `oos_export_sha256`; an empty `[m9] row_key_columns` is refused (exit 2) before the fit.
- **`runs/attrs.py`**: `TrialAttrs.oos_export_sha256` (optional).
- **`registry/promote.py`**: `_refit` dispatches an `[m9]` re-fit to `_refit_batches` (per-day batches, the trial's `split_seed` / `xgb_seed`); refusals (PR-050, PR-051) unchanged.
- **Tests (new):** `tests/cli/test_m9_oos_export.py` (9), `tests/cli/test_m9_promote_batches.py` (4).
- Docs: `docs/ARCHITECTURE.md` (§ the out-of-sample row export and the batch promote re-fit; the promote flow), `CHANGELOG.md`, `docs/0.3/RESEARCH-BACKLOG.md`.

### The export's schema (the interface program PR-026 / PR-027 H2 consume)

`oos_rows.parquet`, an Optuna artifact of the trial (`<runs.artifacts_root>/<study>/<artifact_id>`; `fold_meta.json` `oos_export.artifact_id`), schema version 1, one row per val / test row, val first, each partition in the batches' order (no meaning):

| column | dtype | meaning |
|---|---|---|
| `partition` | String | `"val"` or `"test"` |
| the `[m9] row_key_columns` (M9 configs: `stamp_ms`, `coin`, `side`, `kind`, `p_bp`, `q_usd`, `h_ms`) | the source's (Categorical / Enum → String) | the row's key |
| `prediction` | Float32 | what the fit scored (`XGBModel.predict` up to the best iteration) |
| `realized` | Float64 | the target cast to float64, null → NaN (NaN rows are outside the test score) |

Key-value metadata `rux_ml.oos_export`: schema version, columns, key / target columns, partitions, the fit's `split_definition`. The trial record: `fold_meta.json` `oos_export` = `{filename, artifact_id, sha256, bytes, rows: {val, test}, columns, key_columns, target_column, schema_version}` and user attribute `oos_export_sha256`.

**The train mask:** a row of the same view is train iff it passes `split_definition.row_filter_non_null` (non-null; not NaN when float), its key is not in the export, and it is not purged — time-block `stamp + split_embargo < cuts.val_first_stamp`; a prefix fit `stamp <= train_prefix_last_stamp`. Rows sharing a key share a partition under every regime, so the complement is exact.

## Dependencies

PR-054 (merged, `b1a3186`).

## Architecture section implemented

`docs/ARCHITECTURE.md` § "The out-of-sample row export and the batch promote re-fit (per PR-055)"; the `registry promote` data-flow block.

## Verification criteria

- [x] **Failing first.** With the new tests on the pre-PR source (`git stash` of `src/`): **13 failed** — 8 × `KeyError: 'oos_export'` (no export in the trial record), 4 × `AssertionError: the [m9] promote re-fit collected the whole source`, 1 × `assert 1 == 2` (the key-less fit ran instead of being refused).
- [x] **The export equals the partitions** (3 regimes): its val / test keys and realized targets equal `make_splits` on the whole frame; its predictions equal `predict_fold`'s bit for bit, in order; the recorded val score (`metrics.json`) and test score (`fold_meta.json` `oos`) recompute exactly from it; the downloaded file's sha256 equals `fold_meta.json` `oos_export.sha256` and the user attribute.
- [x] **The train mask rule** reproduces the train partition row for row (3 regimes at the full prefix, time-block and row-random at a 0.5 prefix).
- [x] **The promote re-fit is bit-identical** to the in-memory re-fit (the pre-PR `_refit` copied verbatim into the test) on the C6 fixture with `subsample` = `colsample_bytree` = 0.8, all three regimes: prediction sha256 through the scorer's route, `best_iteration`, rounds, the pipeline's input names; `materialize` patched to raise. Sensitivity: another trial's seeds give other predictions.
- [x] **Promote end to end** on `m9_fill_frac` through the CLI with `materialize` refused: the bundle saves, with PR-051's harness manifest. PR-050 / PR-051 refusal tests unchanged and green.
- [x] Default suite **730 passed, 3 skipped** (717 + 13 new); `ruff check`; `ruff format --check src tests`; `basedpyright src/` 0 errors.
- [x] **Desktop measurement** (2026-09-28, 21:43:33–21:46:29 CDT; 2 min 56 s under `~/rumpy-heavy.lock`, inside `timeout 1200`). Setup: branch @ `010ac19` in a detached worktree `~/projects/rux-ml-pr055-measure`, xgboost 3.2.0 with CUDA, Polars 1.40.1, pyarrow 24.0.0. Each step ran in `systemd-run --user --scope -p MemoryHigh=20G -p MemoryMax=20G -p MemorySwapMax=0` under `/usr/bin/time -v`, `nvidia-smi` sampled every 0.5 s. Input: the 156-day view `f6a544e240b5-83a175d4b812.s3-1777593600000` `fill/` (156 day files, now on the desktop), `m9_fill_frac`, its time-block split, GPU. Step 1 is the whole `rux-ml train` (it writes the export); step 2 the whole `rux-ml registry promote` of that trial (its `data_hash` + harness manifest checks, then the batch re-fit, then the bundle).

  | | `train` (writes the export) | `registry promote` (the batch re-fit) |
  |---|---|---|
  | process peak RSS (`time -v`) | 15.51 GiB (16,260,092 kB) | **18.20 GiB** (19,083,432 kB) |
  | scope `memory.peak` (incl. page cache) | 16.72 GiB | **18.53 GiB** (≤ 20 GiB; `high` 0, `max` 0, `oom_kill` 0, swap 0) |
  | whole-process wall | 96.4 s | **79.0 s** |
  | GPU memory used (max) | 13,060 MiB | 13,068 MiB |
  | result | test Brier 0.157391 (= PR-054's 0.15739), best iteration 99 | bundle `v_2026_09_29_ba094a` (model.ubj 496,162 B) |

  - **The export at scale:** 13,814,732 val + 14,011,016 test rows (= the split record), **83,173,828 bytes (79.3 MiB, ≈ 3.0 B/row)**, sha256 `58a9787e…c410`. Streaming it cost val / test predict 3.8 / 4.3 s against PR-054's 2.9 / 3.0 s; closing + hashing 0.04 s. The train RSS peak equals PR-054's 15.50 GiB: the export adds none.
  - **The promote peak is 2.7 GiB above train's and unattributed** (no per-stage RSS). Read, not measured: `registry promote` never calls `pin_threads` (`train` / `tune` / `solve` do), so its Polars pool and OpenMP run unpinned — a candidate, not a finding.

## Research backing

Program PR-027 §Findings Phase 3 Q7 (the proposed interface), Phase 4 A8, the build plan row R3 and "Build progress 2026-09-28" (the operator's promote ruling); PR-054 (the batch path, its bit-identity proof); xgboost 3.2 `Booster.predict` (all trees by default — the scorer's route, used on both sides of the promote identity).

## Notes — for the program lead

- **H2's read path:** find the trial's `fold_meta.json` `oos_export`, read `<artifacts_root>/<study>/<artifact_id>`, check its sha256 against `oos_export_sha256`, build the train mask by the rule above on the same view, fit the floor, score it on the `partition == "test"` rows. `realized` NaN rows are outside the model's test score.
- **Size (measured):** 27.8M val + test rows for the 156-day fill fit in 79.3 MiB, not Q7's ≈ 0.7 GB best-guess.
- **Promote headroom is thin:** 18.53 GiB of the 20 GiB scope at time-block, 1.47 GiB left. The symbol-holdout train partition is ≈ 73 % of the filtered rows against time-block's ≈ 70 %, so its re-fit is the tightest case. That case was not measured. If the headroom matters, a per-stage RSS attribution comes next. The first candidate is the unpinned promote above; it is pre-existing, and this PR does not change it.
- **Promote vs trial at scale is unverified.** The bit-identity here is batch re-fit vs in-memory re-fit, in one process with the same threads. Whether the promoted booster reproduces the trial's predictions at 156 days was not checked. The two runs used different thread pinning, so this is open. The export makes a cheap check possible (R4 candidate): predict the export's test rows with the bundle and compare to `prediction`.
- **Tune** keeps the in-memory path and writes no export (the Stage 3 fits are `rux-ml train` runs).
