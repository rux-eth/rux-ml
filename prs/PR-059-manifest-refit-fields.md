# PR-059: the promoted model manifest records how to reproduce the evaluated model

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1.** A track of the rux-capital program lead (2026-09-30). The research is program PR-028a (`prs/PR-028a-registry-model.md`): Phase 2 items **Q4e / Q6 / Q16**, Phase 3 **DG-1 / DG-2** (desktop, evidence `docs/0.3/evidence/pr028a-dg{1,2}-20260930.json`), and Phase 4 **item 3**, operator-approved 2026-09-30 ("go", after 12:09 CDT). Item 3's walk honesty form follows the operator's ruling of 2026-09-30 on program `docs/0.3/RESEARCH-m9-walk-underdeduct.md` decision 3: *"whatever is most honest. we never make dishonest predictions just to flatter ourselves."* The merge is the program lead's.

## Research findings

### State Assessment (2026-09-30 ≈ 12:20 CDT)

Baseline: rux-ml `origin/dev` @ `0bce381` (PR-058 merged), the base program PR-028a read. The default suite there is 748 passed, 3 skipped (with `uv sync --all-extras`).

**The program's reads, re-checked at `0bce381` (no drift):**
- `ModelManifest` is `extra="forbid"` and has no row count, no iteration range, no device and no out-of-sample record (`registry/manifest.py:72-110`).
- Promote hashes `cfg.data.source_path` (`registry/promote.py:239`, `data_hashes` → `compute_data_hash` → `iter_parquet_files`, `data/versioning.py:149-172`) and the batch re-fit opens `SourceBatches(source_path).files = iter_parquet_files(path)` (`registry/promote.py:195-197`, `data/partitions.py:210`). The lists are equal by construction and nothing asserts it (Q6).
- `fit_batches` records `best_iteration` and `iteration_range = (0, best_iteration + 1)` and `predict_fold` passes the range (`training/xgboost/batches.py:255-263, :287`); promote returned only the booster and dropped both. DG-2 measured the consequence: `best_iteration` survives `model.ubj`, but XGBoost 3.2.0's predict default uses all 100 trees, 1.048 bp off the evaluated walk model on the worst row (Q4e).
- The trial's TEST record lives in its `fold_meta.json` artifact (`oos`, `cli/train.py:515`), while the manifest's `metric_value` is the Optuna value, the VAL MAE (0.4551 on the walk; DG-2). The trial's `oos_export_sha256` and `booster_device` / `booster_nthread` are in `TrialAttrs` (`runs/attrs.py:94-100`).
- **Item 3 is small.** PR-058's `signed_error_honesty` already reports both forms under `forms`, and `verdict_form` names the one the verdict keys carry. Switching the walk's verdict to `clipped_at_zero` is one TOML value; the signed verdict stays in `forms.signed`, and the markout and fill records do not change.

**Decisions the build forced (leans, each one line to change):**
1. **Shape.** The re-fit fields are one closed block, `refit`: `train_rows`, `val_rows`, `best_iteration`, `iteration_range`, `device`, `nthread`. `iteration_range` is validated against `best_iteration`: `(0, best + 1)` when early stopping ran, else XGBoost's `(0, 0)` (every round), the rule `XGBModel._get_iteration_range` and `fit_batches` use.
2. **The `oos` block is the trial's record verbatim.** It is read from the trial's `fold_meta.json` before the re-fit. Its named keys are typed (`partition`, `n_rows`, `n_scored`, `metric`, `score`, `honesty`) and every other key is kept as written (`extra="allow"`). A trial with no record promotes with `oos = None`; a trial with several records or several `fold_meta.json` artifacts refuses, because which one the model is promoted on would be a guess.
3. **Q6's list is the hash's own**, returned by the one enumeration the hash makes (`compute_data_hash_and_files`), never a second enumeration. `rux-ml data hash`'s output is unchanged. The assertion covers the batch path, as the program asked; the in-memory path reads through `load_parquet`, whose files the bridge certifies (program Q6 residual (a)).
4. **No new equality gate on the trees.** Promote does not refuse when the re-fit's `best_iteration` differs from the trial's. Promote ≡ trial is asserted by the tests and was measured by DG-2; a refusal would be new behaviour the approval did not name. Flagged for the lead.
5. **No `test` row count in `refit`.** The approval names train / val; the TEST count is `oos.n_rows`.

### Phases 2–4 (light)

- **No web round.** Every question was answered by program PR-028a's reads and DG-2's measurement at XGBoost 3.2.0.
- **Budget.** Promote gains one `fold_meta.json` download (kilobytes) and one list comparison. The hash still enumerates the source once, and nothing runs per row. The fit and predict paths are untouched: `training/` has no change.

### Synthesis / Gate

**Outcome: Confirm (program PR-028a Phase 4 item 3 as approved; the leans above).** Gate: covered by the operator's Phase 4 approval of 2026-09-30. The merge is the program lead's.

---

## Scope

- **`registry/manifest.py`**: new `RefitRecord` (closed, with the range validator) and `OosRecord` (named keys typed, the rest verbatim). New optional `ModelManifest` fields `refit`, `oos_export_sha256` and `oos`. Old manifests load with all three `None`.
- **`registry/promote.py`**:
  - `_refit` / `_refit_batches` return the `RefitRecord` beside the pipeline and booster, and take `hashed_files`. The batch path refuses with `PromoteFileSetError` (a `ValueError`, exit 2) unless `SourceBatches.files` equals the hashed list, before the plan or any fit.
  - `trial_oos_record` and `_trial_fold_meta` read the trial's `fold_meta.json` `oos` before the re-fit.
  - The manifest carries `refit`, `attrs.oos_export_sha256` and `oos`.
- **`data/versioning.py`** `compute_data_hash_and_files`, and **`runs/provenance.py`** `data_hashes_and_files`: the hashes plus the file list they covered. `compute_data_hash` and `data_hashes` return what they returned before.
- **`configs/problems/m9_walk_bp.toml`**: `honesty_verdict_form = "clipped_at_zero"`.
- **Tests**:
  - New `tests/registry/test_manifest_refit_fields.py` (4): the round trip through `save_bundle` / `load_bundle`, an old manifest loads, an unknown key still refuses, and the range validator.
  - New `tests/cli/test_m9_promote_manifest_refit.py` (4): an `[m9]` train + promote end to end, with every field held to the trial's own record (`split_definition.rows`, `row_count`, `TrialAttrs`, `fold_meta` `booster` / `oos_export` / `oos`) and to the loaded booster. Also the no-record and several-records cases, and the file-set refusal for a file added after the hash and for a file the hash did not cover.
  - `tests/registry/test_promote.py` (+1): the in-memory re-fit's record against an independent `make_splits`.
  - `tests/config/test_m9_problem_configs.py`: the walk's form.
  - `tests/cli/test_m9_promote_batches.py`: the bit-identity test passes the hashed list (the call's signature; the assertions are unchanged).
- **Docs**: `docs/ARCHITECTURE.md` (the promote flow; the A7 honesty forms bullet; the new paragraph after the harness-manifest-id paragraph), `CHANGELOG.md`, `docs/0.3/RESEARCH-BACKLOG.md`.

## Dependencies

None open: PR-051 (the harness manifest id), PR-054 (the batch fit), PR-055 (the export and the batch promote) and PR-058 (the honesty forms) are merged in `dev` @ `0bce381`.

## Architecture section implemented

`docs/ARCHITECTURE.md`: the new paragraph "The model manifest records how to reproduce the evaluated model", and the promote flow diagram's re-fit and compose steps.

## Verification criteria

- [x] The new tests failed before the code. On `0bce381`'s `src` + `configs` with this PR's tests: 3 collection errors (`OosRecord`, `data_hashes_and_files` did not exist), `test_promote_records_the_in_memory_refit` failed, and the walk's config test failed. They pass after.
- [x] An `[m9]` promote records the re-fit's train / val rows equal to the trial's partition plan, `best_iteration` equal to the trial's and to the loaded booster's, `iteration_range = (0, best_iteration + 1)`, the fit's device / nthread equal to the trial's, the trial's `oos_export_sha256`, and its TEST record verbatim (end to end, `m9_markout_bp`).
- [x] The fields round-trip through `save_bundle` / `load_bundle`, and a manifest without them still loads.
- [x] A changed file set refuses (a file added after the hash; a file the hash did not cover), naming the file.
- [x] Promote ≡ trial is untouched: `training/` has no change, and `test_the_batch_refit_predicts_bit_identically_to_the_in_memory_refit` passes for all three regimes with its assertions unchanged.
- [x] The walk's verdict is `clipped_at_zero`, and the signed form is still reported (PR-058's `forms`, unchanged).
- [x] Mac: `uv run pytest` gives **757 passed, 3 skipped, 0 failed**. `ruff check .`, `ruff format --check src tests` and `basedpyright src/` are clean.
- [ ] Desktop: `make test-golden`. It is unaffected by construction: `golden_v1` promotes through the in-memory path, which now also writes `refit`, and its assertions are on predictions, not on the manifest. It was not run here.
- [ ] The program lead's merge.

## Research backing

Program PR-028a §Phase 2 (Q4e :82, Q6 :58-63, Q16 :71-79), §Phase 3 DG-1 / DG-2 (:164-264, "The rux-ml manifest PR's fields — confirmed by DG-2"), §Phase 4 item 3 (:271). Program `docs/0.3/RESEARCH-m9-walk-underdeduct.md` decision 3, ruled 2026-09-30.

## Notes

- **The walk's hash.** `m9_walk_bp`'s `root_cfg_hash` changes, because `honesty_verdict_form` is in the hashed `[m9]` layer. Promote does not compare a trial's `root_cfg_hash` with the stack's, so the existing walk trial still promotes. The version id and the manifest's hashes are the trial's.
- **The existing walk trial's TEST record has no clipped form.** Study `m9_walk_bp_m9_regime_time_block_20260929T155106Z` trial 0 was fitted at `48ffd4e`, before PR-058, so its `fold_meta.json` `oos.honesty` is PR-044's signed record (0.695) without `forms`. Promoting it at this PR's code carries that record verbatim. The clipped figure (0.726) appears in a bundle only if the trial is re-run at this code, or if the program computes the verdict on the served values (PR-028b). That is the lead's call.
- **For consumers (PR-028b):** predict with `iteration_range=tuple(manifest.refit.iteration_range)`. The loaded booster's device is `cpu`, and `refit.device` is the fit's.
- **Pre-existing, not this PR's:** `basedpyright src/` reports missing-import errors unless the worktree is synced with `--all-extras` (LightGBM / CatBoost).
