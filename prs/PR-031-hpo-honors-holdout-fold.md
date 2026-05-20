# PR-031: HPO objective honors holdout fold

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** per memory `project_cv_strategy_tier2` (anything touching CV/splits requires Tier-2). The architectural sub-decision D1 (holdout semantics — truly held out vs post-HPO sanity check) is **locked by the v0.3 design session** before this PR's Phase 1 begins. Phase 1 state assessment is mandatory regardless of design-time research.

This PR triggers a MINOR bump (v0.3.0) per `docs/VERSIONING.md §1` — behavior change in HPO substrate. Ships under v0.3.0 alongside PR-032–PR-035.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Open research questions** (must be resolved before implementation):

1. **Sklearn / Nixtla / mlfinlab / AutoGluon TS convention** on global-holdout-vs-CV-substrate. Is the workbench's current behavior (CV operates on full df) the convention, or the exception?
2. **Metric shift quantification** on crypto-h3 baseline pre/post change. Run a 10-trial baseline study under PR-031 semantics on the same splits + seeds as the PR-025 calibration; compare to the RMSE 0.027101 receipt.
3. **`data.split_kind == "random"` consistency**: should the test fold also be excluded for the random-split path (for symmetry with `time_ordered`)?
4. **Backward compatibility**: existing study TOMLs will produce different metrics after this PR. Does the `studies/studies.db` storage need a schema migration, or does the CV-substrate change live entirely in the objective (per-trial behavior — no DB schema impact)?

## Scope

Modify `tuning/objective.py` so HPO CV operates on `splits["train"] + splits["val"]` instead of `df_full`. The test fold becomes truly held out from HP search.

### Item 1 — `build_objective` substrate change

`build_objective(base_cfg)` currently materializes `df_full = materialize(load_parquet(source_path))` and uses the full dataframe as the CV substrate. Change to materialize, then carve via `make_splits(base_cfg, df_full, seed=base_cfg.tuning.entropy)` (or a deterministic-once seed per the temporal path) and concatenate `splits["train"] + splits["val"]` as the new CV substrate. The test fold is set aside and unused by HPO.

The seed for `make_splits` must be deterministic across trials — the global substrate is shared. For `time_ordered`, the split is seed-independent (deterministic). For `random`, use `base_cfg.tuning.entropy` directly (or a fresh `make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=-1)` constant — to be decided in Phase 2/4).

### Item 2 — `cli/train.py` consistency (already-honors-test path; no change)

`cli/train.py:73` already calls `make_splits` and uses only `["train"] + ["val"]` for fitting (test fold ignored). No code change required at the CLI level.

### Item 3 — Re-fit at promote stays unchanged

`registry/promote.py:110` already uses `splits["train"] + splits["val"]` for re-fit. No code change required at the registry level.

### Item 4 — Crypto-h3 baseline metric receipt refresh

Bundle a baseline-receipt-refresh into the PR: re-run the 10-trial baseline study (`configs/studies/crypto_breakout_h3_baseline.toml`) under PR-031 semantics; commit the new metric receipt to `prs/PR-031-baseline-receipt.json`. The PR-025 calibration RMSE 0.027101 was measured with the OLD CV substrate (100% of data) — the new RMSE will differ. Documenting both prevents the metric shift from being mistaken for regression.

### Item 5 — Docs update

- `docs/ARCHITECTURE.md` "Decision Rules" → "Optuna sampler / pruner" subsection: update the CV-substrate description to note the test-fold exclusion.
- `docs/CONVENTIONS.md` "HPO objective shape" subsection: same.
- `docs/ARCHITECTURE.md` Storage table or Data section: clarify that `splits["test"]` is now actually held out, not vestigially carved.

### Out of scope

- Adding a CLI verb to score on `splits["test"]` (PR-032's territory).
- ExtMem path activation (PR-033's territory).
- Optuna Artifacts Store wiring (PR-034's territory).

## Dependencies

- **PR-030** (sprint scaffolding) lands first.
- **v0.3 design session** runs and locks D1 (holdout semantics) before this PR's Phase 1.

## Architecture section implemented

`docs/ARCHITECTURE.md` "Decision Rules" → Optuna sampler/pruner + CV-strategy-by-data-shape subsections. The change makes the carved test fold semantically meaningful for the first time since PR-024.

## Verification criteria

Populated after Phase 1 state assessment. Initial sketch:

- [ ] `tuning/objective.py:build_objective` substrate is `splits["train"] + splits["val"]`, not `df_full`.
- [ ] Existing CV tests under `tests/tuning/` updated for new substrate row counts.
- [ ] New behavior test: a study run with `data.split_kind="time_ordered"` confirms test-fold rows are NOT seen by any fold's `x_te` indices.
- [ ] `prs/PR-031-baseline-receipt.json` committed with the new crypto-h3 baseline metric under PR-031 semantics (+ delta vs PR-025 calibration's 0.027101 for transparency).
- [ ] `docs/ARCHITECTURE.md` + `docs/CONVENTIONS.md` updated in the same commit.
- [ ] `docs/0.3/ROADMAP.md` PR-031 row flipped `[ ]` → `[x]`.
- [ ] `CHANGELOG.md [Unreleased] ### Changed` entry documenting the behavior change loudly.

## Research backing

Tier-2 — locked by v0.3 design session D1 before Phase 1.

Anchored on:
- `make_splits` dispatcher (PR-024)
- `temporal_train_val_test_split` determinism (PR-024)
- v0.3 design session D1 (to be written)

Phase 1 must verify:
- `tuning/objective.py:259-261` still uses `df_full = materialize(load_parquet(source_path))` at HEAD (zero drift since PR-025).
- No subsequent v0.3 PR has shifted the substrate semantics.

## Notes

- **Triggers MINOR bump** (v0.3.0) per `docs/VERSIONING.md §1`.
- **Behavior-changing PR**: existing study TOMLs produce different metrics after this PR. CHANGELOG entry must be loud.
- **Pairs with PR-032**: PR-032's holdout-scorer is only semantically meaningful AFTER PR-031 makes the test fold truly unseen by HPO. PRs are independently mergeable but bundled in v0.3 cut.
- Per memory `feedback_roadmap_flip_in_pr`: PR-031's implementation commit must flip `docs/0.3/ROADMAP.md` PR-031 row `[ ]` → `[x]` in the same commit.
- The crypto-h3 RMSE under PR-031 will be a fresh empirical receipt — the 0.027101 number from PR-025 calibration is NOT a regression target; it's a different-substrate measurement.
