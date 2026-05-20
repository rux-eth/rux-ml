# PR-032: `rux-ml registry score` CLI verb + holdout scorer

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** by inheritance from PR-031 (holdout semantics) per memory `feedback_tier_inheritance`. The scorer module is new code consuming `splits["test"]` — multiple sub-decisions (CLI verb signature; receipt JSON schema; predictions parquet layout; whether `compute_score` works for raw booster + DMatrix).

This PR triggers a MINOR bump (v0.3.0) per `docs/VERSIONING.md §1` — new CLI verb + new public scorer module. Ships under v0.3.0 alongside PR-031 + PR-033 + PR-034 + PR-035.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Open research questions** (must be resolved before implementation):

1. **CLI verb signature** — positional vs flag args; default `version = "champion"`; required `--problem`; optional `--output`. Survey MLflow/Aim/W&B for "score promoted model on holdout" CLI conventions.
2. **Receipt JSON schema** — mirror `prs/PR-025-calibration-results.json` (study-level receipt) or new shape (single-bundle-score receipt)? Fields: `bundle_version`, `promoted_from`, `holdout` (row count + time range + split kind), `score`, `scored_at`, `git_sha`, `manifest_git_sha`, `preds_path`.
3. **Predictions parquet column layout** — minimal (`y_true`, `y_pred`) or wider (timestamp + asset + y_true + y_pred)? Larger panels (20.7M rows for crypto-h3 → 15% test = ~3M rows) — wider columns multiply storage cost.
4. **`compute_score` compatibility** — `compute_score(metric, trainer, x_eval, y_eval)` assumes a Trainer wrapper with `predict_proba` / `predict`. For a loaded bundle (raw `xgb.Booster`), we need `booster.predict(xgb.DMatrix(x))`. Either add a Booster-aware `compute_score` overload OR build a thin shim wrapping the Booster as a Trainer.
5. **Test fold reconstruction seed** — for `time_ordered`, deterministic (no seed needed). For `random`, need `bag.split_seed` reconstructed from the trial's `entropy_hex` (manifest doesn't carry `entropy_hex`; must re-open the Optuna DB via `load_run(study, trial)`).
6. **Receipts directory convention** — gitignored or tracked? PR-025 calibration receipt is tracked at `prs/PR-025-calibration-results.json`. Direction-4 chat consensus: tracked under `receipts/` at repo root; gitignore `receipts/*.parquet` (large, regeneratable); track `receipts/*.json`.

## Scope

New CLI verb + scorer module. Loads a promoted bundle, reconstructs the holdout test fold, scores the bundle, writes receipts.

### Item 1 — `rux-ml registry score` CLI verb

```
rux-ml registry score \
    --problem <name> \
    [--version <version>] \     # defaults to champion
    [--output <dir>]            # defaults to receipts/
```

Implementation in `src/rux_ml/cli/registry.py`:

1. Load `RuxMLConfig` via `_load_cfg(ctx)`.
2. Resolve version: explicit arg > `read_champion(...)["version"]`.
3. Load bundle: `pipeline, booster, manifest = load_bundle(version_dir(...))`.
4. Materialize source parquet, call `make_splits(cfg, df, seed=<resolved>)`.
5. Score on `splits["test"]` via the new `score_bundle_on_holdout` helper.
6. Write receipt JSON + predictions parquet under `--output`.

### Item 2 — `score_bundle_on_holdout` helper

New function in `src/rux_ml/registry/scorer.py` (or extended `bundle.py`):

```python
def score_bundle_on_holdout(
    cfg: RuxMLConfig,
    pipeline: Pipeline,
    booster: xgb.Booster,
    manifest: ModelManifest,
    df: pl.DataFrame,
) -> HoldoutScoreReceipt:
    ...
```

Returns a Pydantic `HoldoutScoreReceipt` model that JSON-serializes to the receipt schema (defined in Phase 4 after Q2 resolution).

### Item 3 — `promote.py` stale comment cleanup

Remove the misleading docstring at `registry/promote.py:97`:

> "The test fold is unused — held out for future golden-regression evaluation (PR-014)."

Replace with: "The test fold is unused — consumed by `rux-ml registry score` for held-out-window evaluation (PR-032)."

### Item 4 — Receipts directory + gitignore

- `receipts/` at repo root, git-tracked.
- `.gitignore` entry: `receipts/*.parquet` (large, regeneratable; receipts JSON still tracked).

### Item 5 — Docs + tests

- `docs/ARCHITECTURE.md` Storage table: add `receipts/` row.
- `docs/CONVENTIONS.md`: subsection on the receipts convention.
- Unit tests for `score_bundle_on_holdout` (synthetic small bundle).
- CLI test: `test_registry_score_writes_receipt` in `tests/cli/test_registry_subcommands.py`.

### Out of scope

- Multi-bundle scoring sweep (one bundle at a time).
- Per-asset / per-time-period breakdown (point-estimate RMSE only; diagnostic deep-dive is a separate follow-up).

## Dependencies

- **PR-030** (sprint scaffolding) lands first.
- **PR-031** (HPO honors holdout) — semantic prerequisite. The holdout score is only meaningfully "held out" after PR-031 stops HPO from seeing the test fold.

## Architecture section implemented

`docs/ARCHITECTURE.md` Storage section (new `receipts/` store) + Components table (new scorer module under `registry/`).

## Verification criteria

Populated after Phase 1 state assessment. Initial sketch:

- [ ] `rux-ml registry score --problem crypto_breakout_h3` writes a valid receipt JSON + preds parquet end-to-end on a synthetic-but-real bundle in `tests/`.
- [ ] Receipt JSON validates against the new `HoldoutScoreReceipt` Pydantic schema.
- [ ] `compute_score` Booster-aware path (or shim) returns the same RMSE for a known-deterministic bundle.
- [ ] `promote.py:97` stale comment replaced.
- [ ] `receipts/*.parquet` is gitignored; `receipts/*.json` is tracked.
- [ ] `docs/ARCHITECTURE.md` + `docs/CONVENTIONS.md` updated in the same commit.
- [ ] `docs/0.3/ROADMAP.md` PR-032 row flipped `[ ]` → `[x]`.
- [ ] `CHANGELOG.md [Unreleased] ### Added` entry for the new verb + module.

## Research backing

Tier-2 — pattern locked by existing `cli/registry.py` verbs (promote, list, rollback) + `load_bundle` API. Sub-decisions open per the 6 questions in `## Research findings` above.

Anchored on:
- `cli/registry.py` precedent (promote, list, rollback)
- `load_bundle` API (PR-010)
- `compute_score` metric registry (PR-014 + PR-006)
- `make_splits` dispatcher (PR-024)

## Notes

- **Triggers MINOR bump** (v0.3.0) per `docs/VERSIONING.md §1`.
- **Semantically paired with PR-031**: the score's "held out" claim is only honest after PR-031.
- Per memory `feedback_roadmap_flip_in_pr`: PR-032's implementation commit must flip `docs/0.3/ROADMAP.md` PR-032 row `[ ]` → `[x]` in the same commit.
- A future PR may add per-asset / per-time-period breakdown to the receipt for richer diagnostics (out of v0.3 scope).
