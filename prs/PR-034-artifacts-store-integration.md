# PR-034: Optuna Artifacts Store integration (diagnostic-only)

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-2** — the architectural sub-decision D3 (artifacts: what to upload) is **locked by the v0.3 design session** before this PR's Phase 1 begins. Lean: diagnostic-only. The alternative (per-trial bundles replace re-fit) reverses D8/PR-010 sub-decision A1 and is out of scope.

This PR triggers a MINOR bump (v0.3.0) per `docs/VERSIONING.md §1` — new per-trial artifact upload behavior + new `TrialAttrs` field (if Q3 lands as "record artifact IDs").

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Open research questions** (must be resolved before implementation):

1. **Optuna `FileSystemArtifactStore` API surface** at the currently pinned Optuna version. The module was renamed once between Optuna 3.x versions; Phase 1 verifies the current install's actual import path. Required: `optuna.artifacts.FileSystemArtifactStore` + `upload_artifact` + `download_artifact` (or equivalents).
2. **When to upload per trial** — in `tuning/objective.py` post-fit (inside the objective closure) or in `_internal/trial_runner.py` post-`study.optimize` (outside the objective)? Lean: in objective post-fit, so the upload is tied to the trial's success and the artifact lands BEFORE `trial.report` finalizes.
3. **What to upload** — Phase 4-locked options:
   - **Minimum**: per-fold predictions CSV (per-fold `y_true`, `y_pred`) + fold-scores JSON.
   - **Medium**: + per-fold confusion matrix / scatter plot PNG (if relevant for the metric).
   - **Maximum**: + per-trial model bundle (replaces re-fit-at-promote — out of scope per D3 lean).
   - **Lean**: Minimum — preds CSV + fold-scores JSON only.
4. **`TrialAttrs.artifact_ids: list[str]` field** — new field for artifact IDs, or query Optuna by trial number at retrieval time? Lean: query by trial number (avoids `TrialAttrs` schema churn; Optuna's artifact-store API supports this natively).
5. **Cleanup policy** — do we ever delete old artifacts? Disk-space implications on the workbench (per-trial CSVs across 100s of trials add up). Lean: no auto-cleanup at v0.3; document manual cleanup via `studies/artifacts/` rm; revisit if disk pressure surfaces.
6. **Per-trial provenance hash impact** — `artifacts_root` is currently path-elided at `config/root.py:60` (excluded from config hashing). After PR-034, does anything change? Lean: no — artifacts_root is still path; still elided. The artifact CONTENT is per-trial, not per-config.

## Scope

Wire up `optuna.artifacts.FileSystemArtifactStore` so per-trial diagnostic artifacts are actually stored. Completes phantom #3 from the 2026-05-20 audit.

### Item 1 — `FileSystemArtifactStore` initialization

New helper in `src/rux_ml/runs/artifacts.py`:

```python
def make_artifact_store(cfg: RuxMLConfig) -> FileSystemArtifactStore:
    """Initialize the artifact store at cfg.runs.artifacts_root."""
    cfg.runs.artifacts_root.mkdir(parents=True, exist_ok=True)
    return FileSystemArtifactStore(str(cfg.runs.artifacts_root))
```

### Item 2 — Per-trial artifact upload

Modify `tuning/objective.py:_fold_scores` (or wrap in a post-`_fold_scores` hook in `objective` closure) to:

1. After all folds complete (before `return statistics.fmean(scores)`):
2. Concatenate per-fold `(y_true, y_pred)` into a single CSV per trial (`trial_{number}_preds.csv`).
3. Serialize fold-scores list into a JSON sidecar (`trial_{number}_fold_scores.json`).
4. Upload via `upload_artifact(study, trial.number, ...)` (or the equivalent at the locked Optuna API).

### Item 3 — `cli/train.py` artifact upload (one-off baseline)

Symmetric upload for the one-fit baseline at `cli/train.py:_fit_and_score`:
- Single (y_true, y_pred) CSV (no folds).
- Single val-fold-score JSON.

### Item 4 — `cli/solve.py` skipped

Solver trials produce `(x_star, objective_value)`, not predictions. No diagnostic CSV; existing `solver_status` + `objective_value` provenance fields are sufficient. Document the exception in CONVENTIONS.md.

### Item 5 — `rux-ml runs show` integration

Augment `cli/runs.py:show` to list artifact IDs (or paths) for the queried trial. New section in the show output: "Diagnostic artifacts".

### Item 6 — Docs update

- `docs/ARCHITECTURE.md` Storage table: artifact-store row updates from documented-aspiration to current-active.
- `docs/CONVENTIONS.md`: subsection on artifact retention + manual cleanup.
- `docs/ARCHITECTURE.md` Data Flow diagram: add per-trial artifact-upload step in the child-process flow.

### Out of scope

- Per-trial bundle uploads (would replace re-fit-at-promote; out of scope per D3 lean).
- Plot generation (PNG plots — Medium tier; not v0.3).
- Auto-cleanup (lean: no auto-cleanup at v0.3).

## Dependencies

- **PR-030** (sprint scaffolding) lands first.
- **v0.3 design session** runs and locks D3 before this PR's Phase 1.

## Architecture section implemented

`docs/ARCHITECTURE.md` Storage table (artifact-store row) + Data Flow diagram (per-trial upload step) + Components table (`runs/artifacts.py` module).

## Verification criteria

Populated after Phase 1 state assessment. Initial sketch:

- [ ] `FileSystemArtifactStore` instantiated in production (`tuning/objective.py` + `cli/train.py` paths).
- [ ] Per-trial preds CSV + fold-scores JSON uploaded after fold loop.
- [ ] `studies/artifacts/<study>/<trial_number>/...` directories populated for a 2-trial test study.
- [ ] Round-trip test: upload + download via Optuna API returns identical bytes.
- [ ] `rux-ml runs show` lists artifact paths.
- [ ] `docs/ARCHITECTURE.md` Storage table updated.
- [ ] `docs/0.3/ROADMAP.md` PR-034 row flipped `[ ]` → `[x]`.
- [ ] `CHANGELOG.md [Unreleased] ### Added` entry for the new behavior.
- [ ] `D8/PR-010 sub-decision A1 (re-fit at promote)` is preserved — verify by inspecting `registry/promote.py` for no behavior change.

## Research backing

Tier-2 — locked by v0.3 design session D3 + Phase 1 verification of Optuna API surface.

Anchored on:
- Optuna `FileSystemArtifactStore` docs (version-locked)
- `RunsConfig.artifacts_root` field declaration (PR-007)
- D8/PR-010 A1 (re-fit at promote — preserved)
- v0.3 design session D3 (to be written)

## Notes

- **Triggers MINOR bump** (v0.3.0) per `docs/VERSIONING.md §1`.
- **D8/PR-010 A1 preserved**: this PR does NOT change `registry/promote.py`'s re-fit behavior. Artifacts are diagnostic only.
- Per memory `feedback_roadmap_flip_in_pr`: PR-034's implementation commit must flip `docs/0.3/ROADMAP.md` PR-034 row `[ ]` → `[x]` in the same commit.
- Disk-space: each trial's preds CSV is roughly `n_test_rows × n_folds × 2 floats × 8 bytes`. For crypto-h3 at 20.7M rows × 10 folds × 2 × 8 = ~3.3 GB per trial. **This is too large for "diagnostic only".** Phase 4 must downscale: per-trial summary stats only (per-fold RMSE / count / etc.) OR per-fold sub-sampled preds (e.g., 1% sample). Lock in design session.
