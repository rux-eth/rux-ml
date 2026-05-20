# PR-035: `rux-ml tune retry-trial` integration test

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**This PR is Tier-1** — test-only PR; pattern inherited from existing `tests/cli/test_tune_subcommands.py` test functions (start/resume/status). No architecture decisions. The architectural sub-decision D4 (test-only vs behavior change) is **locked by the v0.3 design session** before this PR's Phase 1 begins; lean is test-only.

This PR triggers a PATCH bump (rolls into v0.3.0 cut bundle) per `docs/VERSIONING.md §1` — test-only addition, no behavior change.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`. Do not begin implementation until this section exists with completed findings from all required phases._

**Open research questions** (must be resolved before implementation):

1. **Body drift check** — verify `cli/tune.py:206-228` retry-trial body has not drifted since the 2026-05-20 audit. Specifically:
   - Still loads study via `optuna.load_study(...)` (line ~217)
   - Still validates trial exists with `next(t for t in study_obj.trials if t.number == trial_id)` (line ~221)
   - Still enqueues with `study_obj.enqueue_trial(prior.params)` (line ~227)
   - Still calls `study_obj.optimize(objective, n_trials=1)` (line ~228)
2. **Test fixture pattern** — existing `tune_workdir` fixture in `tests/cli/test_tune_subcommands.py` provides a tmp_path-scoped workdir + TOML setup. Reuse directly for `test_tune_retry_trial`.
3. **Failed-trial state simulation** — how to produce a "failed trial" in the test? Options: (a) mock the objective to raise on first invocation; (b) use Optuna's `study.tell(trial, state=TrialState.FAIL)` to inject a failed trial directly; (c) configure a search-space value that produces NaN. Lean: (b) — cleanest, no implementation dependency.

## Scope

Add `test_tune_retry_trial_*` to `tests/cli/test_tune_subcommands.py`. Exercises enqueue → optimize → verify. No production code change.

### Item 1 — `test_tune_retry_trial_enqueues_prior_params`

```python
def test_tune_retry_trial_enqueues_prior_params(
    runner: CliRunner, tune_workdir: Path
) -> None:
    # 1. Start a 2-trial study; one trial succeeds with known params P1; second fails (state=FAIL).
    # 2. Run `rux-ml tune retry-trial <study_name> 1` against the failed trial number.
    # 3. Assert: study now has 3 trials; trial 2 params match the failed trial's prior params.
    # 4. Assert: trial 2 state is COMPLETE (or FAIL with new metric if objective still rejects).
```

### Item 2 — `test_tune_retry_trial_missing_trial_raises_bad_parameter`

```python
def test_tune_retry_trial_missing_trial_raises_bad_parameter(
    runner: CliRunner, tune_workdir: Path
) -> None:
    # 1. Start a 2-trial study.
    # 2. Run `rux-ml tune retry-trial <study_name> 99` (non-existent trial number).
    # 3. Assert: CLI exits with BadParameter (exit code 2 in Typer convention).
    # 4. Assert: error message includes "trial #99 not found".
```

### Item 3 — `test_tune_retry_trial_missing_study_raises_bad_parameter`

```python
def test_tune_retry_trial_missing_study_raises_bad_parameter(
    runner: CliRunner, tune_workdir: Path
) -> None:
    # 1. No study created.
    # 2. Run `rux-ml tune retry-trial nonexistent_study 0`.
    # 3. Assert: CLI exits with BadParameter; error message includes "study 'nonexistent_study' not found".
```

### Item 4 — Test-file docstring correction

`tests/cli/test_tune_subcommands.py:1` claims coverage for `{start, resume, status, retry-trial}`. The claim was correct in intent (the docstring was written when retry-trial verb was added) but the test functions were never added. After this PR, the docstring matches reality.

### Out of scope

- Behavior change to `cli/tune.py:retry_trial` body (per D4 lean; test-only).
- Refactor of the test fixture (`tune_workdir` already adequate).

## Dependencies

- **PR-030** (sprint scaffolding) lands first.
- **v0.3 design session** locks D4 before Phase 1 (mostly formality; lean is test-only).

## Architecture section implemented

`docs/ARCHITECTURE.md` Testing Strategy table (CLI row gets coverage backfill for an existing verb).

## Verification criteria

Populated after Phase 1 state assessment. Initial sketch:

- [ ] 3 new test functions in `tests/cli/test_tune_subcommands.py`: enqueue-prior-params + missing-trial + missing-study.
- [ ] All 3 tests pass under `uv run pytest tests/cli/test_tune_subcommands.py`.
- [ ] Default suite count moves from 367 → 370.
- [ ] `cli/tune.py:retry_trial` body is unchanged.
- [ ] `docs/0.3/ROADMAP.md` PR-035 row flipped `[ ]` → `[x]`.
- [ ] `CHANGELOG.md [Unreleased] ### Fixed` (test coverage) entry — backfill for an existing-but-untested verb.

## Research backing

Tier-1 — test-only PR. Phase 1 verifies the retry body hasn't drifted; Phases 2-4 are formality.

Anchored on:
- `cli/tune.py:206-228` (existing retry-trial verb body)
- `tests/cli/test_tune_subcommands.py` (existing test pattern for start/resume/status)
- Optuna's `study.tell(trial, state=TrialState.FAIL)` API for fixture setup

## Notes

- **PATCH bump only** (rolls into v0.3.0 bundle).
- Per memory `feedback_roadmap_flip_in_pr`: PR-035's implementation commit must flip `docs/0.3/ROADMAP.md` PR-035 row `[ ]` → `[x]` in the same commit.
- Smallest scope in v0.3. Quick win to start the sprint and establish the v0.3 ROADMAP-flip cadence.
- The test-file docstring at `tests/cli/test_tune_subcommands.py:1` is corrected (or rather, made true) by this PR.
