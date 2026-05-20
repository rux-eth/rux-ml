# PR-035: `rux-ml tune retry-trial` integration test

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**This PR is Tier-1** — test-only PR; pattern inherited from existing `tests/cli/test_tune_subcommands.py` test functions (start/resume/status). No architecture decisions. The architectural sub-decision D4 (test-only vs behavior change) is **resolved by this PR's Phase 1 state assessment (2026-05-20)** — verdict: test-only. The verb body at `cli/tune.py:206-231` is logically correct; the gap is purely test coverage.

This PR triggers a PATCH bump (rolls into v0.3.0 cut bundle) per `docs/VERSIONING.md §1` — test-only addition, no behavior change.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

### Phase 1 — State Assessment (2026-05-20)

**Current state** (HEAD = `6b8fb45`, post-PR-030 merge on `dev`):

- `src/rux_ml/cli/tune.py:206-231` retry-trial body verified intact, unchanged since 2026-05-20 audit. Three failure paths handled: missing study (`KeyError` → `BadParameter`); missing trial (`StopIteration` → `BadParameter`); successful retry (`enqueue_trial` + `optimize(n_trials=1)`).
- `tests/cli/test_tune_subcommands.py` fixture (`tune_workdir`, lines 16-76) provides synthetic 200-row classification dataset + base.toml (kfold n_splits=3, n_trials=2, sampler=tpe, pruner=wilcoxon). Reusable directly.
- Existing test functions follow established idioms: `runner.invoke(app, _argv(...), catch_exceptions=False)`; reload study from SQLite for state verification; assert `result.exit_code != 0` for failure tests (no specific exit-code value).
- Module docstring at line 1 claims coverage for `{start, resume, status, retry-trial}` — overstates reality pre-PR-035; closed by this PR.

**Drift assessment**: `cli/tune.py` last touched PR-017 (~7 days ago); test file last touched PR-017. Well within 60-day staleness. **Zero drift since 2026-05-20 audit.**

**Stale assumptions**:
1. PR-035 stub references "v0.3 design session locks D4" — no design session runs per option-1 from 2026-05-20 chat. **D4 resolved by this Phase 1: test-only.** The verb body is correct; the gap is coverage.
2. Stub references body at lines `206-228`; actual body spans `206-231`. Mechanical correction in the implementation commit.
3. Stub `Open research questions` Q3 listed failed-trial-state simulation options. **Re-scoped via Phase 1 body-read**: retry verb is state-agnostic (no `t.state` check); tests use successful trials directly. No simulation method needed.

**Prior-art audit** (per post-PR-028/29 procedure):
- `cli/tune.py` history: PR-017 → PR-013 → PR-011 → PR-008 → PR-007 → PR-003. Last touched PR-017 (Trainer registry refactor).
- `test_tune_subcommands.py` history: PR-017 → PR-013 → PR-011 → PR-008 → PR-007. Patterns stable since PR-007.
- Known-good patterns to carry forward: `_argv(workdir, *args)` helper; `catch_exceptions=False` for success tests (default `True` for failure tests); SQLite reload for state verification; assertion message includes `result.stderr or result.stdout` for diagnostic.
- Known-bad patterns surfaced: none in this PR's area.

**New constraints**:
1. `study.enqueue_trial(params)` + `study.optimize(n_trials=1)` is the validated retry idiom (used by the production body itself). No external research needed to validate the combination.
2. The `runner` fixture lives in `tests/cli/conftest.py:9` (`@pytest.fixture def runner() -> CliRunner: return CliRunner()`).

**Exit decision**: Premise still valid; no prerequisite PRs surfaced; stale assumptions resolved at Phase 1; proceed to Phase 2 (light per Tier-1).

### Phase 2 — Scope the Research (2026-05-20, light)

**Must-answer**:

1. **Q1 — Failed-trial state simulation method.** **Re-scoped**: the retry verb body at `cli/tune.py:220-227` doesn't check trial state; it just looks up `prior.params` by `trial_id`. **No simulation needed.** Tests use successful trials directly. Resolved by Phase 1 body-read; research skipped per Phase 2 "may be skipped" clause.

2. **Q2 — Test scope (which test functions to add)**. **Lean (convention-matching, in-repo cited)**: three tests, one per distinct code path in `cli/tune.py:206-231` (success + missing study + missing trial). Mirrors the existing `test_tune_status_reports_best_value` + `test_tune_status_missing_study_raises_bad_parameter` pattern. No external research needed.

**Dependencies**: Q1 and Q2 are independent; both resolved at Phase 2 without dispatching agents.

**Explicitly excluded from this round** (nice-to-have):
- Parametrized failure-mode tests (over-engineering for 3 paths).
- Specific BadParameter exit-code assertion (Typer convention is `exit_code != 0`; existing tests don't assert specific values).
- Re-verification of `peak_rss_mb` / `user_attrs` after retry (already end-to-end covered by `test_tune_start_runs_2_trials_and_records_user_attrs`).
- Behavior change to `cli/tune.py:retry_trial` (locked test-only at Phase 1).
- Multi-retry chain test (fragile; not a documented use case).

### Phase 3 — Findings (skipped per documented exception)

Per `PROCEDURE-pr-research.md` Phase 2 note ("**When research may be skipped**: if a decision is trivial, well-known, or covered by a prior research-backed decision, document the reasoning explicitly"):

- **Q1 skipped**: resolved by Phase 1 body-read (verb is state-agnostic).
- **Q2 skipped**: convention-matching with in-repo cited example (`test_tune_status_*` pattern).

**Group D MCP Verification** (run at Phase 1 for Tier-1 audit-light path):

| Probe | Scope filter | Verdict |
|---|---|---|
| Schema-Integrity | N/A — no new identifiers introduced | Filtered out |
| Synthesis-Verification | In-scope — test combines CliRunner + `tune_workdir` + `study.load_study` patterns | **Verified**: `test_tune_resume_appends_trials:149` uses the exact 3-element combination |
| Binding-at-creation | N/A — pure test additions, no state-registration surface | Filtered out |

### Phase 4 — Synthesis (2026-05-20)

**Outcome**: **Confirm** — Phase 1 + Phase 2 findings support the original PR-035 spec. Three minor stub corrections applied in same commit (line range, failed-trial framing, D4 lock-source); none change the PR's premise or scope.

**Changes to this PR from research**:
- Stub line-range corrected `206-228` → `206-231`.
- Item 1 test body in original stub said "second [trial] fails (state=FAIL)" — scrubbed; tests use successful trials directly.
- D4 lock-source: "v0.3 design session" → "PR-035 Phase 1 state assessment 2026-05-20".

**Changes to ARCHITECTURE.md**: None — test coverage backfill; no architecture change.

**Changes to CONSTRAINTS.md**: None.

**Changes to CONVENTIONS.md**: None.

**Changes to docs/0.3/ROADMAP.md**: PR-035 row flips `[ ]` → `[x]` in this commit.

**Changes to docs/0.3/RESEARCH-BACKLOG.md**: PR-035 row updates from `design-research ✓` to `state-assessed 2026-05-20` + `implementation-cleared 2026-05-20`.

**Changes to docs/0.3/DESIGN-log.md**: No new session entry — Tier-1 with light Phase 2 doesn't warrant a design-log session.

**New PRs that must come first**: None.

**Research-backed details now locked in this PR**:
1. **D4 verdict**: test-only.
2. **Failed-trial simulation**: not needed; retry verb is state-agnostic.
3. **Test count**: 3, one per distinct code path.
4. **Test fixture**: reuse existing `tune_workdir`.
5. **Verification idiom**: reload study from SQLite; assert on params equality / exit code.
6. **Retry idiom**: `study.enqueue_trial(params) + study.optimize(n_trials=1)` (production-path cited).

### Phase 5 — Gate Check (2026-05-20)

- Premise still valid: ✓
- No prerequisite PRs surfaced: ✓
- No design-planning loop needed: ✓
- User approved updated spec: ✓ (2026-05-20)
- Implementation cleared: ✓ (2026-05-20)

---

## Scope

Add three integration tests for `rux-ml tune retry-trial` to `tests/cli/test_tune_subcommands.py`. No production code change. Phase 4 corrections applied to this stub in the same commit.

### Item 1 — Three test functions

#### `test_tune_retry_trial_enqueues_prior_params`

Start a 2-trial study via `rux-ml tune start ... --n-trials 2` (both succeed). Run `rux-ml tune retry-trial smoke_retry 0`. Reload the study from SQLite; assert `len(trials) == 3` and `trials[2].params == trials[0].params`. Retry verb is state-agnostic, so no failure injection needed — the verb retries any trial by trial_id.

#### `test_tune_retry_trial_missing_trial_raises_bad_parameter`

Start a 2-trial study. Run `rux-ml tune retry-trial smoke_retry_missing_trial 99` (non-existent trial). Assert `result.exit_code != 0` (matches existing `test_tune_status_missing_study_raises_bad_parameter` pattern at line 181).

#### `test_tune_retry_trial_missing_study_raises_bad_parameter`

No study created. Run `rux-ml tune retry-trial nonexistent_study 0`. Assert `result.exit_code != 0`.

### Item 2 — No production code change

`cli/tune.py:retry_trial` body at lines 206-231 is unchanged. D4 verdict from Phase 1: test-only.

### Item 3 — Test-file module docstring

`tests/cli/test_tune_subcommands.py:1` already claims coverage for `{start, resume, status, retry-trial}` — the docstring was correct in intent. After this PR, the claim matches reality.

### Out of scope

- Behavior change to `cli/tune.py:retry_trial` (D4 = test-only).
- Refactor of the `tune_workdir` fixture (already adequate).
- Per-asset / per-time-period diagnostic backfill (not v0.3 scope).

## Dependencies

- **PR-030** (sprint scaffolding) lands first. ✓ (merged at `6b8fb45`)
- D4 sub-decision resolved by this PR's Phase 1 state assessment (no separate design session per the option-1 plan locked 2026-05-20).

## Architecture section implemented

`docs/ARCHITECTURE.md` Testing Strategy table (CLI row gains coverage backfill for an existing verb).

## Verification criteria

- [x] 3 new test functions in `tests/cli/test_tune_subcommands.py`: enqueue-prior-params + missing-trial + missing-study.
- [x] `uv run pytest tests/cli/test_tune_subcommands.py` — 8 passed, 1 gpu-deselected.
- [x] `cli/tune.py:retry_trial` body unchanged.
- [x] `docs/0.3/ROADMAP.md` PR-035 row flipped `[ ]` → `[x]` in implementation commit.
- [x] `CHANGELOG.md [Unreleased] ### Fixed` test-coverage entry.
- [ ] Default `uv run pytest` suite count moves from 367 → 370.
- [ ] `uv run ruff check .` clean.
- [ ] `uv run basedpyright src/` clean.

## Research backing

Tier-1 — test-only PR. Phase 1 verified the retry body hasn't drifted; Phases 2-4 light per documented research-skip exception; Phase 5 gate-checked.

Anchored on:
- `cli/tune.py:206-231` (existing retry-trial verb body — verified intact)
- `tests/cli/test_tune_subcommands.py` existing test pattern (in-repo cited)
- Optuna's `study.enqueue_trial` + `study.optimize(n_trials=1)` (production-path validated)

## Notes

- **PATCH bump only** (rolls into v0.3.0 bundle at PR-036).
- Per memory `feedback_roadmap_flip_in_pr`: PR-035's implementation commit flips `docs/0.3/ROADMAP.md` PR-035 row `[ ]` → `[x]` in the same commit. Streak preserved.
- Smallest scope in v0.3 sprint; chosen as the warm-up to establish v0.3 ROADMAP-flip cadence.
- The test-file module docstring at line 1 was already accurate in intent; this PR closes the gap between intent and reality.
- D4 sub-decision (test-only vs behavior change) resolved by Phase 1 body-read on 2026-05-20: retry verb body is correct; only test coverage is missing.
