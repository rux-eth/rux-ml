# PR-030: v0.3 sprint scaffolding

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**This PR is Tier-1.** It is sprint-bookkeeping only — no code change, no architecture decision. Pattern inherited from the v0.2 sprint precedent (`docs/0.2/{DESIGN-log, ROADMAP, RESEARCH-BACKLOG}.md` were created at the start of the v0.2 design session; PR-026 was the explicit version-cut sibling). v0.3 starts from a phantom audit rather than a real-dataset run, so the scaffolding PR is explicit.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

### State Assessment (2026-05-20)

**Current state** (HEAD = `1af27d9`, post-PR-029 merge on `dev`):

- v0.2 sprint closed; `docs/0.2/ROADMAP.md` all rows `[x]`. v0.2.0 tagged at `1fa43c0`.
- `CHANGELOG.md [Unreleased]` holds 2 procedural entries (PR-028 sync; PR-029 stack-neutral cleanup). No code-change entries.
- `docs/0.3/` does NOT exist — to be created by this PR.
- `prs/` numbered through PR-029. Next free number: PR-030.
- 4 phantom implementations confirmed by the 2026-05-20 audit (two parallel passes + manual verification of two false-positive candidates):
  1. `splits["test"]` allocated by `make_splits`, read by zero production code paths (grep verified zero hits in `src/rux_ml/`).
  2. ExtMem ingest infrastructure (`ParquetDataIter`, `select_ingest`'s ExtMem branch, `cache_host_ratio` MemoryConfig field, `use_native` XGBoostTraining field, `_check_extmem_compat` guard at `tuning/objective.py:153`) — every component wired in design + docs, never instantiated/read in production. `cli/train.py:87-90` honestly admits "the native ExtMem branch executes via xgb.train + ParquetDataIter once a problem actually exceeds the threshold (see training/ingest.py for the contract)" — confirming this is documented future-use, not active behavior.
  3. Optuna `FileSystemArtifactStore` — declared by `RunsConfig.artifacts_root` + path-elided at `config/root.py:60` + set in test fixtures — never instantiated or used; zero production hits across `src/`.
  4. `rux-ml tune retry-trial` — verb body real (`cli/tune.py:206-228`); no test function in `tests/cli/test_tune_subcommands.py` (only `test_tune_start_*`, `test_tune_resume_*`, `test_tune_status_*` exist despite docstring claiming retry-trial coverage).
- Two false-positive phantom candidates verified clean: subprocess watchdog (covered via `build_objective(cfg)` chain — Watchdog instantiated at `objective.py:302` inside the closure that `trial_runner.py:128` invokes); CLI promote `pydantic.ValidationError` handling (Pydantic v2's `ValidationError` IS a `ValueError` subclass, caught by existing `except ValueError` at `cli/registry.py:42`).

**Assumptions at PR draft time**: none — this is sprint-bookkeeping only.

**Stale assumptions**: none.

**New constraints learned from prior PRs**:
- The `feedback_roadmap_flip_in_pr` discipline says every PR-NNN commit flips its ROADMAP row from `[ ]` to `[x]`. PR-030 creates the v0.3 ROADMAP with PR-030 itself at `[ ]`; the implementation commit flips it to `[x]` per discipline.
- v0.2 design session output landed as a single doc-only commit (no associated numbered PR file at the time). v0.3 differs by giving the scaffolding step its own numbered PR — preferred for paper-trail discipline as the project grows.

**Exit decision**: no drift. Premise still valid; pure-docs PR; proceed to Phase 2 (light).

### Phase 2 — Scope Research (2026-05-20, light)

No external research required — sprint-bookkeeping with locked precedent (PR-021 v0.1.0 cut + PR-026 v0.2.0 cut for version-cut shape; v0.2 design-session doc commit for design-log shape).

**Must-answer:** none (sprint structure decisions are user-approved leans from the 2026-05-20 chat; locked in `docs/0.3/DESIGN-log.md` S1).

**Excluded from this PR**: 
- The v0.3 design session itself (D1–D4 architectural sub-decisions). Runs as a separate activity AFTER PR-030 merges, before PR-031 implementation.
- Any code change. Pure docs + PR stubs.

### Phase 3 — Findings (2026-05-20, no Group D probes required)

No identifiers, no combinations, no vendor bindings — pure project scaffolding with file-creation only. Group D Scope filter: out — no implementation-specific claims; pure-methodology PR. Per the post-PR-028/29 procedure, Group D is mandatory before locking Phase 4; this PR's Group D output is **Scope-filtered: N/A across all 3 probes** (Probe 1 no canonical-documenter targets, Probe 2 no combinations, Probe 3 no state-registration surface).

### Phase 4 — Synthesis (2026-05-20)

**Outcome**: **Confirm** — no architectural decisions to lock; the v0.3 sprint structure is the PR's scope.

**Changes to this PR**: none from research.

**Changes to ARCHITECTURE.md**: none (no behavior change).

**Changes to CONSTRAINTS.md**: none.

**Changes to CONVENTIONS.md**: none.

**Changes to ROADMAP.md**: creates `docs/0.3/ROADMAP.md` with rows for PR-030..PR-036.

**New PRs that must come first**: none.

### Phase 5 — Gate Check (2026-05-20)

- Premise still valid: ✓ (phantom audit findings + v0.3 sprint scoping)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: pending — user to approve PR-030 scope on review
- Implementation cleared: pending

---

## Scope

Pure docs + PR-stub creation. No code change, no test change, no behavior change.

### Files created

- `docs/0.3/ROADMAP.md` — v0.3 PR index (PR-030..PR-036) with status `[ ]`, dependencies, theme description, and design-session prerequisite
- `docs/0.3/RESEARCH-BACKLOG.md` — Tier-1/Tier-2 classifications + per-PR drift-risk notes + 60-day staleness clock
- `docs/0.3/DESIGN-log.md` — placeholder for the upcoming v0.3 design session (S1 sprint structure decision recorded; D1–D4 architectural sub-decisions stated as leans, NOT locked)
- `prs/PR-030-v0-3-sprint-scaffolding.md` — this file
- `prs/PR-031-hpo-honors-holdout-fold.md` — stub
- `prs/PR-032-registry-score-cli-verb.md` — stub
- `prs/PR-033-extmem-path-activation.md` — stub
- `prs/PR-034-artifacts-store-integration.md` — stub
- `prs/PR-035-tune-retry-trial-test.md` — stub
- `prs/PR-036-v0-3-0-cut.md` — stub

### Files modified

- `CHANGELOG.md` — `[Unreleased] ### Added` entry for v0.3 sprint scaffolding (no user-facing behavior change, but sprint-bookkeeping artifact)

## Dependencies

None. PR-030 lands directly on `dev`.

## Architecture section implemented

None — sprint bookkeeping only.

## Verification criteria

- [ ] `docs/0.3/` directory exists with `ROADMAP.md` + `RESEARCH-BACKLOG.md` + `DESIGN-log.md`
- [ ] All 7 v0.3 PR stub files exist in `prs/` with PR-TEMPLATE.md frontmatter shape
- [ ] `docs/0.3/ROADMAP.md` row for PR-030 flips from `[ ]` to `[x]` in this PR's implementation commit (per `feedback_roadmap_flip_in_pr`)
- [ ] `CHANGELOG.md` `[Unreleased]` has a sprint-scaffolding entry
- [ ] `uv run pytest` still passes at the baseline (367 passed, 1 skipped, 15 deselected — no behavior change)
- [ ] `uv run ruff check .` clean
- [ ] `uv run basedpyright src/` clean
- [ ] No file under `src/` is modified

## Research backing

Tier-1 — no architecture decisions; pattern inherited from PR-021 / PR-026 sprint-bookkeeping precedent + v0.2 design-session doc commit. Phase 1 state assessment + Phases 2-4 light.

## Notes

- The v0.3 design session runs as a separate activity AFTER PR-030 merges. Its output replaces the "Pending — to be locked in v0.3 design session" section of `docs/0.3/DESIGN-log.md` with locked decisions for D1–D4.
- The audit transcript (2026-05-20 chat session) is referenced in `docs/0.3/DESIGN-log.md` Session 2026-05-20 for traceability.
- PR-030 is the FIRST scaffolding PR with its own numbered PR file. Prior sprints (v0.2) created their `docs/0.2/` directory in a non-numbered commit; v0.3 favors paper-trail discipline as the project grows.
