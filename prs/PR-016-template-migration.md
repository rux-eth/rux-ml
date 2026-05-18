# PR-016: Template migration to vibe-rails hybrid docs-versioning layout

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**This PR introduces the Per-Phase Approval Gate (NON-NEGOTIABLE) constraint** (added to `docs/CONSTRAINTS.md` in Phase 4). The constraint is observed from this PR forward — Claude halts at every phase boundary and waits for explicit user approval before advancing.

## Research findings

This PR is **Tier-1** — the structural design is already locked in `vibe-rails main` (commits `3991c7f` and `471c7c4`). No web research needed; only project-local state assessment to surface what's specific to `rux-ml`.

### State Assessment (2026-05-18)

**Current state**:
- Project version: `0.0.1` (`pyproject.toml`). No git tags.
- Branch `dev` clean at `31e2826`.
- Stack: Python (uv-managed). No existing `scripts/` directory.
- Temporal docs present at flat root: `docs/DESIGN-log.md`, `docs/RESEARCH-BACKLOG.md`, `docs/ROADMAP.md`.
- Flat SSOT docs present: `docs/ARCHITECTURE.md`, `docs/CONSTRAINTS.md`, `docs/CONVENTIONS.md`.
- Missing: `docs/VERSIONING.md`, `docs/DEPLOYMENT.md`, `/CHANGELOG.md`.
- 15 numbered PR files (PR-001 through PR-015), all merged on `dev`. None carry a `Landed-in:` header.
- `prs/PR-TEMPLATE.md` lacks the `Landed-in:` comment block.
- 60-day staleness threshold already documented in `docs/CONSTRAINTS.md:130` — preserved.
- Cross-reference surface (files mentioning the 3 moved docs): 12 markdown files + 2 Python source files (`src/rux_ml/cli/__init__.py:15`, `src/rux_ml/training/protocol.py:3`).

**Assumptions at PR draft time**:
- Vibe-rails `3991c7f`/`471c7c4` define the canonical target structure (hybrid versioned-docs layout + `VERSIONING.md` + `CHANGELOG.md` + Per-Phase Approval Gate constraint).
- The canonical migrator is `https://github.com/rux-eth/john/blob/main/scripts/rewrite-doc-refs.ts` (TypeScript, 145 lines).
- This project's stack (Python) requires a port rather than a verbatim copy — the idempotency contract (PATH_REWRITES + MOVED_FILE_FIXUPS with negative lookbehind) is the load-bearing property to preserve.

**Stale assumptions** (where current state disagrees with PR assumptions):
- None — vibe-rails `main` was inspected fresh today; the canonical script was fetched fresh via `gh api`. No drift.

**New constraints** (learned from prior PRs or codebase evolution):
- Two Python source files reference `docs/ROADMAP.md` / `docs/DESIGN-log.md` in module docstrings. The canonical TS walker only scans `*.md`; the Python port must extend the walker to also include `*.py` so those references migrate cleanly. The walker filter is widened in the port; the rewrite logic itself stays identical.
- The project has no `node_modules` or JS build dirs but does have `.venv`, `.pytest_cache`, `.ruff_cache`, `.hypothesis`, `.basedpyright_cache`, `.mypy_cache`, `__pycache__`. IGNORE_DIRS expanded accordingly.

### Sub-decisions (locked 2026-05-18, user-approved)

1. **Bump rules** for `docs/VERSIONING.md`:
   - **MAJOR**: CLI flag/command rename, breaking change to a model-registry bundle layout, breaking change to a TOML config schema, breaking change to a manifest field.
   - **MINOR**: any new user-visible CLI verb or flag, any new layer factory (`make_*`), any new built-in Trainer/Splitter family, any new study/problem-level config knob.
   - **PATCH**: bug fixes, copy/typo edits, internal refactors with no behavioural change, dependency bumps that don't change output, doc-only edits.
   - **v1.0 criteria**: the workbench is feature-stable enough that registry bundles + studies could be handed to a future collaborator.

2. **`Landed-in:` value** for the 15 already-merged PRs: **`v0.0.1`**. All v0 PRs land at the current pyproject version. CHANGELOG opens with `## [Unreleased]` (this migration) + retroactive `## [0.0.1] - 2026-05-16` (the date the v0 rollout finished merging per memory `project_v0_rollout_state`).

3. **Version cut**: tag `v0.0.1` at this PR's merge commit. No `pyproject.toml` bump (already `0.0.1`). Tag goes onto post-merge `main` HEAD per VERSIONING §7.

---

## Scope

Migrate `rux-ml` from the pre-overhaul vibe-rails template structure to the post-overhaul structure (vibe-rails `main` at `471c7c4`), preserving all project-specific content. Concretely:

- Move `docs/{DESIGN-log,RESEARCH-BACKLOG,ROADMAP}.md` → `docs/0.0/`.
- Add `docs/VERSIONING.md`, `docs/DEPLOYMENT.md`, `/CHANGELOG.md`.
- Append Per-Phase Approval Gate (NON-NEGOTIABLE) to `docs/CONSTRAINTS.md`.
- Add `Landed-in:` header field to `prs/PR-TEMPLATE.md` and back-fill on all 15 numbered PR files (`v0.0.1`).
- Add `scripts/rewrite_doc_refs.py` (Python port of the canonical TS migrator) + Makefile targets.
- Rewrite cross-references in all 14 affected files (12 .md + 2 .py).
- Cross-link `VERSIONING.md` from `PROCEDURE-design-planning.md` "When to use" + `CLAUDE.md` "Design References".
- Tag `v0.0.1` at the merge commit.

Out of scope: any non-structural doc rewrites, any code changes outside the 2 Python doc-path references, any new test or PR work beyond this migration.

## Dependencies

None — operates on the doc layer + PR registry. All 15 prior PRs are merged.

## Architecture section implemented

Meta — this PR restructures docs themselves rather than implementing an `ARCHITECTURE.md` section. After this lands, the layout described in `docs/VERSIONING.md §3` (Doc-versioning layout — hybrid) becomes canonical.

## Verification criteria

- [ ] `docs/0.0/` contains exactly `DESIGN-log.md`, `RESEARCH-BACKLOG.md`, `ROADMAP.md`.
- [ ] Flat `docs/` contains exactly `ARCHITECTURE.md`, `CONSTRAINTS.md`, `CONVENTIONS.md`, `DEPLOYMENT.md`, `VERSIONING.md`.
- [ ] `/CHANGELOG.md` exists, Keep-a-Changelog 1.1.0 shape, has `[Unreleased]` + `[0.0.1] - 2026-05-16`.
- [ ] `docs/VERSIONING.md` covers all 7 sections (bump triggers, design-planning rule, layout policy, PR numbering, migration script section, changelog, tag-vs-deploy).
- [ ] `docs/CONSTRAINTS.md` includes Per-Phase Approval Gate NON-NEGOTIABLE.
- [ ] `PROCEDURE-design-planning.md` "When to use" cross-links `docs/VERSIONING.md`.
- [ ] All 15 PR files carry a `**Landed-in:** v0.0.1` line directly under the title.
- [ ] `prs/PR-TEMPLATE.md` carries the `Landed-in:` comment block.
- [ ] `scripts/rewrite_doc_refs.py --dry-run` after apply reports 0 replacements (idempotent).
- [ ] `git log --follow` works on each moved file at git's default similarity threshold.
- [ ] `make test` + `uv run basedpyright src/` stay green.
- [ ] `git diff --stat` between pre- and post-refactor commits shows only renames + targeted reference updates — no in-content rewrites of existing docs.
- [ ] Tag `v0.0.1` points at the merge commit of this PR.

## Research backing

- Source of truth: `https://github.com/rux-eth/vibe-rails` `main` at `471c7c4` — `docs/VERSIONING.md`, `docs/CONSTRAINTS.md` (Per-Phase Approval Gate §), `docs/DEPLOYMENT.md`, `CLAUDE.md`, `PROCEDURE-design-planning.md`, `prs/PR-TEMPLATE.md`, `CHANGELOG.md`.
- Canonical migrator: `https://github.com/rux-eth/john/blob/main/scripts/rewrite-doc-refs.ts` (sha `73adf5c3d4001b27be21ebc3e1ed07a741cd8b63`). Idempotency contract = `PATH_REWRITES` for references *to* moved files + `MOVED_FILE_FIXUPS` regex with negative lookbehind for internal-relative-path fixups *within* the moved files. The Python port preserves this contract verbatim; only the file-walk extension set widens (`*.md` + `*.py`).

## Notes

- This PR file itself goes inside `prs/` like every other; the rewrite script will see it during its walk but the only paths it references match `PATH_REWRITES`, so the script's idempotency check (`--dry-run` after apply = 0) covers it too.
- The Python port lives at `scripts/rewrite_doc_refs.py` (snake_case per `CONVENTIONS.md`). Make targets `rewrite-doc-refs` and `rewrite-doc-refs-dry` mirror the vibe-rails `pnpm` invocation pattern adapted to this project's stack.
- ROADMAP row for PR-016 is added under a new "Phase H — Post-v0 template migration" section in `docs/0.0/ROADMAP.md`, flipped to `[x]` in the final commit of this PR per memory `feedback_roadmap_flip_in_pr`.
