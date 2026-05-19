# PR-028: Sync `PROCEDURE-pr-research.md` from vibe-rails (prior-art audit + cite-or-flag + Group D probes)

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-1 light**. The change is fully spec'd by the upstream `vibe-rails` commit `fb0eec9` (2026-05-19), itself propagated from `Casus PR-032`. Both upstreams have already validated the additions; rux-ml's job is to mirror them. Phase 1 confirms zero rux-ml-local additions in `PROCEDURE-pr-research.md` since PR-021 — the file matches the pre-fb0eec9 vibe-rails baseline except for the 4 versioned-path lines that rux-ml owns (`docs/0.2/...` instead of bare `docs/...`).

**Bump classification: PATCH** per `docs/VERSIONING.md` §1 (docs-only / internal procedure refactor; no user-visible behaviour change, no CLI surface, no model bundle change). Ships in `[Unreleased]` until the next minor/major cut bundles it.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

**Pattern locked upstream**. Two-link propagation chain: Casus PR-032 (the originating empirical-failure source) → vibe-rails commit `fb0eec9` (2026-05-19; the template-level sync) → rux-ml PR-028 (this PR). The substantive change is exactly the diff between vibe-rails' `471c7c4..fb0eec9` on `PROCEDURE-pr-research.md`, minus rux-ml's 4 versioned-path differences.

### Phase 1 — State Assessment (2026-05-19)

**Current state** (HEAD = `1fa43c0`, post-v0.2.0 tag; PR-028 branch cut from there):

- **`PROCEDURE-pr-research.md` drift check vs pre-fb0eec9 vibe-rails baseline (`471c7c4`)**: rux-ml's file differs from the baseline at exactly **4 lines** — all are versioned-path references where rux-ml uses `docs/0.2/...` instead of bare `docs/...` (lines 23, 132, 165, 198). Zero substantive rux-ml-local additions to preserve. Confirmed via `diff -u <vibe-rails:471c7c4:PROCEDURE-pr-research.md> rux-ml:PROCEDURE-pr-research.md`.
- **Upstream change scope** (vibe-rails `471c7c4..fb0eec9`, +49 / −2 lines):
  - **Phase 1, new item 6**: prior-art audit step — for every touched file, run `git log -p` + survey the last 3-5 PRs in related infrastructure for known-bad and known-good patterns. Output is a list of "things prior PRs already learned the hard way" that constrains Phase 3 (so research doesn't re-derive) and Phase 4 (so synthesis doesn't drop them).
  - **Phase 3 Rules, new cite-or-flag clause** (non-negotiable; agent prompts must include this clause verbatim): every specific identifier in a recommendation needs a source-of-truth cite; every combination needs a cited working example using the EXACT combination (not synthesis from disparate sources). Missing either → `best-guess-given-constraints` + flag.
  - **Phase 3 Rules, amended BGGC label definition**: adds "Also the required label whenever the cite-or-flag rule above is not fully satisfied."
  - **Phase 3, new Group D MCP-Verification Round** (mandatory before locking Phase 4, ≤30 min, bounded): three named probes — Schema-Integrity (every named identifier exists in the canonical schema documenter), Synthesis-Verification (every combined identifier has a cited working example using the EXACT combination), Binding-at-creation (for PRs introducing vendor-side bindings, confirm registration occurs at the expected lifecycle moment via MCP introspection or `gh api`).
  - **Exit criteria amended**: now requires Group D probes to have run on every load-bearing claim before Phase 4 is locked.
- **Propagation surface to preserve**: rux-ml's 4 versioned-path lines stay at `docs/0.2/...` — vibe-rails uses bare `docs/...` because it's a generic starter; rux-ml's hybrid layout (PR-016 + PR-021 + PR-026) keeps temporal docs version-scoped.
- **No drift on adjacent procedures** since PR-021: `PROCEDURE-design-planning.md` and `PROCEDURE-code-audit.md` are not part of the upstream change scope; they don't need touching.

**Stale assumptions in PR draft**: none — the propagation surface is exhausted by the 3 substantive additions.

**New constraints from prior PRs (post-v0.2.0)**:

- **`feedback_roadmap_flip_in_pr` exception**: post-v0.2.0 cut + pre-v0.3 design session, no active `docs/<x.y>/ROADMAP.md` row exists for in-flight PATCH work. PR-028 has no ROADMAP row to flip — analogous to PR-022's pre-PR-026 interim state. The streak (20 consecutive PRs through PR-027) is preserved by the "no row to flip" semantic; future-Claude should not flag a missed flip. The next minor-cut version PR (analog of PR-021/PR-026) is where this PR's ROADMAP-row backfill (if any) lands.
- **No prerequisite PRs**: PR-028 is independent of any post-v0.2.0 work that may surface. The next v0.3 design session does NOT block this docs-only sync.

**Exit decision**: zero drift; substantive change fully spec'd upstream; rux-ml's local edits (path versioning) preserved by line-level surgery rather than wholesale replacement. **Proceed to light Phase 2 — no scope sharpening needed beyond confirming the preservation set; no Phase 3 web research (the upstream commit IS the research artifact).**

### Phase 2 — Scope Research (2026-05-19, light)

Per `feedback_phase1_scope`: Tier-1 with no drift → straight to Gate Check. No Phase 3 dispatch required — the substantive content is the upstream commit, already validated at Casus PR-032 + vibe-rails fb0eec9. Phase 2 reduces to scope confirmation:

**Must-answer**: none. The 3 substantive additions land verbatim from vibe-rails fb0eec9; the 4 versioned-path lines stay rux-ml-local.

**Phase 3 dispatch plan**: SKIP. No web research needed; the upstream commit + this PR's Phase 1 diff are the entire research artifact.

### Phase 4 — Synthesis (2026-05-19)

**Outcome**: Confirm-with-locked-specifics. Phase 1 diff lists every change line; Phase 5 implements them via line-level Edits against rux-ml's `PROCEDURE-pr-research.md`.

**Final implementation shape**:

1. **Phase 1 section** — after line 25 ("Check `docs/0.2/DESIGN-log.md` for any decisions that might have drifted"), insert new item 6 with the prior-art audit step (verbatim from vibe-rails fb0eec9).
2. **Phase 3 Rules section** — within the bulleted list of rules, insert the cite-or-flag clause as a new bullet after the "Parallel agents for independent questions" bullet (verbatim from vibe-rails fb0eec9).
3. **Phase 3 Rules section** — amend the BGGC label line to add the trailing sentence about "the required label whenever the cite-or-flag rule above is not fully satisfied" (verbatim from vibe-rails fb0eec9).
4. **Phase 3 section** — insert the new Group D subsection between the existing Synthesis output template and the Exit criteria line (verbatim from vibe-rails fb0eec9, including the 3 probe descriptions + the Group D output template).
5. **Phase 3 Exit criteria** — amend the existing exit-criteria sentence to add the trailing clause about Group D probes (verbatim from vibe-rails fb0eec9).
6. **Versioned-path references at lines 23, 132, 165, 198 stay unchanged** — these are rux-ml-local diverges from the generic vibe-rails template per the hybrid layout convention.

**Changes to ARCHITECTURE.md / CONVENTIONS.md / CONSTRAINTS.md from research**: none.

**New PRs that must come first**: none.

**CHANGELOG `[Unreleased]` entry**: PR-028 adds a `### Changed` row documenting the procedural sync (Phase 1 prior-art audit step + Phase 3 cite-or-flag + Group D MCP-Verification probes; propagated from Casus PR-032 via vibe-rails fb0eec9).

### Phase 5 — Gate Check (2026-05-19)

- Premise still valid: ✓ (the substantive change is verbatim-mirrorable from vibe-rails fb0eec9)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (pending Phase 1 user approval; Phase 5 starts after that gate clears)
- Implementation cleared: ✓ (pending pre-commit gate)

### Phase 5 — Implementation outcomes (2026-05-19)

Mini state-assessment: zero days elapsed since Phase 4; `git log` between PR-026 merge (`1fa43c0`) and PR-028 implementation start shows no commits — no drift.

**Code landed**:

- `PROCEDURE-pr-research.md` — 3 substantive additions + 2 amendments applied verbatim from vibe-rails `fb0eec9` (rux-ml's 4 versioned-path lines at lines 23, 132, 165, 198 preserved):
  - **Phase 1, new item 6**: prior-art audit step (`git log -p` + last 3-5 related-area PR survey).
  - **Phase 3 Rules**: new cite-or-flag clause bullet inserted after the "Parallel agents" bullet.
  - **Phase 3 Rules**: BGGC label gets the trailing sentence about being required when cite-or-flag isn't satisfied.
  - **Phase 3, new Group D subsection**: MCP-Verification Round with Schema-Integrity, Synthesis-Verification, and Binding-at-creation probes + the Group D output template.
  - **Phase 3 Exit criteria**: amended to require Group D probes have run on every load-bearing claim.
- `CHANGELOG.md [Unreleased] ### Changed` — one PR-028 row documenting the propagation chain (Casus PR-032 → vibe-rails fb0eec9 → rux-ml PR-028) + the 3 substantive additions + the preservation of rux-ml's 4 versioned-path lines.

**Verification (substance match)**: `diff PROCEDURE-pr-research.md ~/templates/vibe-rails:fb0eec9:PROCEDURE-pr-research.md` shows exactly 4 differences — all at the 4 known versioned-path lines (rux-ml `docs/0.2/...` vs generic `docs/...`). Substance is bit-identical to vibe-rails `fb0eec9` modulo the hybrid-layout divergence.

**No ROADMAP / RESEARCH-BACKLOG flips** — post-v0.2.0 cut + pre-v0.3 design session, no active version-scoped ROADMAP for in-flight PATCH work. `feedback_roadmap_flip_in_pr` streak (20 PRs through PR-027) preserved by the "no row to flip" semantic, per PR-022 precedent.

**Test outcomes**:
- `uv run pytest` — **367 passed, 1 skipped, 15 deselected** (matches dev @ `1fa43c0`; docs-only PR has no test delta).
- `uv run ruff check .` — clean.
- `uv run basedpyright src/` — clean (0 errors).

**Deferred / out of scope**:
- Touching `PROCEDURE-design-planning.md` / `PROCEDURE-code-audit.md` — neither in the upstream change scope.
- Re-running prior PRs' Phase 1-5 with the new procedure — the new procedure only governs future PRs.
- Adding a v0.3 ROADMAP scaffold — no v0.3 design session has happened; premature.

---

## Scope

Sync `PROCEDURE-pr-research.md` from `vibe-rails` commit `fb0eec9` (2026-05-19; itself propagated from `Casus PR-032`):

1. **Phase 1, new item 6**: prior-art audit step.
2. **Phase 3 Rules**: cite-or-flag clause + amended BGGC label definition.
3. **Phase 3, new Group D**: MCP-Verification Round with Schema-Integrity, Synthesis-Verification, and Binding-at-creation probes.
4. **Exit criteria amendment**: Group D probes must run on every load-bearing claim before Phase 4 locks.
5. **CHANGELOG `[Unreleased] ### Changed`**: one row documenting the sync.

Preserve rux-ml's 4 versioned-path references (`docs/0.2/...`) — these diverge from the bare-path generic vibe-rails template per the hybrid layout convention.

### Out of scope

- Touching `PROCEDURE-design-planning.md` or `PROCEDURE-code-audit.md` — neither is part of the upstream change.
- Adding a ROADMAP row for PR-028 — post-v0.2.0 cut + pre-v0.3 design session, no active version-scoped ROADMAP for in-flight PATCH work (PR-022 precedent).
- Re-running prior PRs' research with the new procedure — the new procedure only governs future PRs.

## Dependencies

None. Built on v0.2.0-dev `1fa43c0` (PR-026 merge).

## Architecture section implemented

None — procedural refactor.

## Verification criteria

- [ ] `PROCEDURE-pr-research.md` matches vibe-rails `fb0eec9` substance verbatim (the 3 procedural additions + the BGGC + exit-criteria amendments), with rux-ml's 4 versioned-path lines preserved.
- [ ] `CHANGELOG.md [Unreleased] ### Changed` has one PR-028 row.
- [ ] No tests / src / config changes (procedural refactor only).
- [ ] `uv run pytest` stays green (no test delta expected).
- [ ] `uv run ruff check .` clean.
- [ ] `uv run basedpyright src/` clean.

## Research backing

Tier-1 — anchored on the upstream commit chain:
- `Casus PR-032` (originating empirical-failure source — the agent-synthesis foot-gun this addresses).
- `vibe-rails fb0eec9` (template-level sync; commit message + diff are the canonical research artifact).
- Phase 1 diff above identifies the exact rux-ml-local lines to preserve.

No Phase 3 web research dispatched; the substantive content is mirrored verbatim from a validated upstream.

## Notes

- **PATCH bump** (likely v0.2.1 when the next cut lands) per `docs/VERSIONING.md` §1: docs-only, no design session needed.
- **`feedback_roadmap_flip_in_pr` exception**: post-v0.2.0 cut + pre-v0.3 design session, no active ROADMAP for in-flight PATCH work. PR-028 has no row to flip; the streak (20 consecutive PRs through PR-027) is preserved by the "no row to flip" semantic. Future-Claude should not flag a missed flip — analogous to PR-022's pre-PR-026 interim state.
- **Propagation chain note**: the commit message of vibe-rails fb0eec9 documents the chain back to Casus PR-032. Any future audits asking "where did this procedural rule come from?" trace back via that chain.
- **Selective propagation**: rux-ml's 4 `docs/0.2/...` versioned-path lines diverge from vibe-rails' bare `docs/...` — this is intentional per the hybrid layout convention (`docs/VERSIONING.md` §3). The sync preserves this divergence by line-level surgery rather than wholesale file replacement.
