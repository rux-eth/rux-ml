# PR-029: Sync `PROCEDURE-pr-research.md` to vibe-rails fd0e6dd (stack-neutral)

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

**This PR is Tier-1 light**. The change is fully spec'd by the upstream `vibe-rails` commit `fd0e6dd` (2026-05-19), which scrubbed stack-specific examples from the procedure additions PR-028 mirrored at the earlier `fb0eec9`. PR-029 is the direct counterpart on the rux-ml side. **First PR to operationally exercise the new procedure** (Phase 1 prior-art audit step + Phase 3 cite-or-flag clause + Group D MCP-Verification probes); Group D dogfoods its own rules on a mechanical sync — a useful low-stakes operational test.

**Bump classification: PATCH** per `docs/VERSIONING.md` §1 (docs-only / internal procedure refactor; no user-visible behaviour change). Bundles in `[Unreleased]` alongside PR-028 until the next cut.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

## Research findings

### Phase 1 — State Assessment (2026-05-19)

**Current state** (HEAD = `497e20a`, post-PR-028 merge; PR-029 branch cut from there):

- **Drift check vs propagation target (`vibe-rails fd0e6dd`)**: `diff` shows exactly **6 substantive divergences** (the stack-flavored content rux-ml carried over from `vibe-rails fb0eec9` via PR-028) **plus 4 versioned-path divergences** (rux-ml's `docs/0.2/...` vs generic `docs/...` — preserved per hybrid-layout convention). The 6 substantive divergences are precisely the cleanup vibe-rails landed at `fd0e6dd`.
- **Upstream change scope** (vibe-rails `fb0eec9..fd0e6dd`, +7 / −13 lines):
  - L82 Cite-or-flag clause: parenthetical examples scrubbed of "action input" / "multi-step workflows" / "action+input pairings".
  - L122 Scope filter: "vendor-specific" → "implementation-specific".
  - L124-130 Probe 1: 5-bullet stack-flavored list (webhook payloads / TypeScript `.d.ts` / OpenAPI / `action.yaml` / CLI `--help`) collapses to a single abstract statement.
  - L134 Probe 2 ¶1: "multi-step workflows / action+input pairings" → "multi-step procedures / parameter+value pairings".
  - L136 Probe 2 ¶2: "workflow YAML, configuration sample, integration test, production source" → "configuration sample, integration test, production source, runtime declaration".
  - L139 Probe 3: GitHub-deployment-flavored examples → generic lifecycle-binding examples (factory registrations / hook subscriptions / external-system bindings / indexer entries / persisted configuration); "gh api" / "vendor MCP" → "introspect via whatever surface reflects that state".
  - L150 Group D output table: `client_payload.X` / `vendor/repo/path/to/schema-doc` → `<identifier>` / `<repo/path/to/canonical-documenter>`.

### Phase 1 — Prior-art audit (2026-05-19)

Per the new Phase 1 step 6 (added by PR-028): `git log -p PROCEDURE-pr-research.md` shows 5 commits ever touched this file in rux-ml:

| commit | PR | what it learned |
|---|---|---|
| `7f01bbf` | PR-028 | The vibe-rails `fb0eec9` propagation. **Critical prior-art**: PR-028 established that the 4 versioned-path lines (`docs/0.2/...`) are rux-ml-local and must NOT propagate into edits. PR-029 inherits this preservation contract. |
| `e2532f5` | PR-026 | v0.2.0 version cut. No direct edit to PROCEDURE-pr-research.md content; only path-rewrite to the 4 versioned-path lines (`docs/0.1/* → docs/0.2/*`). Established the rewriter + sentinel pattern. |
| `f8c19d4` | PR-021 | v0.1.0 version cut. Same shape as `e2532f5` (path rewrite, not content edit). |
| `94cafae` | (refactor) | v0.0 cross-reference rewrite. Initial path-versioning migration. |
| `84cb386` | (init) | Initial commit. PR-016 brought in the vibe-rails baseline. |

**Things prior PRs already learned the hard way** (constraints for Phase 4 synthesis):
- **Versioned paths are rux-ml-local**: every cut PR (PR-021, PR-026) rewrites these in the migration step; the rewriter script's OPT_OUT_GLOBS + sentinel pattern locks them. PR-028 confirmed PROCEDURE-pr-research.md is in the active-navigation set (auto-rewrites at version cuts) — not a sentinel-locked historical citation.
- **vibe-rails ↔ rux-ml propagation has a stable contract**: substance mirrors verbatim; versioned-path lines stay rux-ml-local; PR file documents the chain. PR-028 codified the pattern; PR-029 inherits it.
- **No tests / src / config affected by procedure changes**: docs-only refactor in this file; CI runs the full default suite as a regression check.

**Drift assessment**: zero days elapsed since PR-028 merged at `497e20a`. No rux-ml-local edits to `PROCEDURE-pr-research.md` between PR-028 and PR-029. The 4 versioned-path lines are at the same positions as when PR-028 landed.

**Stale assumptions in PR draft**: none — the substance is the upstream commit.

**Exit decision**: zero drift; substantive change fully spec'd upstream; prior-art audit surfaces no new constraints beyond what PR-028 already locked. **Proceed to light Phase 2.**

### Phase 2 — Scope Research (2026-05-19, light)

Per `feedback_phase1_scope`: Tier-1 with no drift → straight to Gate Check. Phase 3 SKIP — the upstream commit IS the research artifact (same shape as PR-028).

**Must-answer**: none.

**Phase 3 dispatch plan**: SKIP.

### Phase 4 — Synthesis (2026-05-19)

**Outcome**: confirm-with-locked-specifics. Phase 1 diff lists every change line; Phase 5 applies them via line-level Edits.

**Final implementation shape**: 6 line-level edits to `PROCEDURE-pr-research.md` (matching vibe-rails fd0e6dd) + 1 CHANGELOG entry. Rux-ml's 4 versioned-path lines stay intact (at lines 23, 132, 165, 198 in the post-PR-028 state — confirmed by Phase 1 prior-art audit).

### Group D — MCP Verification (2026-05-19)

First-ever Group D pass in rux-ml. Per the new procedure (PR-028), this is the mandatory driver pass before locking Phase 4.

**Schema-Integrity Probe:**

| Claim | Identifier | Canonical documenter | Verified? | Notes |
|---|---|---|---|---|
| 6 substantive edits mirror vibe-rails fd0e6dd verbatim | Specific text strings in old_string / new_string Edits | `~/templates/vibe-rails/PROCEDURE-pr-research.md` @ `fd0e6dd` | yes | Each old_string passed to Edit is the exact pre-edit rux-ml text; each new_string is the exact corresponding post-edit vibe-rails text. Post-apply `diff rux-ml/PROCEDURE-pr-research.md vibe-rails:fd0e6dd:PROCEDURE-pr-research.md` will report exactly the 4 versioned-path divergences and nothing else. |

**Synthesis-Verification Probe:**

| Claim | Combined elements | Cited working example | Verified? | Notes |
|---|---|---|---|---|
| The combination "apply 6 vibe-rails fd0e6dd edits + preserve 4 versioned-path lines" produces a valid rux-ml state | 6 vibe-rails edits + 4 versioned-path preservations | `rux-ml @ 497e20a:PROCEDURE-pr-research.md` (the PR-028 round-trip outcome) + vibe-rails @ `fb0eec9..fd0e6dd` (the upstream round-trip) | yes | PR-028 (commit `7f01bbf`) established the exact "mirror substance + preserve versioned-paths" round-trip on the same file. PR-029 is the same shape with cleaner substance. Post-apply diff verifies. |

**Binding-at-creation**: not applicable — PR-029 introduces no state-registration surface. No factory registrations, no external bindings, no persistence writes. Pure docs refactor.

**Reconciliations**: none — every load-bearing claim verified.

### Phase 5 — Gate Check (2026-05-19)

- Premise still valid: ✓ (rux-ml's PROCEDURE-pr-research.md genuinely diverges from vibe-rails fd0e6dd at 6 substantive lines that should mirror)
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (2026-05-19 "sync" reply)
- Group D probes complete: ✓ (Probes 1+2 verified; Probe 3 N/A)
- Implementation cleared: ✓

### Phase 5 — Implementation outcomes (2026-05-19)

Mini state-assessment: zero days elapsed since Phase 4 / Group D; `git log` between PR-028 merge (`497e20a`) and PR-029 implementation start shows no commits — no drift.

**Code landed**:

- `PROCEDURE-pr-research.md` — 6 substantive Edits applied verbatim from `vibe-rails fd0e6dd`. Rux-ml's 4 versioned-path lines (lines 23, 132, 165, 198) preserved.
- `CHANGELOG.md [Unreleased] ### Changed` — new PR-029 entry inserted above the PR-028 entry, documenting the stack-neutral cleanup propagation + the first operational Group D pass.

**Verification (substance match)**: `diff PROCEDURE-pr-research.md ~/templates/vibe-rails:fd0e6dd:PROCEDURE-pr-research.md` shows exactly 4 differences — all at the 4 known versioned-path lines (rux-ml `docs/0.2/...` vs generic `docs/...`). Substance bit-identical to upstream modulo the hybrid-layout divergence.

**No ROADMAP / RESEARCH-BACKLOG flips** — same post-cut precedent as PR-028.

**Test outcomes**: docs-only PR; no test delta expected. To be verified at pre-commit gate.

**Group D operational notes** (first-ever rux-ml Group D pass):
- **Probe 1 (Schema-Integrity)**: identifier-level verification mapped cleanly to "old_string text exists in current rux-ml file" + "new_string text exists at vibe-rails fd0e6dd". Both verified by Edit tool's atomicity contract (Edit fails if old_string isn't a unique exact match) + post-apply diff.
- **Probe 2 (Synthesis-Verification)**: the combination "mirror substance + preserve versioned-path divergences" had a cited working example at `7f01bbf` (PR-028) — the exact same combination on the exact same file. Verified.
- **Probe 3 (Binding-at-creation)**: N/A by Scope filter — no state registered at one lifecycle moment and read at another. Confirms the "Probe 3 cleanly N/A for non-vendor-binding workbench PRs" pattern flagged after PR-028 merged. No workbench-shaped Probe 3 reframe needed for this class of PR.

**Implication for future PRs**: the new procedure's Group D pass adds ≤10 minutes of overhead for a mechanical-sync PR like PR-029 — within tolerable bounds. For research-heavy PRs (Tier-2 with dispatched agents), Group D's Probe 2 (Synthesis-Verification) will be the load-bearing probe and may downgrade some recommendations from CONVENTION to BGGC; that's the intended behavior.

---

## Scope

Mirror the 6 line-level edits from `vibe-rails fd0e6dd` into rux-ml's `PROCEDURE-pr-research.md`:

1. **Cite-or-flag clause** (Phase 3 Rules) — strip stack-flavored parenthetical examples.
2. **Scope filter** (Group D) — "vendor-specific" → "implementation-specific".
3. **Probe 1 — Schema-Integrity** — collapse the 5-bullet stack-flavored list into a single abstract statement.
4. **Probe 2 — Synthesis-Verification** ¶1 — terminology touch-up.
5. **Probe 2 — Synthesis-Verification** ¶2 — terminology touch-up.
6. **Probe 3 — Binding-at-creation** — replace GitHub-deployment-flavored examples with generic lifecycle-binding examples.
7. **Group D output table example values** — `client_payload.X` / `vendor/repo/path/to/schema-doc` → `<identifier>` / `<repo/path/to/canonical-documenter>`.

Preserve rux-ml's 4 versioned-path lines (`docs/0.2/...`) — these diverge from the generic vibe-rails template per the hybrid layout convention.

Add CHANGELOG `[Unreleased] ### Changed` entry documenting the propagation.

### Out of scope

- Touching `PROCEDURE-design-planning.md` / `PROCEDURE-code-audit.md` — neither in the upstream change.
- Adding a ROADMAP row — same precedent as PR-028 (post-v0.2.0 cut + pre-v0.3 design session; no active version-scoped ROADMAP for in-flight PATCH work).

## Dependencies

None. Built on `dev @ 497e20a` (post-PR-028 merge).

## Architecture section implemented

None — procedural refactor.

## Verification criteria

- [ ] `PROCEDURE-pr-research.md` matches `vibe-rails fd0e6dd` substance verbatim (6 substantive edits), with rux-ml's 4 versioned-path lines preserved.
- [ ] `CHANGELOG.md [Unreleased] ### Changed` has one PR-029 row.
- [ ] Post-apply `diff` vs `vibe-rails:fd0e6dd:PROCEDURE-pr-research.md` reports exactly 4 differences (the versioned-path divergences) and nothing else.
- [ ] No tests / src / config changes.
- [ ] `uv run pytest` stays green.
- [ ] `uv run ruff check .` clean.
- [ ] `uv run basedpyright src/` clean.

## Research backing

Tier-1 — anchored on:
- `vibe-rails fd0e6dd` (the propagation target).
- PR-028's `7f01bbf` (the precedent for the mirror-substance + preserve-versioned-paths contract).
- Phase 1 prior-art audit (above) — 5-commit history of `PROCEDURE-pr-research.md` in rux-ml; all 5 are sync-shaped (PR-028) or path-rewrite-shaped (PR-021 / PR-026 / refactor / init).

No Phase 3 web research dispatched.

## Notes

- **PATCH bump**. Bundles in `[Unreleased]` alongside PR-028 until next cut.
- **`feedback_roadmap_flip_in_pr` exception** — same as PR-028 (post-v0.2.0 cut; no active ROADMAP row to flip). Streak preserved by the "no row to flip" semantic.
- **First Group D pass** — exercises the new procedure end-to-end on a low-stakes mechanical sync. Probe 1 + Probe 2 verified cleanly; Probe 3 N/A by Scope filter (no state-registration surface in a docs refactor). Validates the operational uncertainty I flagged after PR-028 merged: **Probe 3 N/A works cleanly for non-vendor-binding PRs**.
- **Propagation chain note**: rux-ml downstream is now in sync with vibe-rails fd0e6dd substance. Future stack-flavored references in rux-ml-specific contexts (e.g., a future PR that genuinely needs to discuss XGBoost / Optuna / Polars examples) belong in `docs/CONVENTIONS.md` or PR-specific writeups, not in the procedure file.
