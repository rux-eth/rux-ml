# Debug PR Standards

How a debugging PR's file is structured, what it must contain, and how it differs from the standard PR template. Branched from the project's standard PR file format (`prs/PR-TEMPLATE.md`). The procedure that produces these artifacts is `PROCEDURE-debug-pr.md`.

---

## Scope of this document

A "debug PR" is a PR whose primary purpose is fixing observed **bugs in code** — deterministic, reproducible defects in code logic. See `PROCEDURE-debug-pr.md` "When to use" for the full dividing line.

**Not in scope here** (use the standard `PROCEDURE-pr-research.md` instead):

- Prompting / LLM-output quality issues (biased outputs, instruction-following failures, hallucinations, etc.). LLM outputs are probabilistic; the TDD-shaped failing-test-first gate in this document doesn't map cleanly. rux-ml does not currently call LLMs in any production path; this exclusion is forward-looking.
- Features, refactors, dependency migrations, infra-setup PRs.

If the PR also adds features, split it.

The project's standard PR file structure still applies for everything not overridden here. This document specifies the deltas.

---

## File Lifecycle (debug-PR)

| Stage | What gets added |
|-------|-----------------|
| **Stub creation** | Title, Problem, Dependencies, Out of scope |
| **Phase 0** | `## Bugs (current vs expected behavior)` section locked with atomic bug pairs |
| **(if relevant)** | `## Code changes since baseline` — enumeration of every change since the last formal PR merge, with cited purpose per change |
| **Phase 1** | `### Phase 1 — Expanded State Assessment` (with import-closure analysis + common-imports) |
| **Phase 2** | `### Phase 2 — Hypothesis Assembly` (ranked hypotheses + confirmation observations) |
| **Phase 3** | `### Phase 3 — Bug Confirmation` (web research + dev-mode empirical evidence per bug) |
| **Phase 4** | `### Phase 4 — Test Design` — ⚠️ **USER SIGNOFF REQUIRED** before Phase 5 |
| **Phase 5** | `### Phase 5 — Test Implementation + Failure Confirmation` (tests committed; current code fails them) |
| **Phase 6** | `### Phase 6 — Fix Design` (convention + project-precedent research, rollback decisions, CONSTRAINTS.md / CONVENTIONS.md amendments drafted) — ⚠️ **USER SIGNOFF REQUIRED** before Phase 7 |
| **Phase 7** | `### Phase 7 — Implementation + Verification` (fix lands, tests turn green, ROADMAP row flipped if applicable, PR link delivered) |
| **Pre-merge** | Verification-run results appended (`uv run pytest` summary, `make test-golden` if relevant, optional workbench end-to-end if the bug needed GPU repro) |

---

## File Structure (debug-PR)

```markdown
# PR-NNN: Title

## Before Implementation (NON-NEGOTIABLE)

[Same enforcement notice as the project's standard PR template, but cites PROCEDURE-debug-pr.md as the gating procedure.]

## Status

Debug-PR (always research-required; no Tier-1 light path)

## Problem

[Symptom-level description: what observable wrong-behavior triggered the PR. Cite logs, runtime traces, study DB excerpts, CLI output — whatever applies. Do NOT specify bugs here — that's the next section.]

## Bugs (current vs expected behavior)

### Bug N — [one-line title]

**Current behavior**

[What the code actually does, with file:line + log / runtime / reproduction evidence.]

**Expected behavior**

[Contract the fix must satisfy, including observable acceptance criteria.]

[Repeat per atomic bug. Sub-bugs use 2a/2b/2c notation.]

## Code changes since baseline (optional)

[Used when this PR is responding to a debugging rabbit-hole that produced
speculative changes. Enumerate every commit since the last formal PR
merge, with file:line ranges and cited purpose per change. Each row's
purpose must come from a cited source — commit message, inline comment,
auto-memory entry — or be marked "No information found for purpose of change".]

## Dependencies

[Same as the project's standard PR template.]

## Out of scope

[Same as the project's standard PR template. Common debug-PR exclusions: unrelated refactors,
broader architecture changes that the bug surface tempted you to make.]

## Research findings

### Phase 1 — Expanded State Assessment (YYYY-MM-DD)
### Phase 2 — Hypothesis Assembly (YYYY-MM-DD)
### Phase 3 — Bug Confirmation (YYYY-MM-DD)
### Phase 4 — Test Design (YYYY-MM-DD)         ← signoff
### Phase 5 — Test Implementation + Failure Confirmation (YYYY-MM-DD)
### Phase 6 — Fix Design (YYYY-MM-DD)          ← signoff
### Phase 7 — Implementation + Verification (YYYY-MM-DD)

## Verification criteria

[Per-bug acceptance tests. Each criterion maps 1:1 to a test from Phase 4
and to a bug in `## Bugs`. A criterion that doesn't map to a bug is out of
scope.]

## Verification run

[Pre-merge results: `uv run pytest` count, `uv run ruff check .` status, `uv run basedpyright src/` status, `make test-golden` (if relevant), optional workbench end-to-end run output (`ssh ssh-desktop` invocation + result excerpt) when the bug needed GPU or large-dataset reproduction.]

### Failed-attempt log (optional, populated during Phase 7)

[If a Phase-6-approved fix didn't turn the Phase-5 test green:
the evidence, the revised hypothesis, and the new fix that landed.
Per PROCEDURE-debug-pr.md failed-attempt protocol.]
```

---

## Section Specifications

### Bugs (current vs expected behavior)

The defining section of a debug PR. Each bug is an atomic pair:

- **Atomic** — one current/expected pair per testable claim. If a bug has multiple independent facets (cadence, concurrency, side effects, config-overlay precedence, persistence-vs-in-process state, etc.), it's multiple sub-bugs (2a, 2b, 2c), not one. The atomicity test: can you write a verification criterion that tests one facet without invoking the others?

- **Current behavior** — what the code actually does, cited:
  - file:line ranges for the relevant code
  - log entries / runtime traces / Optuna trial-attrs / study-DB excerpts / reproduction steps where applicable
  - **NO speculation on root cause** — only observable behavior. Root-cause analysis goes in Phase 1–3.

- **Expected behavior** — the contract the fix must satisfy:
  - Acceptable solution shapes can be enumerated (e.g., "default the CLI flag to `None` + fall back to config, OR remove the flag and force config-only, OR add a sentinel value"). Specific implementations are deferred to research.
  - Must include at least one **observable acceptance criterion** — what would let the user verify the bug is fixed? This becomes a verification-criteria entry post-Phase-7.
  - Should reference any research-locked decisions the contract must respect.

- **Cross-coupling notes** — if a bug's expected behavior depends on another bug being fixed, state the coupling explicitly.

### Code changes since baseline (when used)

Optional section, populated when the debug PR is partially responding to speculative changes already in the codebase that the user wants reviewed for rollback.

Format: per-file, per-change-block, with cited purpose. Citation sources allowed:
- The introducing commit's message
- An inline code comment in the change itself
- An entry in user/auto memory (the rux-ml auto-memory lives at `~/.claude/projects/-Users-maxrux-projects-rux-ml/memory/`)
- A doc file in the repo (`docs/`, `prs/`, `PROCEDURE-*`)

If no source explains the purpose of a specific change, the row says **"No information found for purpose of change"**. Do NOT speculate.

This enumeration is the input to Phase 6's rollback-decisions step.

### Phase 1 — Expanded State Assessment

Differs from the standard PR-research Phase 1:

- **Import closure required** — every file directly named in the bug specs PLUS its transitive imports (≥1 hop). Common imports (used by ≥2 buggy files) are highlighted as candidate root-cause locations.
- **Recent commits filter** — list every commit since the last formal PR merge that touched any file in the import closure.
- **Empirical persisted-telemetry scan REQUIRED** when persisted telemetry exists AND the bug class is observable in it. For rux-ml the canonical persisted telemetry surfaces are: Optuna study databases (`studies/*.db`), `trial.user_attrs` / `trial.system_attrs`, run logs (`logs/*`), workbench `nohup`-redirected job output, and any `receipts/` JSONs from `rux-ml registry score` runs. Write a one-pass script that scans every available surface for each bug's signature, run it against the canonical surface (the local `studies/*.db` or the workbench's `~/projects/rux-ml/studies/*.db` over SSH, whichever holds the relevant runs), and report findings inline in the PR file (see template below). The scan answers four questions Phase 1 must NOT defer to Phase 3:
  1. **Does the bug actually fire?** A bug spec'd from code reading might not fire under current config / CLI-flag-default combinations. That changes test priority.
  2. **At what density?** Cited counts in the bug spec or in auto-memory drift over time. The scan corrects them.
  3. **What's the time window?** Pinpoints whether the bug is recent (recent commit caused it) or longstanding (already firing for weeks).
  4. **Are there cross-bug correlations the per-file read pass would miss?** E.g., a Watchdog-trip pattern revealing a silent-failure path that emits zero log lines.

  Scratch scripts live in `/tmp/` for one-off use. Reusable scans go to `scripts/audit/<topic>.sh`. Either way, the script's invocation goes into the PR file's Phase 1 section so the user can re-run.

  **Conditional N/A clause** — for bug classes that produce no persisted-telemetry signal (CLI option-parsing bugs, pre-trial config validation bugs, docs-only bugs, in-process bugs without persisted state), document `N/A — bug class is not observable in persisted telemetry; reproduction will happen in Phase 3` in the table below and skip the scan. This is the common case for rux-ml CLI / config bugs; the empirical scan is rare and reserved for bugs that affect persisted Optuna state.

- **Approach decision** — by the end of Phase 1 (or early Phase 2 at latest), pick sequential vs parallel handling of the bug set. Document the choice and rationale. **Empirical findings drive Phase 7 implementation order**: bugs firing heavily in persisted runs sequence ahead of theoretical / dead-path bugs.

Phase 1 template additions:

```markdown
**Empirical baseline (persisted telemetry):**

Scan command: `[the actual invocation — e.g., a script path + arguments, or "N/A" with justification]`

| Bug | Status | Persisted-run count | Window | Density | Notes |
|---|---|---|---|---|---|
| 1 | actively-firing | N matches across M study DBs | YYYY-MM-DD → YYYY-MM-DD | X/run | [config / flag context if relevant] |
| 2 | theoretical | 0 occurrences | — | — | defensive; code path is vulnerable but has not yet bitten |
| 3 | dead-path | 0 occurrences | — | — | path unreachable under current logic; confirm in Phase 3 |
| 4 | silent-failure | 0 logged BUT [cross-bug evidence] | — | — | proves the silent path fires; observable via [side-effect, e.g., trial.system_attrs absence] |
| 5 | not-telemetry-observable | N/A | — | — | CLI / config / docs bug; reproduction in Phase 3 |
```

Status taxonomy used in the table:
- **actively-firing** — persisted telemetry contains the bug's signature.
- **theoretical** — code path is vulnerable but produces no persisted-run occurrences. Defensive fix.
- **dead-path** — code path is empirically unreachable. Either remove or harden defensively. Phase 3 must confirm unreachability.
- **silent-failure** — no log lines, but side-effects (missing `user_attrs` fields, absent `system_attrs` artifact records, unexpected `peak_rss_mb` patterns) prove the path fires. Highest priority — the user has no signal today.
- **not-telemetry-observable** — the bug class doesn't produce persisted-telemetry signal at all (CLI option parsing, config validation, docs). Phase 3 dev-mode reproduction is the primary evidence path. Default state for rux-ml CLI/config bugs.

### Phase 2 — Hypothesis Assembly

Per-bug ranked list of root-cause hypotheses with file:line attributions and confirm/refute observations. Provides the input list Phase 3 will work through.

```markdown
### Phase 2 — Hypothesis Assembly (YYYY-MM-DD)

**Bug N hypotheses (ranked):**
1. [hypothesis] — confirm/refute via [observation]
2. [hypothesis] — confirm/refute via [observation]
...

**Dependencies:** [if H2 depends on H1, etc.]
```

### Phase 3 — Bug Confirmation

Web research + empirical testing in dev mode locally. Empirical instrumentation (logging, tracing, third-party tools, profiling) is allowed and may be kept in the codebase if it has lasting observability value.

When Phase 2 leaves ≥2 hypotheses competing on which layer fails, Phase 3 must include **isolation tests** — throwaway scripts or scratch fixtures that vary one variable at a time and hold the rest fixed, until the failure surface is pinned to a specific layer. Standard debugging.

Phase 3 is finished only when each bug has end-to-end mechanism described origin → propagation → symptom, with file:line at each hop, AND log/empirical/runtime evidence of the predicted failure path being taken at runtime.

```markdown
### Phase 3 — Bug Confirmation (YYYY-MM-DD)

**Bug N — confirmed mechanism**

*Hypothesis tested*: [Phase 2 hypothesis]
*Web evidence*: [cited URLs]
*Empirical evidence*: [instrumentation added + log / trial-attrs / DB excerpts]
*Isolation tests* (if applicable): [test name | varied | held | outcome | hypothesis confirmed/refuted]
*End-to-end mechanism*: [origin → propagation → symptom with file:line]
*Status*: confirmed (file:line of originating defect)
```

### Phase 4 — Test Design

Per-bug unit / integration test design. Specifies setup, trigger, assertion, predicted current failure, predicted post-fix pass. For rux-ml CLI bugs the typer `CliRunner`-driven tests in `tests/cli/` are the canonical integration surface.

⚠️ **USER SIGNOFF REQUIRED** before Phase 5.

### Phase 5 — Test Implementation + Failure Confirmation

Tests committed to the branch. Run against current (unfixed) code; each test must fail in its predicted way. If any test passes (or fails differently than predicted), Phase 4 (or Phase 0) loops.

```markdown
### Phase 5 — Test Implementation + Failure Confirmation (YYYY-MM-DD)

**Bug N — test result**
*Test file*: [path]
*Run output*: [excerpt showing failure]
*Predicted failure mode matched*: yes / no
```

### Phase 6 — Fix Design

Per-bug fix proposal with research backing on TWO dimensions: convention (web research, ≥2 cited sources) and project precedent (architecture-reference repo + existing codebase patterns, when either applies). Plus rollback decisions for any pre-existing speculative-fix commits, plus drafted `CONSTRAINTS.md` / `CONVENTIONS.md` amendments if the fix implies a new coding standard.

⚠️ **USER SIGNOFF REQUIRED** before Phase 7.

### Phase 7 — Implementation + Verification

Fix landed exactly as designed in Phase 6. Phase-5 tests turn green. ROADMAP row flipped `[ ] → [x]` in the same commit if an active `docs/<x.y>/ROADMAP.md` exists (per `feedback_roadmap_flip_in_pr`). Project gates (`uv run pytest`, `uv run ruff check .`, `uv run basedpyright src/`, and `make test-golden` if the touched surface affects the golden contract) all pass. PR link delivered.

If a fix doesn't turn its test green: failed-attempt protocol per `PROCEDURE-debug-pr.md`.

### Verification criteria

Maps 1:1 to bugs in `## Bugs` AND to tests from Phase 4. Each criterion is testable, observable, and references the expected-behavior contract:

```markdown
- [ ] Bug N: [observable acceptance criterion in concrete, testable terms]. Test: <path>.
- [ ] Bug N+1: ...
```

### Failed-attempt log (when used)

Populated if, during Phase 7, a fix doesn't turn its Phase-5 test green. Per `PROCEDURE-debug-pr.md` failed-attempt protocol:

```markdown
### Failed-attempt log

#### Attempt 1 (YYYY-MM-DD)
**Phase 6 fix proposal**: [original]
**Test result**: [still failing / failing differently]
**Evidence**: [test output excerpt]
**Revised hypothesis**: [Phase 3 mechanism update or Phase 6 approach change]
**New Phase 6 design** (re-approved YYYY-MM-DD): ...

#### Attempt 2 ...
```

Failed attempts are not committed as standalone commits. They live in the PR file as forensic record.

---

## Anti-patterns specific to debug-PR standards

- **Spec-less PR** — opening a debug PR without `## Bugs` populated to atomic granularity. The PR is not actionable until the contracts are written down.
- **Speculative fix in the stub** — writing "we'll fix this by adding X" in `## Scope` before Phase 6 closes. The Scope section is locked AFTER fix research, not before.
- **Test that doesn't map to a bug** — adding verification criteria that test general-purpose properties unrelated to the spec. Each criterion maps to a single bug in `## Bugs`.
- **Test that passes against unfixed code** — Phase 5 requires every test to fail against current code. A test that passes either tests the wrong thing OR the bug spec was wrong.
- **Living with failed-attempt residue** — leaving in code from a fix attempt that didn't turn its test green. The Phase 6 rollback decision is mandatory per failed-attempt commit.
- **Bundling debug + feature** — using the bug-fix PR as cover for an unrelated improvement. Split the PR.
- **Skipping the empirical-evidence gate in Phase 3** — closing Phase 3 with web-only or code-reading-only certainty. Empirical evidence at runtime is required to advance.
- **Skipping a user signoff** — Phases 4 and 6 require explicit approval. Proceeding without is a hard violation.
