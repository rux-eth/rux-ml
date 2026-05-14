# Procedure: PR Research

## When to use

**Before every PR implementation.** No exceptions. This procedure runs even for PRs that were research-backed at design time — state drifts between design and implementation, and research assumptions must be validated before code.

The research-backed-decisions constraint in `docs/CONSTRAINTS.md` requires every architectural and significant design decision to be backed by research from reputable sources. This procedure is how that requirement is enforced per-PR.

## Output location

Research findings are appended to the PR file itself under the `## Research findings` section. The research travels with the PR in git history. Do not discard research output.

## Phases

### Phase 1: State Assessment

**Goal**: Establish what is actually true *right now* — not what was assumed when the PR was drafted.

1. Read the PR file completely — scope, dependencies, verification criteria, any prior research findings
2. Read the current codebase around the PR's area — what exists vs. what was assumed
3. Read the last N PRs that landed in related areas — what surfaced, what changed
4. Re-read relevant `docs/ARCHITECTURE.md` + `docs/CONSTRAINTS.md` sections — current locked decisions
5. Check `docs/DESIGN-log.md` for any decisions that might have drifted

**Output** (appended to PR's `## Research findings` section):

```markdown
### State Assessment (YYYY-MM-DD)

**Current state**:
- [What code/decisions exist now that are relevant to this PR]

**Assumptions at PR draft time**:
- [What the PR spec assumed when it was written]

**Stale assumptions** (where current state disagrees with PR assumptions):
- [Assumption X turned out to be Y]

**New constraints** (learned from prior PRs or codebase evolution):
- [Constraint Z that restricts this PR's options]
```

**Exit criteria**: stale assumptions flagged, new constraints documented. If stale assumptions are severe enough to change the PR's premise, STOP — loop to `PROCEDURE-design-planning.md`.

### Phase 2: Scope the Research

**Goal**: Define exactly what must be answered before this PR can be implemented.

1. List decisions this PR requires that aren't already made
2. Mark each as **MUST-ANSWER** (blocks implementation) or **NICE-TO-HAVE** (explicitly excluded)
3. For each must-answer: define what form a good answer would take (library name + reasoning, concrete threshold, code pattern, etc.)
4. Identify dependencies between questions — if Q2 depends on Q1's answer, sequence them

**Output** (appended to PR's `## Research findings` section):

```markdown
### Research Questions

**Must-answer:**
1. [Question] — success criteria: [what a good answer looks like]
2. [Question] — success criteria: [...]

**Dependencies:**
- Q2 depends on Q1 (sequential)
- Q3, Q4 independent (parallel)

**Explicitly excluded from this round** (nice-to-have):
- [Question deferred to later PR or later research round]
```

**Exit criteria**: a ranked list of must-answer questions with clear success criteria.

### Phase 3: Run the Research

**Goal**: Answer the must-answer questions using reputable sources, presenting evidence for AND against each candidate option without bias.

**Rules** (non-negotiable):

- **Web search is mandatory** for every must-answer question. Dispatch at least one research agent with `WebSearch` / `WebFetch` capability per independent question. Internal codebase reading, prior design-log citations, or LLM general knowledge alone is not sufficient.
- **Reputable sources only** (per `docs/CONSTRAINTS.md`): production system documentation, official framework source on GitHub, RFCs, published post-mortems / engineering blog posts from serious teams, battle-tested OSS with meaningful adoption. Not sufficient: marketing pages, random StackOverflow answers, personal blogs without engineering weight, LLM intuition dressed up as "common knowledge."
- **Unbiased presentation**: every finding MUST include at least one alternative / competing option with its own cited evidence — not just support for a preferred answer. Actively search for disconfirming evidence against the leaning option. If genuinely none exists after search, say so explicitly and record the search attempts made.
- **Pros/cons for each option** — structured, cited, with concrete implications (performance, correctness, ergonomics, maintainability, security). Avoid vague adjectives ("clean", "simple") — describe the specific technical tradeoff.
- Parallel agents for independent questions; sequential for dependent ones.
- Every finding labeled with its epistemic status:
  - **Proven** — widely deployed with direct cited evidence (production source, RFC, post-mortem showing the choice works / fails).
  - **Convention** — common practice in cited production systems, but without rigorous proof. **Must cite at least 2 independent systems using this convention.** Single-source "convention" is not allowed.
  - **Best-guess-given-constraints** — explicitly flagged when evidence is thin or unavailable after genuine search. Not a default.

**Output** (appended to PR's `## Research findings` section):

```markdown
### Findings

**Q1: [question text]**

*Options considered:*
- **Option A: [name]** — [one-line summary]
  - Sources: [cited URLs]
  - Pros: [concrete list, each tied to a specific technical consequence]
  - Cons: [concrete list, same standard]
- **Option B: [name]** — [one-line summary]
  - Sources: [cited URLs]
  - Pros: [...]
  - Cons: [...]
- [Option C, ...]

*Disconfirming evidence sought:*
- [What counter-arguments / failure modes / known drawbacks were searched for, and what was found — or "none found after searching [specific queries / sources]"]

*Recommendation:* Option [X]
- **Status**: proven / convention / best-guess-given-constraints
- **Why**: [reasoning grounded in the pros/cons above]
- **Risks accepted**: [explicit cons of the chosen option that we're living with]
```

**Exit criteria**: every must-answer question has ≥2 cited options with pros/cons, an explicit disconfirming-evidence section, and a recommendation with status label backed by the sources cited.

### Phase 4: Synthesis

**Goal**: Reconcile findings with the PR spec, the codebase (if backfilling an already-merged PR), and the broader project docs.

**Research Outcome Branch** (determine BEFORE running the synthesis steps below):

- **Confirm**: findings support the current spec / shipped code. Proceed with the synthesis steps; update doc sections with the cited findings.
- **Amend**: findings reveal a significant drawback or a better alternative to what's in the spec / shipped code. **STOP the synthesis. Present findings + amendment options to the user** (keep as-is with documented risk / change per cited recommendation / escalate further). Resume only after an explicit user decision.
- **Escalate**: findings invalidate the PR's premise entirely. **STOP.** Loop to `PROCEDURE-design-planning.md`.

Synthesis steps below execute only on a **Confirm** outcome, or on an **Amend** outcome after the user has approved the specific amendment.

1. What changed vs. the PR's original spec? Update the PR file's scope/verification sections
2. Does anything propagate back to `docs/ARCHITECTURE.md` or `docs/CONSTRAINTS.md`? Update them in the same commit
3. Are there new PRs that must come first? Update `docs/ROADMAP.md` and create new PR files with this procedure marked pending
4. Remove any invented specifics from the PR file — replace with research-backed details

**Output** (appended to PR's `## Research findings` section):

```markdown
### Synthesis

**Outcome**: Confirm / Amend / Escalate — [one-line rationale]

**Changes to this PR** from research:
- [Specific change, e.g., "Library choice locked to X instead of Y", or "none — findings confirm original spec"]

**Changes to ARCHITECTURE.md**:
- [None, or specific section updates committed alongside]

**Changes to CONSTRAINTS.md**:
- [None, or new constraint added]

**New PRs that must come first**:
- [None, or PR-XXX added to roadmap]

**Research-backed details now locked in this PR**:
- [List the specific choices research answered]
```

**Exit criteria**: PR file updated with research-backed specifics; related docs updated if needed.

### Phase 5: Gate Check

**Goal**: Confirm the PR is actually ready to implement.

1. Does research invalidate the PR's premise? → loop to `PROCEDURE-design-planning.md`
2. Did research surface prerequisite PRs? → update `docs/ROADMAP.md`, implement those first
3. Does the user approve the updated PR spec?
4. If all clear → PR is ready to implement

**Output** (appended to PR's `## Research findings` section):

```markdown
### Gate Check

- Premise still valid: ✓
- No prerequisite PRs surfaced: ✓
- User approved updated spec: ✓ (YYYY-MM-DD)
- Implementation cleared
```

**Exit criteria**: explicit user approval to begin implementation, or user directs a replan.

## Anti-patterns

- **Researching without state assessment** — findings are generic and not grounded in current reality. Phase 1 is non-negotiable.
- **Re-running research that's already locked** — wastes effort. If prior research exists and state assessment shows no drift, skip Phase 3 for that question.
- **Accepting "this is convention" without a source** — convention + no source ≠ research-backed. Label as best-guess-given-constraints and make the risk explicit.
- **Padding scope** — researching nice-to-haves during a PR-focused round. Defer them to their own research round.
- **Drift between research and docs** — when research changes a decision, `ARCHITECTURE.md` / `CONSTRAINTS.md` / `DESIGN-log.md` MUST update in the same commit.
- **Skipping state assessment for research-backed PRs** — state may have drifted. Phase 1 always runs; Phases 2-4 may be light if no drift is found.
- **Confirmation bias** — cherry-picking sources that support a predetermined answer. Phase 3 requires ≥1 alternative with pros/cons, an explicit disconfirming-evidence search, and honest reporting when the preferred option has real drawbacks.
- **LLM intuition dressed as "convention"** — labeling a decision "convention" without cited production-system examples. If only general knowledge supports a claim, it must be labeled `best-guess-given-constraints` and the risk flagged.
- **Silent amendment** — updating code or docs to align with research without explicit user decision when research reveals an amendment is warranted. Phase 4's Outcome Branch is mandatory; the user must be informed and approve any code change the research suggests.

## Time-decay policy

If a PR's research was completed more than the project-defined staleness threshold before implementation begins, re-run **Phase 1 (State Assessment)**. If state assessment surfaces stale assumptions, re-run the relevant parts of Phases 2-4 to update.

The threshold is project-defined. Typical range: weeks to months. Faster-moving ecosystems warrant shorter thresholds. Record the chosen threshold in `docs/CONSTRAINTS.md` or `docs/RESEARCH-BACKLOG.md`.

## Relationship to other procedures

- `PROCEDURE-design-planning.md` — used when the PR's premise itself is in doubt (Phase 5 loops here). Design planning sets direction with integrated research; PR research validates specifics within that direction.
- `PROCEDURE-code-audit.md` — used after design changes to catch drift between docs and code. PR research catches drift between the PR spec and current state; code audit catches drift between current state and the docs.
