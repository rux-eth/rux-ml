# Constraints

Hard rules that must never be violated. These are enforced by Claude at all times.

---

## Structural Constraints

These apply to every project built with this template.

### No Phantom Implementations (NON-NEGOTIABLE)

A step is NOT complete if:
- A function exists but returns a default, stub, or placeholder value
- A module is declared but never called from the main flow
- Tests only verify something exists, not that it works correctly

Every PR must include:
1. A test that exercises the **actual behavior**
2. Explicit listing of any stubs or TODOs in the PR description
3. Proof of end-to-end data flow where applicable

### Documentation Accuracy (NON-NEGOTIABLE)

Every code change must include corresponding doc updates in the same commit. Before writing docs, check the actual diff — base doc updates on what changed, not memory. Docs must never describe behavior that doesn't exist in code.

### One PR, One Thing

Each PR is a single, reviewable change. No "while I'm here I'll also add..." — that's scope creep. Each PR references the specific section of the architecture it implements.

### Config Over Hardcoding

All configurable values come from config files. Zero hardcoded parameters for behavior that might change. If a value could reasonably vary between environments or over time, it belongs in config.

### Research-Backed Decisions (NON-NEGOTIABLE)

Every architectural and significant design decision must be backed by research from **reputable sources**. "I think this is right" is not sufficient.

**Reputable sources:**
- Production system documentation
- Official framework source and docs
- Published post-mortems and engineering blog posts from serious engineering teams
- Battle-tested open-source code with meaningful adoption

**Not sufficient:**
- LLM intuition
- Marketing pages
- Personal blog posts without engineering weight
- StackOverflow answers without corroborating evidence

**Process:** decisions without research backing must be explicitly flagged as unresearched in the relevant PR or design doc, and researched before implementation begins. `docs/DESIGN-log.md` tracks which decisions have research and which don't. `PROCEDURE-design-planning.md` integrates research rounds into Phase 2 (Decisions).

### PR Research Procedure Required (NON-NEGOTIABLE)

No PR is implemented until `PROCEDURE-pr-research.md` has been followed and its findings are documented in the PR file's `## Research findings` section.

**Applies to all PRs — including PRs that were research-backed at design time.** State drifts between design and implementation. The procedure's Phase 1 (State Assessment) catches drift before implementation begins.

**Enforcement:**
- Every PR file starts from `prs/PR-TEMPLATE.md`, which includes a `## Before Implementation (NON-NEGOTIABLE)` section requiring this procedure
- Every PR file has a `## Research findings` section that must be populated before implementation
- PRs without completed research findings are rejected
- Research findings include a state-assessment date; if implementation hasn't started within the project's staleness threshold, re-run state assessment per the time-decay policy in `PROCEDURE-pr-research.md`

State drifts. Research must be validated before code.

---

## Domain Constraints

<!-- Define your project-specific non-negotiables here. -->
<!-- These are the rules that, if violated, would make the system fundamentally broken. -->
<!-- Examples: -->
<!-- - No unauthorized access to user data -->
<!-- - All monetary calculations use decimal, never floating point -->
<!-- - Every API endpoint must be authenticated -->
<!-- - No external network calls during unit tests -->

<!-- Project staleness threshold (for PR research time-decay): -->
<!-- Example: "Research is considered stale after 30 days for this project" -->
