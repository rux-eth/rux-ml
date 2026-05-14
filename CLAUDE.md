# CLAUDE.md

<!-- STATUS: uninitialized -->
<!-- When status is "uninitialized", Claude runs the onboarding flow below. -->
<!-- After onboarding, Claude replaces this entire file with project-specific guidance. -->

## Onboarding Instructions (for Claude)

This is a fresh project created from the **vibe-rails** template. The project has not been initialized yet.

**Trigger**: When the user sends "init", "start", "begin", "hello", "hi", or any first message, check the status marker above. If it says `uninitialized`, ask the user **"What do you want to build?"**

Then follow these steps exactly:

### Step 1: Understand the vision

Ask clarifying questions until you understand:
- What the project does (core purpose)
- Who it's for (target users/audience)
- What tech stack they want (language, frameworks, infrastructure)
- Any hard constraints they already know about

Do NOT start suggesting architecture yet. Just listen and ask questions.

### Step 2: Run the design planning procedure

Once you have enough context, follow `PROCEDURE-design-planning.md` exactly:
- **Idea phase**: restate what you heard, confirm understanding
- **Decisions phase (with research)**: work through architectural choices one at a time. For every significant decision, run research rounds (using parallel agents where possible). No decision is made without research backing.
- **Convergence phase**: summarize all decisions + the research that backs each, get user confirmation
- **Docs phase**: populate the doc templates (see Step 3)

### Step 3: Populate project docs

Based on the design session, fill in:

1. **`docs/ARCHITECTURE.md`** — replace the skeleton with actual architecture decisions:
   - System overview
   - Component/service descriptions
   - Data flow
   - Key abstractions/traits/interfaces
   - Storage strategy
   - Testing strategy

2. **`docs/CONSTRAINTS.md`** — keep the structural constraints, add domain-specific non-negotiables the user defined. Set the project's **staleness threshold** for research time-decay.

3. **`docs/ROADMAP.md`** — create phased PR plan with dependencies. Each PR gets a file in `prs/` (copy from `prs/PR-TEMPLATE.md`) with:
   - Mandatory "Before Implementation" section (inherited from template)
   - Scope (what's included)
   - Dependencies (what must be done first)
   - Verification criteria (how to prove it works)

4. **`docs/RESEARCH-BACKLOG.md`** — index every PR with its research status (Tier-1 research-backed vs Tier-2 research-pending), and set the project's staleness threshold.

5. **`docs/DESIGN-log.md`** — capture the design conversation (decisions + rationale + research trail)

### Step 4: Rewrite this file

Replace this entire `CLAUDE.md` with project-specific guidance. Use this structure:

```markdown
# CLAUDE.md

<!-- STATUS: initialized -->

## What Is [Project Name]

[One paragraph description]

## Build & Test Commands

[Language/framework specific commands]

## Architecture

[Summary + pointer to docs/ARCHITECTURE.md]

## Implementation Status

See `docs/ROADMAP.md` for the ordered PR plan. Full PR descriptions in `prs/`.

## Ongoing Behavior (MANDATORY)

- **Every PR runs `PROCEDURE-pr-research.md` before implementation.** No exceptions. Tier-1 PRs get the light path (Phase 1 State Assessment); Tier-2 PRs get the full 5-phase path.
- **Research findings travel with the PR** — appended to the PR file's `## Research findings` section.
- **State drifts** — even research-backed decisions need Phase 1 state assessment before implementation.

## Design References

- `docs/ARCHITECTURE.md` — canonical architecture reference
- `docs/CONSTRAINTS.md` — hard rules
- `docs/ROADMAP.md` — PR index with phases and dependencies
- `docs/RESEARCH-BACKLOG.md` — per-PR research status + drift watch
- `prs/` — full PR descriptions (start new PRs from `prs/PR-TEMPLATE.md`)
- `docs/DESIGN-log.md` — design conversation log
- `PROCEDURE-design-planning.md` — how to run design sessions (with integrated research rounds)
- `PROCEDURE-pr-research.md` — mandatory research procedure before every PR implementation
- `PROCEDURE-code-audit.md` — post-design-session code audit

## Constraints

[List non-negotiables from docs/CONSTRAINTS.md]
```

### Step 5: Confirm completion

Tell the user: "Project initialized. Here's what I set up: [summary]. Ready to start on PR-001?"

Remember: PR-001 cannot begin implementation until `PROCEDURE-pr-research.md` has been followed. Even if it's Tier-1, Phase 1 (State Assessment) runs first.

---

## Ongoing Behavior (post-initialization)

Once initialized, Claude should:

- **Follow the constraints** in `docs/CONSTRAINTS.md` at all times
- **Update docs with code** — every code change includes doc updates in the same commit
- **Use the PR workflow** — work is tracked in `prs/` (new PRs start from `prs/PR-TEMPLATE.md`), status in `docs/ROADMAP.md` and `docs/RESEARCH-BACKLOG.md`
- **Run `PROCEDURE-pr-research.md` before every PR implementation** — no exceptions. State drifts, research must be validated.
- **Run code audits** after design sessions (per `PROCEDURE-code-audit.md`)
- **Run design planning with integrated research** for new features (per `PROCEDURE-design-planning.md`)
- **Never create phantom implementations** — if it's not tested, it's not done
- **All configurable values from config files** — zero hardcoded parameters
- **Every architectural decision is research-backed** — cite reputable sources
