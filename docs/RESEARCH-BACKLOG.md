# Research Backlog

Index of PRs with research status. Every PR runs `PROCEDURE-pr-research.md` before implementation — this document tracks the state of each PR's research and flags drift risk per the time-decay policy.

**Tiers:**
- **Tier 1** — Design-time research exists. Phase 1 (State Assessment) required before implementation; Phases 2-4 may be light if no drift found.
- **Tier 2** — Design-time research is partial or absent. Full procedure required before the PR can be written in final form.

**Status legend:**
- `design-research ✓` — research done at design time
- `design-research ~` — partial design research (some aspects covered, others not)
- `design-research ✗` — no design-time research
- `state-assessed YYYY-MM-DD` — Phase 1 of `PROCEDURE-pr-research.md` completed
- `fully-researched YYYY-MM-DD` — all 5 phases of `PROCEDURE-pr-research.md` completed
- `implementation-cleared YYYY-MM-DD` — Phase 5 Gate Check passed

---

<!-- Populate this document during the design session (Phase 4). Each PR gets a row. -->

## Tier 1 — Implementation-Ready (pending state assessment)

<!-- PRs whose significant decisions are research-backed at design time. -->
<!-- Table columns: | PR | Design research | State assessed | Implementation cleared | -->

## Tier 2 — Research-Pending

<!-- PRs that require full PROCEDURE-pr-research.md before they can be written in final form. -->
<!-- Table columns: | PR | Design research | Required research topics | -->

---

## Drift Watch

Per the time-decay policy in `PROCEDURE-pr-research.md`, any PR marked `fully-researched` or `state-assessed` more than the project's staleness threshold before implementation begins must re-run Phase 1 (State Assessment).

**Project staleness threshold:** <!-- e.g., 30 days — define during design session -->

**Currently watching:** <!-- list PRs approaching staleness -->

## How to update this document

- After Phase 4 (Docs) of a design session, add each new PR as a row with its design-research status
- After running `PROCEDURE-pr-research.md` Phase 1 for a PR, update the `State assessed` column with the date
- After completing all 5 phases, update `Implementation cleared` with the date
- If state assessment surfaces drift that blocks implementation, move the PR back to Tier 2 and document required research
- New PRs added to the roadmap start as Tier 2 unless they explicitly inherit prior research
