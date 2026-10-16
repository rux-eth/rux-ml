# PR-064: The native adapter's out-of-core (ExtMem) path fails when an eval set is passed

**Landed-in:** (not yet landed)

<!-- Stub — opened 2026-10-16 on the rumpy operator's go (≈ 16:19 CDT, "go, approve the cut, merge 063, delete": item 5 "open a stub").
     Found by PR-063's build review; NOT researched, NOT built. Defect fix; its Phase 1 on its own word. -->

**Status:** stub — Phase 1 (State Assessment) not started.

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the
`## Research findings` section below.

## Research findings

_To be populated by `PROCEDURE-pr-research.md`._ What is known (PR-063's review, `prs/PR-063-pin-before-heavy-imports.md` §Build
status, 2026-10-16): on `dev` the native XGBoost adapter's out-of-core path (`ExtMemQuantileDMatrix`, built from the Parquet
iterator) fails whenever an eval set is passed — `ValueError('data did not contain feature names…')`: the training matrix built from the
iterator carries no feature names while the pandas eval matrix does. It fails the same way before and after PR-063 (not introduced by it);
PR-063's byte check could therefore not cover the ExtMem arms.

---

## Scope

- Reproduce on a minimal fixture (ExtMem + an eval set) — a test that fails on `dev`.
- Make the iterator-built matrix and the eval matrix agree on feature names (or both carry none), whichever the adapter's contract
  requires; no change to fitted bytes on the in-memory paths.

**Not.** No change to `memory.omp_threads` / thread handling (PR-063); no config default change.

## Dependencies

PR-063 (merged, `dev` `d51e89f`) — touches the same adapter (`nthread` on every matrix).

## Verification criteria

- [ ] The failing test first (ExtMem + eval set → the error on `dev`), passing after the fix.
- [ ] The ExtMem arms of PR-063's byte check run and are recorded (model / prediction bytes reproducible).
- [ ] The rux-ml suites as PR-062 / PR-063 ran them (Mac + desktop).

## Notes

- Opened as a stub only; tier and estimate at its Phase 1.
