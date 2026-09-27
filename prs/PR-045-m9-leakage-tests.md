# PR-045: The leakage audit and the three M9 evaluation regimes

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** Ruled by program v0.3 D41 (three regimes) and ACCEPTANCE C9 ("leakage tests on the real set (no holdout coin's row in train; no training stamp inside a test stamp's embargo) and the fold definitions committed"). R-14 track, 2026-09-26; **stacked on PR-044**; operator-gated merge.

## Research findings

### State Assessment (2026-09-26)

Baseline: `pr-044/m9-honesty-metric` @ `0948610`.
- PR-042 gives the regimes' splitters and records `split_definition`; rux-ml's CV leakage tests (`tests/data/test_cv_leakage.py`, PR-023) cover the CV splitters, not one-off splits.
- Nothing measured the separation of a one-off split independently of the code that made it.
- The real set (program PR-024) is not landed; per the lead's note (PR-024 amendment A13, 2026-09-26) it becomes two subtrees, `fill/` and `walk/`.

**Decisions** (light; the rule is the program's, the mechanism is internal):
- Stamp test = C9's sentence as a measurement: a later row violates when a fitted stamp lies within the window, `|s − t| ≤ window` (symmetric, so it also measures the row-random regime). Test is checked against train + val because val early-stops the fit. Window = `[m9] h_max_ms` (the label horizon is what leaks), else `split_embargo`.
- Independent algorithm (binary search over sorted unique stamps; set intersection) so the audit checks the splitters rather than restating them. **Proven** on hand-computed counts.
- Enforced only for `[m9]` fits: a legacy `time_ordered` problem without an embargo promises no stamp separation and must keep training.
- "The fold definitions committed": the regimes as committed study overlays; each fit's `split_definition` (groups per partition / embargo + purged rows) and `leakage` are the per-fit record.

### Synthesis / Gate

**Outcome: Confirm.** ARCHITECTURE gains "The leakage audit and the M9 regimes". Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `rux_ml.data.leakage`: `leakage_audit`, `assert_regime_clean`, `LeakageError(ValueError)`.
- `train` records `leakage` for every fit; an `[m9]` fit is refused (exit 2) when its regime's promise fails.
- `configs/studies/m9_regime_row_random.toml`, `m9_regime_time_block.toml`, `m9_regime_symbol_holdout.toml`.

## Dependencies

PR-044 (stack).

## Architecture section implemented

`docs/ARCHITECTURE.md` "The leakage audit and the M9 regimes" (new).

## Verification criteria

- [x] Hand-computed counts: fitted stamps {0, 10, 20} + {22}, test {25, 26, 60}, window 5 → 2 test violations, 1 val violation; group overlaps 1 / 0 / 0 (`tests/data/test_leakage.py`).
- [x] On a 24-coin × 50-stamp panel: row-random leaks on both axes; the embargoed time-block split has 0 violations while the legacy slicing leaks and is refused; the symbol holdout has 0 group overlap and a planted coin is refused; an unmeasurable promised axis is refused.
- [x] CLI, real `m9_walk_bp` + each regime overlay on the C6-shaped set: row-random records violations on both axes (window 14,400,000 = `h_max_ms`); time-block records 0 and a positive purge; symbol holdout records 0 overlaps and the exact coin lists (`tests/cli/test_m9_train_cli.py`).
- [x] Default suite 549 → 557 passed, 1 skipped, 0 failed; ruff clean; basedpyright: no new error.
- [ ] On the real set: `rux-ml --problem m9_<target> --study m9_regime_<regime> train` for each (target, regime) — the program's Stage 3 run; the audit rides in each fit's `fold_meta.json`.
