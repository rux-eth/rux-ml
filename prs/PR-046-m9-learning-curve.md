# PR-046: The nested-prefix learning curve and D43's "history-limited" verdict

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** Ruled by program v0.3 D43 (R1-f best-guess: nested prefixes ¼ / ½ / ¾ / full, "history-limited" when the OOS metric still improves from ¾ to full by more than the tolerance) and ACCEPTANCE C9 (the curve and the verdict "against the signed tolerance"). R-14 track, 2026-09-26; **stacked on PR-045**; operator-gated merge.

## Research findings

### State Assessment (2026-09-26)

Baseline: `pr-045/m9-leakage-tests` @ `0fe8e8b`.
- No learning-curve mechanism existed. Each `train` run lands in its own timestamped study (`runs.study_name_template`), so a curve spans several studies (or several trials of one when two runs share a second).
- PR-044 records every `[m9]` fit's held-out `oos` block; PR-044's reader already returns `m9_learning_curve_tolerance_rel`.

**Decisions** (light; flagged where the program is silent):
- *What is a prefix.* D43 says "of the corpus, purged". Cutting the corpus and re-splitting would move the test set with every prefix, and the curve would then compare different OOS sets. Built: prefixes of the **train window** after the regime split (val / test fixed), nested by construction. **best-guess-given-constraints** — Q18.
- *Which scalar.* D43's "the OOS honesty metric" is not a single number (the honesty test has a fraction and a bound). Built: a config path into the `oos` record, set to `score` (MAE in bp for markout / walk, Brier for the fill fraction), direction `minimize`. **best-guess** — Q17.
- *Verdict arithmetic.* Relative improvement of the last step `(prev − last) / |prev|` (mirrored to maximize), fired when `> tolerance`. Follows D43's sentence; a zero `prev` refuses.

### Synthesis / Gate

**Outcome: Confirm (with the flagged definitions).** ARCHITECTURE gains "The nested-prefix learning curve". Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `data.train_prefix_frac` (0 < f ≤ 1; requires `time_column`; hash-neutral while unset); `rux_ml.data.splits.train_prefix`; recorded in `split_definition`.
- `[m9] learning_curve_fractions / learning_curve_metric / learning_curve_direction`; set in the three M9 problems to `[0.25, 0.5, 0.75, 1.0]`, `score`, `minimize`.
- `rux_ml.runs.learning_curve.learning_curve_report`, `LearningCurveError`; verb `rux-ml runs learning-curve --study … --output …` (gates read fail-closed; exit 2 on an incomparable curve).

## Dependencies

PR-045 (stack).

## Architecture section implemented

`docs/ARCHITECTURE.md` "The nested-prefix learning curve" (new).

## Verification criteria

- [x] `train_prefix`: 12 unique stamps, f = ½ keeps 0..5 (ceil 6), f = ¼ keeps 3 stamps, f = 1 keeps all; prefixes of one split nest strictly (`tests/runs/test_learning_curve.py`).
- [x] Verdict: MAE 3.0 / 2.4 / 2.0 / 1.8 → (2.0 − 1.8) / 2.0 = 0.10 > 0.05 fires; 1.95 → 0.025 does not; a worse full fit → −0.10; a maximized metric 0.8 → 0.9 → 0.125 fires.
- [x] Refused: a missing or repeated fraction, a second regime, a different test row count, a missing value.
- [x] CLI: four prefix fits of the real `m9_walk_bp` on the C6-shaped set → the report orders ¼ … full, echoes the signed tolerance and the gates sha256, and its verdict equals the hand formula on its own values; a single fit exits 2 and writes nothing (`tests/cli/test_m9_train_cli.py`; both fail without the verb).
- [x] Default suite 557 → 569 passed, 1 skipped, 0 failed; ruff clean; basedpyright: no new error.
- [ ] The real curves — the program's Stage 3 (per target: `train --study m9_regime_<r> --set data.train_prefix_frac=<f>` × 4, then `runs learning-curve`).

## Notes — questions for the program lead

- Q17: the scalar "the OOS honesty metric" of D43 — built as the held-out `score` (MAE bp / Brier), configurable to any `oos` path (e.g. `honesty.no_underdeduct_frac` with `maximize`).
- Q18: prefixes of the train window with a fixed OOS set (built) vs prefixes of the whole corpus re-split per prefix — and on which regime the D43 verdict is taken (C9 lists all three).
