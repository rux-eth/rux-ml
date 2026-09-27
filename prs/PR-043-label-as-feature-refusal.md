# PR-043: Label quarantine — a `y__` label is never a feature (extends PR-040)

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** The rule is ruled by rux-capital program v0.3 D38 #3 ("a `y__` column is a supervised target consumed only as `target_column`; rux-ml keeps refusing `oracle__` and gains a refusal of any `y__` column among the features") and ACCEPTANCE C6 ("rux-ml refuses an `oracle__` set and a `y__`-as-feature set on real files (exit 2, recorded)"); the mechanism is PR-040's (Tier-2, researched 2026-09-24), extended. R-14 track session, 2026-09-26; **stacked on PR-042**; operator-gated merge.

## Research findings

### State Assessment (2026-09-26)

Baseline: `pr-042/symbol-holdout-split` @ `30e29af`.

**Current state**:
- PR-040: `[data.oracle] namespace / tag_file`; `check_oracle_quarantine` in `load_parquet` + `compute_data_hash` + the CLI preflight `refuse_oracle_source` (train, tune ×3); `OracleQuarantineError(ValueError)` → `typer.BadParameter` → exit 2; `data.oracle` hash-elided; a missing table refuses.
- **The feature pipeline selects exactly `features.spec` columns** (`features/pipeline.py` `select_columns`), so the spec is the only route a column takes into a model.
- **Probe (this PR's failing tests, run before the implementation):** on a C6-shaped set under the real `m9_walk_bp` problem, `features.spec` listing `y__time_to_fill_ms` or `Y__WALK_BP_ALIAS` **trained with exit 0**; listing the diagnostic `y__fill_frac` exited **1** by traceback (PR-041's config validator — pydantic `ValidationError` is not mapped by the CLI).
- The oracle refusal on the C6 layout (a day-partitioned directory with one file carrying an `oracle__` column) already exits 2 through PR-040, unchanged.

**New constraints**:
- C6 wants exit 2 → the refusal must be a named runtime check, not a config validator (config errors exit 1). PR-041's "diagnostic is a feature" rule moves into the same check.
- "Extend, never duplicate": reuse PR-040's table, error family and exit-2 mapping; one function, called where the config is at hand before each load.

### Research (light — Tier-1)

- *Where to check: the feature spec (chosen) vs the loaded frame's columns vs a config validator.* The frame legitimately carries `y__` columns (the target and the diagnostics), so a frame-level namespace refusal (PR-040's shape) would refuse every M9 set; the spec is what enters the model. A config validator exits 1 in the CLI and does not satisfy C6. **Proven** by the probes above.
- *Case-insensitive prefix match*: PR-040's convention (its research: stricter than the harness's case-sensitive `startswith`), kept for consistency.

### Synthesis / Gate

**Outcome: Confirm.** ARCHITECTURE gains "Label quarantine"; PR-041's validator rule relocated (documented). Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `OracleQuarantineConfig.label_namespace` (required, `min_length=1`); `configs/base.toml` `label_namespace = "y__"`; hash-neutral (`data.oracle` elided).
- `LabelAsFeatureError(OracleQuarantineError)`; `check_label_quarantine(features, oracle, diagnostics=…)`; `check_feature_labels(cfg)`.
- Called in `refuse_oracle_source` (train / tune start / resume / retry-trial preflight, first, regardless of `source_path`), `registry/promote.py` re-fit, `registry/scorer.py`, `tuning/objective.py` — each before its load.
- PR-041's config-time diagnostic ∩ features rule removed from the `RuxMLConfig` validator (now in the runtime check).

Not covered: `scripts/calibrate_pr025.py` (a one-off calibration script for `crypto_breakout_h3`; no `y__` columns).

## Dependencies

PR-042 (stack); PR-040 (merged).

## Architecture section implemented

`docs/ARCHITECTURE.md` "Label quarantine: a label is never a feature" (new).

## Verification criteria

- [x] Unit: a `y__` feature (and `Y__…`) refused with `LabelAsFeatureError`, an `OracleQuarantineError`; a listed diagnostic refused; no `[data.oracle]` refuses; clean passes (`tests/data/test_quarantine.py`).
- [x] CLI on a C6-shaped day-partitioned set under the real `m9_walk_bp` problem: clean trains (exit 0, `score (mae)`); three label-feature variants exit 2 with "label quarantine" and **0 trial rows**; `tune start` exits 2 before any child; the promote re-fit exits 2 and writes no bundle; an `oracle__` column exits 2 through PR-040 (`tests/cli/test_label_quarantine_cli.py`, 7 tests; 5 fail without this PR).
- [x] PR-040 pins unchanged; default suite 514 → 524 passed, 1 skipped, 0 failed; ruff clean; basedpyright: no new error.
- [ ] The recorded real-file run (C6 evidence) — the program's, once PR-024's set exists (Q9).

## Research backing

Program D38 #2–#3; ACCEPTANCE C6, C13; PR-040's research record.

## Notes — questions for the program lead

- Q9: C6's "on real files (exit 2, recorded)" — rux-ml's refusal is proven on a C6-shaped synthetic set; the recorded real-file run needs PR-024's set (commands: `rux-ml --config configs/base.toml --problem m9_walk_bp --set data.source_path=<set> --set 'features.spec.numeric_columns=["p_bp","q_usd","h_ms","y__fill_frac"]' train` → exit 2; and the set with an `oracle__` column → exit 2).
- Q10: is `y__` the harness's final label namespace, and will the harness carry it as a config key rux-ml should mirror (as `oracle__` mirrors `config/harness.toml [oracle]`)?
