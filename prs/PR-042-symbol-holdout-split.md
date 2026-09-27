# PR-042: The symbol-holdout one-off split kind and the embargoed temporal split

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier: Tier-1 proposed (operator's call).** The regimes are ruled by rux-capital program v0.3 D41 ("row-random (diagnostic only), time-block purged with embargo ≥ the longest label horizon, and symbol holdout — a new one-off rux-ml split kind by coin"; research R1-g proven) and ACCEPTANCE C9. Built by the R-14 track session (2026-09-26); **stacked on PR-041** (uses `[m9]`), merge after it; operator-gated merge.

## Research findings

### State Assessment (2026-09-26)

Baseline: `pr-041/m9-problem-configs` @ `b476e52` (on `dev` `fcdbf3f`).

**Current state**:
- `data.split_kind ∈ {random, time_ordered}` (PR-024), dispatched by `make_splits` to `train_val_test_split` (seeded shuffle) or `temporal_train_val_test_split` (sort + row-count slice). Consumers: `cli/train.py`, `registry/promote.py`, `registry/scorer.py` (seed recovered from the trial for non-`time_ordered` kinds), `tuning/objective.py`, `scripts/calibrate_pr025.py` (random / time_ordered only — a one-off calibration script, left as is).
- **The one-off temporal split has no embargo**: its docstring says adjacent partitions may share the boundary timestamp and points to the CV layer's purge knobs for timestamp-atomic separation.
- `GroupKFoldCV` (CV) and `PanelCombinatorialPurgedCV` (CV; purge/embargo in timestamp *counts* via `target_horizon_bars` / `embargo_pct`) exist; there is no one-off group split.
- The split seed is per trial (`SeedBag.split_seed`, spawned from `trial.number`), so a seeded shuffle of coins would hold out different coins in every fit of a study.

**Stale assumptions / new constraints**:
1. C9's run schedule counts **36 fits = 3 targets × 3 regimes × 4 prefixes** — one fit per cell, i.e. one-off splits, while C9's text says the time-block regime runs "via the panel purged splitter" (a CV splitter with n_folds × paths). This PR supplies the one-off embargoed split; the panel splitter remains available for CV. **Question Q7.**
2. The learning curve (D43) compares nested prefixes on the same holdout: the holdout assignment must not depend on the coin universe or the trial seed → per-coin hash assignment from a config seed.
3. New `DataConfig` fields would move every recorded `data_cfg_hash` / `root_cfg_hash` (PR-024 precedent) → hash-neutral while None; the PR-040 pins (`tests/config/test_oracle_config.py`) guard it and still pass.
4. h_max: program D45 #4 — "a label is written only when t + h_max < … the split; D41's embargo ≥ h_max is the same rule at the split". The program defines `markout_horizons_ms` (max 14,400,000) but **no fill-horizon grid** (`h_ms`), so h_max is not fully defined. **Question Q6.**

### Research (light — Tier-1)

- *Symbol holdout by stable hashing vs a seeded shuffle of present groups vs explicit lists.* Stable hashing (chosen): assignment independent of universe and order, config-driven, sizes approximate (binomial). Seeded shuffle (sklearn `GroupShuffleSplit` style): exact sizes but the assignment changes when a prefix has a different coin set — breaks D43's nested-prefix comparison. Explicit lists: most auditable but ~180 coins hand-maintained per universe change. Hash-based deterministic bucketing is a common pattern for stable holdouts (e.g. hashing an ID into train/test so the split survives dataset growth); **best-guess-given-constraints** (no web search in this track; the property it needs is proven by the tests).
- *Embargo semantics.* `t + e < first stamp of the next partition` is D45 #4's rule verbatim applied at the split — **proven** against the program's own ruling; `embargo=0` = timestamp-atomic.

### Group D (local probes)

- `make_splits` dispatch, `split_definition` record, config refusals and the repo M9 configs are asserted by tests below; `promote` and `registry score` reach the same partition (the kind uses the config seed, not the trial seed).

### Synthesis / Gate

**Outcome: Confirm.** ARCHITECTURE gains "One-off split kinds". No CONSTRAINTS change. Gate: track-autonomous; operator approval pending at merge.

---

## Scope

- `data.split_kind = "symbol_holdout"` + `data.group_column` + `data.symbol_holdout_seed` (both required by the kind); `symbol_holdout_split` (whole groups, sha256 `"<seed>:<group>"` → [0, 1) vs cumulative ratios; refuses missing / null group column and an empty positive-ratio partition).
- `data.split_embargo` (int ≥ 0, time-column units) for `time_ordered`: keep `t + e <` the next partition's first stamp; integer columns only; a fully purged train refused. None = legacy.
- `[m9] h_max_ms`; with `[m9]` and `time_ordered`, `h_max_ms` is required and `split_embargo ≥ h_max_ms` (refused at load).
- `train` writes `split_definition` into `fold_meta.json` (groups per partition / embargo + purged rows / rows per partition).
- The three M9 configs: `split_embargo = h_max_ms = 14400000` (best-guess lower bound, Q6), `group_column = "coin"`, `symbol_holdout_seed = 20260926` (arbitrary, operator-reviewable).
- The three new data fields hash-neutral while None.

## Dependencies

PR-041 (the `[m9]` layer).

## Architecture section implemented

`docs/ARCHITECTURE.md` "One-off split kinds: symbol holdout and the embargoed temporal split" (new).

## Verification criteria

- [x] Each coin lands whole in the partition its hash names (expected values computed from the contract in the test, independently of the implementation); no coin in two partitions; all rows placed.
- [x] A coin's partition is identical on a sub-universe in reverse row order; a different seed changes the test set.
- [x] Refusals: missing / null group column; an empty test partition (2 coins); config without `group_column` / `symbol_holdout_seed`; negative embargo; non-integer time column; fully purged train.
- [x] Embargo: hand-computed row sets on 0..19 (e = 3: train 0..8, val {12}, test 16..19) and on a stacked panel (e = 0 separates the shared boundary stamp; None keeps legacy slicing).
- [x] `[m9]` + `time_ordered`: refused without `h_max_ms`, refused at embargo 99 < 100, accepted at 100; the repo M9 configs satisfy it.
- [x] `train` records the exact coin lists (and rows) of a symbol-holdout fit and the embargo + 10 purged rows of a temporal fit.
- [x] PR-040's pinned data/root hashes unchanged; default suite 499 → 514 passed, 1 skipped, 0 failed; ruff clean; basedpyright: no new error.
- [ ] Leakage tests on the real set — PR-045 (needs program PR-024's set).

## Research backing

Program `docs/0.3/DESIGN-log.md` D41, D43, D45 #4; `docs/0.3/ACCEPTANCE.md` C9 and its run schedule.

## Notes — questions for the program lead

- Q6: h_max — the longest label horizon over all labels. D45 #3 gives `markout_horizons_ms` (max 4 h) but the fill-horizon grid (`h_ms`) and the walk latency L are undefined; `h_max_ms = split_embargo = 14400000` is a lower bound. Raise both if any horizon is longer.
- Q7: C9 names the panel purged splitter for the time-block regime but schedules one fit per (target, regime, prefix). Is the time-block regime this one-off embargoed split (as built), or a panel-CPCV study (n paths per cell)?
- Q8: symbol holdout vs cross-sectional contemporaneous leakage — test coins share timestamps with training coins, so market-wide moves are learnable across coins. D41 lists the symbol holdout as its own regime (built as such); gen-2's design notes mention "purged time-block × symbol holdout". Should a combined regime exist?
