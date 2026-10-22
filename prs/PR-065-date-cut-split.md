# PR-065: The `[m9]` time-block split with the cut given as a date

<!-- Number: next free at origin/dev d51e89f — PR-064 is the ExtMem eval-set stub (branch pr-064-extmem-eval-stub, 9f24fe6);
     no prs/ file, local or remote branch, or doc names PR-065 (checked 2026-10-22). Confirmed at the lead's gate. -->

**Landed-in:** (not yet landed)

## Before Implementation (NON-NEGOTIABLE)

This PR MUST NOT be implemented until `PROCEDURE-pr-research.md` has been completed in full and its output appended to the `## Research findings` section below.

**Tier-1 PRs** (research-backed at design time): Phase 1 (State Assessment) is required to catch drift. Phases 2-4 may be light if no drift is found.

**Tier-2 PRs** (research-pending): all 5 phases of `PROCEDURE-pr-research.md` must run before this PR is written in final form.

Skipping the PR research procedure is a hard violation of the research-backed-decisions constraint in `docs/CONSTRAINTS.md`.

**Tier (lean): Tier-1.** The rule is fixed by the consumer (program v0.7 D63 as amended; the harness's
`probe/partition.py` `partition_at_cut`, program PR-102) and Phase 1 shows rux-ml already applies it once the cut is
an input (below); what is left is config shape — operator decisions, not external research. Phases 2–4 light.

## Research findings

### State Assessment (2026-10-22)

Run as a track of the rumpy program (program branch `track/ruxml-date-cut`, the operator's go ≈ 12:18 CDT,
item 10 of v0.7's Stage 1 go-list), under rux-ml's own procedure. Base: `origin/dev` = `d51e89f` (unchanged by the
fetch). Labels: **proven** (read in code or executed), **convention** (repo precedent), **best-guess**.

**Why this PR (the program's ask).** Program v0.7 fits two things through rux-ml under a calendar split — the feature
probe (program PR-110) and, conditionally, one training (program PR-111): train = rows whose label ends before
2025-11-26, test = rows stamped on or after it, in blocks of 28 days. rux-ml's split takes only fractions
(`data.split_ratios`), so landing the cut on a date means choosing `split_ratios.train` per cell so that the stamp at
sorted position `int(n · train)` is the date — fragile (float64 `int(n · frac)` rounding, ties on the boundary stamp,
re-chosen per cell and per row filter). The program asks for the cut as a DATE, row-identical to the harness's
`partition_at_cut` (harness `b440fb0`): train = valid & `stamp + h·DAY_MS < cut`; test = valid & `stamp ≥ cut`; ties
to test; an event's stamp = `(day + 1) · DAY_MS`.

**Current state:**
- **The fraction split is the only cut source (proven).** `config/data.py:65` `split_kind: Literal["random",
  "time_ordered", "symbol_holdout"]`; `split_ratios` (default 0.7 / 0.15 / 0.15); `split_embargo` (int, the time
  column's own units, `ge=0`); `train_prefix_frac`. No date- or stamp-valued cut field anywhere in `src/`.
- **An `[m9]` split already IS a stamp-predicate rule (proven).** Since PR-054, `make_splits` with `[m9]` set runs
  `rux_ml.data.partitions.plan_partitions` over the frame as one batch; `_plan_time` turns the ratios into two stamps
  (`val_first_stamp` = the stamp at sorted position `int(n·train)`, `test_first_stamp` = at `int(n·(train+val))`) and
  `PartitionPlan._time_folds` assigns: train = `s + emb < t_val`; val = `s ≥ t_val & s + emb < t_test`; test =
  `s ≥ t_test`; the rest `PURGED`. Boundary ties go to the later partition (`split_record` `cuts.boundary_ties =
  "later_partition"`). The per-day batch fit (`cli/train.py:616`) and promote's re-fit (`registry/promote.py:293`) use
  the same plan, so every consumer shares the membership by construction.
- **Val-less mode (proven; PR-060).** `split_ratios.val == 0` → `t_val == t_test`, val is empty by the inequality, no
  `eval_set`, early stopping must be off (`cli/train.py:249` `_val_less` refuses otherwise), `[m9]` only.
- **The embargo floor (proven; PR-042).** `root.py:199-210`: an `[m9]` `time_ordered` problem needs `m9.h_max_ms`
  and `data.split_embargo ≥ h_max_ms` ("a label reaches the split"); `_plan_time` refuses a missing embargo and
  "split embargo … purged every train row" when `int(n·train) ≥ 1` but no train row survives.
- **The record and its readers key on `time_ordered` (proven).** `split_record` writes `cuts.val_first_stamp /
  test_first_stamp`; the OOS export's train mask (`runs/oos_export.py:35-40`, the rule the harness recomputes) is
  `stamp + split_embargo < cuts.val_first_stamp`; `leakage.assert_regime_clean("time_ordered", …)` checks stamp
  separation within `h_max_ms`; `registry/scorer.py:204` uses seed 0 for `time_ordered`; `runs/learning_curve.py` and
  `cli/runs.py` read `split_kind` / `split_definition.kind`.
- **Hash stability for new optional fields (proven; convention from PR-042).** `root.py:82-86`
  `_HASH_OPTIONAL_FIELDS["data"]` drops listed fields from the hashed dump while `None`, so existing `data_cfg_hash` /
  `root_cfg_hash` pins do not move.
- **The harness mirror (proven, harness `b440fb0`).** `probe/partition.py` `partition` = rux-ml's val-less `[m9]`
  time-block partition, row-identical at `814cc15` (program PR-082: 352 checks, 0 mismatches, 32 / 32 refusals on a
  committed fixture, `tests/fixtures/pr082/`, generated by rux-ml's own `make_splits`); `partition_at_cut` (program
  PR-102) = the same rule with the cut given, no prefix, refusing `partition_embargo_purged_train` only when valid rows
  lie below the cut but the embargo keeps none — an empty side below the cut is returned empty, not refused.
- **Executed check (proven, 2026-10-22, synthetic data only).** rux-ml's existing `PartitionPlan.assign` at `d51e89f`,
  with `val_first_stamp = test_first_stamp = cut` and `embargo = h · 86,400,000`, against the harness's
  `partition_at_cut` loaded from the harness checkout: **400 sets, 0 mismatches** in train / test positions —
  607 rows tied on the cut (all test), 595 rows at exactly `cut − h·DAY` (all purged), h ∈ {0, 1, 7, 14}, day-stamped
  and intra-day stamps, ~15 % null targets; on the 31 sets the harness refuses, rux-ml's train is empty too.
  Script: the program track's scratchpad `datecut_identity.py` (not committed; the PR's tests replace it).

**Assumptions at PR draft time** (the program's ROADMAP row, 2026-10-21):
- rux-ml needs a new split mode with its own rule; the seam is `make_splits`.
- The cut is "given as a date".

**Stale assumptions:**
1. **No new rule is needed — only a new source for the cut.** The `[m9]` fold rule at a given cut is already
   row-identical to the harness (executed check above). The PR is a config field + `_plan_time` taking the cut from it
   + validators + the record + tests; `_time_folds`, the per-day batch path, promote, the OOS export mask and the
   leakage audit stay as they are.
2. **The seam is `partitions.plan_partitions`, not `make_splits`** — since PR-054 `make_splits`' `[m9]` branch
   delegates to it; the legacy in-memory `temporal_train_val_test_split` (non-`[m9]` problems) is not on v0.7's path.
3. **"A date" needs a unit contract.** rux-ml's time column has no declared unit (the embargo and `h_max_ms` are "the
   column's own units — epoch ms for an Int64 stamp"); a date becomes a stamp only under that contract
   (2025-11-26 → 1,764,115,200,000 ms at UTC midnight).

**New constraints** (from prior PRs — the prior-art audit, `git log` on `partitions.py`, `splits.py`,
`config/data.py`, `config/root.py`: PR-024, PR-041, PR-042 ×2, PR-043, PR-046, PR-054):
- Ties go to the later partition (PR-054); the strict `<` in `s + emb < cut` (a train row whose label ends exactly at
  the cut is purged) is the shared rule — row identity requires keeping it.
- `split_embargo ≥ m9.h_max_ms` stays (PR-042); for identity with the harness the cell's config must set
  `split_embargo = h · 86,400,000` exactly (h 7 → 604,800,000; h 14 → 1,209,600,000) — the harness's config to write.
- Identity also needs `m9.row_filter_non_null` = exactly the target column (the harness's `target_valid` is the
  target alone) and the set's time column = the harness's event stamp `(day + 1)·DAY_MS` — both the harness's side.
- A new optional `data` field joins `_HASH_OPTIONAL_FIELDS` (PR-042), with a test that existing hash pins hold.
- The plan-over-day-batches = plan-over-frame equivalence test (PR-054) extends to the new mode.
- Val-less refusals (PR-060) are unchanged and apply.
- No change to `docs/0.3/DESIGN-log.md` decisions (nothing there on a date cut; no drift). Staleness threshold 60 days
  (`docs/CONSTRAINTS.md`): this assessment is good to 2026-12-21.

**Performance budget (the program's Build to Budget, stated before the build).** The per-row path (`assign`) is
unchanged — vectorised NumPy, no per-row Python. `_plan_time` in date mode drops the sorted-position lookup and keeps
one key-column pass (the "rows below the cut" check for the refusal). Budget: plan + assign ≤ the fraction mode's on
the same set, and ≤ 1 s for a 1,000,000-row synthetic set on the Mac (v0.7's largest cell is ≈ 27–28 k rows);
closing measurement in the PR, and on the first real cell set at program PR-110's sample.

**Phase 1 gate — decisions for the operator (each a lean with basis and confidence):**
1. **Shape — an optional field on `time_ordered`, not a new `split_kind`** (lean). Basis: every `time_ordered` branch
   (embargo floor, leakage audit, scorer seed, OOS mask, learning curve, runs listing) then works unchanged; a new kind
   re-opens ~8 sites. Confidence high (proven by reading + the executed check).
2. **Config type — a TOML date, `data.split_cut_date = 2025-11-26`** (lean), converted to UTC-midnight epoch ms for an
   Int64 epoch-ms time column; the split record writes the date and the stamp. Alternative: an int `split_cut_ms` like
   `split_embargo`. Basis: the operator's rule is a date; a hand-typed 13-digit stamp is the error-prone form. Confidence
   medium (rux-ml cannot check the column's unit — the same contract `h_max_ms` already relies on).
3. **Val-less only** (lean): the date mode requires `split_ratios.val == 0` and refuses `val > 0` by name; a val date
   is added when a consumer needs one. Basis: the harness partition and v0.7's probe are val-less. Confidence medium —
   **open:** program PR-111's training must also be val-less (early stopping off) for this to cover it.
4. **The ratios in date mode** (lean): still validated (sum to 1), not used for the cut; the record says
   `cuts.source = "date"`. Confidence medium-high.
5. **Refusals** (lean): the purged-train refusal exactly as today ("split embargo … purged every train row") when rows
   lie below the cut but none survive; plus an empty-train refusal by name when no row lies below the cut (the harness
   returns that case empty for its dry count; a fit cannot run on it). Confidence medium-high.
6. **Scope** (lean): `[m9]` problems only; refused with a non-`time_ordered` kind, without `[m9]`, and with
   `train_prefix_frac` (the harness's `partition_at_cut` has no prefix; a later consumer lifts it with its own test).
   Confidence medium.
7. **Row identity with the harness — two sides** (lean): rux-ml's tests (date mode = fraction mode where the cut is the
   fraction-chosen `test_first_stamp`; batches = frame; ties, the `cut − h·DAY` row, every refusal; hash pins) plus a
   committed fixture in PR-082's pattern — made-up parquet tables + rux-ml's date-mode output at the merged commit as
   `partition_at_cut_expected.json` — for a harness test to replay through `partition_at_cut`. The harness test is a
   harness change: **its owner is the program lead's call** (lean: program PR-110's build, which needs it first).
   Confidence high on the method (precedent), open on the owner.

**Size (re-floored):** Tier-1; Phases 2–4 light in the main loop ≈ 0.05–0.1M; build one agent ≈ 0.2M (68k + 3k × ~35
calls + a ~30k pack); review ≈ 0.1M — **≈ 0.35–0.5M, $0, no desktop run**; plus ≈ 0.1M for the harness-side fixture
replay in whichever harness PR owns it.

**Exit:** stale assumptions flagged, new constraints documented; the premise holds (smaller than drafted). HALT at the
Phase 1 gate.

---

## Scope

- `data.split_cut_date` (per gate 2) on `DataConfig`, in `_HASH_OPTIONAL_FIELDS["data"]`; validators per gates 3, 5, 6.
- `partitions._plan_time`: with the field set, `val_first_stamp = test_first_stamp = ` the cut stamp; the refusals.
- `split_record` / `split_definition`: the cut's source and date.
- Tests (gate 7, rux-ml side) and the committed fixture for the harness replay.
- Docs: `docs/ARCHITECTURE.md` (the split section, ~lines 232–255), `CHANGELOG.md` `[Unreleased]`, `configs/` example
  if one is added.

## Dependencies

None in rux-ml (base `d51e89f`). Consumers: program PR-110 (the probe) and PR-111 (the training), which land after it.

## Architecture section implemented

`docs/ARCHITECTURE.md` — the one-off split / `[m9]` partition rule (`time_ordered`, `split_embargo`, the partition plan).

## Verification criteria

- [ ] Date mode = fraction mode, row for row, wherever the date equals the fraction-chosen `test_first_stamp`.
- [ ] Rows tied on the cut are test; a row at `cut − split_embargo` is purged; null targets are in no partition.
- [ ] The per-day batch plan equals the frame plan in date mode.
- [ ] Each refusal fires by name (gates 3, 5, 6); existing `data_cfg_hash` / `root_cfg_hash` pins are unchanged.
- [ ] The committed fixture reproduces through the harness's `partition_at_cut` (harness-side test, its owner's PR).
- [ ] Performance budget met and measured (above).

## Research backing

This file's State Assessment; program `prs/PR-102-stage-a-dry-count-new-extract.md` (gap A), program
`docs/0.7/ROADMAP.md` (the rux-ml date-cut row), harness `b440fb0` `src/rumpy_harness/probe/partition.py` and
`tests/test_pr102_stage_a_calendar.py`, `tests/test_pr082_mirror_partition.py`.

## Notes

- Out of scope: a val date; a prefix with a date cut; the legacy non-`[m9]` `temporal_train_val_test_split`; any
  harness change.
- PR-064 (ExtMem eval-set) is independent: a val-less fit passes no eval set.
